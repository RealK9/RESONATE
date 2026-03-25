#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <thread>
#include <mutex>

ResonateBridgeProcessor::ResonateBridgeProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    // When RESONATE requests audio capture:
    // 1. Snapshot the ring buffer (under snapshotMutex — NOT captureLock)
    // 2. Dispatch WAV encode + socket send to a detached thread
    // processBlock is NEVER blocked by any of this.
    bridge.onCaptureRequest = [this]()
    {
        juce::AudioBuffer<float> snapshot;
        double sr;
        getCapturedAudio(snapshot, sr);

        if (snapshot.getNumSamples() == 0)
        {
            bridge.sendJson("{\"type\":\"audioCapture\",\"error\":\"no audio captured\"}\n");
            return;
        }

        // Heap-allocate the snapshot so the detached thread owns it
        auto* heapBuf = new juce::AudioBuffer<float>(std::move(snapshot));
        auto sampleRate = sr;
        auto* bridgePtr = &bridge;

        std::thread([heapBuf, sampleRate, bridgePtr]()
        {
            std::unique_ptr<juce::AudioBuffer<float>> buf(heapBuf);

            juce::MemoryOutputStream wavStream;
            {
                auto writer = std::unique_ptr<juce::AudioFormatWriter>(
                    juce::WavAudioFormat().createWriterFor(
                        &wavStream, sampleRate, buf->getNumChannels(),
                        16, {}, 0));
                if (writer)
                {
                    writer->writeFromAudioSampleBuffer(*buf, 0, buf->getNumSamples());
                    writer->flush();
                }
            }

            auto headerJson = juce::String::formatted(
                "{\"type\":\"audioCapture\",\"sampleRate\":%.0f,\"channels\":%d,"
                "\"samples\":%d,\"wavSize\":%d}\n",
                sampleRate,
                buf->getNumChannels(),
                buf->getNumSamples(),
                (int)wavStream.getDataSize());

            bridgePtr->sendJson(headerJson);
            bridgePtr->sendRawBytes(wavStream.getData(), (int)wavStream.getDataSize());
        }).detach();
    };
}

ResonateBridgeProcessor::~ResonateBridgeProcessor() = default;

void ResonateBridgeProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    currentSampleRate = sampleRate;
    const int totalSamples = static_cast<int>(sampleRate * CAPTURE_SECONDS);
    captureBuffer.setSize(2, totalSamples);
    captureBuffer.clear();
    writePos.store(0, std::memory_order_relaxed);
    samplesWritten.store(0, std::memory_order_release);
}

void ResonateBridgeProcessor::releaseResources()
{
    // No-op
}

void ResonateBridgeProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer& /*midiMessages*/)
{
    // ── Pure pass-through — never modify audio ──
    juce::ScopedNoDenormals noDenormals;

    // Read transport from DAW host
    if (auto* playHead = getPlayHead())
    {
        if (auto posInfo = playHead->getPosition())
        {
            BridgeClient::TransportState state;

            if (auto bpm = posInfo->getBpm())
                state.bpm = *bpm;

            if (auto timeSig = posInfo->getTimeSignature())
            {
                state.timeSigNum = timeSig->numerator;
                state.timeSigDen = timeSig->denominator;
            }

            state.isPlaying = posInfo->getIsPlaying();

            if (auto timeInSeconds = posInfo->getTimeInSeconds())
                state.positionInSeconds = *timeInSeconds;

            lastTransport = state;

            // Throttle: only push to bridge every ~10 blocks (~100ms at 44100/512)
            if (++blockCounter >= 10)
            {
                bridge.updateTransport(state);
                blockCounter = 0;
            }
        }
    }

    // ── Lock-free ring buffer write ──
    // Only this thread writes writePos/captureBuffer — no lock needed.
    const int numSamples = buffer.getNumSamples();
    const int bufferSize = captureBuffer.getNumSamples();

    if (bufferSize > 0 && numSamples > 0)
    {
        const int numChannels = juce::jmin(buffer.getNumChannels(), 2);
        int wp = writePos.load(std::memory_order_relaxed);

        for (int ch = 0; ch < numChannels; ++ch)
        {
            const float* src = buffer.getReadPointer(ch);
            float* dst = captureBuffer.getWritePointer(ch);

            if (wp + numSamples <= bufferSize)
            {
                std::memcpy(dst + wp, src, sizeof(float) * numSamples);
            }
            else
            {
                const int firstPart = bufferSize - wp;
                std::memcpy(dst + wp, src, sizeof(float) * firstPart);
                std::memcpy(dst, src + firstPart, sizeof(float) * (numSamples - firstPart));
            }
        }

        int newPos = (wp + numSamples) % bufferSize;
        writePos.store(newPos, std::memory_order_release);
        samplesWritten.fetch_add(numSamples, std::memory_order_release);
    }

    // Clear any extra output channels
    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i)
        buffer.clear(i, 0, buffer.getNumSamples());
}

void ResonateBridgeProcessor::getCapturedAudio(juce::AudioBuffer<float>& output,
                                                double& outSampleRate)
{
    // This runs on the bridge thread — take snapshotMutex (never held by audio thread)
    std::lock_guard<std::mutex> lock(snapshotMutex);

    outSampleRate = currentSampleRate;
    const int bufferSize = captureBuffer.getNumSamples();

    if (bufferSize == 0)
    {
        output.setSize(2, 0);
        return;
    }

    // Read atomics with acquire to see latest audio thread writes
    const int64_t totalWritten = samplesWritten.load(std::memory_order_acquire);
    const int wp = writePos.load(std::memory_order_acquire);

    if (totalWritten == 0)
    {
        output.setSize(2, 0);
        return;
    }

    // How much valid data do we have?
    const bool wrapped = totalWritten >= bufferSize;

    if (!wrapped)
    {
        // Haven't filled the buffer yet — copy 0..wp
        const int validSamples = static_cast<int>(juce::jmin((int64_t)bufferSize, totalWritten));
        output.setSize(2, validSamples);
        for (int ch = 0; ch < 2; ++ch)
            std::memcpy(output.getWritePointer(ch),
                        captureBuffer.getReadPointer(ch),
                        sizeof(float) * validSamples);
    }
    else
    {
        // Buffer has wrapped — full bufferSize of audio available
        // Read chronologically: wp..end, then 0..wp
        output.setSize(2, bufferSize);
        const int firstPart = bufferSize - wp;
        for (int ch = 0; ch < 2; ++ch)
        {
            std::memcpy(output.getWritePointer(ch),
                        captureBuffer.getReadPointer(ch) + wp,
                        sizeof(float) * firstPart);
            std::memcpy(output.getWritePointer(ch) + firstPart,
                        captureBuffer.getReadPointer(ch),
                        sizeof(float) * wp);
        }
    }
}

juce::AudioProcessorEditor* ResonateBridgeProcessor::createEditor()
{
    return new ResonateBridgeEditor(*this);
}

// Entry point — tells the DAW about this plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ResonateBridgeProcessor();
}
