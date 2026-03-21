#include "PluginProcessor.h"
#include "PluginEditor.h"

ResonateBridgeProcessor::ResonateBridgeProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    // When RESONATE requests audio capture, send the ring buffer
    bridge.onCaptureRequest = [this]()
    {
        juce::AudioBuffer<float> captured;
        double sampleRate;
        getCapturedAudio(captured, sampleRate);

        if (captured.getNumSamples() == 0)
        {
            bridge.sendJson("{\"type\":\"audioCapture\",\"error\":\"no audio captured\"}\n");
            return;
        }

        // Write WAV to memory
        juce::MemoryOutputStream wavStream;
        {
            auto writer = std::unique_ptr<juce::AudioFormatWriter>(
                juce::WavAudioFormat().createWriterFor(
                    &wavStream, sampleRate, captured.getNumChannels(),
                    16, {}, 0));
            if (writer)
            {
                writer->writeFromAudioSampleBuffer(captured, 0, captured.getNumSamples());
                writer->flush();
            }
        }

        // Send as: header line + raw bytes
        auto headerJson = juce::String::formatted(
            "{\"type\":\"audioCapture\",\"sampleRate\":%.0f,\"channels\":%d,"
            "\"samples\":%d,\"wavSize\":%d}\n",
            sampleRate,
            captured.getNumChannels(),
            captured.getNumSamples(),
            (int)wavStream.getDataSize());

        bridge.sendJson(headerJson);
        bridge.sendRawBytes(wavStream.getData(), (int)wavStream.getDataSize());
    };
}

ResonateBridgeProcessor::~ResonateBridgeProcessor() = default;

void ResonateBridgeProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    currentSampleRate = sampleRate;
    const int totalSamples = static_cast<int>(sampleRate * CAPTURE_SECONDS);
    captureBuffer.setSize(2, totalSamples);
    captureBuffer.clear();
    captureWritePos = 0;
}

void ResonateBridgeProcessor::releaseResources()
{
    // No-op
}

void ResonateBridgeProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midiMessages*/)
{
    // Pure pass-through — don't touch the audio
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

            // Throttle: only send every ~10 blocks (~100ms at 44100/512)
            if (++blockCounter >= 10)
            {
                bridge.updateTransport(state);
                blockCounter = 0;
            }
        }
    }

    // Capture audio into ring buffer (non-blocking, lock-free write)
    {
        const juce::ScopedLock sl(captureLock);
        const int numSamples = buffer.getNumSamples();
        const int bufferSize = captureBuffer.getNumSamples();

        if (bufferSize > 0)
        {
            const int numChannels = juce::jmin(buffer.getNumChannels(), 2);
            int writePos = captureWritePos;

            for (int ch = 0; ch < numChannels; ++ch)
            {
                const float* src = buffer.getReadPointer(ch);
                float* dst = captureBuffer.getWritePointer(ch);

                if (writePos + numSamples <= bufferSize)
                {
                    std::memcpy(dst + writePos, src, sizeof(float) * numSamples);
                }
                else
                {
                    // Wrap around
                    const int firstPart = bufferSize - writePos;
                    std::memcpy(dst + writePos, src, sizeof(float) * firstPart);
                    std::memcpy(dst, src + firstPart, sizeof(float) * (numSamples - firstPart));
                }
            }

            captureWritePos = (writePos + numSamples) % bufferSize;
        }
    }

    // Ensure output matches input (pass-through)
    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i)
        buffer.clear(i, 0, buffer.getNumSamples());
}

void ResonateBridgeProcessor::getCapturedAudio(juce::AudioBuffer<float>& output, double& outSampleRate)
{
    const juce::ScopedLock sl(captureLock);
    outSampleRate = currentSampleRate;
    const int bufferSize = captureBuffer.getNumSamples();

    if (bufferSize == 0 || captureWritePos == 0)
    {
        output.setSize(2, 0);
        return;
    }

    // Copy the ring buffer in chronological order
    const int totalSamples = juce::jmin(captureWritePos, bufferSize);
    output.setSize(2, totalSamples);

    if (captureWritePos <= bufferSize)
    {
        // Buffer hasn't wrapped yet — simple copy
        for (int ch = 0; ch < 2; ++ch)
            std::memcpy(output.getWritePointer(ch),
                        captureBuffer.getReadPointer(ch),
                        sizeof(float) * totalSamples);
    }
    else
    {
        // Buffer has wrapped — copy from writePos to end, then start to writePos
        const int startPos = captureWritePos % bufferSize;
        const int firstPart = bufferSize - startPos;
        for (int ch = 0; ch < 2; ++ch)
        {
            std::memcpy(output.getWritePointer(ch),
                        captureBuffer.getReadPointer(ch) + startPos,
                        sizeof(float) * firstPart);
            std::memcpy(output.getWritePointer(ch) + firstPart,
                        captureBuffer.getReadPointer(ch),
                        sizeof(float) * startPos);
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
