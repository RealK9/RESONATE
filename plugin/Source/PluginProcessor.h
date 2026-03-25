#pragma once
#include <JuceHeader.h>
#include "BridgeClient.h"
#include <atomic>

/**
 * RESONATE Bridge — Audio Processor.
 * Pass-through plugin that reads DAW transport and sends to RESONATE app.
 * Captures audio ring buffer for "Analyze from DAW" feature.
 * No audio modification — pure bridge functionality.
 *
 * THREAD SAFETY:
 *   processBlock() runs on the real-time audio thread and must NEVER block.
 *   Audio capture uses a lock-free single-producer (audio thread) /
 *   single-consumer (capture thread) ring buffer. The consumer copies
 *   under a separate mutex that processBlock never touches.
 */
class ResonateBridgeProcessor : public juce::AudioProcessor
{
public:
    ResonateBridgeProcessor();
    ~ResonateBridgeProcessor() override;

    // AudioProcessor interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "RESONATE Bridge"; }

    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    // Bridge access
    BridgeClient& getBridge() { return bridge; }
    const BridgeClient::TransportState& getLastTransport() const { return lastTransport; }

    // Audio capture for analysis — called from bridge thread, NOT audio thread
    void getCapturedAudio(juce::AudioBuffer<float>& output, double& outSampleRate);
    bool hasCapturedAudio() const { return samplesWritten.load(std::memory_order_acquire) > 0; }

private:
    BridgeClient bridge;
    BridgeClient::TransportState lastTransport;
    int blockCounter = 0;

    // ── Lock-free ring buffer for audio capture ──
    // Only the audio thread writes; only the capture thread reads.
    static constexpr int CAPTURE_SECONDS = 30;
    juce::AudioBuffer<float> captureBuffer;       // fixed size after prepareToPlay
    std::atomic<int> writePos { 0 };              // written by audio thread only
    std::atomic<int64_t> samplesWritten { 0 };    // monotonic counter
    double currentSampleRate = 44100.0;

    // Snapshot mutex — taken by getCapturedAudio (bridge thread).
    // processBlock NEVER touches this.
    std::mutex snapshotMutex;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ResonateBridgeProcessor)
};
