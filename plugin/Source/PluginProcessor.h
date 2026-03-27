#pragma once
#include <JuceHeader.h>
#include "BridgeClient.h"

/**
 * RESONATE Bridge — Audio Processor.
 * Pass-through plugin that reads DAW transport and sends to RESONATE app.
 * Captures audio ring buffer for "Analyze from DAW" feature.
 * No audio modification — pure bridge functionality.
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

    // Audio capture for analysis
    void getCapturedAudio(juce::AudioBuffer<float>& output, double& outSampleRate);
    bool hasCapturedAudio() const { return captureWritePos > 0; }

private:
    BridgeClient bridge;
    BridgeClient::TransportState lastTransport;
    int blockCounter = 0;

    // Ring buffer for audio capture (30 seconds stereo)
    static constexpr int CAPTURE_SECONDS = 30;
    juce::AudioBuffer<float> captureBuffer;
    int captureWritePos = 0;
    bool captureWrapped = false;   // True once the ring buffer has wrapped at least once
    double currentSampleRate = 44100.0;
    juce::CriticalSection captureLock;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ResonateBridgeProcessor)
};
