#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

/**
 * RESONATE Bridge — Plugin Editor (UI).
 * Minimal UI showing connection status and live DAW transport info.
 */
class ResonateBridgeEditor : public juce::AudioProcessorEditor,
                              public juce::Timer
{
public:
    explicit ResonateBridgeEditor(ResonateBridgeProcessor&);
    ~ResonateBridgeEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    ResonateBridgeProcessor& processorRef;

    juce::Label titleLabel;
    juce::Label statusLabel;
    juce::Label bpmLabel;
    juce::Label timeSigLabel;
    juce::Label transportLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ResonateBridgeEditor)
};
