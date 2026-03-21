#include "PluginEditor.h"

ResonateBridgeEditor::ResonateBridgeEditor(ResonateBridgeProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    setSize(320, 200);

    // Title
    titleLabel.setText("RESONATE Bridge", juce::dontSendNotification);
    titleLabel.setFont(juce::FontOptions(20.0f, juce::Font::bold));
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setColour(juce::Label::textColourId, juce::Colour(0xFFE8E8EC));
    addAndMakeVisible(titleLabel);

    // Connection status
    statusLabel.setFont(juce::FontOptions(12.0f));
    statusLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(statusLabel);

    // BPM
    bpmLabel.setFont(juce::FontOptions(28.0f, juce::Font::bold));
    bpmLabel.setJustificationType(juce::Justification::centred);
    bpmLabel.setColour(juce::Label::textColourId, juce::Colour(0xFFE8E8EC));
    addAndMakeVisible(bpmLabel);

    // Time signature
    timeSigLabel.setFont(juce::FontOptions(14.0f));
    timeSigLabel.setJustificationType(juce::Justification::centred);
    timeSigLabel.setColour(juce::Label::textColourId, juce::Colour(0xFF8888AA));
    addAndMakeVisible(timeSigLabel);

    // Transport state
    transportLabel.setFont(juce::FontOptions(11.0f));
    transportLabel.setJustificationType(juce::Justification::centred);
    transportLabel.setColour(juce::Label::textColourId, juce::Colour(0xFF666680));
    addAndMakeVisible(transportLabel);

    startTimerHz(10);  // 10 FPS UI refresh
}

ResonateBridgeEditor::~ResonateBridgeEditor()
{
    stopTimer();
}

void ResonateBridgeEditor::paint(juce::Graphics& g)
{
    // Dark background matching RESONATE's theme
    g.fillAll(juce::Colour(0xFF0D0D12));

    // Subtle border
    g.setColour(juce::Colour(0xFF2A2A35));
    g.drawRect(getLocalBounds(), 1);

    // Connection indicator dot
    auto isConnected = processorRef.getBridge().isConnected();
    auto dotColour = isConnected ? juce::Colour(0xFF22C55E) : juce::Colour(0xFF666680);
    g.setColour(dotColour);
    g.fillEllipse(148.0f, 56.0f, 8.0f, 8.0f);
}

void ResonateBridgeEditor::resized()
{
    auto area = getLocalBounds();

    titleLabel.setBounds(area.removeFromTop(40).reduced(10, 8));

    auto statusArea = area.removeFromTop(24);
    statusLabel.setBounds(statusArea.reduced(10, 0));

    bpmLabel.setBounds(area.removeFromTop(44).reduced(10, 4));
    timeSigLabel.setBounds(area.removeFromTop(24).reduced(10, 0));
    transportLabel.setBounds(area.removeFromTop(24).reduced(10, 0));
}

void ResonateBridgeEditor::timerCallback()
{
    auto& bridge = processorRef.getBridge();
    auto& transport = processorRef.getLastTransport();

    // Update status
    auto isConnected = bridge.isConnected();
    statusLabel.setText(bridge.getStatus(), juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId,
                          isConnected ? juce::Colour(0xFF22C55E) : juce::Colour(0xFF888899));

    // Update BPM
    bpmLabel.setText(juce::String(transport.bpm, 1) + " BPM", juce::dontSendNotification);

    // Update time sig
    timeSigLabel.setText(juce::String(transport.timeSigNum) + "/" + juce::String(transport.timeSigDen),
                         juce::dontSendNotification);

    // Update transport
    auto mins = (int)(transport.positionInSeconds / 60.0);
    auto secs = (int)transport.positionInSeconds % 60;
    transportLabel.setText(
        (transport.isPlaying ? "Playing" : "Stopped") +
        juce::String(" — ") + juce::String(mins) + ":" +
        juce::String(secs).paddedLeft('0', 2),
        juce::dontSendNotification
    );

    repaint();
}
