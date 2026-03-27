#pragma once
#include <JuceHeader.h>

/**
 * RESONATE Bridge — TCP client.
 * Sends DAW transport state (BPM, time sig, position) to the RESONATE app.
 * Receives key change and capture requests from RESONATE.
 *
 * Thread-safe design:
 *   - `socketLock` protects all socket access (timer thread, bg thread, caller thread)
 *   - `stateLock` protects transport state reads/writes
 *   - `connected` atomic for fast non-blocking checks
 */
class BridgeClient : public juce::Thread,
                     public juce::Timer
{
public:
    BridgeClient();
    ~BridgeClient() override;

    // Transport state from DAW
    struct TransportState
    {
        double bpm = 120.0;
        int timeSigNum = 4;
        int timeSigDen = 4;
        bool isPlaying = false;
        double positionInSeconds = 0.0;
    };

    void updateTransport(const TransportState& state);
    bool isConnected() const { return connected.load(); }
    juce::String getStatus() const;

    // Callbacks for messages received from RESONATE
    std::function<void(const juce::String&)> onKeyChange;
    std::function<void()> onCaptureRequest;

    // Thread-safe send methods (used by PluginProcessor for audio capture)
    void sendJson(const juce::String& json);
    void sendRawBytes(const void* data, int size);

private:
    void run() override;           // Thread: connection + read loop
    void timerCallback() override;  // Timer: send throttled updates

    void connectToServer();
    bool readResponse(juce::String& out);
    void disconnect();              // Thread-safe disconnect helper
    bool parseJsonField(const juce::String& json, const juce::String& key, juce::String& value);

    std::atomic<bool> connected { false };
    TransportState latestState;
    juce::CriticalSection stateLock;
    juce::CriticalSection socketLock;   // Protects ALL socket access
    std::unique_ptr<juce::StreamingSocket> socket;

    static constexpr int PORT = 9876;
    static constexpr int SEND_INTERVAL_MS = 100;
    static constexpr int RECONNECT_DELAY_MS = 2000;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BridgeClient)
};
