#include "BridgeClient.h"

BridgeClient::BridgeClient()
    : Thread("ResonateBridge")
{
    startThread();
    startTimer(SEND_INTERVAL_MS);
}

BridgeClient::~BridgeClient()
{
    stopTimer();
    signalThreadShouldExit();
    disconnect();
    stopThread(2000);
}

void BridgeClient::updateTransport(const TransportState& state)
{
    const juce::ScopedLock sl(stateLock);
    latestState = state;
}

juce::String BridgeClient::getStatus() const
{
    if (connected.load())
        return "Connected to RESONATE";
    return "Waiting for RESONATE...";
}

// ---------------------------------------------------------------------------
// Thread-safe disconnect
// ---------------------------------------------------------------------------
void BridgeClient::disconnect()
{
    const juce::ScopedLock sl(socketLock);
    connected.store(false);
    if (socket != nullptr)
    {
        socket->close();
        socket.reset();
    }
}

// ---------------------------------------------------------------------------
// Safe JSON field parser (replaces raw indexOf hacking)
// ---------------------------------------------------------------------------
bool BridgeClient::parseJsonField(const juce::String& json,
                                   const juce::String& key,
                                   juce::String& value)
{
    // Look for "key":"value" pattern
    auto pattern = "\"" + key + "\"";
    auto idx = json.indexOf(pattern);
    if (idx < 0)
        return false;

    // Find the colon after the key
    auto colonIdx = json.indexOf(idx + pattern.length(), ":");
    if (colonIdx < 0)
        return false;

    // Find opening quote of value
    auto openQuote = json.indexOf(colonIdx + 1, "\"");
    if (openQuote < 0)
        return false;

    // Find closing quote of value
    auto closeQuote = json.indexOf(openQuote + 1, "\"");
    if (closeQuote < 0 || closeQuote <= openQuote)
        return false;

    value = json.substring(openQuote + 1, closeQuote);
    return value.isNotEmpty();
}

// ---------------------------------------------------------------------------
// Background thread: connect + read loop
// ---------------------------------------------------------------------------
void BridgeClient::run()
{
    while (!threadShouldExit())
    {
        if (!connected.load())
        {
            connectToServer();
        }

        if (connected.load())
        {
            juce::String response;
            if (readResponse(response))
            {
                if (response.contains("keyChange"))
                {
                    juce::String key;
                    if (parseJsonField(response, "key", key))
                    {
                        if (onKeyChange)
                            onKeyChange(key);
                    }
                }
                else if (response.contains("captureAudio"))
                {
                    if (onCaptureRequest)
                    {
                        // Invoke on a separate thread to avoid blocking the read loop
                        juce::MessageManager::callAsync([this]()
                        {
                            if (onCaptureRequest)
                                onCaptureRequest();
                        });
                    }
                }
            }
        }

        wait(50);
    }
}

// ---------------------------------------------------------------------------
// Connect to RESONATE backend
// ---------------------------------------------------------------------------
void BridgeClient::connectToServer()
{
    auto newSocket = std::make_unique<juce::StreamingSocket>();

    if (newSocket->connect("127.0.0.1", PORT, 1000))
    {
        {
            const juce::ScopedLock sl(socketLock);
            socket = std::move(newSocket);
        }
        connected.store(true);
        DBG("ResonateBridge: Connected to RESONATE on port " + juce::String(PORT));
    }
    else
    {
        // Wait before retry (interruptible)
        for (int i = 0; i < RECONNECT_DELAY_MS / 50 && !threadShouldExit(); ++i)
            wait(50);
    }
}

// ---------------------------------------------------------------------------
// Timer callback: send throttled transport updates (runs on message thread)
// ---------------------------------------------------------------------------
void BridgeClient::timerCallback()
{
    if (!connected.load())
        return;

    TransportState ts;
    {
        const juce::ScopedLock sl(stateLock);
        ts = latestState;
    }

    auto json = juce::String::formatted(
        "{\"type\":\"transport\",\"bpm\":%.2f,\"timeSigNum\":%d,\"timeSigDen\":%d,"
        "\"playing\":%s,\"position\":%.4f}\n",
        ts.bpm,
        ts.timeSigNum,
        ts.timeSigDen,
        ts.isPlaying ? "true" : "false",
        ts.positionInSeconds
    );

    sendJson(json);
}

// ---------------------------------------------------------------------------
// Thread-safe send (protected by socketLock)
// ---------------------------------------------------------------------------
void BridgeClient::sendJson(const juce::String& json)
{
    const juce::ScopedLock sl(socketLock);

    if (socket == nullptr || !socket->isConnected())
    {
        connected.store(false);
        return;
    }

    auto data = json.toRawUTF8();
    auto len = (int)json.getNumBytesAsUTF8();
    int totalWritten = 0;

    // Retry partial writes
    while (totalWritten < len)
    {
        int written = socket->write(data + totalWritten, len - totalWritten);
        if (written <= 0)
        {
            DBG("ResonateBridge: Write failed, disconnecting");
            connected.store(false);
            socket->close();
            socket.reset();
            return;
        }
        totalWritten += written;
    }
}

void BridgeClient::sendRawBytes(const void* data, int size)
{
    const juce::ScopedLock sl(socketLock);

    if (socket == nullptr || !socket->isConnected())
    {
        connected.store(false);
        return;
    }

    const auto* bytes = static_cast<const char*>(data);
    int totalWritten = 0;

    while (totalWritten < size)
    {
        int written = socket->write(bytes + totalWritten, size - totalWritten);
        if (written <= 0)
        {
            DBG("ResonateBridge: Raw write failed, disconnecting");
            connected.store(false);
            socket->close();
            socket.reset();
            return;
        }
        totalWritten += written;
    }
}

// ---------------------------------------------------------------------------
// Non-blocking read (protected by socketLock)
// ---------------------------------------------------------------------------
bool BridgeClient::readResponse(juce::String& out)
{
    const juce::ScopedLock sl(socketLock);

    if (socket == nullptr || !socket->isConnected())
    {
        connected.store(false);
        return false;
    }

    if (!socket->waitUntilReady(true, 0))
        return false;

    char buffer[2048];
    int bytesRead = socket->read(buffer, sizeof(buffer) - 1, false);
    if (bytesRead > 0)
    {
        buffer[bytesRead] = '\0';
        out = juce::String::fromUTF8(buffer, bytesRead);
        return true;
    }
    else if (bytesRead < 0)
    {
        DBG("ResonateBridge: Read error, disconnecting");
        connected.store(false);
        socket->close();
        socket.reset();
    }

    return false;
}
