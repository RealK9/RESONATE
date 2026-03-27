#include "BridgeClient.h"

BridgeClient::BridgeClient()
    : Thread("ResonateBridge-TCP")
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
    std::lock_guard<std::mutex> lock(socketMutex);
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

        // Check for incoming messages (key changes, capture requests)
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
                    // Fire capture on a detached thread to avoid blocking the read loop
                    if (onCaptureRequest)
                    {
                        auto cb = onCaptureRequest;
                        std::thread([cb]() { cb(); }).detach();
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
            std::lock_guard<std::mutex> lock(socketMutex);
            socket = std::move(newSocket);
        }
        connected.store(true);
        DBG("ResonateBridge: Connected to RESONATE on port " + juce::String(PORT));
    }
    else
    {
        // Wait before retry — but check for exit periodically
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
// Thread-safe send (protected by socketMutex)
// ---------------------------------------------------------------------------
void BridgeClient::sendJson(const juce::String& json)
{
    std::lock_guard<std::mutex> lock(socketMutex);

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
            connected.store(false);
            socket->close();
            socket.reset();
            DBG("ResonateBridge: Connection lost (write failed)");
            return;
        }
        totalWritten += written;
    }
}

void BridgeClient::sendRawBytes(const void* data, int size)
{
    std::lock_guard<std::mutex> lock(socketMutex);

    if (socket == nullptr || !socket->isConnected())
    {
        connected.store(false);
        return;
    }

    // Send in chunks to avoid blocking too long
    const int CHUNK = 32768;
    const char* ptr = static_cast<const char*>(data);
    int remaining = size;

    while (remaining > 0)
    {
        int toSend = juce::jmin(remaining, CHUNK);
        int written = socket->write(ptr, toSend);
        if (written <= 0)
        {
            connected.store(false);
            socket->close();
            socket.reset();
            DBG("ResonateBridge: Connection lost (raw write failed)");
            return;
        }
        ptr += written;
        remaining -= written;
    }
}

// ---------------------------------------------------------------------------
// Non-blocking read (protected by socketMutex)
// ---------------------------------------------------------------------------
bool BridgeClient::readResponse(juce::String& out)
{
    std::lock_guard<std::mutex> lock(socketMutex);

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
