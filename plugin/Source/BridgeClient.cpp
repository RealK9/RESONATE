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

    {
        std::lock_guard<std::mutex> lock(socketMutex);
        if (socket != nullptr)
            socket->close();
    }

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
                    auto keyStart = response.indexOf("\"key\"") + 6;
                    auto keyEnd = response.indexOf(keyStart + 1, "\"");
                    if (keyStart > 5 && keyEnd > keyStart)
                    {
                        auto key = response.substring(keyStart + 1, keyEnd);
                        if (onKeyChange)
                            onKeyChange(key);
                    }
                }
                else if (response.contains("captureAudio"))
                {
                    // Fire capture on a separate thread to avoid blocking the read loop
                    if (onCaptureRequest)
                    {
                        auto cb = onCaptureRequest;
                        std::thread([cb]() { cb(); }).detach();
                    }
                }
            }
        }

        wait(50);  // 20Hz check rate
    }
}

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

void BridgeClient::timerCallback()
{
    if (!connected.load())
        return;

    TransportState state;
    {
        const juce::ScopedLock sl(stateLock);
        state = latestState;
    }

    auto json = juce::String::formatted(
        "{\"type\":\"transport\",\"bpm\":%.2f,\"timeSigNum\":%d,\"timeSigDen\":%d,"
        "\"playing\":%s,\"position\":%.4f}\n",
        state.bpm,
        state.timeSigNum,
        state.timeSigDen,
        state.isPlaying ? "true" : "false",
        state.positionInSeconds
    );

    sendJson(json);
}

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

    int written = socket->write(data, len);
    if (written < 0)
    {
        connected.store(false);
        socket.reset();
        DBG("ResonateBridge: Connection lost (write failed)");
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
        if (written < 0)
        {
            connected.store(false);
            socket.reset();
            DBG("ResonateBridge: Connection lost (raw write failed)");
            return;
        }
        ptr += written;
        remaining -= written;
    }
}

bool BridgeClient::readResponse(juce::String& out)
{
    std::lock_guard<std::mutex> lock(socketMutex);

    if (socket == nullptr || !socket->isConnected())
        return false;

    // Non-blocking check if data available
    if (!socket->waitUntilReady(true, 0))
        return false;

    char buffer[1024];
    int bytesRead = socket->read(buffer, sizeof(buffer) - 1, false);
    if (bytesRead > 0)
    {
        buffer[bytesRead] = '\0';
        out = juce::String::fromUTF8(buffer, bytesRead);
        return true;
    }
    else if (bytesRead < 0)
    {
        connected.store(false);
        socket.reset();
    }

    return false;
}
