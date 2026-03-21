"""
RESONATE — Bridge Server.
TCP socket server that receives transport state from the VST3 Bridge plugin
and exposes it to the FastAPI app via shared state.
Supports audio capture for "Analyze from DAW" feature.
"""

import asyncio
import json
import threading
from typing import Optional

import state

# Bridge state — updated by plugin, read by frontend
bridge_state = {
    "connected": False,
    "bpm": 120.0,
    "timeSigNum": 4,
    "timeSigDen": 4,
    "playing": False,
    "position": 0.0,
}

_clients = {}  # writer -> reader mapping
_server: Optional[asyncio.AbstractServer] = None

# Audio capture state
_capture_event: Optional[asyncio.Event] = None
_captured_wav: Optional[bytes] = None


async def request_audio_capture() -> Optional[bytes]:
    """Request audio capture from the connected bridge plugin. Returns WAV bytes or None."""
    global _capture_event, _captured_wav
    if not _clients:
        return None

    _capture_event = asyncio.Event()
    _captured_wav = None

    # Send capture request to all connected plugins
    await send_to_plugin({"type": "captureAudio"})

    # Wait up to 10 seconds for the plugin to respond with audio
    try:
        await asyncio.wait_for(_capture_event.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        print("  ✗ Bridge: Audio capture timed out")
        _capture_event = None
        return None

    result = _captured_wav
    _capture_event = None
    _captured_wav = None
    return result


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single bridge plugin connection."""
    global _captured_wav, _capture_event

    addr = writer.get_extra_info("peername")
    print(f"  ✓ Bridge: VST3 plugin connected from {addr}")
    _clients[writer] = reader
    bridge_state["connected"] = True

    buffer = b""
    awaiting_wav_bytes = 0  # How many raw WAV bytes we're expecting

    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break

            buffer += data

            # If we're in the middle of reading raw WAV data
            if awaiting_wav_bytes > 0:
                if len(buffer) >= awaiting_wav_bytes:
                    wav_data = buffer[:awaiting_wav_bytes]
                    buffer = buffer[awaiting_wav_bytes:]
                    awaiting_wav_bytes = 0

                    _captured_wav = wav_data
                    if _capture_event:
                        _capture_event.set()

                    print(f"  ✓ Bridge: Received {len(wav_data)} bytes of audio from DAW")
                continue

            # Process complete JSON messages (newline-delimited)
            text_buffer = buffer.decode("utf-8", errors="ignore")
            while "\n" in text_buffer:
                line, text_buffer = text_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                    if msg.get("type") == "transport":
                        bridge_state["bpm"] = msg.get("bpm", 120.0)
                        bridge_state["timeSigNum"] = msg.get("timeSigNum", 4)
                        bridge_state["timeSigDen"] = msg.get("timeSigDen", 4)
                        bridge_state["playing"] = msg.get("playing", False)
                        bridge_state["position"] = msg.get("position", 0.0)

                        state.daw_bpm = bridge_state["bpm"]
                        state.daw_playing = bridge_state["playing"]

                    elif msg.get("type") == "audioCapture":
                        if msg.get("error"):
                            print(f"  ✗ Bridge: Capture error: {msg['error']}")
                            if _capture_event:
                                _capture_event.set()
                        else:
                            wav_size = msg.get("wavSize", 0)
                            if wav_size > 0:
                                awaiting_wav_bytes = wav_size
                                print(f"  ✓ Bridge: Expecting {wav_size} bytes of WAV audio...")
                except json.JSONDecodeError:
                    pass

            buffer = text_buffer.encode("utf-8")

    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        del _clients[writer]
        bridge_state["connected"] = len(_clients) > 0
        print(f"  ✗ Bridge: VST3 plugin disconnected from {addr}")
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def send_to_plugin(message: dict):
    """Send a message to all connected bridge plugins."""
    data = (json.dumps(message) + "\n").encode("utf-8")
    dead = set()
    for writer in _clients:
        try:
            writer.write(data)
            await writer.drain()
        except Exception:
            dead.add(writer)
    for w in dead:
        _clients.pop(w, None)


async def start_bridge_server(port: int = 9876):
    """Start the bridge TCP server."""
    global _server
    try:
        _server = await asyncio.start_server(handle_client, "127.0.0.1", port)
        print(f"  ✓ Bridge server listening on port {port}")
        async with _server:
            await _server.serve_forever()
    except OSError as e:
        print(f"  ✗ Bridge server failed to start: {e}")


def run_bridge_in_thread(port: int = 9876):
    """Run the bridge server in a background thread."""
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_bridge_server(port))

    t = threading.Thread(target=_run, daemon=True, name="bridge-server")
    t.start()
    return t
