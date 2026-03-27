"""
RESONATE — Bridge Server.
TCP socket server that receives transport state from the VST3 Bridge plugin
and exposes it to the FastAPI app via shared state.
Supports audio capture for "Analyze from DAW" feature.

Thread safety:
  - _clients_lock protects the _clients dict
  - _capture_lock protects capture state (_capture_event, _captured_wav)
  - bridge_state dict updates are atomic (single-key assignments)
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

_clients: dict[asyncio.StreamWriter, asyncio.StreamReader] = {}
_clients_lock = asyncio.Lock()
_server: Optional[asyncio.AbstractServer] = None

# Audio capture state (protected by _capture_lock)
_capture_lock = asyncio.Lock()
_capture_event: Optional[asyncio.Event] = None
_captured_wav: Optional[bytes] = None


async def request_audio_capture() -> Optional[bytes]:
    """Request audio capture from the connected bridge plugin. Returns WAV bytes or None."""
    global _capture_event, _captured_wav

    async with _clients_lock:
        if not _clients:
            return None

    async with _capture_lock:
        _capture_event = asyncio.Event()
        _captured_wav = None

    await send_to_plugin({"type": "captureAudio"})

    try:
        async with _capture_lock:
            evt = _capture_event
        if evt:
            await asyncio.wait_for(evt.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        print("  ✗ Bridge: Audio capture timed out (10s)")

    async with _capture_lock:
        result = _captured_wav
        _capture_event = None
        _captured_wav = None

    return result


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single bridge plugin connection."""
    global _captured_wav, _capture_event

    addr = writer.get_extra_info("peername")
    print(f"  ✓ Bridge: VST3 plugin connected from {addr}")

    async with _clients_lock:
        _clients[writer] = reader
        bridge_state["connected"] = True

    buffer = b""
    awaiting_wav_bytes = 0

    try:
        while True:
            try:
                data = await asyncio.wait_for(reader.read(65536), timeout=30.0)
            except asyncio.TimeoutError:
                # No data in 30s — check if connection is still alive
                if writer.is_closing():
                    break
                continue

            if not data:
                break

            buffer += data

            # If we're in the middle of reading raw WAV data
            if awaiting_wav_bytes > 0:
                if len(buffer) >= awaiting_wav_bytes:
                    wav_data = buffer[:awaiting_wav_bytes]
                    buffer = buffer[awaiting_wav_bytes:]
                    awaiting_wav_bytes = 0

                    async with _capture_lock:
                        _captured_wav = wav_data
                        if _capture_event:
                            _capture_event.set()

                    print(f"  ✓ Bridge: Received {len(wav_data)} bytes of audio from DAW")
                continue

            # Process complete JSON messages (newline-delimited).
            # Find the boundary between text and potential binary data.
            # Only decode up to the last newline to avoid mangling binary bytes.
            last_newline = buffer.rfind(b"\n")
            if last_newline < 0:
                # No complete line yet — keep buffering
                # Safety: if buffer is huge without a newline, it's garbage
                if len(buffer) > 1_000_000:
                    print("  ✗ Bridge: Buffer overflow (no newline in 1MB), clearing")
                    buffer = b""
                continue

            text_part = buffer[:last_newline + 1]
            buffer = buffer[last_newline + 1:]

            # Now safely decode only the text portion
            try:
                text_lines = text_part.decode("utf-8")
            except UnicodeDecodeError:
                # Binary data mixed in — skip this chunk
                continue

            for line in text_lines.split("\n"):
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "transport":
                    bridge_state["bpm"] = msg.get("bpm", 120.0)
                    bridge_state["timeSigNum"] = msg.get("timeSigNum", 4)
                    bridge_state["timeSigDen"] = msg.get("timeSigDen", 4)
                    bridge_state["playing"] = msg.get("playing", False)
                    bridge_state["position"] = msg.get("position", 0.0)
                    state.daw_bpm = bridge_state["bpm"]
                    state.daw_playing = bridge_state["playing"]

                elif msg_type == "audioCapture":
                    if msg.get("error"):
                        print(f"  ✗ Bridge: Capture error: {msg['error']}")
                        async with _capture_lock:
                            if _capture_event:
                                _capture_event.set()
                    else:
                        wav_size = msg.get("wavSize", 0)
                        if wav_size > 0:
                            awaiting_wav_bytes = wav_size
                            print(f"  ✓ Bridge: Expecting {wav_size} bytes of WAV audio...")

    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        async with _clients_lock:
            _clients.pop(writer, None)
            bridge_state["connected"] = len(_clients) > 0

        # Reset DAW state on disconnect
        if not bridge_state["connected"]:
            state.daw_bpm = 0.0
            state.daw_playing = False

        print(f"  ✗ Bridge: VST3 plugin disconnected from {addr}")
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def send_to_plugin(message: dict):
    """Send a message to all connected bridge plugins."""
    data = (json.dumps(message) + "\n").encode("utf-8")
    dead = []

    async with _clients_lock:
        writers = list(_clients.keys())

    for writer in writers:
        try:
            writer.write(data)
            await asyncio.wait_for(writer.drain(), timeout=2.0)
        except Exception:
            dead.append(writer)

    if dead:
        async with _clients_lock:
            for w in dead:
                _clients.pop(w, None)
            bridge_state["connected"] = len(_clients) > 0


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
