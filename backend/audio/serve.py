"""
RESONATE — Audio file serving with range request support.
"""

from fastapi.responses import FileResponse


_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".m4a": "audio/mp4",
}


def serve_audio(filepath):
    """Serve an audio file with range request support."""
    fsize = filepath.stat().st_size
    media = _MIME_TYPES.get(filepath.suffix.lower(), "application/octet-stream")

    return FileResponse(
        path=str(filepath),
        media_type=media,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(fsize),
            "Access-Control-Allow-Origin": "*",
        }
    )
