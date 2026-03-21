"""
RESONATE — Audio file serving with range request support.
"""

from fastapi.responses import FileResponse


def serve_audio(filepath):
    """Serve an audio file with range request support."""
    fsize = filepath.stat().st_size
    media = "audio/wav" if filepath.suffix.lower() == ".wav" else "audio/mpeg"

    return FileResponse(
        path=str(filepath),
        media_type=media,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(fsize),
            "Access-Control-Allow-Origin": "*",
        }
    )
