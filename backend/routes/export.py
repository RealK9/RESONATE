"""
RESONATE — Batch Export Route.
Multi-select samples and export as a zip.
"""

import io
import zipfile
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

import config
from routes.samples import find_sample_file

router = APIRouter()


class ExportRequest(BaseModel):
    sample_paths: List[str]


@router.post("/samples/export")
async def export_samples(req: ExportRequest):
    """Export selected samples as a zip file."""
    if not req.sample_paths:
        raise HTTPException(status_code=400, detail="No samples selected")
    if len(req.sample_paths) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 samples per export")

    buf = io.BytesIO()
    found = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sp in req.sample_paths:
            fp = find_sample_file(sp)
            if fp and fp.exists():
                zf.write(fp, arcname=fp.name)
                found += 1

    if found == 0:
        raise HTTPException(status_code=404, detail="No sample files found")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=resonate-kit-{found}-samples.zip"},
    )
