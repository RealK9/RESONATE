"""
RESONATE — Version Tracking Routes.
Save, list, and compare track version snapshots.
"""

from fastapi import APIRouter, HTTPException, Request

import state
from db.versions import (
    init,
    save_version,
    get_versions,
    get_latest,
    compare_versions,
    list_projects,
    next_label,
)

router = APIRouter()

# Ensure table exists on import
init()


# ═══════════════════════════════════════════════════════════════════════════
# VERSION ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/versions")
async def create_version(request: Request):
    """Save the current analysis as a new version snapshot."""
    body = await request.json()
    project_name = body.get("project_name")
    if not project_name:
        raise HTTPException(status_code=400, detail="project_name is required")

    version_label = body.get("version_label") or next_label(project_name)
    filepath = body.get("filepath", "")

    # Pull readiness data from latest gap result
    readiness_score = None
    gap_summary = None
    chart_potential = None
    missing_roles = None

    gap = state.latest_gap_result
    if gap:
        readiness_score = getattr(gap, "readiness_score", None)
        if readiness_score is None and isinstance(gap, dict):
            readiness_score = gap.get("readiness_score")
        gap_summary = getattr(gap, "gap_summary", None)
        if gap_summary is None and isinstance(gap, dict):
            gap_summary = gap.get("gap_summary")
        chart_potential = getattr(gap, "chart_potential", None)
        if chart_potential is None and isinstance(gap, dict):
            chart_potential = gap.get("chart_potential")
        missing = getattr(gap, "missing_roles", None)
        if missing is None and isinstance(gap, dict):
            missing = gap.get("missing_roles")
        if missing is not None:
            missing_roles = list(missing) if not isinstance(missing, list) else missing

    # Full analysis snapshot from mix profile
    analysis_json = None
    mp = state.latest_mix_profile
    if mp:
        analysis_json = mp if isinstance(mp, dict) else (
            mp.__dict__ if hasattr(mp, "__dict__") else None
        )

    version_id = save_version(
        project_name=project_name,
        version_label=version_label,
        filepath=filepath,
        readiness_score=readiness_score,
        gap_summary=gap_summary,
        chart_potential=chart_potential,
        missing_roles=missing_roles,
        analysis_json=analysis_json,
    )

    return {
        "id": version_id,
        "version_label": version_label,
        "readiness_score": readiness_score,
        "gap_summary": gap_summary,
        "status": "saved",
    }


@router.get("/versions/projects")
async def get_all_projects():
    """List all projects with version counts."""
    return {"projects": list_projects()}


@router.get("/versions/compare")
async def compare(id_a: int, id_b: int):
    """Compare two versions by id."""
    result = compare_versions(id_a, id_b)
    if result is None:
        raise HTTPException(status_code=404, detail="One or both versions not found")
    return result


@router.get("/versions/{project_name}")
async def get_project_versions(project_name: str):
    """List all versions for a project."""
    versions = get_versions(project_name)
    return {"versions": versions, "total": len(versions)}
