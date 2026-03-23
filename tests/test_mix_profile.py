import json
import numpy as np
from backend.ml.models.mix_profile import (
    MixProfile,
    MixLevelAnalysis,
    SpectralOccupancy,
    StereoWidth,
    SourceRolePresence,
    StyleCluster,
    NeedOpportunity,
)


def test_default_construction():
    p = MixProfile()
    assert p.filepath == ""
    assert p.filename == ""
    assert p.analysis.bpm == 0.0
    assert p.analysis.loudness_lufs == -100.0
    assert p.analysis.key == ""
    assert p.spectral_occupancy.time_frames == 0
    assert p.spectral_occupancy.bands == []
    assert p.stereo_width.overall_width == 0.0
    assert p.source_roles.roles == {}
    assert p.style.primary_cluster == ""
    assert p.needs == []
    assert p.density_map == []


def test_to_dict_roundtrip():
    p = MixProfile(filepath="/mix.wav", filename="mix.wav")
    p.analysis.bpm = 128.0
    p.analysis.bpm_confidence = 0.95
    p.analysis.key = "Cm"
    p.analysis.loudness_lufs = -8.5
    p.analysis.section_energy = [0.3, 0.7, 0.9, 0.5]
    p.spectral_occupancy.bands = ["sub", "bass", "low_mid"]
    p.spectral_occupancy.mean_by_band = [0.6, 0.8, 0.4]
    p.stereo_width.overall_width = 0.72
    p.stereo_width.correlation = 0.85
    p.source_roles.roles = {"kick": 0.95, "bass": 0.88, "lead": 0.6}
    p.style.cluster_probabilities = {"melodic_techno": 0.7, "2020s_melodic_house": 0.2}
    p.style.primary_cluster = "melodic_techno"
    p.density_map = [0.3, 0.5, 0.8, 0.9, 0.6]

    d = p.to_dict()
    restored = MixProfile.from_dict(d)

    assert restored.filepath == "/mix.wav"
    assert restored.analysis.bpm == 128.0
    assert restored.analysis.key == "Cm"
    assert restored.analysis.loudness_lufs == -8.5
    assert restored.analysis.section_energy == [0.3, 0.7, 0.9, 0.5]
    assert restored.spectral_occupancy.bands == ["sub", "bass", "low_mid"]
    assert restored.stereo_width.overall_width == 0.72
    assert restored.source_roles.roles["kick"] == 0.95
    assert restored.style.primary_cluster == "melodic_techno"
    assert restored.density_map == [0.3, 0.5, 0.8, 0.9, 0.6]


def test_json_serialization():
    p = MixProfile(filepath="/mix.wav")
    p.analysis.bpm = 126.0
    p.analysis.section_energy = np.array([0.2, 0.5, 0.8], dtype=np.float32).tolist()
    p.spectral_occupancy.mean_by_band = np.array([0.1, 0.4, 0.7]).tolist()
    p.density_map = [0.3, 0.6, 0.9]

    j = p.to_json()
    parsed = json.loads(j)

    assert parsed["filepath"] == "/mix.wav"
    assert parsed["analysis"]["bpm"] == 126.0
    assert parsed["density_map"] == [0.3, 0.6, 0.9]

    # Roundtrip through JSON
    restored = MixProfile.from_dict(parsed)
    assert restored.analysis.bpm == 126.0
    assert restored.density_map == [0.3, 0.6, 0.9]


def test_need_opportunity_fields():
    need = NeedOpportunity(
        category="spectral",
        description="Sub-bass region is empty below 60Hz",
        severity=0.85,
        recommendation_policy="fill_missing_role",
    )
    assert need.category == "spectral"
    assert need.severity == 0.85
    assert need.recommendation_policy == "fill_missing_role"
    assert "empty" in need.description


def test_from_dict_with_needs_list():
    d = {
        "filepath": "/mix.wav",
        "filename": "mix.wav",
        "analysis": {"bpm": 140.0, "key": "Am"},
        "needs": [
            {
                "category": "role",
                "description": "No hi-hat presence detected",
                "severity": 0.7,
                "recommendation_policy": "fill_missing_role",
            },
            {
                "category": "spatial",
                "description": "Mix is very narrow in mid frequencies",
                "severity": 0.5,
                "recommendation_policy": "reinforce_existing",
            },
        ],
        "density_map": [0.4, 0.6, 0.8],
    }
    p = MixProfile.from_dict(d)

    assert p.analysis.bpm == 140.0
    assert p.analysis.key == "Am"
    assert len(p.needs) == 2
    assert p.needs[0].category == "role"
    assert p.needs[0].severity == 0.7
    assert p.needs[1].category == "spatial"
    assert p.needs[1].recommendation_policy == "reinforce_existing"
    assert p.density_map == [0.4, 0.6, 0.8]
