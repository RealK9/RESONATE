import json
import numpy as np
from ml.models.sample_profile import (
    SampleProfile, CoreDescriptors, SpectralDescriptors,
    HarmonicDescriptors, TransientDescriptors, PerceptualDescriptors,
    Embeddings, PredictedLabels,
)


def test_default_construction():
    p = SampleProfile()
    assert p.filepath == ""
    assert p.core.duration == 0.0
    assert p.labels.role == "unknown"


def test_to_dict_roundtrip():
    p = SampleProfile(filepath="/test.wav", filename="test.wav")
    p.core.duration = 2.5
    p.core.lufs = -14.0
    p.spectral.centroid = 3000.0
    p.labels.role = "kick"
    p.labels.genre_affinity = {"trap": 0.8, "house": 0.3}
    d = p.to_dict()
    restored = SampleProfile.from_dict(d)
    assert restored.filepath == "/test.wav"
    assert restored.core.duration == 2.5
    assert restored.labels.role == "kick"
    assert restored.labels.genre_affinity["trap"] == 0.8


def test_json_serialization_with_numpy():
    p = SampleProfile(filepath="/test.wav")
    p.spectral.contrast = np.array([1.0, 2.0, 3.0])
    p.harmonic.chroma_profile = np.zeros(12, dtype=np.float32).tolist()
    j = p.to_json()
    parsed = json.loads(j)
    assert parsed["spectral"]["contrast"] == [1.0, 2.0, 3.0]


def test_from_dict_ignores_unknown_keys():
    d = {"filepath": "/x.wav", "core": {"duration": 1.0, "unknown_field": 999}}
    p = SampleProfile.from_dict(d)
    assert p.core.duration == 1.0
    assert p.filepath == "/x.wav"
