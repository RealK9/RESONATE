"""
Tests for CandidateGenerator -- Phase 4 candidate selection.

Fixtures populate a SampleStore with 20+ diverse SampleProfiles and
create MixProfiles with specific needs to verify that the generator
correctly selects, filters, and prioritizes candidates.
"""
from __future__ import annotations

import pytest

from ml.db.sample_store import SampleStore
from ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    NeedOpportunity,
    SourceRolePresence,
    StyleCluster,
)
from ml.models.sample_profile import (
    CoreDescriptors,
    HarmonicDescriptors,
    PredictedLabels,
    SampleProfile,
)
from ml.recommendation.candidate_generator import CandidateGenerator


# ---------------------------------------------------------------------------
# Sample library fixture — 24 diverse samples
# ---------------------------------------------------------------------------

_SAMPLE_DEFS: list[dict] = [
    # Kicks (3)
    {"fp": "/lib/kick_01.wav", "role": "kick", "tonal": False, "cr": 0.7},
    {"fp": "/lib/kick_02.wav", "role": "kick", "tonal": False, "cr": 0.8},
    {"fp": "/lib/kick_03.wav", "role": "kick", "tonal": False, "cr": 0.1},  # low quality
    # Snares (3)
    {"fp": "/lib/snare_01.wav", "role": "snare", "tonal": False, "cr": 0.6},
    {"fp": "/lib/snare_02.wav", "role": "snare", "tonal": False, "cr": 0.75},
    {"fp": "/lib/snare_03.wav", "role": "snare", "tonal": False, "cr": 0.2},  # low quality
    # Claps (1)
    {"fp": "/lib/clap_01.wav", "role": "clap", "tonal": False, "cr": 0.65},
    # Hats (3)
    {"fp": "/lib/hat_01.wav", "role": "hat", "tonal": False, "cr": 0.7},
    {"fp": "/lib/hat_02.wav", "role": "hat", "tonal": False, "cr": 0.6},
    {"fp": "/lib/hat_03.wav", "role": "hat", "tonal": False, "cr": 0.15},  # low quality
    # Basses (3) — tonal
    {"fp": "/lib/bass_01.wav", "role": "bass", "tonal": True, "cr": 0.8, "key": "C"},
    {"fp": "/lib/bass_02.wav", "role": "bass", "tonal": True, "cr": 0.7, "key": "G"},
    {"fp": "/lib/bass_03.wav", "role": "bass", "tonal": True, "cr": 0.65, "key": "F#"},
    # Leads (2) — tonal
    {"fp": "/lib/lead_01.wav", "role": "lead", "tonal": True, "cr": 0.75, "key": "Am"},
    {"fp": "/lib/lead_02.wav", "role": "lead", "tonal": True, "cr": 0.5, "key": "Eb"},
    # Pads (3) — tonal
    {"fp": "/lib/pad_01.wav", "role": "pad", "tonal": True, "cr": 0.8, "key": "C"},
    {"fp": "/lib/pad_02.wav", "role": "pad", "tonal": True, "cr": 0.7, "key": "Dm"},
    {"fp": "/lib/pad_03.wav", "role": "pad", "tonal": True, "cr": 0.6, "key": "Bb"},
    # Textures (2)
    {"fp": "/lib/texture_01.wav", "role": "texture", "tonal": False, "cr": 0.55},
    {"fp": "/lib/texture_02.wav", "role": "texture", "tonal": False, "cr": 0.5},
    # FX (2)
    {"fp": "/lib/fx_01.wav", "role": "fx", "tonal": False, "cr": 0.6},
    {"fp": "/lib/fx_02.wav", "role": "fx", "tonal": False, "cr": 0.45},
    # Vocals (1) — tonal
    {"fp": "/lib/vocal_01.wav", "role": "vocal", "tonal": True, "cr": 0.85, "key": "C"},
    # Extra low quality to ensure filtering works broadly
    {"fp": "/lib/pad_lowq.wav", "role": "pad", "tonal": True, "cr": 0.05, "key": "C"},
]


def _make_sample(d: dict) -> SampleProfile:
    """Create a SampleProfile from a compact definition dict."""
    fp = d["fp"]
    filename = fp.rsplit("/", 1)[-1]
    key = d.get("key", "")
    # Build chroma profile if a key is specified.
    chroma: list[float] = []
    pitch_conf = 0.0
    if key:
        note_names = [
            "C", "Db", "D", "Eb", "E", "F",
            "F#", "G", "Ab", "A", "Bb", "B",
        ]
        root = key.rstrip("m")
        chroma = [0.1] * 12
        if root in note_names:
            chroma[note_names.index(root)] = 0.9
        pitch_conf = 0.8

    return SampleProfile(
        filepath=fp,
        filename=filename,
        core=CoreDescriptors(duration=0.5, sample_rate=44100, channels=1),
        harmonic=HarmonicDescriptors(
            chroma_profile=chroma,
            pitch_confidence=pitch_conf,
        ),
        labels=PredictedLabels(
            role=d["role"],
            role_confidence=0.8,
            tonal=d["tonal"],
            commercial_readiness=d["cr"],
        ),
    )


@pytest.fixture
def store(tmp_path):
    """SampleStore populated with 24 diverse samples."""
    db_path = tmp_path / "test.db"
    s = SampleStore(str(db_path))
    s.init()
    for d in _SAMPLE_DEFS:
        s.save(_make_sample(d))
    return s


@pytest.fixture
def generator(store):
    """CandidateGenerator with no vector index."""
    return CandidateGenerator(sample_store=store)


def _mix_with_needs(
    *,
    key: str = "",
    needs: list[NeedOpportunity] | None = None,
) -> tuple[MixProfile, list[NeedOpportunity]]:
    """Helper to create a MixProfile and needs list."""
    profile = MixProfile(
        filepath="/mixes/test_mix.wav",
        filename="test_mix.wav",
        analysis=MixLevelAnalysis(key=key, key_confidence=0.9 if key else 0.0),
        style=StyleCluster(primary_cluster="modern_trap"),
        source_roles=SourceRolePresence(roles={
            "kick": 0.1,
            "snare_clap": 0.1,
            "hats_tops": 0.05,
            "bass": 0.6,
            "lead": 0.0,
            "pad": 0.0,
        }),
    )
    if needs is None:
        needs = [
            NeedOpportunity(
                category="role",
                description="Weak attack support -- kick and snare presence is very low",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
        ]
    return profile, needs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCandidateGenerator:
    """Test suite for CandidateGenerator."""

    def test_generates_candidates(self, generator):
        """generate() returns a list of SampleProfile objects."""
        mix, needs = _mix_with_needs()
        candidates = generator.generate(mix, needs)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(c, SampleProfile) for c in candidates)

    def test_respects_max_candidates(self, generator):
        """Respects the max_candidates limit."""
        mix, needs = _mix_with_needs()
        candidates = generator.generate(mix, needs, max_candidates=3)
        assert len(candidates) <= 3

    def test_prioritizes_needed_roles(self, generator):
        """If mix needs kick, candidates include kicks."""
        needs = [
            NeedOpportunity(
                category="role",
                description="Weak attack support -- kick and snare presence is very low",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
        ]
        mix, _ = _mix_with_needs(needs=needs)
        candidates = generator.generate(mix, needs)
        roles = {c.labels.role for c in candidates}
        assert "kick" in roles
        assert "snare" in roles

    def test_filters_low_quality(self, generator):
        """Samples below the quality threshold are excluded."""
        mix, needs = _mix_with_needs()
        candidates = generator.generate(mix, needs)
        # kick_03 (cr=0.1), snare_03 (cr=0.2), hat_03 (cr=0.15),
        # pad_lowq (cr=0.05) should all be excluded.
        low_quality_paths = {
            "/lib/kick_03.wav",
            "/lib/snare_03.wav",
            "/lib/hat_03.wav",
            "/lib/pad_lowq.wav",
        }
        candidate_paths = {c.filepath for c in candidates}
        assert candidate_paths.isdisjoint(low_quality_paths)

    def test_empty_store_returns_empty(self, tmp_path):
        """Gracefully returns empty list when store has no samples."""
        db_path = tmp_path / "empty.db"
        empty_store = SampleStore(str(db_path))
        empty_store.init()
        gen = CandidateGenerator(sample_store=empty_store)
        mix, needs = _mix_with_needs()
        candidates = gen.generate(mix, needs)
        assert candidates == []

    def test_multiple_needs_covered(self, generator):
        """Candidates span multiple needed roles when multiple needs exist."""
        needs = [
            NeedOpportunity(
                category="role",
                description="Weak attack support -- kick and snare presence is very low",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
            NeedOpportunity(
                category="role",
                description="No glue texture -- pads and ambient layers are missing",
                severity=0.7,
                recommendation_policy="fill_missing_role",
            ),
            NeedOpportunity(
                category="role",
                description="No rhythmic sparkle -- hi-hats and top percussion absent",
                severity=0.6,
                recommendation_policy="enhance_groove",
            ),
        ]
        mix, _ = _mix_with_needs(needs=needs)
        candidates = generator.generate(mix, needs)
        roles = {c.labels.role for c in candidates}
        # Should include percussion AND textural roles.
        assert "kick" in roles
        assert "pad" in roles
        assert "hat" in roles

    def test_tonal_compatibility_filtering(self, generator):
        """Tonal samples with incompatible keys are filtered out when mix has a key."""
        # Mix is in C major.  bass_03 is in F# (distant on circle of fifths)
        # and lead_02 is in Eb (also distant).
        needs = [
            NeedOpportunity(
                category="role",
                description="Harmonic layer too thin -- needs bass and lead",
                severity=0.8,
                recommendation_policy="fill_missing_role",
            ),
        ]
        mix, _ = _mix_with_needs(key="C", needs=needs)
        candidates = generator.generate(mix, needs)
        candidate_paths = {c.filepath for c in candidates}
        # bass_03 (F#) should be excluded -- 6 steps on circle of fifths.
        assert "/lib/bass_03.wav" not in candidate_paths
        # bass_01 (C) and bass_02 (G) should be included.
        assert "/lib/bass_01.wav" in candidate_paths
        assert "/lib/bass_02.wav" in candidate_paths

    def test_no_needs_returns_empty(self, generator):
        """Empty needs list results in no candidates."""
        mix, _ = _mix_with_needs()
        candidates = generator.generate(mix, [])
        assert candidates == []

    def test_non_tonal_samples_always_pass_key_filter(self, generator):
        """Non-tonal samples (drums, fx) pass through even when mix has a key."""
        needs = [
            NeedOpportunity(
                category="role",
                description="Weak attack support -- kick",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
        ]
        mix, _ = _mix_with_needs(key="C", needs=needs)
        candidates = generator.generate(mix, needs)
        kick_candidates = [c for c in candidates if c.labels.role == "kick"]
        # kick_01 and kick_02 are non-tonal, should both be included.
        assert len(kick_candidates) == 2

    def test_candidates_ordered_by_need_severity(self, generator):
        """Higher-severity needs produce candidates earlier in the list."""
        needs = [
            NeedOpportunity(
                category="role",
                description="No glue texture -- pads and ambient layers are missing",
                severity=0.3,
                recommendation_policy="fill_missing_role",
            ),
            NeedOpportunity(
                category="role",
                description="Weak attack support -- kick and snare presence is very low",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
        ]
        mix, _ = _mix_with_needs(needs=needs)
        candidates = generator.generate(mix, needs)
        # First candidates should be kicks/snares (severity 0.9),
        # not pads (severity 0.3).
        first_few_roles = [c.labels.role for c in candidates[:4]]
        assert any(r in ("kick", "snare", "clap") for r in first_few_roles)
