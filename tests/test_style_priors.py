"""
Tests for Phase 3 — StylePriorsTrainer and NeedsEngine corpus integration.

Covers:
  - trainer add + train
  - trainer load_or_default (no file -> defaults)
  - trainer save and load round-trip
  - NeedsEngine accepts custom corpus
  - NeedsEngine works with no corpus arg (default behaviour)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ml.analysis.needs_engine import NeedsEngine
from backend.ml.analysis.reference_profiles import (
    DefaultPriors,
    ReferenceProfileBuilder,
)
from backend.ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    NeedOpportunity,
    SourceRolePresence,
    SpectralOccupancy,
    StereoWidth,
    StyleCluster,
)
from backend.ml.models.reference_profile import ReferenceCorpus, StylePrior
from backend.ml.training.style_priors import StylePriorsTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BAND_NAMES = [
    "sub", "bass", "low_mid", "mid", "upper_mid",
    "presence", "brilliance", "air", "ultra_high", "ceiling",
]

_ROLE_NAMES = [
    "kick", "snare_clap", "hats_tops", "bass", "lead",
    "chord_support", "pad", "vocal_texture", "fx_transitions", "ambience",
]


def _make_mix_profile(
    cluster: str = "modern_trap",
    spectral: list[float] | None = None,
    width: list[float] | None = None,
    roles: dict[str, float] | None = None,
    density_map: list[float] | None = None,
    section_energy: list[float] | None = None,
    harmonic_density: float = 0.4,
) -> MixProfile:
    """Build a minimal but valid MixProfile for testing."""
    if spectral is None:
        spectral = [0.5] * 10
    if width is None:
        width = [0.4] * 10
    if roles is None:
        roles = {r: 0.5 for r in _ROLE_NAMES}
    if density_map is None:
        density_map = [0.5] * 16
    if section_energy is None:
        section_energy = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5]

    return MixProfile(
        filepath="/tmp/test.wav",
        filename="test.wav",
        analysis=MixLevelAnalysis(
            bpm=140.0,
            bpm_confidence=0.9,
            key="C minor",
            key_confidence=0.85,
            harmonic_density=harmonic_density,
            section_energy=section_energy,
            dynamic_range=10.0,
        ),
        spectral_occupancy=SpectralOccupancy(
            bands=_BAND_NAMES,
            time_frames=16,
            mean_by_band=spectral,
        ),
        stereo_width=StereoWidth(
            bands=_BAND_NAMES,
            width_by_band=width,
            overall_width=0.5,
        ),
        source_roles=SourceRolePresence(roles=roles),
        style=StyleCluster(
            cluster_probabilities={cluster: 0.8},
            primary_cluster=cluster,
            era_estimate="2020s",
        ),
        density_map=density_map,
    )


# ---------------------------------------------------------------------------
# Tests — StylePriorsTrainer
# ---------------------------------------------------------------------------


class TestTrainerAddAndTrain:
    """test_trainer_add_and_train — add a reference, train, get corpus."""

    def test_add_reference_and_train(self) -> None:
        trainer = StylePriorsTrainer()

        # Manually feed via the underlying builder (bypasses audio I/O)
        profile = _make_mix_profile(cluster="melodic_techno")
        trainer._builder.add_reference(profile)

        corpus = trainer.train()

        assert isinstance(corpus, ReferenceCorpus)
        prior = corpus.get_prior("melodic_techno")
        assert prior is not None
        assert prior.cluster_name == "melodic_techno"
        assert prior.reference_count == 1

    def test_train_multiple_clusters(self) -> None:
        trainer = StylePriorsTrainer()

        for cluster in ["modern_trap", "lo_fi_chill", "dnb"]:
            trainer._builder.add_reference(
                _make_mix_profile(cluster=cluster)
            )

        corpus = trainer.train()
        assert corpus.total_references == 3
        for cluster in ["modern_trap", "lo_fi_chill", "dnb"]:
            assert corpus.get_prior(cluster) is not None


class TestTrainerLoadOrDefault:
    """test_trainer_load_or_default — no file -> returns defaults."""

    def test_no_file_returns_defaults(self, tmp_path: Path) -> None:
        # Point to a non-existent path
        trainer = StylePriorsTrainer(
            corpus_path=str(tmp_path / "does_not_exist.json")
        )
        corpus = trainer.load_or_default()

        # Should be the default corpus with all 14 clusters
        assert isinstance(corpus, ReferenceCorpus)
        assert len(corpus.priors) == 14

    def test_no_path_returns_defaults(self) -> None:
        trainer = StylePriorsTrainer()  # no corpus_path
        corpus = trainer.load_or_default()
        assert isinstance(corpus, ReferenceCorpus)
        assert len(corpus.priors) == 14


class TestTrainerSaveAndLoad:
    """test_trainer_save_and_load — train, save, load roundtrip."""

    def test_save_and_reload(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "test_corpus.json"
        trainer = StylePriorsTrainer(corpus_path=str(corpus_file))

        # Add references and train (saves to disk)
        for i in range(3):
            spectral = [0.3 + i * 0.1] * 10
            trainer._builder.add_reference(
                _make_mix_profile(cluster="afro_house", spectral=spectral)
            )

        original = trainer.train()

        # Verify file was written
        assert corpus_file.exists()

        # Reload from disk via a fresh trainer
        fresh_trainer = StylePriorsTrainer(corpus_path=str(corpus_file))
        loaded = fresh_trainer.load_or_default()

        assert isinstance(loaded, ReferenceCorpus)
        assert set(loaded.priors.keys()) == set(original.priors.keys())

        orig_prior = original.get_prior("afro_house")
        load_prior = loaded.get_prior("afro_house")
        assert orig_prior is not None
        assert load_prior is not None
        assert load_prior.target_spectral_mean == pytest.approx(
            orig_prior.target_spectral_mean, abs=1e-6
        )
        assert load_prior.reference_count == orig_prior.reference_count

    def test_saved_file_is_valid_json(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.json"
        trainer = StylePriorsTrainer(corpus_path=str(corpus_file))
        trainer._builder.add_reference(_make_mix_profile())
        trainer.train()

        data = json.loads(corpus_file.read_text())
        assert "priors" in data
        assert "version" in data
        assert "total_references" in data


# ---------------------------------------------------------------------------
# Tests — NeedsEngine with corpus
# ---------------------------------------------------------------------------


class TestNeedsEngineWithCorpus:
    """test_needs_engine_with_corpus — NeedsEngine accepts custom corpus."""

    def test_engine_uses_corpus_norms(self) -> None:
        """A corpus with custom spectral norms should change the diagnosis."""
        # Build a corpus with a very specific spectral norm for modern_trap:
        # all bands at 0.9.  A mix with all bands at 0.5 should then show
        # deficits that wouldn't appear with default norms.
        prior = StylePrior(
            cluster_name="modern_trap",
            target_spectral_mean=[0.90] * 10,
            target_spectral_std=[0.05] * 10,
            reference_count=10,
            confidence=0.8,
        )
        corpus = ReferenceCorpus(
            priors={"modern_trap": prior},
            total_references=10,
        )

        # Mix with moderate values — should have big deficits vs 0.9 norms
        profile = _make_mix_profile(
            cluster="modern_trap",
            spectral=[0.50] * 10,
        )

        engine_with = NeedsEngine(corpus=corpus)
        needs_with = engine_with.diagnose(profile)

        # Without corpus (uses hardcoded norms which are closer to 0.5 in mids)
        engine_without = NeedsEngine()
        needs_without = engine_without.diagnose(profile)

        # The corpus version should detect more/different spectral issues
        spectral_with = [n for n in needs_with if n.category == "spectral"]
        spectral_without = [n for n in needs_without if n.category == "spectral"]

        # The high-norm corpus should produce at least some spectral needs
        # that the default norms wouldn't trigger (since default modern_trap
        # norms are already close to 0.50 in the mids)
        assert len(spectral_with) > 0, (
            "Custom corpus with 0.9 norms should trigger spectral needs "
            "for a mix at 0.5"
        )

    def test_engine_falls_back_for_unknown_cluster(self) -> None:
        """If the corpus has no prior for the cluster, fall back to hardcoded."""
        corpus = ReferenceCorpus(
            priors={"some_other_genre": StylePrior(cluster_name="some_other_genre")},
            total_references=1,
        )

        profile = _make_mix_profile(
            cluster="modern_trap",
            spectral=[0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.10, 0.05, 0.05, 0.02],
        )

        engine = NeedsEngine(corpus=corpus)
        needs = engine.diagnose(profile)

        # Should still work and detect top-end issues via hardcoded fallback
        spectral = [n for n in needs if n.category == "spectral"]
        descriptions = " ".join(n.description.lower() for n in spectral)
        assert "top-end too sparse" in descriptions


class TestNeedsEngineDefaultCorpus:
    """test_needs_engine_default_corpus — NeedsEngine works with no corpus arg."""

    def test_no_corpus_argument(self) -> None:
        """NeedsEngine() with no args should work as before."""
        engine = NeedsEngine()
        profile = _make_mix_profile(
            cluster="modern_trap",
            spectral=[0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.10, 0.05, 0.05, 0.02],
        )
        needs = engine.diagnose(profile)

        assert isinstance(needs, list)
        assert len(needs) > 0
        for n in needs:
            assert isinstance(n, NeedOpportunity)
            assert 0.0 <= n.severity <= 1.0

    def test_none_corpus_argument(self) -> None:
        """Explicitly passing corpus=None should behave the same as no arg."""
        engine = NeedsEngine(corpus=None)
        profile = _make_mix_profile(cluster="pop_production")
        needs = engine.diagnose(profile)
        assert isinstance(needs, list)
