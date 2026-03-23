import numpy as np
from backend.ml.analysis.transient_descriptors import extract_transient_descriptors
from backend.ml.models.sample_profile import TransientDescriptors


def test_returns_transient_descriptors(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert isinstance(result, TransientDescriptors)


def test_kick_has_onset(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.onset_count >= 1


def test_hihat_has_onset(hihat_like):
    result = extract_transient_descriptors(str(hihat_like))
    assert result.onset_count >= 1


def test_onset_positions_in_range(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    for pos in result.transient_positions:
        assert 0.0 <= pos <= 0.2  # kick is 150ms


def test_onset_strength_positive(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.onset_strength_mean > 0


def test_kick_sharp_attack(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.attack_sharpness > 0.3


def test_pad_soft_attack(pad_like):
    result = extract_transient_descriptors(str(pad_like))
    assert result.attack_sharpness < 0.5


def test_silence_no_onsets(silence):
    result = extract_transient_descriptors(str(silence))
    assert result.onset_count == 0


def test_transient_density_reasonable(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.transient_density >= 0
