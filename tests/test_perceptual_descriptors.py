import numpy as np
from ml.analysis.perceptual_descriptors import extract_perceptual_descriptors
from ml.models.sample_profile import PerceptualDescriptors


def test_returns_perceptual_descriptors(sine_440hz):
    result = extract_perceptual_descriptors(str(sine_440hz))
    assert isinstance(result, PerceptualDescriptors)


def test_all_values_in_range(sine_440hz):
    result = extract_perceptual_descriptors(str(sine_440hz))
    for field_name in ["brightness", "warmth", "air", "punch", "body",
                       "bite", "smoothness", "width", "depth_impression"]:
        val = getattr(result, field_name)
        assert 0.0 <= val <= 1.0, f"{field_name}={val} out of range"


def test_noise_has_brightness(stereo_noise):
    result = extract_perceptual_descriptors(str(stereo_noise))
    # White noise has lots of high-frequency content
    assert result.brightness > 0.3


def test_kick_has_punch(kick_like):
    result = extract_perceptual_descriptors(str(kick_like))
    assert result.punch > 0.2


def test_kick_has_body(kick_like):
    result = extract_perceptual_descriptors(str(kick_like))
    assert result.body > 0.1


def test_pad_has_warmth(pad_like):
    result = extract_perceptual_descriptors(str(pad_like))
    assert result.warmth > 0.1


def test_stereo_has_width(pad_like):
    result = extract_perceptual_descriptors(str(pad_like))
    assert result.width > 0.1


def test_mono_low_width(sine_440hz):
    result = extract_perceptual_descriptors(str(sine_440hz))
    assert result.width < 0.1


def test_hihat_has_bite(hihat_like):
    result = extract_perceptual_descriptors(str(hihat_like))
    assert result.bite > 0.1
