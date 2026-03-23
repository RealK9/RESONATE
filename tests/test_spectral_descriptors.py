import numpy as np
from backend.ml.analysis.spectral_descriptors import extract_spectral_descriptors
from backend.ml.models.sample_profile import SpectralDescriptors


def test_returns_spectral_descriptors(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result, SpectralDescriptors)


def test_centroid_for_sine(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    # 440Hz sine should have centroid near 440Hz
    assert 400 < result.centroid < 500


def test_centroid_for_noise(stereo_noise):
    result = extract_spectral_descriptors(str(stereo_noise))
    # White noise has centroid roughly at sr/4
    assert result.centroid > 2000


def test_flatness_high_for_noise(stereo_noise):
    result = extract_spectral_descriptors(str(stereo_noise))
    # Noise is spectrally flat
    assert result.flatness > 0.1


def test_flatness_low_for_sine(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    # Pure sine is spectrally concentrated
    assert result.flatness < 0.05


def test_rolloff_reasonable(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert result.rolloff > 0


def test_bandwidth_positive(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert result.bandwidth >= 0


def test_contrast_is_list(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result.contrast, list)
    assert len(result.contrast) >= 6  # librosa default 7 bands


def test_sub_to_bass_ratio_for_kick(kick_like):
    result = extract_spectral_descriptors(str(kick_like))
    # Kick has sub/bass energy
    assert result.sub_to_bass_ratio > 0


def test_harshness_zones_list(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result.harshness_zones, list)
    assert len(result.harshness_zones) >= 3  # multiple harsh bands


def test_resonant_peaks_list(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result.resonant_peaks, list)
