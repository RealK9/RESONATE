# tests/test_harmonic_descriptors.py
import numpy as np
from backend.ml.analysis.harmonic_descriptors import extract_harmonic_descriptors
from backend.ml.models.sample_profile import HarmonicDescriptors


def test_returns_harmonic_descriptors(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert isinstance(result, HarmonicDescriptors)


def test_f0_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    # Should detect ~440Hz
    assert 420 < result.f0 < 460


def test_pitch_confidence_high_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert result.pitch_confidence > 0.7


def test_pitch_confidence_low_for_noise(stereo_noise):
    result = extract_harmonic_descriptors(str(stereo_noise))
    assert result.pitch_confidence < 0.5


def test_chroma_profile_length(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert len(result.chroma_profile) == 12


def test_chroma_a_dominant_for_440(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    # A=440Hz, chroma index 9 (A)
    chroma = result.chroma_profile
    a_idx = 9
    assert chroma[a_idx] == max(chroma) or chroma[a_idx] > 0.5


def test_hnr_high_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert result.harmonic_to_noise_ratio > 10  # dB


def test_hnr_low_for_noise(stereo_noise):
    result = extract_harmonic_descriptors(str(stereo_noise))
    assert result.harmonic_to_noise_ratio < 10


def test_tonalness_high_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert result.tonalness > 0.5


def test_noisiness_high_for_noise(stereo_noise):
    result = extract_harmonic_descriptors(str(stereo_noise))
    assert result.noisiness > 0.3


def test_dissonance_range(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert 0.0 <= result.dissonance <= 1.0


def test_roughness_range(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert 0.0 <= result.roughness <= 1.0
