import numpy as np
from ml.analysis.core_descriptors import extract_core_descriptors
from ml.models.sample_profile import CoreDescriptors


def test_sine_duration(sine_440hz, sample_rate):
    result = extract_core_descriptors(str(sine_440hz))
    assert isinstance(result, CoreDescriptors)
    assert abs(result.duration - 1.0) < 0.01


def test_sine_sample_rate(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert result.sample_rate == 44100


def test_mono_detection(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert result.channels == 1


def test_stereo_detection(stereo_noise):
    result = extract_core_descriptors(str(stereo_noise))
    assert result.channels == 2


def test_rms_nonzero_for_signal(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert result.rms > 0.1


def test_rms_near_zero_for_silence(silence):
    result = extract_core_descriptors(str(silence))
    assert result.rms < 0.001


def test_lufs_reasonable_range(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    # 440Hz sine at 0.5 amplitude should be roughly -10 to -6 LUFS
    assert -20.0 < result.lufs < 0.0


def test_peak_amplitude(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert 0.4 < result.peak < 0.6  # sine at 0.5 amplitude


def test_crest_factor_sine(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    # Sine wave crest factor = sqrt(2) ≈ 3.01 dB
    assert 2.0 < result.crest_factor < 5.0


def test_kick_fast_attack(kick_like):
    result = extract_core_descriptors(str(kick_like))
    assert result.attack_time < 0.01  # kick has near-instant attack


def test_pad_slow_attack(pad_like):
    result = extract_core_descriptors(str(pad_like))
    assert result.attack_time > 0.05  # pad has slow attack


def test_kick_fast_decay(kick_like):
    result = extract_core_descriptors(str(kick_like))
    assert result.decay_time < 0.2


def test_sustain_level_range(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert 0.0 <= result.sustain_level <= 1.0
