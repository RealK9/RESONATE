"""Shared fixtures for all tests. Generates synthetic audio for deterministic testing."""
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

SR = 44100


@pytest.fixture(scope="session")
def sample_rate():
    return SR


@pytest.fixture(scope="session")
def sine_440hz(tmp_path_factory):
    """1-second 440Hz sine wave, mono, 44.1kHz."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "sine_440.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def stereo_noise(tmp_path_factory):
    """0.5-second stereo white noise."""
    rng = np.random.default_rng(42)
    audio = (0.3 * rng.standard_normal((SR // 2, 2))).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "stereo_noise.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def kick_like(tmp_path_factory):
    """Synthetic kick drum: short sine sweep with fast decay."""
    duration = 0.15
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 150 * np.exp(-30 * t) + 40  # pitch sweep down
    phase = 2 * np.pi * np.cumsum(freq) / SR
    envelope = np.exp(-20 * t)
    audio = (0.9 * envelope * np.sin(phase)).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "kick.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def silence(tmp_path_factory):
    """1-second silence."""
    audio = np.zeros(SR, dtype=np.float32)
    path = tmp_path_factory.mktemp("audio") / "silence.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def pad_like(tmp_path_factory):
    """2-second rich pad: stacked detuned sines with slow attack."""
    duration = 2.0
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    envelope = 1.0 - np.exp(-2 * t)
    freqs = [220, 220.5, 330, 440.3, 554, 660.2]
    audio = np.zeros(n, dtype=np.float64)
    for f in freqs:
        audio += np.sin(2 * np.pi * f * t) / len(freqs)
    audio = (0.4 * envelope * audio).astype(np.float32)
    left = audio
    right = np.roll(audio, int(0.003 * SR))
    stereo = np.stack([left, right], axis=-1)
    path = tmp_path_factory.mktemp("audio") / "pad.wav"
    sf.write(str(path), stereo, SR)
    return path


@pytest.fixture(scope="session")
def hihat_like(tmp_path_factory):
    """Synthetic hi-hat: filtered noise burst."""
    duration = 0.08
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(n)
    envelope = np.exp(-60 * t)
    audio = (0.5 * envelope * noise).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "hihat.wav"
    sf.write(str(path), audio, SR)
    return path
