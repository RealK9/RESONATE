import numpy as np
import soundfile as sf
from ml.analysis.loop_detection import detect_loop


def test_oneshot_kick(kick_like):
    is_loop, confidence = detect_loop(str(kick_like))
    assert is_loop is False
    assert confidence > 0.5


def test_oneshot_hihat(hihat_like):
    is_loop, confidence = detect_loop(str(hihat_like))
    assert is_loop is False


def test_loop_detection_on_repeated_pattern(tmp_path, sample_rate):
    """A repeated pattern should be detected as a loop."""
    # Create a 2-bar repeating pattern
    bar_len = sample_rate // 2  # 0.5s per bar
    t = np.linspace(0, 0.5, bar_len, endpoint=False)
    pattern = (0.3 * np.sin(2 * np.pi * 200 * t) *
               np.exp(-4 * (t % 0.125) / 0.125)).astype(np.float32)
    # Repeat 4 times
    audio = np.tile(pattern, 4)
    path = tmp_path / "loop_pattern.wav"
    sf.write(str(path), audio, sample_rate)
    is_loop, confidence = detect_loop(str(path))
    assert is_loop is True


def test_confidence_range(sine_440hz):
    _, confidence = detect_loop(str(sine_440hz))
    assert 0.0 <= confidence <= 1.0
