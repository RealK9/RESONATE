import pytest
import soundfile as sf
import numpy as np
from backend.ml.pipeline.batch_processor import BatchProcessor


@pytest.fixture
def sample_dir(tmp_path, sample_rate):
    """Create a directory with multiple test audio files."""
    for name in ["kick.wav", "snare.wav", "hat.wav"]:
        t = np.linspace(0, 0.1, sample_rate // 10, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        sf.write(str(tmp_path / name), audio, sample_rate)
    return tmp_path


def test_batch_discovers_files(sample_dir):
    processor = BatchProcessor(skip_embeddings=True)
    files = processor.discover_audio_files(str(sample_dir))
    assert len(files) == 3


def test_batch_processes_all(sample_dir, tmp_path):
    db_path = tmp_path / "test_batch.db"
    processor = BatchProcessor(skip_embeddings=True, db_path=str(db_path))
    results = processor.process_directory(str(sample_dir))
    assert results["processed"] == 3
    assert results["failed"] == 0


def test_batch_reports_progress(sample_dir, tmp_path):
    db_path = tmp_path / "test_batch.db"
    processor = BatchProcessor(skip_embeddings=True, db_path=str(db_path))
    progress_updates = []
    processor.process_directory(
        str(sample_dir),
        on_progress=lambda done, total: progress_updates.append((done, total))
    )
    assert len(progress_updates) == 3
    assert progress_updates[-1] == (3, 3)
