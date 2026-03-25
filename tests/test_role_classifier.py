import pytest
from ml.classifiers.role_classifier import RoleClassifier


@pytest.fixture(scope="module")
def classifier():
    return RoleClassifier()


VALID_ROLES = {"kick", "snare", "clap", "hat", "bass", "lead", "pad",
               "fx", "texture", "vocal", "percussion", "unknown"}


def test_returns_role_and_confidence(classifier, kick_like):
    role, confidence = classifier.classify(str(kick_like))
    assert role in VALID_ROLES
    assert 0.0 <= confidence <= 1.0


def test_kick_classified(classifier, kick_like):
    role, _ = classifier.classify(str(kick_like))
    assert role in {"kick", "percussion", "bass"}  # Allow reasonable confusion


def test_hihat_classified(classifier, hihat_like):
    role, _ = classifier.classify(str(hihat_like))
    assert role in {"hat", "percussion"}


def test_pad_classified(classifier, pad_like):
    role, _ = classifier.classify(str(pad_like))
    assert role in {"pad", "texture", "lead"}


def test_classify_with_filename_hint(classifier, kick_like):
    role, conf = classifier.classify(str(kick_like), filename_hint="808_kick_hard.wav")
    assert role == "kick"
    assert conf > 0.5
