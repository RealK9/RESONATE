from ml.classifiers.quality_scorer import QualityScorer


def test_returns_float(kick_like):
    scorer = QualityScorer()
    score = scorer.score(str(kick_like))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_silence_low_quality(silence):
    scorer = QualityScorer()
    score = scorer.score(str(silence))
    assert score < 0.3


def test_real_sound_higher_quality(kick_like, silence):
    scorer = QualityScorer()
    kick_score = scorer.score(str(kick_like))
    silence_score = scorer.score(str(silence))
    assert kick_score > silence_score
