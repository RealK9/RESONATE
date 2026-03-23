from backend.ml.classifiers.genre_era_classifier import GenreEraClassifier


def test_genre_affinity_returns_dict(kick_like):
    clf = GenreEraClassifier()
    genres = clf.classify_genre(str(kick_like))
    assert isinstance(genres, dict)
    for genre, score in genres.items():
        assert 0.0 <= score <= 1.0


def test_era_affinity_returns_dict(kick_like):
    clf = GenreEraClassifier()
    eras = clf.classify_era(str(kick_like))
    assert isinstance(eras, dict)


def test_genre_keys_are_strings(sine_440hz):
    clf = GenreEraClassifier()
    genres = clf.classify_genre(str(sine_440hz))
    for key in genres:
        assert isinstance(key, str)
