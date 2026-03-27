from ml.classifiers.style_tagger import StyleTagger

EXPECTED_TAGS = {"bright", "dark", "wide", "punchy", "analog", "digital",
                 "gritty", "clean", "warm", "airy", "tight", "loose"}


def test_returns_dict(kick_like):
    tagger = StyleTagger()
    tags = tagger.tag(str(kick_like))
    assert isinstance(tags, dict)
    for tag, score in tags.items():
        assert isinstance(tag, str)
        assert 0.0 <= score <= 1.0


def test_kick_is_punchy(kick_like):
    tagger = StyleTagger()
    tags = tagger.tag(str(kick_like))
    # A kick should have some "punchy" quality
    assert "punchy" in tags
    assert tags["punchy"] > 0.1


def test_noise_is_not_clean(stereo_noise):
    tagger = StyleTagger()
    tags = tagger.tag(str(stereo_noise))
    if "clean" in tags:
        assert tags["clean"] < 0.5
