"""
Genre and era affinity classifier.
Phase 1: rule-based using spectral/rhythmic features + PANNs tags.
Phase 3 will replace with learned style-cluster models.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf

GENRES = [
    "trap", "hip-hop", "drill", "edm", "house", "techno",
    "pop", "r&b", "lo-fi", "cinematic", "dnb", "afro",
    "melodic-techno", "bass-music", "ambient"
]

ERAS = ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]

# Genre spectral signatures (centroid range, sub_bass_ratio, transient_density)
GENRE_SIGNATURES = {
    "trap": {"centroid": (800, 3000), "sub_bass": (0.2, 0.6), "transient": (2, 8)},
    "house": {"centroid": (1500, 4000), "sub_bass": (0.1, 0.3), "transient": (4, 10)},
    "techno": {"centroid": (1000, 3500), "sub_bass": (0.1, 0.4), "transient": (5, 12)},
    "drill": {"centroid": (800, 2500), "sub_bass": (0.2, 0.5), "transient": (3, 8)},
    "lo-fi": {"centroid": (500, 2000), "sub_bass": (0.05, 0.2), "transient": (1, 4)},
    "cinematic": {"centroid": (1000, 5000), "sub_bass": (0.05, 0.3), "transient": (0, 3)},
    "ambient": {"centroid": (500, 3000), "sub_bass": (0.02, 0.2), "transient": (0, 2)},
}


class GenreEraClassifier:
    """Classify genre and era affinity from audio features."""

    def classify_genre(self, filepath: str,
                       panns_tags: dict[str, float] | None = None) -> dict[str, float]:
        """Return genre affinity scores (0-1) for each genre."""
        features = self._extract_features(filepath)
        scores = {}

        for genre in GENRES:
            score = 0.0
            if genre in GENRE_SIGNATURES:
                sig = GENRE_SIGNATURES[genre]
                # Centroid match
                c_lo, c_hi = sig["centroid"]
                if c_lo <= features["centroid"] <= c_hi:
                    score += 0.3
                # Sub-bass match
                sb_lo, sb_hi = sig["sub_bass"]
                if sb_lo <= features["sub_bass_ratio"] <= sb_hi:
                    score += 0.3
                # Transient density match
                t_lo, t_hi = sig["transient"]
                if t_lo <= features["transient_density"] <= t_hi:
                    score += 0.2
            else:
                score = 0.1  # Uniform low prior for genres without signatures

            # Boost from PANNs tags
            if panns_tags:
                for tag, conf in panns_tags.items():
                    tag_lower = tag.lower()
                    if genre in tag_lower or tag_lower in genre:
                        score += conf * 0.3

            scores[genre] = round(min(score, 1.0), 4)

        return scores

    def classify_era(self, filepath: str) -> dict[str, float]:
        """Return era affinity scores. Rule-based on spectral characteristics."""
        features = self._extract_features(filepath)
        scores = {}

        centroid = features["centroid"]
        bandwidth = features["bandwidth"]

        # Older eras: narrower bandwidth, lower centroid
        # Newer eras: wider bandwidth, higher centroid (generally)
        for era in ERAS:
            decade = int(era[:4])
            # Simple model: more modern = brighter, wider bandwidth
            modernity = (decade - 1970) / 50.0  # 0 to 1
            centroid_score = 1.0 - abs(centroid / 6000.0 - modernity)
            bw_score = 1.0 - abs(bandwidth / 5000.0 - modernity)
            scores[era] = round(float(np.clip((centroid_score + bw_score) / 2, 0, 1)), 4)

        return scores

    def _extract_features(self, filepath: str) -> dict:
        """Quick feature extraction for classification."""
        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        duration = len(mono) / sr

        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total = mean_spec.sum()

        centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())

        sub_mask = (freqs >= 20) & (freqs < 100)
        sub_bass_ratio = float(mean_spec[sub_mask].sum() / total) if total > 0 else 0

        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        transient_density = len(onsets) / duration if duration > 0 else 0

        return {
            "centroid": centroid,
            "bandwidth": bandwidth,
            "sub_bass_ratio": sub_bass_ratio,
            "transient_density": transient_density,
            "duration": duration,
        }
