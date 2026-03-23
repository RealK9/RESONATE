"""
Style tag predictor: bright/dark/wide/punchy/analog/digital/gritty/clean/warm/airy/tight/loose.
Maps from perceptual and spectral features to subjective style descriptors.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf


class StyleTagger:
    """Predict subjective style tags from audio."""

    def tag(self, filepath: str) -> dict[str, float]:
        """Return style tags with confidence scores (0-1)."""
        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        n_channels = audio.shape[1]
        mono = audio.mean(axis=1)

        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total = mean_spec.sum()

        if total < 1e-10:
            return {}

        def band_ratio(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(mean_spec[mask].sum() / total)

        high_energy = band_ratio(4000, sr / 2)
        low_energy = band_ratio(20, 300)
        mid_energy = band_ratio(300, 4000)
        presence = band_ratio(2000, 5000)

        # Onset analysis
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        transient_sharpness = 0.0
        if len(onset_env) > 0 and onset_env.mean() > 0:
            transient_sharpness = onset_env.max() / onset_env.mean()

        # Spectral flatness
        flatness = float(librosa.feature.spectral_flatness(S=S).mean())

        # Width
        if n_channels >= 2:
            corr = np.corrcoef(audio[:, 0], audio[:, 1])[0, 1]
            width_score = 1.0 - abs(corr) if not np.isnan(corr) else 0.0
        else:
            width_score = 0.0

        tags = {}

        # Bright vs Dark
        tags["bright"] = round(float(np.clip(high_energy * 5, 0, 1)), 4)
        tags["dark"] = round(float(np.clip(1.0 - high_energy * 3, 0, 1)), 4)

        # Wide
        tags["wide"] = round(float(np.clip(width_score, 0, 1)), 4)

        # Punchy
        tags["punchy"] = round(float(np.clip(transient_sharpness / 8.0, 0, 1)), 4)

        # Warm
        warm_band = band_ratio(200, 800)
        tags["warm"] = round(float(np.clip(warm_band * 4, 0, 1)), 4)

        # Airy
        air_band = band_ratio(10000, sr / 2)
        tags["airy"] = round(float(np.clip(air_band * 10, 0, 1)), 4)

        # Gritty vs Clean
        tags["gritty"] = round(float(np.clip(flatness * 3 + presence * 2, 0, 1)), 4)
        tags["clean"] = round(float(np.clip(1.0 - flatness * 2, 0, 1)), 4)

        # Analog vs Digital (approximation: analog = less perfect, warmer)
        spectral_regularity = float(np.std(np.diff(mean_spec)) / (mean_spec.mean() + 1e-10))
        tags["analog"] = round(float(np.clip(warm_band * 2 + (1 - flatness), 0, 1)), 4)
        tags["digital"] = round(float(np.clip(high_energy * 3 + flatness, 0, 1)), 4)

        # Tight vs Loose
        tags["tight"] = round(float(np.clip(transient_sharpness / 6.0, 0, 1)), 4)
        tags["loose"] = round(float(np.clip(1.0 - transient_sharpness / 8.0, 0, 1)), 4)

        return tags
