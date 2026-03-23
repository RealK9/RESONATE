"""
Sound role classifier: kick/snare/clap/hat/bass/lead/pad/fx/texture/vocal/percussion.
Uses a combination of:
1. Filename-based heuristics (high confidence when keywords match)
2. Audio feature rules (spectral centroid, duration, transients, pitch)
3. PANNs tag mapping (when available)

Later phases will add a trained neural classifier on top.
"""
from __future__ import annotations
import re
import numpy as np
from pathlib import Path

# Keyword -> role mapping (prioritized)
ROLE_KEYWORDS: dict[str, list[str]] = {
    "kick": ["kick", "kik", "808", "bd"],
    "snare": ["snare", "snr", "sd", "rim"],
    "clap": ["clap", "clp", "handclap"],
    "hat": ["hat", "hh", "hihat", "hi-hat", "openhat", "closedhat", "oh", "ch"],
    "bass": ["bass", "sub", "808bass", "reese", "bassline"],
    "lead": ["lead", "ld", "synth", "pluck", "stab"],
    "pad": ["pad", "atmosphere", "atmo", "ambient", "drone"],
    "fx": ["fx", "effect", "riser", "downlifter", "sweep", "impact", "whoosh", "transition"],
    "texture": ["texture", "foley", "noise", "grain", "field"],
    "vocal": ["vocal", "vox", "voice", "acapella", "adlib", "chant", "choir"],
    "percussion": ["perc", "percussion", "conga", "bongo", "shaker", "tambourine", "tom", "cymbal", "crash", "ride"],
}

# PANNs AudioSet tag -> role mapping
PANNS_TAG_MAP: dict[str, str] = {
    "Bass drum": "kick", "Kick drum": "kick",
    "Snare drum": "snare",
    "Clapping": "clap",
    "Hi-hat": "hat", "Cymbal": "hat",
    "Bass guitar": "bass", "Bass": "bass",
    "Synthesizer": "lead",
    "Singing": "vocal", "Speech": "vocal", "Voice": "vocal",
    "Drum": "percussion", "Drum kit": "percussion",
}


class RoleClassifier:
    """Classify the role/function of an audio sample."""

    def classify(self, filepath: str, filename_hint: str | None = None,
                 panns_tags: dict[str, float] | None = None) -> tuple[str, float]:
        """
        Classify sample role.
        Returns (role, confidence).
        """
        filename = filename_hint or Path(filepath).name
        scores: dict[str, float] = {role: 0.0 for role in ROLE_KEYWORDS}
        scores["unknown"] = 0.1  # small prior

        # Stage 1: Filename heuristics
        filename_lower = re.sub(r"[_\-\.]", " ", filename.lower())
        tokens = set(filename_lower.split())
        for role, keywords in ROLE_KEYWORDS.items():
            for kw in keywords:
                if kw in tokens or kw in filename_lower:
                    scores[role] += 0.7
                    break

        # Stage 2: Audio features (lightweight analysis)
        try:
            feature_scores = self._feature_based_scores(filepath)
            for role, score in feature_scores.items():
                scores[role] = scores.get(role, 0) + score * 0.5
        except Exception:
            pass

        # Stage 3: PANNs tags (if provided)
        if panns_tags:
            for tag, conf in panns_tags.items():
                if tag in PANNS_TAG_MAP and conf > 0.1:
                    role = PANNS_TAG_MAP[tag]
                    if role in scores:
                        scores[role] += conf * 0.4

        # Pick winner
        best_role = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_role]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0

        return best_role, round(confidence, 4)

    def _feature_based_scores(self, filepath: str) -> dict[str, float]:
        """Score roles based on audio features."""
        import soundfile as sf
        import librosa

        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        duration = len(mono) / sr

        # Quick spectral analysis
        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total_energy = mean_spec.sum()

        if total_energy < 1e-10:
            return {}

        def band_ratio(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float(mean_spec[mask].sum() / total_energy)

        sub_bass = band_ratio(20, 100)
        low_mid = band_ratio(100, 500)
        mid = band_ratio(500, 2000)
        high_mid = band_ratio(2000, 6000)
        high = band_ratio(6000, sr / 2)

        # Onset detection
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        onset_peak = onset_env.max() if len(onset_env) > 0 else 0
        onset_mean = onset_env.mean() if len(onset_env) > 0 else 0
        transient_ratio = onset_peak / onset_mean if onset_mean > 0 else 0

        scores: dict[str, float] = {}

        # Kick: short, sub-heavy, strong transient
        if duration < 1.0 and sub_bass > 0.3:
            scores["kick"] = 0.5 + min(transient_ratio / 10, 0.3)

        # Snare: short, mid-focused, strong transient
        if duration < 0.8 and mid > 0.2 and transient_ratio > 3:
            scores["snare"] = 0.4

        # Hat: short, high-frequency dominant
        if duration < 0.5 and high > 0.3:
            scores["hat"] = 0.5

        # Bass: longer, sub/low dominant, tonal
        if sub_bass + low_mid > 0.5 and duration > 0.3:
            scores["bass"] = 0.4

        # Pad: long, spread spectrum, slow
        if duration > 1.0 and transient_ratio < 3:
            scores["pad"] = 0.4

        # Lead: mid-focused, tonal
        if mid + high_mid > 0.4 and 0.1 < duration < 3.0:
            scores["lead"] = 0.3

        # FX: anything weird or long with lots of spectral change
        spectral_flux = np.mean(np.diff(S, axis=1) ** 2)
        if spectral_flux > mean_spec.mean() * 0.5:
            scores["fx"] = 0.3

        return scores
