"""
Commercial readiness / quality scorer.
Evaluates: signal integrity, spectral balance, dynamic range, noise floor,
transient clarity, tonal quality.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf


class QualityScorer:
    """Score the commercial readiness / quality of an audio sample."""

    def score(self, filepath: str) -> float:
        """
        Return a quality score from 0.0 (unusable) to 1.0 (professional grade).
        """
        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)

        if len(mono) < 100:
            return 0.0

        scores = []

        # 1. Signal presence (not silence)
        rms = np.sqrt(np.mean(mono ** 2))
        peak = np.max(np.abs(mono))
        if rms < 1e-6:
            return 0.0
        scores.append(min(rms * 10, 1.0))

        # 2. Dynamic range (crest factor in reasonable range)
        crest = peak / rms if rms > 0 else 0
        # Good dynamic range: crest between 3-20 dB
        crest_db = 20 * np.log10(crest) if crest > 0 else 0
        if 3 < crest_db < 20:
            scores.append(0.8)
        elif crest_db <= 3:
            scores.append(0.4)  # Too compressed
        else:
            scores.append(0.5)  # Too dynamic / sparse

        # 3. Spectral balance (not all energy in one band)
        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total = mean_spec.sum()
        if total > 0:
            # Split into 4 bands and check balance
            bands = [
                (20, 250), (250, 2000), (2000, 8000), (8000, sr / 2)
            ]
            ratios = []
            for lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                ratios.append(mean_spec[mask].sum() / total)
            # Entropy-based balance score
            ratios = np.array(ratios)
            ratios = ratios[ratios > 0]
            entropy = -np.sum(ratios * np.log2(ratios + 1e-10))
            max_entropy = np.log2(len(bands))
            scores.append(entropy / max_entropy)
        else:
            scores.append(0.0)

        # 4. No DC offset
        dc_offset = abs(np.mean(mono))
        if dc_offset < 0.01:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - dc_offset * 10))

        # 5. No clipping
        clip_ratio = np.mean(np.abs(mono) > 0.99)
        scores.append(max(0, 1.0 - clip_ratio * 20))

        # 6. Noise floor quality (SNR proxy)
        # Sort RMS frames, compare bottom 10% to top 10%
        hop = max(1, sr // 100)
        n_frames = len(mono) // hop
        if n_frames > 10:
            frame_rms = np.array([
                np.sqrt(np.mean(mono[i*hop:(i+1)*hop] ** 2))
                for i in range(n_frames)
            ])
            sorted_rms = np.sort(frame_rms)
            noise_floor = np.mean(sorted_rms[:max(1, n_frames // 10)])
            signal_level = np.mean(sorted_rms[-max(1, n_frames // 10):])
            if noise_floor > 0:
                snr = signal_level / noise_floor
                scores.append(min(snr / 100, 1.0))
            else:
                scores.append(1.0)
        else:
            scores.append(0.5)

        return round(float(np.mean(scores)), 4)
