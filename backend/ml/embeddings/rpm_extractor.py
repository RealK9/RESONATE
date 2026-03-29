"""
RESONATE Production Model — Production Inference Extractor.

Replaces the old EmbeddingManager (CLAP + PANNs + AST) with a single RPM model.
Supports both PyTorch and ONNX Runtime backends for maximum flexibility.

Usage:
    extractor = RPMExtractor()  # auto-detects ONNX or PyTorch
    result = extractor.extract("path/to/audio.wav")
    embedding = result.embedding        # 768-d numpy array
    labels = result.labels              # all predicted labels
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────

ROLE_NAMES = [
    "kick", "snare", "clap", "hat", "perc",
    "bass", "lead", "pad", "fx", "texture", "vocal",
]

ERA_NAMES = [
    "1950s", "1960s", "1970s", "1980s",
    "1990s", "2000s", "2010s", "2020s",
]

KEY_NAMES = [
    "C major", "C# major", "D major", "D# major", "E major", "F major",
    "F# major", "G major", "G# major", "A major", "A# major", "B major",
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
]

MODE_NAMES = [
    "Ionian", "Dorian", "Phrygian", "Lydian",
    "Mixolydian", "Aeolian", "Locrian",
]

PERCEPTUAL_NAMES = [
    "brightness", "warmth", "air", "punch",
    "body", "bite", "smoothness", "width", "depth",
]


@dataclass
class RPMResult:
    """Complete analysis result from the RPM model."""

    # Core embedding for FAISS retrieval
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(768, dtype=np.float32))

    # Head 1: Role
    role: str = "unknown"
    role_confidence: float = 0.0
    role_distribution: dict[str, float] = field(default_factory=dict)

    # Head 2: Genre
    genre_top: str = "unknown"
    genre_top_confidence: float = 0.0
    genre_sub: str = "unknown"
    genre_sub_confidence: float = 0.0
    genre_distribution: dict[str, float] = field(default_factory=dict)

    # Head 3: Instruments
    instruments: list[tuple[str, float]] = field(default_factory=list)  # (name, confidence) pairs

    # Head 4: Theory
    key: str = "unknown"
    key_confidence: float = 0.0
    chord_quality: str = "unknown"
    mode: str = "unknown"

    # Head 5: Perceptual
    perceptual: dict[str, float] = field(default_factory=dict)

    # Head 6: Era
    era: str = "unknown"
    era_confidence: float = 0.0
    era_distribution: dict[str, float] = field(default_factory=dict)

    # Head 7: Chart potential
    chart_potential: float = 0.0

    def to_dict(self) -> dict:
        """Convert to serializable dict (no numpy)."""
        return {
            "embedding": self.embedding.tolist(),
            "role": self.role,
            "role_confidence": self.role_confidence,
            "role_distribution": self.role_distribution,
            "genre_top": self.genre_top,
            "genre_top_confidence": self.genre_top_confidence,
            "genre_sub": self.genre_sub,
            "genre_sub_confidence": self.genre_sub_confidence,
            "genre_distribution": self.genre_distribution,
            "instruments": self.instruments,
            "key": self.key,
            "key_confidence": self.key_confidence,
            "chord_quality": self.chord_quality,
            "mode": self.mode,
            "perceptual": self.perceptual,
            "era": self.era,
            "era_confidence": self.era_confidence,
            "era_distribution": self.era_distribution,
            "chart_potential": self.chart_potential,
        }


# ──────────────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────────────

class RPMExtractor:
    """
    Production inference with the RESONATE Production Model.
    Supports ONNX Runtime (fast) and PyTorch (flexible) backends.
    """

    def __init__(self, model_dir: str = "~/.resonate/rpm_models",
                 backend: str = "auto", device: str = "auto",
                 genre_labels: Optional[dict] = None,
                 instrument_labels: Optional[list] = None):
        """
        Args:
            model_dir: directory containing rpm_full.onnx or rpm_final.pt
            backend: "onnx", "pytorch", or "auto" (prefers ONNX)
            device: "auto", "cuda", "mps", or "cpu"
            genre_labels: optional dict mapping genre IDs to names
            instrument_labels: optional list of instrument names
        """
        self.model_dir = Path(model_dir).expanduser()
        self.genre_labels = genre_labels or {}
        self.instrument_labels = instrument_labels or []
        self._backend = None
        self._model = None
        self._session = None
        self._feature_extractor = None

        # Resolve device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        # Resolve backend
        if backend == "auto":
            if (self.model_dir / "rpm_full.onnx").exists() or (self.model_dir / "rpm_embedding.onnx").exists():
                try:
                    import onnxruntime
                    self._backend = "onnx"
                except ImportError:
                    self._backend = "pytorch"
            else:
                self._backend = "pytorch"
        else:
            self._backend = backend

        logger.info(f"RPMExtractor: backend={self._backend}, device={self._device}")

    @property
    def feature_extractor(self):
        """Lazy-load the AST feature extractor."""
        if self._feature_extractor is None:
            from transformers import ASTFeatureExtractor
            self._feature_extractor = ASTFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
        return self._feature_extractor

    def _load_onnx(self):
        """Load ONNX Runtime session."""
        import onnxruntime as ort

        model_path = self.model_dir / "rpm_full.onnx"
        if not model_path.exists():
            # Fall back to embedding-only model
            model_path = self.model_dir / "rpm_embedding.onnx"

        providers = []
        if self._device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        logger.info(f"ONNX model loaded from {model_path}")

    def _load_pytorch(self):
        """Load PyTorch model."""
        import torch
        from ml.training.rpm_model import RPMModel, RPMConfig

        # Try final model first, then best checkpoint
        for name in ["rpm_final.pt", "rpm_best.pt"]:
            path = self.model_dir / name
            if path.exists():
                cfg = RPMConfig(freeze_backbone=False)
                self._model = RPMModel(cfg)
                state = torch.load(str(path), map_location=self._device, weights_only=False)
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                # Strip _orig_mod. prefix from torch.compile'd checkpoints
                cleaned = {}
                for k, v in state.items():
                    cleaned[k.replace("_orig_mod.", "")] = v
                self._model.load_state_dict(cleaned, strict=False)
                self._model = self._model.to(self._device)
                self._model.eval()
                logger.info(f"PyTorch model loaded from {path}")
                return

        raise FileNotFoundError(f"No model found in {self.model_dir}")

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._backend == "onnx" and self._session is None:
            self._load_onnx()
        elif self._backend == "pytorch" and self._model is None:
            self._load_pytorch()

    def _load_audio(self, filepath: str) -> np.ndarray:
        """Load audio at 16kHz mono, max 10 seconds."""
        import librosa
        audio, _ = librosa.load(filepath, sr=16000, mono=True, duration=10.0)
        return audio

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram features using ASTFeatureExtractor."""
        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        )
        return inputs.input_values  # [1, T, F]

    def extract(self, filepath: str) -> RPMResult:
        """
        Full extraction: audio → embedding + all labels.

        Args:
            filepath: path to audio file

        Returns:
            RPMResult with embedding and all predictions
        """
        self._ensure_loaded()
        result = RPMResult()

        try:
            audio = self._load_audio(filepath)
            features = self._extract_features(audio)

            if self._backend == "onnx":
                result = self._run_onnx(features)
            else:
                result = self._run_pytorch(features)

        except Exception as e:
            logger.error(f"RPM extraction failed for {filepath}: {e}")

        return result

    def extract_embedding_only(self, filepath: str) -> np.ndarray:
        """Extract only the 768-d embedding (faster — skips head decoding)."""
        self._ensure_loaded()

        try:
            audio = self._load_audio(filepath)
            features = self._extract_features(audio)

            if self._backend == "onnx":
                outputs = self._session.run(["embedding"], {"input_values": features})
                return outputs[0].flatten().astype(np.float32)
            else:
                import torch
                with torch.no_grad():
                    input_tensor = torch.from_numpy(features).to(self._device)
                    embedding = self._model.get_embedding(input_tensor)
                    return embedding.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"RPM embedding extraction failed for {filepath}: {e}")
            return np.zeros(768, dtype=np.float32)

    def _run_onnx(self, features: np.ndarray) -> RPMResult:
        """Run ONNX inference and decode outputs."""
        output_names = [o.name for o in self._session.get_outputs()]
        outputs = self._session.run(output_names, {"input_values": features})

        result = RPMResult()

        # Map outputs by name
        out_map = dict(zip(output_names, outputs))

        # Embedding
        if "embedding" in out_map:
            result.embedding = out_map["embedding"].flatten().astype(np.float32)

        # Role
        if "role_logits" in out_map:
            self._decode_role(out_map["role_logits"], result)

        # Genre
        if "genre_top_logits" in out_map:
            self._decode_genre(out_map["genre_top_logits"], out_map.get("genre_sub_logits"), result)

        # Instruments
        if "instrument_probs" in out_map:
            self._decode_instruments(out_map["instrument_probs"], result)

        # Theory
        if "key_logits" in out_map:
            self._decode_theory(out_map["key_logits"], out_map.get("chord_logits"),
                              out_map.get("mode_logits"), result)

        # Perceptual
        if "perceptual" in out_map:
            self._decode_perceptual(out_map["perceptual"], result)

        # Era
        if "era_probs" in out_map:
            self._decode_era(out_map["era_probs"], result)

        # Chart
        if "chart_potential" in out_map:
            result.chart_potential = float(out_map["chart_potential"].flatten()[0])

        return result

    def _run_pytorch(self, features: np.ndarray) -> RPMResult:
        """Run PyTorch inference and decode outputs."""
        import torch

        result = RPMResult()

        with torch.no_grad():
            input_tensor = torch.from_numpy(features).to(self._device)
            outputs = self._model(input_tensor)

        # Convert all to numpy
        np_outputs = {k: v.cpu().numpy() for k, v in outputs.items()}

        result.embedding = np_outputs["embedding"].flatten().astype(np.float32)
        self._decode_role(np_outputs["role_logits"], result)
        self._decode_genre(np_outputs["genre_top_logits"], np_outputs.get("genre_sub_logits"), result)
        self._decode_instruments(
            1 / (1 + np.exp(-np_outputs["instrument_logits"])),  # sigmoid
            result
        )
        self._decode_theory(np_outputs["key_logits"], np_outputs.get("chord_logits"),
                          np_outputs.get("mode_logits"), result)
        self._decode_perceptual(np_outputs["perceptual"], result)
        self._decode_era(np_outputs["era_probabilities"], result)
        result.chart_potential = float(np_outputs["chart_potential"].flatten()[0])

        return result

    # ── Decoders ──

    def _decode_role(self, logits: np.ndarray, result: RPMResult):
        probs = _softmax(logits.flatten())
        idx = int(np.argmax(probs))
        result.role = ROLE_NAMES[idx] if idx < len(ROLE_NAMES) else "unknown"
        result.role_confidence = float(probs[idx])
        result.role_distribution = {
            ROLE_NAMES[i]: float(probs[i])
            for i in range(min(len(probs), len(ROLE_NAMES)))
        }

    def _decode_genre(self, top_logits: np.ndarray, sub_logits: Optional[np.ndarray],
                      result: RPMResult):
        top_probs = _softmax(top_logits.flatten())
        top_idx = int(np.argmax(top_probs))
        result.genre_top = self.genre_labels.get(top_idx, f"genre_{top_idx}")
        result.genre_top_confidence = float(top_probs[top_idx])
        result.genre_distribution = {
            self.genre_labels.get(i, f"genre_{i}"): float(top_probs[i])
            for i in range(len(top_probs))
        }

        if sub_logits is not None:
            sub_probs = _softmax(sub_logits.flatten())
            sub_idx = int(np.argmax(sub_probs))
            result.genre_sub = self.genre_labels.get(sub_idx, f"subgenre_{sub_idx}")
            result.genre_sub_confidence = float(sub_probs[sub_idx])

    def _decode_instruments(self, probs: np.ndarray, result: RPMResult):
        probs = probs.flatten()
        # Get instruments with confidence > 0.3
        top_instruments = []
        for i, p in enumerate(probs):
            if p > 0.3:
                name = self.instrument_labels[i] if i < len(self.instrument_labels) else f"instrument_{i}"
                top_instruments.append((name, float(p)))
        top_instruments.sort(key=lambda x: x[1], reverse=True)
        result.instruments = top_instruments[:10]  # top 10

    def _decode_theory(self, key_logits: np.ndarray, chord_logits: Optional[np.ndarray],
                       mode_logits: Optional[np.ndarray], result: RPMResult):
        key_probs = _softmax(key_logits.flatten())
        key_idx = int(np.argmax(key_probs))
        result.key = KEY_NAMES[key_idx] if key_idx < len(KEY_NAMES) else "unknown"
        result.key_confidence = float(key_probs[key_idx])

        if chord_logits is not None:
            chord_probs = _softmax(chord_logits.flatten())
            CHORD_NAMES = ["major", "minor", "dom7", "maj7", "min7", "dim",
                          "aug", "sus2", "sus4", "dim7", "minmaj7", "other"]
            chord_idx = int(np.argmax(chord_probs))
            result.chord_quality = CHORD_NAMES[chord_idx] if chord_idx < len(CHORD_NAMES) else "unknown"

        if mode_logits is not None:
            mode_probs = _softmax(mode_logits.flatten())
            mode_idx = int(np.argmax(mode_probs))
            result.mode = MODE_NAMES[mode_idx] if mode_idx < len(MODE_NAMES) else "unknown"

    def _decode_perceptual(self, values: np.ndarray, result: RPMResult):
        values = values.flatten()
        result.perceptual = {
            PERCEPTUAL_NAMES[i]: float(values[i])
            for i in range(min(len(values), len(PERCEPTUAL_NAMES)))
        }

    def _decode_era(self, probs: np.ndarray, result: RPMResult):
        probs = probs.flatten()
        idx = int(np.argmax(probs))
        result.era = ERA_NAMES[idx] if idx < len(ERA_NAMES) else "unknown"
        result.era_confidence = float(probs[idx])
        result.era_distribution = {
            ERA_NAMES[i]: float(probs[i])
            for i in range(min(len(probs), len(ERA_NAMES)))
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
