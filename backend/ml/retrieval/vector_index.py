"""
FAISS-based vector index for fast nearest-neighbor sample retrieval.
Supports add, remove, search, save/load.
"""
from __future__ import annotations
import json
import numpy as np
import faiss
from pathlib import Path


class VectorIndex:
    """FAISS vector index mapping filepaths to embedding vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        # Use IndexFlatIP (inner product) for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(dim)
        self._id_to_filepath: list[str] = []
        self._filepath_to_id: dict[str, int] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, filepath: str, vector: np.ndarray):
        """Add a vector for a filepath. Overwrites if exists."""
        if filepath in self._filepath_to_id:
            self.remove(filepath)

        vec = vector.astype(np.float32).reshape(1, -1)
        # L2 normalize for cosine similarity via inner product
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        idx = len(self._id_to_filepath)
        self.index.add(vec)
        self._id_to_filepath.append(filepath)
        self._filepath_to_id[filepath] = idx
        self._vectors[filepath] = vec.flatten()

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Search for k nearest neighbors. Returns [(filepath, score), ...]."""
        if self.index.ntotal == 0:
            return []

        q = query.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._id_to_filepath):
                results.append((self._id_to_filepath[idx], float(score)))
        return results

    def get_vector(self, filepath: str) -> np.ndarray | None:
        """Get the stored vector for a filepath."""
        return self._vectors.get(filepath)

    def remove(self, filepath: str):
        """Remove a filepath from the index. Rebuilds index."""
        if filepath not in self._filepath_to_id:
            return
        del self._vectors[filepath]
        self._rebuild()

    def size(self) -> int:
        return self.index.ntotal

    def _rebuild(self):
        """Rebuild the FAISS index from stored vectors."""
        self.index = faiss.IndexFlatIP(self.dim)
        self._id_to_filepath = []
        self._filepath_to_id = {}

        for fp, vec in self._vectors.items():
            idx = len(self._id_to_filepath)
            self.index.add(vec.reshape(1, -1))
            self._id_to_filepath.append(fp)
            self._filepath_to_id[fp] = idx

    def save(self, path: str):
        """Save index to disk (FAISS index + metadata)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        metadata = {
            "dim": self.dim,
            "filepaths": self._id_to_filepath,
        }
        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f)
        # Save vectors for rebuild capability
        np.savez(str(p / "vectors.npz"), **{
            fp.replace("/", "__SLASH__"): vec
            for fp, vec in self._vectors.items()
        })

    @classmethod
    def load(cls, path: str) -> VectorIndex:
        """Load index from disk."""
        p = Path(path)
        with open(p / "metadata.json") as f:
            metadata = json.load(f)

        idx = cls(dim=metadata["dim"])
        idx.index = faiss.read_index(str(p / "index.faiss"))
        idx._id_to_filepath = metadata["filepaths"]
        idx._filepath_to_id = {fp: i for i, fp in enumerate(idx._id_to_filepath)}

        # Load vectors
        data = np.load(str(p / "vectors.npz"))
        for key in data:
            fp = key.replace("__SLASH__", "/")
            idx._vectors[fp] = data[key]

        return idx
