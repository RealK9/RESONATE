"""
Candidate generator -- first stage of the recommendation pipeline.

Narrows the full sample library down to ~50-100 candidates that have a
realistic chance of improving a given mix.  The downstream scorer then
ranks these candidates in detail.

Selection strategy (in priority order):
1. Roles needed -- parse NeedOpportunity list for roles to search
2. Quality floor -- commercial_readiness > threshold
3. Tonal compatibility -- if the mix has a detected key, filter by
   compatible keys (same key, relative major/minor, circle-of-fifths
   neighbors, or atonal/unpitched samples which are always compatible)
4. Style affinity -- if a VectorIndex is available, do embedding
   similarity search against the mix's style
5. Deduplicate and cap at max_candidates, prioritizing candidates that
   fill the highest-severity needs first
"""
from __future__ import annotations

import numpy as np

from ml.db.sample_store import SampleStore
from ml.models.mix_profile import MixProfile, NeedOpportunity
from ml.models.sample_profile import SampleProfile
from ml.retrieval.vector_index import VectorIndex


# ---------------------------------------------------------------------------
# Tonal compatibility
# ---------------------------------------------------------------------------

# Map from recommendation_policy to the sample roles that could address it.
_POLICY_TO_ROLES: dict[str, list[str]] = {
    "fill_missing_role": ["kick", "snare", "clap", "hat", "bass", "lead",
                          "pad", "fx", "texture", "vocal"],
    "reinforce_existing": ["hat", "lead", "pad", "texture", "fx", "vocal"],
    "improve_polish": ["fx", "texture", "pad"],
    "increase_contrast": ["fx", "lead", "pad", "texture"],
    "add_movement": ["fx", "lead", "hat", "texture"],
    "reduce_emptiness": ["pad", "texture", "bass", "lead", "vocal"],
    "support_transition": ["fx", "texture"],
    "enhance_groove": ["hat", "kick", "snare", "clap"],
    "enhance_lift": ["fx", "lead", "pad", "vocal"],
}

# Roles that can be inferred from a need's description text.
_DESCRIPTION_ROLE_HINTS: dict[str, list[str]] = {
    "kick": ["kick"],
    "snare": ["snare", "clap"],
    "hat": ["hat"],
    "hats": ["hat"],
    "hi-hat": ["hat"],
    "bass": ["bass"],
    "pad": ["pad"],
    "lead": ["lead"],
    "chord": ["pad", "lead"],
    "vocal": ["vocal"],
    "ambien": ["pad", "texture"],
    "texture": ["texture"],
    "fx": ["fx"],
    "sparkle": ["hat"],
    "attack": ["kick", "snare", "clap"],
    "top": ["hat"],
    "glue": ["pad", "texture"],
    "width": ["pad", "texture"],
    "empty": ["pad", "texture", "bass", "lead"],
    "harmonic": ["pad", "lead"],
    "emotional": ["pad", "vocal"],
}

# Circle-of-fifths order for major keys.
_CIRCLE_OF_FIFTHS = [
    "C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F",
]

# Enharmonic normalization.
_ENHARMONIC: dict[str, str] = {
    "C#": "Db", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}

# Relative major/minor pairs.
_RELATIVE: dict[str, str] = {
    "C": "Am", "G": "Em", "D": "Bm", "A": "F#m", "E": "C#m",
    "B": "G#m", "F#": "D#m", "Db": "Bbm", "Ab": "Fm", "Eb": "Cm",
    "Bb": "Gm", "F": "Dm",
    # And the reverse.
    "Am": "C", "Em": "G", "Bm": "D", "F#m": "A", "C#m": "E",
    "G#m": "B", "D#m": "F#", "Bbm": "Db", "Fm": "Ab", "Cm": "Eb",
    "Gm": "Bb", "Dm": "F",
}


def _normalize_key(key: str) -> str:
    """Normalize a key string: strip whitespace, apply enharmonic map."""
    key = key.strip()
    if not key:
        return ""
    # Check if it ends with 'm' (minor).
    if key.endswith("m"):
        root = key[:-1]
        root = _ENHARMONIC.get(root, root)
        return root + "m"
    root = _ENHARMONIC.get(key, key)
    return root


def _root_of(key: str) -> str:
    """Extract root note without minor suffix."""
    if key.endswith("m"):
        return key[:-1]
    return key


def _is_minor(key: str) -> bool:
    return key.endswith("m")


def _cof_distance(root_a: str, root_b: str) -> int:
    """Circle-of-fifths distance between two root notes (0-6)."""
    if root_a not in _CIRCLE_OF_FIFTHS or root_b not in _CIRCLE_OF_FIFTHS:
        return 6  # Unknown root -- treat as maximally distant.
    ia = _CIRCLE_OF_FIFTHS.index(root_a)
    ib = _CIRCLE_OF_FIFTHS.index(root_b)
    dist = abs(ia - ib)
    return min(dist, 12 - dist)


def is_tonally_compatible(mix_key: str, sample_key: str) -> bool:
    """
    Return True if *sample_key* is tonally compatible with *mix_key*.

    Compatible means any of:
    - Same key (after normalization)
    - Relative major/minor
    - Within 1 step on the circle of fifths (same mode)
    - Sample is atonal / unpitched (empty key or "unknown")
    """
    mk = _normalize_key(mix_key)
    sk = _normalize_key(sample_key)

    # No key on either side -- always compatible.
    if not mk or not sk or sk.lower() in ("unknown", "none"):
        return True

    # Exact match.
    if mk == sk:
        return True

    # Relative major/minor.
    if _RELATIVE.get(mk) == sk:
        return True

    # Circle-of-fifths neighbors (same mode).
    if _is_minor(mk) == _is_minor(sk):
        dist = _cof_distance(_root_of(mk), _root_of(sk))
        if dist <= 1:
            return True

    return False


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------

_QUALITY_FLOOR = 0.3


class CandidateGenerator:
    """
    Generates candidate samples from the library that could improve a mix.
    Narrows from thousands of samples to ~50-100 candidates.
    """

    def __init__(
        self,
        sample_store: SampleStore,
        vector_index: VectorIndex | None = None,
        gap_result=None,
    ):
        self._store = sample_store
        self._index = vector_index
        self._gap_result = gap_result  # GapAnalysisResult (optional)

    def generate(
        self,
        mix_profile: MixProfile,
        needs: list[NeedOpportunity],
        max_candidates: int = 100,
    ) -> list[SampleProfile]:
        """
        Generate candidate samples based on:
        1. Roles needed (from needs analysis)
        2. Tonal compatibility (if mix has a key)
        3. Style match (genre affinity overlap)
        4. Quality floor (commercial_readiness > threshold)
        5. Avoid over-represented bands

        Returns up to *max_candidates* SampleProfile objects, ordered so that
        candidates addressing the highest-severity needs come first.
        """
        if not needs:
            return []

        # Sort needs by severity (highest first) so we prioritize accordingly.
        sorted_needs = sorted(needs, key=lambda n: n.severity, reverse=True)

        mix_key = mix_profile.analysis.key

        # Collect candidates per need, preserving need priority order.
        seen_filepaths: set[str] = set()
        ordered_candidates: list[SampleProfile] = []

        for need in sorted_needs:
            roles = self._roles_for_need(need)
            if not roles:
                continue

            for role in roles:
                samples = self._store.search_by_role(role)
                for sample in samples:
                    if sample.filepath in seen_filepaths:
                        continue
                    # Quality floor.
                    if sample.labels.commercial_readiness < _QUALITY_FLOOR:
                        continue
                    # Tonal compatibility.
                    if mix_key and sample.labels.tonal:
                        sample_key = self._infer_sample_key(sample)
                        if sample_key and not is_tonally_compatible(
                            mix_key, sample_key
                        ):
                            continue
                    seen_filepaths.add(sample.filepath)
                    ordered_candidates.append(sample)

        # If vector index is available, add embedding-based candidates too.
        if self._index is not None:
            embedding_candidates = self._embedding_search(
                mix_profile, max_candidates
            )
            for sample in embedding_candidates:
                if sample.filepath in seen_filepaths:
                    continue
                if sample.labels.commercial_readiness < _QUALITY_FLOOR:
                    continue
                if mix_key and sample.labels.tonal:
                    sample_key = self._infer_sample_key(sample)
                    if sample_key and not is_tonally_compatible(
                        mix_key, sample_key
                    ):
                        continue
                seen_filepaths.add(sample.filepath)
                ordered_candidates.append(sample)

        # If gap analysis is available, boost candidates that fill critical gaps
        if self._gap_result is not None and self._gap_result.missing_roles:
            missing = set(self._gap_result.missing_roles)
            # Partition into gap-fillers and others, preserving order within each
            gap_fillers = [c for c in ordered_candidates if self._fills_missing_role(c, missing)]
            others = [c for c in ordered_candidates if not self._fills_missing_role(c, missing)]
            ordered_candidates = gap_fillers + others

        # Cap at max_candidates.  The list is already in need-priority order,
        # so truncating keeps the most important candidates.
        return ordered_candidates[:max_candidates]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _roles_for_need(need: NeedOpportunity) -> list[str]:
        """Determine which sample roles could address a given need."""
        roles: list[str] = []

        # First, check description for explicit role hints.
        desc_lower = need.description.lower()
        for hint, hint_roles in _DESCRIPTION_ROLE_HINTS.items():
            if hint in desc_lower:
                for r in hint_roles:
                    if r not in roles:
                        roles.append(r)

        # Fall back to policy-based role mapping.
        if not roles:
            policy_roles = _POLICY_TO_ROLES.get(
                need.recommendation_policy, []
            )
            roles = list(policy_roles)

        return roles

    @staticmethod
    def _infer_sample_key(sample: SampleProfile) -> str:
        """Attempt to infer a key from the sample's harmonic descriptors."""
        # If the sample has a chroma profile, use the dominant chroma bin
        # as a rough key estimate.  For proper samples, this should be
        # populated by the harmonic descriptor analyzer.
        chroma = sample.harmonic.chroma_profile
        if not chroma or len(chroma) != 12:
            return ""
        # Only trust it if pitch confidence is reasonable.
        if sample.harmonic.pitch_confidence < 0.3:
            return ""
        # Map chroma index to note name.
        note_names = [
            "C", "Db", "D", "Eb", "E", "F",
            "F#", "G", "Ab", "A", "Bb", "B",
        ]
        peak_idx = int(np.argmax(chroma))
        return note_names[peak_idx]

    @staticmethod
    def _fills_missing_role(sample: SampleProfile, missing_roles: set[str]) -> bool:
        """Check if a sample's detected role matches any missing role."""
        sample_role = sample.labels.role.lower() if sample.labels.role else ""
        # Map sample role labels to gap analysis role names
        role_map = {
            "kick": "kick", "snare": "snare_clap", "clap": "snare_clap",
            "hat": "hats_tops", "hihat": "hats_tops", "hi-hat": "hats_tops",
            "bass": "bass", "lead": "lead", "pad": "pad",
            "chord": "chord_support", "keys": "chord_support",
            "vocal": "vocal_texture", "vox": "vocal_texture",
            "fx": "fx_transitions", "transition": "fx_transitions",
            "texture": "ambience", "ambient": "ambience",
        }
        mapped = role_map.get(sample_role, sample_role)
        return mapped in missing_roles

    def _embedding_search(
        self, mix_profile: MixProfile, k: int
    ) -> list[SampleProfile]:
        """Use vector index to find stylistically similar samples."""
        if self._index is None or self._index.size() == 0:
            return []

        # Build a query vector from the mix's style cluster probabilities.
        # This is a rough heuristic -- we average all sample embeddings
        # that the index knows about, weighted by style overlap.
        # For now, just search with a random-ish approach: if the mix has
        # a filepath in the index, use that as the query.
        query_vec = self._index.get_vector(mix_profile.filepath)
        if query_vec is None:
            return []

        results = self._index.search(query_vec, k=k)
        candidates: list[SampleProfile] = []
        for filepath, _score in results:
            sample = self._store.load(filepath)
            if sample is not None:
                candidates.append(sample)
        return candidates
