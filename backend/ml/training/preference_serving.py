"""
Preference model serving — loads a trained UserTasteModel and provides
scoring for the reranker's ``user_preference`` (theta) component.
"""
from __future__ import annotations

from backend.ml.models.preference import UserTasteModel
from backend.ml.training.preference_dataset import PreferenceDataset


def _rescale_bias(bias_value: float) -> float:
    """Rescale a bias value from [-1, 1] to [0, 1].

    -1 maps to 0, 0 maps to 0.5, +1 maps to 1.
    """
    return max(0.0, min(1.0, (bias_value + 1.0) / 2.0))


class PreferenceServer:
    """Serve preference scores from a trained UserTasteModel.

    Used by the reranker to populate the ``user_preference`` (theta)
    scoring component that was previously hard-coded to 0.0.

    Parameters
    ----------
    dataset : PreferenceDataset
        Source of persisted taste models.
    """

    def __init__(self, dataset: PreferenceDataset) -> None:
        self._dataset = dataset
        self._model: UserTasteModel | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, user_id: str = "default") -> bool:
        """Load the taste model for *user_id*.

        Returns ``True`` if a model was found and loaded, ``False``
        otherwise.
        """
        model = self._dataset.load_taste_model(user_id)
        if model is None:
            self._model = None
            self._loaded = False
            return False
        self._model = model
        self._loaded = True
        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        sample_filepath: str,
        sample_role: str,
        context_style: str,
    ) -> float:
        """Return a preference score in [0, 1] for a candidate sample.

        Combines:

        * **Role bias** — how much the user likes this role, rescaled
          from [-1, 1] to [0, 1].
        * **Style bias** — how much the user likes this style context,
          rescaled the same way.

        Returns ``0.5`` (neutral) if no model is loaded.
        """
        if not self._loaded or self._model is None:
            return 0.5

        role_score = 0.5
        style_score = 0.5

        # Role component
        if sample_role and sample_role in self._model.role_bias:
            role_score = _rescale_bias(self._model.role_bias[sample_role])

        # Style component
        if context_style and context_style in self._model.style_bias:
            style_score = _rescale_bias(
                self._model.style_bias[context_style]
            )

        # Weighted average — role and style contribute equally when both
        # are available; if only one has a bias entry the other stays at
        # 0.5 which pulls the average toward neutral.
        has_role = sample_role in (self._model.role_bias or {})
        has_style = context_style in (self._model.style_bias or {})

        if has_role and has_style:
            return 0.5 * role_score + 0.5 * style_score
        if has_role:
            return role_score
        if has_style:
            return style_score
        return 0.5

    # ------------------------------------------------------------------
    # Weight adjustments
    # ------------------------------------------------------------------

    def get_weight_adjustments(self) -> dict[str, float]:
        """Return weight deltas to be ADDED to the reranker's defaults.

        Returns an empty dict if no model is loaded.
        """
        if not self._loaded or self._model is None:
            return {}
        return dict(self._model.weight_deltas)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether a taste model has been successfully loaded."""
        return self._loaded

    @property
    def model_version(self) -> int:
        """Version of the currently loaded model (0 if none)."""
        return self._model.model_version if self._model else 0
