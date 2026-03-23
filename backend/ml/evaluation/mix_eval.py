"""
Mix-level evaluation metrics for RESONATE's analysis pipeline.

Measures source-role detection quality, need inference accuracy,
and style classification accuracy.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from backend.ml.analysis.needs_engine import NeedsEngine
from backend.ml.analysis.style_classifier import StyleClassifier
from backend.ml.models.mix_profile import MixProfile, NeedOpportunity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RoleDetectionReport:
    """Results of source-role detection evaluation."""

    precision: float
    """Fraction of detected roles that are actually present in the ground truth."""

    recall: float
    """Fraction of ground-truth roles that were detected."""

    f1: float
    """Harmonic mean of precision and recall."""

    per_role_detail: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-role metrics: {role: {precision, recall, f1, true_positive, false_positive, false_negative}}."""


@dataclass
class NeedInferenceReport:
    """Results of need/deficiency inference evaluation."""

    recall: float
    """Fraction of ground-truth deficiencies that were correctly identified."""

    false_positive_rate: float
    """Fraction of inferred needs that do not correspond to any ground-truth deficiency."""

    true_positives: int
    """Number of correctly identified deficiencies."""

    false_negatives: int
    """Number of ground-truth deficiencies that were missed."""

    false_positives: int
    """Number of inferred needs that were not in the ground truth."""


@dataclass
class StyleClassificationReport:
    """Results of style classification evaluation."""

    top1_accuracy: float
    """Fraction of mixes where the primary cluster matches the ground-truth style."""

    top3_accuracy: float
    """Fraction of mixes where the ground-truth style appears in the top 3 clusters."""

    errors: list[tuple[str, str, list[str]]] = field(default_factory=list)
    """List of (filepath, true_style, predicted_top3_styles) for every top-1 miss."""


# ---------------------------------------------------------------------------
# MixAnalysisEval
# ---------------------------------------------------------------------------

class MixAnalysisEval:
    """Evaluation harness for RESONATE's mix-analysis pipeline."""

    # ------------------------------------------------------------------
    # 1. Source-role detection quality
    # ------------------------------------------------------------------

    def evaluate_role_detection(
        self,
        items: list[tuple[MixProfile, list[str]]],
        threshold: float = 0.15,
    ) -> RoleDetectionReport:
        """Evaluate source-role detection quality.

        For each mix profile, the roles with confidence above *threshold*
        are treated as "detected".  These are compared against the
        ground-truth role list.

        Parameters
        ----------
        items:
            List of ``(mix_profile, ground_truth_roles)`` pairs.
            ``ground_truth_roles`` is a list of role name strings that
            are known to be present in the mix (e.g. ``["kick", "bass",
            "lead"]``).
        threshold:
            Minimum confidence value in
            :attr:`~backend.ml.models.mix_profile.SourceRolePresence.roles`
            for a role to be considered detected.

        Returns
        -------
        RoleDetectionReport
            Precision, recall, F1, and per-role detail.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Per-role accumulators
        role_tp: dict[str, int] = defaultdict(int)
        role_fp: dict[str, int] = defaultdict(int)
        role_fn: dict[str, int] = defaultdict(int)

        for mix_profile, gt_roles in items:
            gt_set = set(gt_roles)
            detected_roles = {
                role
                for role, conf in mix_profile.source_roles.roles.items()
                if conf >= threshold
            }

            tp = gt_set & detected_roles
            fp = detected_roles - gt_set
            fn = gt_set - detected_roles

            total_tp += len(tp)
            total_fp += len(fp)
            total_fn += len(fn)

            for role in tp:
                role_tp[role] += 1
            for role in fp:
                role_fp[role] += 1
            for role in fn:
                role_fn[role] += 1

        # Aggregate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Per-role detail
        all_roles = sorted(set(role_tp) | set(role_fp) | set(role_fn))
        per_role_detail: dict[str, dict[str, float]] = {}

        for role in all_roles:
            r_tp = role_tp.get(role, 0)
            r_fp = role_fp.get(role, 0)
            r_fn = role_fn.get(role, 0)

            r_prec = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0.0
            r_rec = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0.0
            r_f1 = (
                2.0 * r_prec * r_rec / (r_prec + r_rec)
                if (r_prec + r_rec) > 0
                else 0.0
            )

            per_role_detail[role] = {
                "precision": round(r_prec, 4),
                "recall": round(r_rec, 4),
                "f1": round(r_f1, 4),
                "true_positive": float(r_tp),
                "false_positive": float(r_fp),
                "false_negative": float(r_fn),
            }

        return RoleDetectionReport(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            per_role_detail=per_role_detail,
        )

    # ------------------------------------------------------------------
    # 2. Need inference quality
    # ------------------------------------------------------------------

    def evaluate_need_inference(
        self,
        items: list[tuple[MixProfile, list[str]]],
        engine: NeedsEngine | None = None,
    ) -> NeedInferenceReport:
        """Evaluate how well the NeedsEngine identifies mix deficiencies.

        Parameters
        ----------
        items:
            List of ``(mix_profile, ground_truth_deficiencies)`` pairs.
            Each ``ground_truth_deficiencies`` is a list of deficiency
            category strings (matching
            :attr:`~backend.ml.models.mix_profile.NeedOpportunity.category`)
            that are known to be present.
        engine:
            A :class:`NeedsEngine` instance.  If ``None``, a default
            engine is instantiated.

        Returns
        -------
        NeedInferenceReport
            Recall, false-positive rate, and raw counts.
        """
        if engine is None:
            engine = NeedsEngine()

        total_tp = 0
        total_fn = 0
        total_fp = 0

        for mix_profile, gt_deficiencies in items:
            gt_set = set(gt_deficiencies)

            diagnosed: list[NeedOpportunity] = engine.diagnose(mix_profile)
            diagnosed_categories = {need.category for need in diagnosed}

            tp = gt_set & diagnosed_categories
            fn = gt_set - diagnosed_categories
            fp = diagnosed_categories - gt_set

            total_tp += len(tp)
            total_fn += len(fn)
            total_fp += len(fp)

        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        total_inferred = total_tp + total_fp
        fpr = total_fp / total_inferred if total_inferred > 0 else 0.0

        return NeedInferenceReport(
            recall=round(recall, 4),
            false_positive_rate=round(fpr, 4),
            true_positives=total_tp,
            false_negatives=total_fn,
            false_positives=total_fp,
        )

    # ------------------------------------------------------------------
    # 3. Style classification accuracy
    # ------------------------------------------------------------------

    def evaluate_style_classification(
        self,
        items: list[tuple[MixProfile, str]],
        classifier: StyleClassifier | None = None,
    ) -> StyleClassificationReport:
        """Evaluate style classification accuracy.

        Parameters
        ----------
        items:
            List of ``(mix_profile, ground_truth_style)`` pairs.  The
            ground-truth style should match one of the cluster names used
            by :class:`StyleClassifier` (e.g. ``"modern_trap"``).
        classifier:
            A :class:`StyleClassifier` instance.  If ``None``, a default
            classifier is instantiated.

        Returns
        -------
        StyleClassificationReport
            Top-1 accuracy, top-3 accuracy, and per-item errors.
        """
        if classifier is None:
            classifier = StyleClassifier()

        top1_correct = 0
        top3_correct = 0
        errors: list[tuple[str, str, list[str]]] = []

        for mix_profile, gt_style in items:
            result = classifier.classify(mix_profile)

            # Top-1: primary cluster
            primary = result.primary_cluster

            # Top-3: sorted by probability descending
            sorted_clusters = sorted(
                result.cluster_probabilities.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            top3_names = [name for name, _prob in sorted_clusters[:3]]

            is_top1 = primary == gt_style
            is_top3 = gt_style in top3_names

            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1

            if not is_top1:
                errors.append((mix_profile.filepath, gt_style, top3_names))

        total = len(items) if items else 1
        return StyleClassificationReport(
            top1_accuracy=round(top1_correct / total, 4),
            top3_accuracy=round(top3_correct / total, 4),
            errors=errors,
        )
