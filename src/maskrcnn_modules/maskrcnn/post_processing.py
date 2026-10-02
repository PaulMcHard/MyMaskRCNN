"""Post-processor for detector-derived anomaly maps.

anomalib's :class:`PostProcessor` fits its pixel threshold with
``F1AdaptiveThreshold``, which stores *every* validation pixel and sorts them
all. On the pooled 3D-ADAM validation split at native resolution that is
~5e8 pixels and roughly 20 GB of host memory per fit (measured: ~1 GB per 20
images of 1024x1280).

A detector's anomaly map is exactly zero outside its detections, so almost
all of those pixels are identical ``(score 0, label 0)`` entries.
:class:`_SparseF1AdaptiveThreshold` stores the non-zero-score pixels
individually and only *counts* the zero-score ones, which gives the same
threshold in a fraction of the memory.
"""

from __future__ import annotations

import logging

import torch
from anomalib.metrics.base import AnomalibMetric
from anomalib.post_processing import PostProcessor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

log = logging.getLogger(__name__)


class _SparseF1AdaptiveThreshold(Metric):
    """F1-maximising threshold, equal to anomalib's ``F1AdaptiveThreshold``.

    Same precision-recall curve, same F1 formula, same tie-breaking (the
    lowest threshold among equal-F1 candidates) and the same fallbacks when
    one class is absent. Only the storage differs.
    """

    full_state_update = False

    preds: list[torch.Tensor]
    target: list[torch.Tensor]
    zero_score_positives: torch.Tensor
    zero_score_negatives: torch.Tensor

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("zero_score_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("zero_score_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.flatten()
        target = target.flatten() > 0
        scored = preds != 0
        self.preds.append(preds[scored])
        self.target.append(target[scored])
        self.zero_score_positives += (target & ~scored).sum()
        self.zero_score_negatives += (~target & ~scored).sum()

    def compute(self) -> torch.Tensor:
        # One entry per stored pixel, plus a single entry standing in for all
        # zero-score pixels, weighted by how many positives / negatives it holds.
        scores = dim_zero_cat(self.preds) if self.preds else torch.zeros(0, device=self.device)
        target = dim_zero_cat(self.target) if self.target else torch.zeros(0, dtype=torch.bool, device=self.device)
        positives, negatives = target.long(), (~target).long()
        if self.zero_score_positives + self.zero_score_negatives > 0:
            scores = torch.cat([scores, scores.new_zeros(1)])
            positives = torch.cat([positives, self.zero_score_positives.reshape(1)])
            negatives = torch.cat([negatives, self.zero_score_negatives.reshape(1)])

        # Same fallbacks as anomalib when the validation set lacks a class.
        if positives.sum() == 0:
            return scores.max()
        if negatives.sum() == 0:
            return scores.min()

        # Highest score first; one threshold candidate per distinct value.
        order = torch.argsort(scores, descending=True)
        scores, positives, negatives = scores[order], positives[order], negatives[order]
        last_of_value = torch.ones_like(scores, dtype=torch.bool)
        last_of_value[:-1] = scores[1:] != scores[:-1]
        true_positives = torch.cumsum(positives, dim=0)[last_of_value]
        false_positives = torch.cumsum(negatives, dim=0)[last_of_value]
        thresholds = scores[last_of_value]

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / true_positives[-1]
        f1_score = torch.nan_to_num((2 * precision * recall) / (precision + recall + 1e-10), nan=0.0)

        # Candidates run from the highest threshold down; on equal F1 anomalib
        # keeps the lowest threshold, i.e. the last maximum in this order.
        best = f1_score.numel() - 1 - int(torch.argmax(f1_score.flip(0)))
        return thresholds[best]


class SparseF1AdaptiveThreshold(AnomalibMetric, _SparseF1AdaptiveThreshold):  # type: ignore[misc]
    """Batch-aware :class:`_SparseF1AdaptiveThreshold` (reads its fields from an anomalib batch)."""


class DetectionPostProcessor(PostProcessor):
    """anomalib's post-processor with two changes for detector outputs.

    1. A memory-bounded pixel threshold fit (:class:`SparseF1AdaptiveThreshold`).
    2. Normalisation that survives a validation set with a single score value.
       A detector that finds nothing on validation (an early or under-trained
       checkpoint) gives ``min == max == 0``; anomalib then divides by zero and
       every test score becomes NaN, which silently corrupts every metric.

    Everything else (image threshold, min-max normalisation, how thresholds
    are applied at test time) is the stock behaviour, so operating points are
    chosen exactly as they are for the one-class models.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pixel_threshold_metric = SparseF1AdaptiveThreshold(fields=["anomaly_map", "gt_mask"], strict=False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        super().on_validation_epoch_end(trainer, pl_module)
        if self.enable_normalization and not self.image_max.isnan() and self.image_max <= self.image_min:
            log.warning(
                "Every validation image got the same anomaly score (%.4f): the model detected nothing above "
                "box_score_thresh. Test metrics will describe a model that flags nothing.",
                float(self.image_max),
            )

    @staticmethod
    def _normalize(
        preds: torch.Tensor | None,
        norm_min: torch.Tensor,
        norm_max: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor | None:
        """Stock min-max normalisation, with a unit range when validation saw only one value.

        With the unit range, scores keep their order and anything above the
        threshold still maps above the 0.5 operating point.
        """
        if preds is not None and not norm_min.isnan() and not norm_max.isnan() and norm_max <= norm_min:
            norm_max = norm_min + 1
        return PostProcessor._normalize(preds, norm_min, norm_max, threshold)
