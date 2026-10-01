# Vendored from SuperDefectExperiments (src/anomaly_modules/shared/metrics.py, commit 11c1c57).
# Do not edit here: re-copy from the source repo so the two do not drift.
# Docstring references to ``anomaly_modules`` refer to that repo.
"""Shared evaluator for anomaly detection approaches (SuperAD, Dinomaly, ...).

Adds pixel-level Average Precision, IoU, and Dice on top of anomalib's built-in
AUROC/F1Max/AUPRO, using anomalib's own ``AnomalibMetric`` pattern (any
torchmetrics ``Metric`` becomes batch-aware by subclassing ``AnomalibMetric``
alongside it) so every approach reports the same metric set via one shared
``configure_evaluator()``.

Also reports the **false-alarm rate** at each model's actual operating point
(see :class:`FalseAlarmRate`), which is the headline diagnostic for the EVT
work (``evt-validity-plan.md`` section 1a) and is deliberately reported for
*every* model rather than only the EVT-thresholded ones -- the stage 1 claim
compares an EVT threshold against a label-supervised adaptive one, and that
comparison needs the same column in both arms' ``metrics.csv``.
"""

from typing import Optional

import numpy as np
import torch
from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Max, F1Score
from anomalib.metrics.base import AnomalibMetric
from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision, BinaryF1Score, BinaryJaccardIndex


class AP(AnomalibMetric, BinaryAveragePrecision):
    """Average precision (step-function AUPRC), operating on the raw ``anomaly_map``.

    Unlike anomalib's ``AUPR`` (a trapezoidal AUC over precision/recall), this is
    the sklearn/COCO-style step-function AP.
    """


class IoU(AnomalibMetric, BinaryJaccardIndex):
    """Intersection-over-union of the thresholded ``pred_mask`` against ``gt_mask``."""


class Dice(AnomalibMetric, BinaryF1Score):
    """Dice/Sorensen coefficient of the thresholded ``pred_mask`` against ``gt_mask``.

    Dice = 2*TP / (2*TP + FP + FN) is algebraically identical to the F1 score for
    binary masks, so this reuses ``BinaryF1Score`` under the ``Dice`` name.
    """


class _FalseAlarmRate(Metric):
    """FP / (FP + TN): the fraction of ground-truth-*normal* samples flagged anomalous.

    Deliberately a plain counter rather than ``1 - BinarySpecificity``: the two
    states sum cleanly across batches and across distributed ranks, and an
    all-anomalous batch (no normals at all) yields 0 rather than a NaN that
    would poison the epoch aggregate.

    ``preds`` may be a boolean ``pred_label``/``pred_mask`` or any numeric
    tensor of the same shape as ``target``; anything ``> 0`` counts as a
    positive prediction, matching how anomalib's own thresholded outputs are
    encoded.
    """

    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("false_alarms", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("normals", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        normal = target == 0
        self.normals += normal.sum()
        self.false_alarms += (preds[normal] > 0).sum()

    def compute(self) -> torch.Tensor:
        # clamp(min=1) guards the no-normal-samples case; `false_alarms` is
        # then necessarily 0, so the reported rate is 0 rather than NaN.
        return self.false_alarms.float() / self.normals.clamp(min=1)


class FalseAlarmRate(AnomalibMetric, _FalseAlarmRate):
    """Batch-aware false-alarm rate.

    Pair with ``pred_label``/``gt_label`` for the image head, or
    ``pred_mask``/``gt_mask`` for the pixel head. The pixel variant reduces
    over every ``gt_mask == 0`` pixel, *including* the normal pixels of
    anomalous images -- which is precisely the quantity
    :class:`anomaly_modules.evt.post_processing.EVTPostProcessor`'s
    ``pixel_threshold_scope`` controls (``evt-validity-plan.md`` section 2).
    """


def _extract_boundary(mask_np: np.ndarray) -> np.ndarray:
    """Boundary of a 2D binary mask via 4-connected morphological gradient."""
    m = mask_np.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def _dilate_bool(mask_np: np.ndarray, radius: int) -> np.ndarray:
    """Boolean dilation by ``radius`` under an L-inf (square) structuring element."""
    if radius <= 0:
        return mask_np.astype(bool)
    m = mask_np.astype(bool)
    r = int(radius)
    padded = np.pad(m, r, mode="constant", constant_values=False)
    out = np.zeros_like(m)
    height, width = m.shape
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            out |= padded[r + di : r + di + height, r + dj : r + dj + width]
    return out


class _BFScore(Metric):
    """Boundary F-score at a fixed pixel tolerance.

    Extracts boundary pixels of ``pred_mask``/``gt_mask`` and matches them
    within ``tolerance`` pixels; accumulates pooled precision/recall across
    the loader and reports their F1, which is more stable than averaging
    per-image F1 when many images have empty boundaries.
    """

    higher_is_better = True
    full_state_update = False

    tp_p: torch.Tensor
    p_pred: torch.Tensor
    tp_r: torch.Tensor
    p_true: torch.Tensor

    def __init__(self, tolerance: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tolerance = int(tolerance)
        self.add_state("tp_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("p_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("p_true", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_b = (pred > 0.5) if pred.is_floating_point() else (pred > 0)
        tgt_b = target > 0
        pred_b = pred_b.squeeze()
        tgt_b = tgt_b.squeeze()
        if pred_b.ndim == 2:
            pred_b = pred_b.unsqueeze(0)
            tgt_b = tgt_b.unsqueeze(0)
        pred_np = pred_b.detach().cpu().numpy().astype(bool)
        tgt_np = tgt_b.detach().cpu().numpy().astype(bool)
        for bi in range(pred_np.shape[0]):
            bp = _extract_boundary(pred_np[bi])
            bt = _extract_boundary(tgt_np[bi])
            if not bp.any() and not bt.any():
                continue
            bp_d = _dilate_bool(bp, self.tolerance)
            bt_d = _dilate_bool(bt, self.tolerance)
            match_pred = bp & bt_d
            match_true = bt & bp_d
            self.tp_p = self.tp_p + torch.tensor(float(match_pred.sum()))
            self.p_pred = self.p_pred + torch.tensor(float(bp.sum()))
            self.tp_r = self.tp_r + torch.tensor(float(match_true.sum()))
            self.p_true = self.p_true + torch.tensor(float(bt.sum()))

    def compute(self) -> torch.Tensor:
        prec = self.tp_p / self.p_pred.clamp_min(1)
        rec = self.tp_r / self.p_true.clamp_min(1)
        denom = prec + rec
        if denom.item() <= 0:
            return torch.tensor(float("nan"))
        return 2 * prec * rec / denom


class BFScore1(AnomalibMetric, _BFScore):
    """Boundary F-score at 1px tolerance -- the tightest boundary-localisation check."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("tolerance", 1)
        super().__init__(**kwargs)


class BFScore2(AnomalibMetric, _BFScore):
    """Boundary F-score at 2px tolerance."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("tolerance", 2)
        super().__init__(**kwargs)


class BFScore3(AnomalibMetric, _BFScore):
    """Boundary F-score at 3px tolerance."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("tolerance", 3)
        super().__init__(**kwargs)


class BFScore5(AnomalibMetric, _BFScore):
    """Boundary F-score at 5px tolerance -- the loosest of the four reported tolerances."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("tolerance", 5)
        super().__init__(**kwargs)


class MacroDice(Metric):
    """Per-class Dice + macro-Dice (background excluded by default).

    Not wired into :func:`configure_evaluator` -- ``compute()`` returns a
    dict rather than a scalar, which anomalib's evaluator can't log
    directly. Available for offline/manual multi-class analysis, e.g. over
    per-defect-type masks rather than this repo's current binary
    anomaly/normal masks.
    """

    higher_is_better = True
    full_state_update = False

    inter2: torch.Tensor
    denom: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        ignore_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.ignore_index = ignore_index
        self.add_state("inter2", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("denom", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.flatten().to(torch.int64)
        target = target.flatten().to(torch.int64)
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            pred, target = pred[keep], target[keep]
        for c in range(self.num_classes):
            p = pred == c
            t = target == c
            self.inter2[c] += 2.0 * (p & t).sum().float()
            self.denom[c] += (p.sum() + t.sum()).float()

    def compute(self) -> dict:
        per_class = torch.where(
            self.denom > 0,
            self.inter2 / self.denom.clamp_min(1),
            torch.full_like(self.denom, float("nan")),
        )
        start = 0 if self.include_background else 1
        defect = per_class[start:]
        finite = defect[~torch.isnan(defect)]
        macro = finite.mean() if finite.numel() > 0 else torch.tensor(float("nan"))
        return {"per_class": per_class, "macro": macro}


class MeanIoU(Metric):
    """Per-class IoU + mean IoU over defect classes (background excluded by default).

    Same offline-only status as :class:`MacroDice` -- ``compute()`` returns
    a dict, so this is not part of :func:`configure_evaluator`.
    """

    higher_is_better = True
    full_state_update = False

    intersection: torch.Tensor
    union: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        ignore_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.ignore_index = ignore_index
        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"MeanIoU: pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
        pred = pred.flatten().to(torch.int64)
        target = target.flatten().to(torch.int64)
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            pred, target = pred[keep], target[keep]
        for c in range(self.num_classes):
            p = pred == c
            t = target == c
            self.intersection[c] += (p & t).sum().float()
            self.union[c] += (p | t).sum().float()

    def compute(self) -> dict:
        iou = torch.where(
            self.union > 0,
            self.intersection / self.union.clamp_min(1),
            torch.full_like(self.union, float("nan")),
        )
        start = 0 if self.include_background else 1
        defect_iou = iou[start:]
        finite = defect_iou[~torch.isnan(defect_iou)]
        miou = finite.mean() if finite.numel() > 0 else torch.tensor(float("nan"))
        return {"per_class": iou, "miou": miou}


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Wraps ``scipy.ndimage.label`` to avoid an eager import at module load."""
    from scipy.ndimage import label as cc_label

    labeled, num_features = cc_label(mask)
    return labeled, int(num_features)


def _match_and_score(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Yield ``(IoU, Dice)`` for each matched (pred, gt) instance pair.

    Greedy matching: for each GT instance, pick the predicted instance with
    max IoU; each predicted instance is used at most once. Unmatched GT
    instances count as ``(0, 0)``; unmatched predictions don't contribute.
    """
    pred_lab, n_p = _label_components(pred_mask)
    gt_lab, n_g = _label_components(gt_mask)
    if n_g == 0 and n_p == 0:
        return
    used_pred: set[int] = set()
    for gi in range(1, n_g + 1):
        g = gt_lab == gi
        best_iou = 0.0
        best_dice = 0.0
        best_p = -1
        for pi in range(1, n_p + 1):
            if pi in used_pred:
                continue
            p = pred_lab == pi
            inter = int(np.logical_and(p, g).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(p, g).sum())
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                denom = int(p.sum() + g.sum())
                best_dice = (2.0 * inter / denom) if denom > 0 else 0.0
                best_p = pi
        if best_p >= 0:
            used_pred.add(best_p)
            yield best_iou, best_dice
        else:
            yield 0.0, 0.0


class InstanceIoU(Metric):
    """Mean IoU over connected-component-matched instances, plus matched-fraction @ IoU=0.5.

    Offline-only like :class:`MacroDice`/:class:`MeanIoU` -- ``compute()``
    returns a dict.
    """

    higher_is_better = True
    full_state_update = False

    sum_iou: torch.Tensor
    n_gt_instances: torch.Tensor
    n_matched_at_05: torch.Tensor

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = float(threshold)
        self.add_state("sum_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_gt_instances", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_matched_at_05", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_b = (pred > self.threshold) if pred.is_floating_point() else (pred > 0)
        pred_b = pred_b.cpu().numpy()
        tgt_b = (target > 0).cpu().numpy()
        pred_b = np.squeeze(pred_b)
        tgt_b = np.squeeze(tgt_b)
        if pred_b.ndim == 2:
            pred_b = pred_b[None]
            tgt_b = tgt_b[None]
        for bi in range(pred_b.shape[0]):
            for iou, _dice in _match_and_score(pred_b[bi], tgt_b[bi]):
                self.sum_iou = self.sum_iou + torch.tensor(float(iou))
                self.n_gt_instances = self.n_gt_instances + 1.0
                if iou >= 0.5:
                    self.n_matched_at_05 = self.n_matched_at_05 + 1.0

    def compute(self) -> dict:
        n = self.n_gt_instances.item()
        if n <= 0:
            return {"mean_iou": torch.tensor(float("nan")), "matched_frac": torch.tensor(float("nan"))}
        return {"mean_iou": self.sum_iou / n, "matched_frac": self.n_matched_at_05 / n}


class InstanceDice(Metric):
    """Mean Dice over connected-component-matched instances (companion to :class:`InstanceIoU`)."""

    higher_is_better = True
    full_state_update = False

    sum_dice: torch.Tensor
    n_gt_instances: torch.Tensor

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = float(threshold)
        self.add_state("sum_dice", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_gt_instances", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_b = (pred > self.threshold) if pred.is_floating_point() else (pred > 0)
        pred_b = pred_b.cpu().numpy()
        tgt_b = (target > 0).cpu().numpy()
        pred_b = np.squeeze(pred_b)
        tgt_b = np.squeeze(tgt_b)
        if pred_b.ndim == 2:
            pred_b = pred_b[None]
            tgt_b = tgt_b[None]
        for bi in range(pred_b.shape[0]):
            for _iou, dice in _match_and_score(pred_b[bi], tgt_b[bi]):
                self.sum_dice = self.sum_dice + torch.tensor(float(dice))
                self.n_gt_instances = self.n_gt_instances + 1.0

    def compute(self) -> torch.Tensor:
        n = self.n_gt_instances.item()
        if n <= 0:
            return torch.tensor(float("nan"))
        return self.sum_dice / n


def configure_evaluator() -> Evaluator:
    """Image AUROC/F1Score/F1Max/FalseAlarmRate and pixel AUROC/F1Score/F1Max/AUPRO/AP/IoU/Dice/FalseAlarmRate/BFScore.

    ``F1Score`` is computed at the model's actual (adaptive) threshold via
    ``pred_label``/``pred_mask``, while ``F1Max`` reports the best-case F1 over
    all thresholds -- kept alongside it for reference.

    ``FalseAlarmRate`` is likewise measured at the model's actual threshold. It
    is the number an EVT threshold is *specified* by (a nominal ``alpha``),
    and the number an F1-adaptive threshold leaves unconstrained -- so it is
    the honest basis for comparing the two.

    ``BFScore`` is reported at four tolerances (1/2/3/5px) against
    ``pred_mask``/``gt_mask`` -- the boundary-localisation quality that a
    region-overlap metric like IoU/Dice can't distinguish.
    """
    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
        FalseAlarmRate(fields=["pred_label", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        AP(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        IoU(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        Dice(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        FalseAlarmRate(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        BFScore1(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        BFScore2(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        BFScore3(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        BFScore5(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
    ]
    return Evaluator(test_metrics=test_metrics)
