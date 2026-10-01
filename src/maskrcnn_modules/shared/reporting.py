# Vendored from SuperDefectExperiments (src/anomaly_modules/shared/reporting.py, commit 11c1c57).
# Do not edit here: re-copy from the source repo so the two do not drift.
# Docstring references to ``anomaly_modules`` refer to that repo.
"""Per-image score capture for post-hoc bootstrap/significance reporting.

anomalib's ``Evaluator`` (see :mod:`anomaly_modules.shared.metrics`) only
ever reduces to epoch-aggregated scalars -- exactly what ``main.py`` writes
to ``metrics.csv``. :func:`compute_per_image_metrics` instead turns the
per-batch predictions from ``engine.predict()`` into one row per image, so
:mod:`anomaly_modules.shared.stats` (``bootstrap_ci``/``wilcoxon_paired``)
has per-image score distributions to operate on.
"""

from types import SimpleNamespace
from typing import Any, Iterable

import torch

from maskrcnn_modules.shared.metrics import AP, Dice, IoU


def _single_image_pixel_metric(metric_cls: type, pred_field: str, pred: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Instantiate a fresh pixel metric and score exactly one image."""
    metric = metric_cls(fields=[pred_field, "gt_mask"])
    metric.update(SimpleNamespace(**{pred_field: pred.unsqueeze(0), "gt_mask": gt_mask.unsqueeze(0)}))
    value = metric.compute()
    return float(value.item()) if torch.isfinite(value) else float("nan")


def compute_per_image_metrics(predictions: Iterable[Any]) -> list[dict]:
    """Flatten ``engine.predict()``'s batches into one row per image.

    Each row carries ``image_path``, ``gt_label``, ``pred_label``,
    ``pred_score``, ``correct``, and -- when the batch carries mask fields
    (``anomaly_map``/``pred_mask``/``gt_mask``) -- per-image
    ``pixel_iou``/``pixel_dice``/``pixel_ap``.
    """
    rows: list[dict] = []
    for batch in predictions:
        image_paths = list(batch.image_path)
        gt_label = batch.gt_label
        pred_label = batch.pred_label
        pred_score = batch.pred_score
        anomaly_map = getattr(batch, "anomaly_map", None)
        pred_mask = getattr(batch, "pred_mask", None)
        gt_mask = getattr(batch, "gt_mask", None)
        has_masks = anomaly_map is not None and pred_mask is not None and gt_mask is not None

        for i, image_path in enumerate(image_paths):
            row: dict[str, Any] = {
                "image_path": str(image_path),
                "gt_label": int(gt_label[i]),
                "pred_label": int(pred_label[i]),
                "pred_score": float(pred_score[i]),
            }
            row["correct"] = row["gt_label"] == row["pred_label"]
            if has_masks:
                row["pixel_iou"] = _single_image_pixel_metric(IoU, "pred_mask", pred_mask[i], gt_mask[i])
                row["pixel_dice"] = _single_image_pixel_metric(Dice, "pred_mask", pred_mask[i], gt_mask[i])
                row["pixel_ap"] = _single_image_pixel_metric(AP, "anomaly_map", anomaly_map[i], gt_mask[i])
            rows.append(row)
    return rows
