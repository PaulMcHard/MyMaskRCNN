from types import SimpleNamespace

import torch

from maskrcnn_modules.shared.reporting import compute_per_image_metrics


def test_compute_per_image_metrics_image_only_batch():
    # 0.5/0.25 are exactly representable in float32, so the round-trip
    # through torch -> float() -> dict equality below is exact.
    batch = SimpleNamespace(
        image_path=["a.png", "b.png"],
        gt_label=torch.tensor([0, 1]),
        pred_label=torch.tensor([0, 0]),
        pred_score=torch.tensor([0.5, 0.25]),
    )

    rows = compute_per_image_metrics([batch])

    assert rows == [
        {"image_path": "a.png", "gt_label": 0, "pred_label": 0, "pred_score": 0.5, "correct": True},
        {"image_path": "b.png", "gt_label": 1, "pred_label": 0, "pred_score": 0.25, "correct": False},
    ]


def test_compute_per_image_metrics_includes_pixel_scores_when_masks_present():
    # Image 0: perfect pixel prediction (IoU/Dice = 1). Image 1: partial overlap.
    gt_mask = torch.tensor([[[1, 1], [0, 0]], [[1, 1], [0, 0]]])
    pred_mask = torch.tensor([[[1, 1], [0, 0]], [[1, 0], [1, 0]]])
    anomaly_map = pred_mask.float()

    batch = SimpleNamespace(
        image_path=["a.png", "b.png"],
        gt_label=torch.tensor([1, 1]),
        pred_label=torch.tensor([1, 1]),
        pred_score=torch.tensor([0.9, 0.6]),
        anomaly_map=anomaly_map,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
    )

    rows = compute_per_image_metrics([batch])

    assert rows[0]["pixel_iou"] == 1.0
    assert rows[0]["pixel_dice"] == 1.0
    # Image 1: intersection=1, union=3 -> IoU = 1/3.
    assert abs(rows[1]["pixel_iou"] - (1 / 3)) < 1e-6


def test_compute_per_image_metrics_flattens_across_multiple_batches():
    batch1 = SimpleNamespace(
        image_path=["a.png"],
        gt_label=torch.tensor([0]),
        pred_label=torch.tensor([0]),
        pred_score=torch.tensor([0.1]),
    )
    batch2 = SimpleNamespace(
        image_path=["b.png"],
        gt_label=torch.tensor([1]),
        pred_label=torch.tensor([1]),
        pred_score=torch.tensor([0.8]),
    )

    rows = compute_per_image_metrics([batch1, batch2])

    assert [row["image_path"] for row in rows] == ["a.png", "b.png"]
