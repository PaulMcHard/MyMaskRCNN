from types import SimpleNamespace

import pytest
import torch
from anomalib.metrics import F1AdaptiveThreshold

from maskrcnn_modules.maskrcnn.post_processing import DetectionPostProcessor, SparseF1AdaptiveThreshold


def _both_thresholds(batches):
    """Fit anomalib's threshold and the sparse one on the same batches."""
    stock = F1AdaptiveThreshold(fields=["anomaly_map", "gt_mask"], strict=False)
    sparse = SparseF1AdaptiveThreshold(fields=["anomaly_map", "gt_mask"], strict=False)
    for anomaly_map, gt_mask in batches:
        batch = SimpleNamespace(anomaly_map=anomaly_map, gt_mask=gt_mask)
        stock.update(batch)
        sparse.update(batch)
    return stock.compute(), sparse.compute()


def _detector_like_batch(generator, quantise):
    """Mostly-zero maps with a few scored boxes that partly overlap the ground truth."""
    anomaly_map = torch.zeros(2, 48, 64)
    gt_mask = torch.zeros(2, 48, 64, dtype=torch.bool)
    for b in range(2):
        for _ in range(3):
            y, x = (int(torch.randint(0, 36, (1,), generator=generator)) for _ in range(2))
            gt_mask[b, y : y + 8, x : x + 8] = True
            scores = torch.rand(10, 10, generator=generator)
            if quantise:
                # Few distinct values, so many candidates tie on score (and some on F1).
                scores = (scores * 4).round() / 4
            anomaly_map[b, y + 3 : y + 13, x + 3 : x + 13] = scores
    return anomaly_map, gt_mask


@pytest.mark.parametrize("quantise", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_sparse_threshold_equals_anomalib_threshold(seed, quantise):
    generator = torch.Generator().manual_seed(seed)
    batches = [_detector_like_batch(generator, quantise) for _ in range(3)]

    stock, sparse = _both_thresholds(batches)

    assert sparse == stock


def test_sparse_threshold_equals_anomalib_on_dense_maps():
    generator = torch.Generator().manual_seed(0)
    anomaly_map = torch.rand(2, 16, 16, generator=generator)
    gt_mask = torch.rand(2, 16, 16, generator=generator) > 0.7

    stock, sparse = _both_thresholds([(anomaly_map, gt_mask)])

    assert sparse == stock


def test_sparse_threshold_fallbacks_match_anomalib_when_a_class_is_missing():
    anomaly_map = torch.zeros(1, 8, 8)
    anomaly_map[0, :2, :2] = torch.tensor([[0.2, 0.9], [0.4, 0.6]])

    no_positives = _both_thresholds([(anomaly_map, torch.zeros(1, 8, 8, dtype=torch.bool))])
    no_negatives = _both_thresholds([(anomaly_map, torch.ones(1, 8, 8, dtype=torch.bool))])

    assert no_positives[0] == no_positives[1] == 0.9
    assert no_negatives[0] == no_negatives[1] == 0.0


def test_sparse_threshold_stores_only_scored_pixels():
    metric = SparseF1AdaptiveThreshold(fields=["anomaly_map", "gt_mask"], strict=False)
    anomaly_map = torch.zeros(1, 100, 100)
    anomaly_map[0, :3, :3] = 0.5
    gt_mask = torch.zeros(1, 100, 100, dtype=torch.bool)
    gt_mask[0, :5, :5] = True
    metric.update(SimpleNamespace(anomaly_map=anomaly_map, gt_mask=gt_mask))

    assert sum(chunk.numel() for chunk in metric.preds) == 9
    assert metric.zero_score_positives == 16
    assert metric.zero_score_negatives == 10_000 - 25


def test_detection_post_processor_uses_the_sparse_pixel_threshold():
    post_processor = DetectionPostProcessor()

    assert isinstance(post_processor._pixel_threshold_metric, SparseF1AdaptiveThreshold)
    assert isinstance(post_processor._image_threshold_metric, F1AdaptiveThreshold)
