from types import SimpleNamespace

import torch

from maskrcnn_modules.shared.metrics import (
    AP,
    BFScore1,
    BFScore2,
    Dice,
    InstanceDice,
    InstanceIoU,
    IoU,
    MacroDice,
    MeanIoU,
    configure_evaluator,
)


def test_ap_matches_manual_average_precision():
    # 2x2 "images": scores rank the two true-positive pixels above the two negatives,
    # so average precision is 1.0 (perfect ranking).
    anomaly_map = torch.tensor([[[0.9, 0.1], [0.8, 0.2]]])
    gt_mask = torch.tensor([[[1, 0], [1, 0]]])

    metric = AP(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    metric.update(SimpleNamespace(anomaly_map=anomaly_map, gt_mask=gt_mask))

    assert metric.name == "pixel_AP"
    assert torch.isclose(metric.compute(), torch.tensor(1.0))


def test_ap_penalizes_poor_ranking():
    # Worst-case ranking: both true-positive pixels score lowest.
    anomaly_map = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]])
    gt_mask = torch.tensor([[[1, 0], [1, 0]]])

    metric = AP(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    metric.update(SimpleNamespace(anomaly_map=anomaly_map, gt_mask=gt_mask))

    assert metric.compute() < 1.0


def test_iou_perfect_partial_and_no_overlap():
    gt_mask = torch.tensor([[[1, 1], [0, 0]]])

    perfect = IoU(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    perfect.update(SimpleNamespace(pred_mask=torch.tensor([[[1, 1], [0, 0]]]), gt_mask=gt_mask))
    assert perfect.name == "pixel_IoU"
    assert torch.isclose(perfect.compute(), torch.tensor(1.0))

    none = IoU(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    none.update(SimpleNamespace(pred_mask=torch.tensor([[[0, 0], [1, 1]]]), gt_mask=gt_mask))
    assert torch.isclose(none.compute(), torch.tensor(0.0))

    partial = IoU(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    # Intersection = 1 pixel, union = 3 pixels -> IoU = 1/3.
    partial.update(SimpleNamespace(pred_mask=torch.tensor([[[1, 0], [1, 0]]]), gt_mask=gt_mask))
    assert torch.isclose(partial.compute(), torch.tensor(1 / 3))


def test_dice_matches_f1_formula_on_partial_overlap():
    gt_mask = torch.tensor([[[1, 1], [0, 0]]])
    pred_mask = torch.tensor([[[1, 0], [1, 0]]])  # TP=1, FP=1, FN=1

    metric = Dice(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    metric.update(SimpleNamespace(pred_mask=pred_mask, gt_mask=gt_mask))

    assert metric.name == "pixel_Dice"
    # Dice = 2*TP / (2*TP + FP + FN) = 2 / 4 = 0.5
    assert torch.isclose(metric.compute(), torch.tensor(0.5))


def test_configure_evaluator_reports_pixel_ap_iou_dice_and_aupro():
    evaluator = configure_evaluator()
    names = {metric.name for metric in evaluator.test_metrics}

    assert {"pixel_AP", "pixel_IoU", "pixel_Dice", "pixel_AUPRO"} <= names


def test_false_alarm_rate_counts_positives_among_normals_only():
    from maskrcnn_modules.shared.metrics import FalseAlarmRate

    pred_label = torch.tensor([1, 0, 1, 0, 1])
    gt_label = torch.tensor([0, 0, 1, 1, 0])  # normals: idx 0,1,4 -> false alarms at 0 and 4

    metric = FalseAlarmRate(fields=["pred_label", "gt_label"], prefix="image_")
    metric.update(SimpleNamespace(pred_label=pred_label, gt_label=gt_label))

    assert metric.name == "image_FalseAlarmRate"
    assert torch.isclose(metric.compute(), torch.tensor(2 / 3))


def test_false_alarm_rate_is_zero_with_no_normal_samples():
    from maskrcnn_modules.shared.metrics import FalseAlarmRate

    pred_label = torch.tensor([1, 1])
    gt_label = torch.tensor([1, 1])

    metric = FalseAlarmRate(fields=["pred_label", "gt_label"], prefix="image_")
    metric.update(SimpleNamespace(pred_label=pred_label, gt_label=gt_label))

    assert torch.isclose(metric.compute(), torch.tensor(0.0))


def test_false_alarm_rate_pixel_variant_reduces_over_all_normal_pixels():
    from maskrcnn_modules.shared.metrics import FalseAlarmRate

    # One anomalous image with a normal region that has a false positive, plus
    # one fully normal image with none -- the pixel FAR must reduce over every
    # gt_mask == 0 pixel, including those inside the anomalous image.
    pred_mask = torch.tensor([[[1, 0], [1, 0]], [[0, 0], [0, 0]]])
    gt_mask = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 0]]])

    metric = FalseAlarmRate(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    metric.update(SimpleNamespace(pred_mask=pred_mask, gt_mask=gt_mask))

    # gt_mask==0 pixels: 7 total; pred_mask==1 among them: 1 (the (0,1,0) cell).
    assert torch.isclose(metric.compute(), torch.tensor(1 / 7))


def test_configure_evaluator_reports_false_alarm_rate():
    evaluator = configure_evaluator()
    names = {metric.name for metric in evaluator.test_metrics}

    assert {"image_FalseAlarmRate", "pixel_FalseAlarmRate"} <= names


def test_bf_score_perfect_boundary_match():
    mask = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

    metric = BFScore2(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    metric.update(SimpleNamespace(pred_mask=mask, gt_mask=mask))

    assert metric.name == "pixel_BFScore2"
    assert torch.isclose(metric.compute(), torch.tensor(1.0))


def test_bf_score_penalizes_spurious_and_missed_regions():
    # Block A (rows/cols 0-1) is in both masks -> perfectly matched boundary.
    # Block B (rows/cols 4-5) is gt-only (missed); block C (rows 4-5, cols 0-1)
    # is pred-only (spurious) and far enough from B that tolerance=1 can't
    # match them -- so precision and recall are both fractional, not 0 or 1.
    gt_mask = torch.zeros(1, 6, 6, dtype=torch.int64)
    gt_mask[0, 0:2, 0:2] = 1
    gt_mask[0, 4:6, 4:6] = 1
    pred_mask = torch.zeros(1, 6, 6, dtype=torch.int64)
    pred_mask[0, 0:2, 0:2] = 1
    pred_mask[0, 4:6, 0:2] = 1

    metric = BFScore1(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    metric.update(SimpleNamespace(pred_mask=pred_mask, gt_mask=gt_mask))

    score = metric.compute()
    assert 0.0 < score < 1.0


def test_macro_dice_excludes_background_by_default():
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 1, 1])

    metric = MacroDice(num_classes=2)
    metric.update(pred, target)
    result = metric.compute()

    # Defect class (1): TP=2, FP=0, FN=1 -> Dice = 2*2 / (2*2 + 0 + 1) = 4/5.
    assert torch.isclose(result["per_class"][1], torch.tensor(0.8))
    assert torch.isclose(result["macro"], torch.tensor(0.8))


def test_mean_iou_excludes_background_by_default():
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 1, 1])

    metric = MeanIoU(num_classes=2)
    metric.update(pred, target)
    result = metric.compute()

    # Defect class (1): intersection=2, union=3 -> IoU = 2/3.
    assert torch.isclose(result["per_class"][1], torch.tensor(2 / 3))
    assert torch.isclose(result["miou"], torch.tensor(2 / 3))


def _two_instance_masks() -> torch.Tensor:
    mask = torch.zeros(1, 5, 5, dtype=torch.int64)
    mask[0, 0:2, 0:2] = 1
    mask[0, 3:5, 3:5] = 1
    return mask


def test_instance_iou_perfect_prediction_matches_both_instances():
    mask = _two_instance_masks()

    metric = InstanceIoU()
    metric.update(mask, mask)
    result = metric.compute()

    assert torch.isclose(result["mean_iou"], torch.tensor(1.0))
    assert torch.isclose(result["matched_frac"], torch.tensor(1.0))


def test_instance_dice_perfect_prediction_scores_one():
    mask = _two_instance_masks()

    metric = InstanceDice()
    metric.update(mask, mask)

    assert torch.isclose(metric.compute(), torch.tensor(1.0))


def test_configure_evaluator_reports_bf_score_tolerances():
    evaluator = configure_evaluator()
    names = {metric.name for metric in evaluator.test_metrics}

    assert {"pixel_BFScore1", "pixel_BFScore2", "pixel_BFScore3", "pixel_BFScore5"} <= names
