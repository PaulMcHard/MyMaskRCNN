import pytest
import torch
from anomalib.engine import Engine
from lightning.pytorch import seed_everything

from maskrcnn_modules.data import CocoInstanceDataModule
from maskrcnn_modules.maskrcnn import DetectionPostProcessor, MaskRCNN
from maskrcnn_modules.maskrcnn.torch_model import MaskRCNNModel, detections_to_anomaly_map

EVALUATOR_METRICS = {
    "image_AUROC", "image_F1Score", "image_F1Max", "image_FalseAlarmRate",
    "pixel_AUROC", "pixel_F1Score", "pixel_F1Max", "pixel_AUPRO", "pixel_AP", "pixel_IoU", "pixel_Dice",
    "pixel_FalseAlarmRate", "pixel_BFScore1", "pixel_BFScore2", "pixel_BFScore3", "pixel_BFScore5",
}  # fmt: skip
MAP_METRICS = {"bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "segm_mAP", "segm_mAP_50", "segm_mAP_75"}


def _detection(masks, scores):
    return {"masks": masks, "scores": torch.tensor(scores)}


def test_anomaly_map_is_zero_without_detections():
    detections = [_detection(torch.zeros(0, 1, 4, 6), [])]

    anomaly_map = detections_to_anomaly_map(detections, (4, 6))

    assert anomaly_map.shape == (1, 4, 6)
    assert not anomaly_map.any()


def test_anomaly_map_takes_per_pixel_max_of_score_times_mask():
    masks = torch.zeros(2, 1, 4, 6)
    masks[0, 0, :2, :] = 1.0  # confident instance over the top half
    masks[1, 0, 1:, :] = 0.5  # soft instance over rows 1-3, overlapping row 1
    detections = [_detection(masks, [0.8, 0.6])]

    anomaly_map = detections_to_anomaly_map(detections, (4, 6))[0]

    assert torch.allclose(anomaly_map[0], torch.full((6,), 0.8))
    assert torch.allclose(anomaly_map[1], torch.full((6,), 0.8))  # max(0.8 * 1.0, 0.6 * 0.5)
    assert torch.allclose(anomaly_map[2:], torch.full((2, 6), 0.3))


def test_coco_weights_are_rejected_for_other_backbones():
    with pytest.raises(ValueError, match="COCO detector weights only exist for resnet50"):
        MaskRCNNModel(backbone="resnet18", pretrained="coco")


def test_module_uses_shared_evaluator_and_detection_post_processor():
    model = MaskRCNN(pretrained="none", min_size=64, max_size=80, visualizer=False)

    assert {metric.name for metric in model.evaluator.test_metrics} == EVALUATOR_METRICS
    assert isinstance(model.post_processor, DetectionPostProcessor)
    assert model.pre_processor is None
    # Three anchor sizes x three aspect ratios per location, for small defects.
    assert model.model.detector.rpn.anchor_generator.num_anchors_per_location() == [9] * 5


def _engine(tmp_path, **kwargs):
    return Engine(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def test_test_before_thresholds_are_fitted_is_refused(tmp_path, datamodule_kwargs):
    model = MaskRCNN(pretrained="none", min_size=64, max_size=80, visualizer=False)
    datamodule = CocoInstanceDataModule(**datamodule_kwargs)

    with pytest.raises(RuntimeError, match="no thresholds yet"):
        _engine(tmp_path).test(model=model, datamodule=datamodule)


def test_too_few_classes_for_the_annotations_is_refused(tmp_path, datamodule_kwargs):
    model = MaskRCNN(num_classes=8, pretrained="none", min_size=64, max_size=80, visualizer=False)
    datamodule = CocoInstanceDataModule(**datamodule_kwargs)

    with pytest.raises(ValueError, match="set num_classes to at least 12"):
        _engine(tmp_path, max_steps=1).fit(model=model, datamodule=datamodule)


def test_fit_validate_test_predict_through_engine(tmp_path, datamodule_kwargs):
    seed_everything(0)
    model = MaskRCNN(pretrained="none", min_size=64, max_size=80, visualizer=False)
    datamodule = CocoInstanceDataModule(**datamodule_kwargs)
    engine = _engine(tmp_path, max_steps=2, val_check_interval=2, check_val_every_n_epoch=None)

    engine.fit(model=model, datamodule=datamodule)
    logged = set(engine.trainer.callback_metrics)
    assert {"train/loss", "train/loss_mask", "val/loss", "val/segm_mAP", "val/bbox_mAP"} <= logged

    engine.validate(model=model, datamodule=datamodule)
    assert not model.post_processor.image_threshold.isnan()
    assert not model.post_processor.pixel_threshold.isnan()

    results = engine.test(model=model, datamodule=datamodule)[0]
    assert set(results) == EVALUATOR_METRICS | MAP_METRICS
    # Good images in the test split give the image-level metrics their negatives.
    assert all(torch.isfinite(torch.tensor(results[name])) for name in EVALUATOR_METRICS if name.startswith("image_"))

    predictions = engine.predict(model=model, datamodule=datamodule)
    assert sum(batch.batch_size for batch in predictions) == 8
    assert predictions[0].pred_mask.dtype == torch.bool
    assert predictions[0].pred_mask.shape == (2, 64, 80)
