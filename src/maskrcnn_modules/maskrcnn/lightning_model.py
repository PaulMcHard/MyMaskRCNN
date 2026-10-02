"""Mask R-CNN as an anomalib module.

anomalib's models are one-class by convention, but an ``AnomalibModule`` is a
``LightningModule`` plus a post-processor, evaluator and visualizer, so a
supervised ``training_step`` fits without changes to the framework. Wrapping
Mask R-CNN this way puts it through the same ``Engine``, threshold fitting
and evaluator as the one-class methods it is compared against.

Example:
    >>> from anomalib.engine import Engine
    >>> from maskrcnn_modules.data import CocoInstanceDataModule
    >>> from maskrcnn_modules.maskrcnn import MaskRCNN

    >>> datamodule = CocoInstanceDataModule(root="...", train_ann_file="...", ...)
    >>> model = MaskRCNN()
    >>> engine = Engine(max_steps=20000)
    >>> engine.fit(model=model, datamodule=datamodule)
    >>> engine.validate(model=model, datamodule=datamodule)  # fits the thresholds
    >>> engine.test(model=model, datamodule=datamodule)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from anomalib import LearningType
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.visualization import Visualizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchmetrics.detection import MeanAveragePrecision

from maskrcnn_modules.data.dataclasses import InstanceBatch
from maskrcnn_modules.maskrcnn.post_processing import DetectionPostProcessor
from maskrcnn_modules.maskrcnn.torch_model import (
    DEFAULT_ANCHOR_SIZES,
    DEFAULT_ASPECT_RATIOS,
    MaskRCNNModel,
    detections_to_anomaly_map,
)
from maskrcnn_modules.shared.metrics import configure_evaluator as build_shared_evaluator

log = logging.getLogger(__name__)

_MAP_IOU_TYPES = ("bbox", "segm")
_MAP_VARIANTS = ("", "_50", "_75")


class MaskRCNN(AnomalibModule):
    """Supervised Mask R-CNN Lightning module.

    Args:
        num_classes: Number of classes including background (11 defect
            types + background for 3D-ADAM). Must exceed the largest
            category id in the annotations; checked when training starts.
        backbone / pretrained / trainable_backbone_layers / min_size /
            max_size / anchor_sizes / aspect_ratios / box_score_thresh /
            box_nms_thresh / box_detections_per_img: See
            :class:`~maskrcnn_modules.maskrcnn.torch_model.MaskRCNNModel`.
        lr / momentum / weight_decay: SGD hyper-parameters.
        warmup_steps: Linear warm-up length, capped at 5% of training.
        min_lr_ratio: Cosine schedule floor as a fraction of ``lr``.
        gradient_clip_val: Gradient-norm clip passed to the trainer.
        mask_threshold: Soft-mask cut-off used for mask mAP only. The anomaly
            outputs use thresholds fitted by the post-processor instead.
        log_val_loss: Also compute the training losses on the validation
            split (one extra forward pass per validation batch).
        pre_processor: Disabled by default: torchvision's Mask R-CNN
            normalises and resizes internally.
        post_processor / evaluator / visualizer: Standard anomalib slots.
    """

    def __init__(
        self,
        num_classes: int = 12,
        backbone: str = "resnet50",
        pretrained: str = "coco",
        trainable_backbone_layers: int = 3,
        min_size: int = 1024,
        max_size: int = 1280,
        anchor_sizes: tuple[tuple[int, ...], ...] = DEFAULT_ANCHOR_SIZES,
        aspect_ratios: tuple[float, ...] = DEFAULT_ASPECT_RATIOS,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1.0e-3,
        warmup_steps: int = 500,
        min_lr_ratio: float = 0.01,
        gradient_clip_val: float = 35.0,
        mask_threshold: float = 0.5,
        log_val_loss: bool = True,
        pre_processor: nn.Module | bool = False,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        # The base __init__ already saved hyper-parameters, including the
        # component instances; re-save without them so checkpoints and the
        # logger only carry plain values.
        self.save_hyperparameters(ignore=["pre_processor", "post_processor", "evaluator", "visualizer"])

        self.model = MaskRCNNModel(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
        )

        self._lr = float(lr)
        self._momentum = float(momentum)
        self._weight_decay = float(weight_decay)
        self._warmup_steps = int(warmup_steps)
        self._min_lr_ratio = float(min_lr_ratio)
        self._gradient_clip_val = float(gradient_clip_val)
        self._mask_threshold = float(mask_threshold)
        self._log_val_loss = bool(log_val_loss)

        # COCO mAP is an instance-level metric, so it can't go through the
        # anomalib evaluator (which only sees the collapsed anomaly fields).
        self.val_map = MeanAveragePrecision(iou_type=_MAP_IOU_TYPES)
        self.test_map = MeanAveragePrecision(iou_type=_MAP_IOU_TYPES)

    # ------------------------------------------------------------------ #
    # Static configuration hooks
    # ------------------------------------------------------------------ #

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Image AUROC/F1/F1Max and pixel AUROC/F1/F1Max/AUPRO/AP/IoU/Dice/BFScore + FAR."""
        return build_shared_evaluator()

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """anomalib's post-processor with a memory-bounded pixel threshold fit."""
        return DetectionPostProcessor()

    def configure_optimizers(self) -> dict[str, Any]:
        """SGD with linear warm-up then cosine decay, stepped every iteration.

        Mirrors the MMDetection schedule in ``train_tb.py``.
        """
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            parameters, lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, min(self._warmup_steps, total_steps // 20))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=self._lr * self._min_lr_ratio,
                ),
            ],
            milestones=[warmup_steps],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # ------------------------------------------------------------------ #
    # Training / eval
    # ------------------------------------------------------------------ #

    def training_step(self, batch: InstanceBatch, *args, **kwargs) -> STEP_OUTPUT:
        """Supervised step: Mask R-CNN's RPN + RoI-head losses on the instance targets."""
        del args, kwargs
        loss_dict = self.model.compute_losses(batch.image, batch.targets())
        total = sum(loss_dict.values())
        self.log("train/loss", total, prog_bar=True, batch_size=batch.batch_size)
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, batch_size=batch.batch_size)
        return {"loss": total}

    def validation_step(self, batch: InstanceBatch, *args, **kwargs) -> STEP_OUTPUT:
        """Predict, accumulate mAP, and optionally log the validation loss."""
        del args, kwargs
        if self._log_val_loss:
            loss_dict = self.model.compute_losses(batch.image, batch.targets())
            self.log("val/loss", sum(loss_dict.values()), batch_size=batch.batch_size)
            for name, value in loss_dict.items():
                self.log(f"val/{name}", value, batch_size=batch.batch_size)
        return self._predict_and_score(batch, self.val_map)

    def test_step(self, batch: InstanceBatch, *args, **kwargs) -> STEP_OUTPUT:
        """Predict and accumulate mAP; the anomalib evaluator scores the returned batch."""
        del args, kwargs
        return self._predict_and_score(batch, self.test_map)

    def on_fit_start(self) -> None:
        """Check the model has an output for every category in the training annotations.

        Otherwise torchvision fails mid-step with an index error that doesn't
        name the cause.
        """
        categories = getattr(getattr(self.trainer.datamodule, "train_data", None), "categories", None)
        if categories and max(categories) >= self.model.num_classes:
            msg = (
                f"The annotations use category ids up to {max(categories)} but num_classes={self.model.num_classes}; "
                f"set num_classes to at least {max(categories) + 1} (categories plus background)."
            )
            raise ValueError(msg)

    def on_validation_epoch_end(self) -> None:
        """Log validation mAP."""
        self._log_map(self.val_map, prefix="val/")

    def on_test_start(self) -> None:
        """Refuse to test before thresholds exist.

        Without them the post-processor passes scores through unthresholded,
        and every thresholded metric (F1Score, IoU, Dice, FalseAlarmRate,
        BFScore) is silently computed on the wrong thing.
        """
        post_processor = self.post_processor
        if (
            isinstance(post_processor, PostProcessor)
            and post_processor.enable_thresholding
            and post_processor.image_threshold.isnan()
            and post_processor.pixel_threshold.isnan()
        ):
            msg = (
                "The post-processor has no thresholds yet. Run `engine.validate(model, datamodule)` "
                "(or load a checkpoint saved after a validation loop) before `engine.test(...)`."
            )
            raise RuntimeError(msg)

    def on_test_epoch_end(self) -> None:
        """Log test mAP, unprefixed like the evaluator's ``image_*`` / ``pixel_*`` metrics."""
        self._log_map(self.test_map, prefix="")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _predict_and_score(self, batch: InstanceBatch, map_metric: MeanAveragePrecision) -> InstanceBatch:
        """Run detection once and use it for both mAP and the anomaly outputs.

        The batch is updated in place: anomalib's evaluator reads the batch
        object it was handed, not the step's return value.
        """
        detections = self.model.detect(batch.image)
        map_metric.update(
            [
                {
                    "boxes": detection["boxes"],
                    "scores": detection["scores"],
                    "labels": detection["labels"],
                    "masks": detection["masks"][:, 0] > self._mask_threshold,
                }
                for detection in detections
            ],
            [
                {"boxes": target["boxes"], "labels": target["labels"], "masks": target["masks"].bool()}
                for target in batch.targets()
            ],
        )
        anomaly_map = detections_to_anomaly_map(detections, tuple(batch.image.shape[-2:]))
        return batch.update(pred_score=anomaly_map.amax(dim=(-2, -1)), anomaly_map=anomaly_map)

    def _log_map(self, map_metric: MeanAveragePrecision, prefix: str) -> None:
        results = map_metric.compute()
        for iou_type in _MAP_IOU_TYPES:
            for variant in _MAP_VARIANTS:
                self.log(f"{prefix}{iou_type}_mAP{variant}", results[f"{iou_type}_map{variant}"].float())
        map_metric.reset()

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Gradient clipping as in the MMDetection config; no sanity validation."""
        return {"gradient_clip_val": self._gradient_clip_val, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """``ONE_CLASS``, although the model is supervised.

        anomalib has no supervised learning type. This value only selects the
        ordinary fit path in ``Engine`` (as opposed to the zero/few-shot one),
        which is what a trained model needs; ``SuperDefect`` and anomalib's
        own SuperSimpleNet make the same choice.
        """
        return LearningType.ONE_CLASS
