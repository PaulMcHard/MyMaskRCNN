"""torchvision Mask R-CNN, plus the mapping from its detections to anomaly outputs.

Mask R-CNN returns a *set of instances* per image (box, class, score, soft
mask); anomalib's post-processor, evaluator and visualizer consume a *dense*
``anomaly_map`` and a scalar ``pred_score``. :func:`detections_to_anomaly_map`
is the bridge between the two.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from anomalib.data import InferenceBatch
from torch import nn
from torchvision.models import get_model_weights
from torchvision.models.detection import MaskRCNN as TorchvisionMaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

log = logging.getLogger(__name__)

# One tuple per FPN level (strides 4/8/16/32/64). Equivalent to the MMDetection
# config's ``scales=[2, 4, 8]`` over those strides: three anchor sizes per
# level starting at 8 px, chosen because the median 3D-ADAM defect is ~17 px
# across. torchvision's default (one size per level, from 32 px) misses them.
DEFAULT_ANCHOR_SIZES = ((8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256), (128, 256, 512))
DEFAULT_ASPECT_RATIOS = (0.5, 1.0, 2.0)


def detections_to_anomaly_map(detections: list[dict[str, torch.Tensor]], image_size: tuple[int, int]) -> torch.Tensor:
    """Collapse per-image detections into a dense ``[B, H, W]`` anomaly map.

    Each pixel takes the maximum over detections of ``score * soft_mask``, so
    a pixel is as anomalous as the most confident instance covering it. The
    predicted class is ignored. Images without detections map to all zeros.
    """
    maps = []
    for detection in detections:
        masks, scores = detection["masks"], detection["scores"]
        if masks.shape[0] == 0:
            maps.append(torch.zeros(image_size, dtype=torch.float32, device=masks.device))
        else:
            maps.append((masks[:, 0] * scores[:, None, None]).amax(dim=0))
    return torch.stack(maps)


def _load_matching_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load every tensor whose name and shape match; report what was skipped.

    The COCO checkpoint's RPN head (3 anchors per location) and box/mask
    predictors (91 classes) don't fit a model with custom anchors and defect
    classes, so those layers keep their fresh initialisation.
    """
    own_state = model.state_dict()
    matching = {k: v for k, v in state_dict.items() if k in own_state and v.shape == own_state[k].shape}
    skipped = sorted(k for k in state_dict if k not in matching)
    model.load_state_dict(matching, strict=False)
    log.info("Loaded %d/%d pretrained tensors; re-initialised: %s", len(matching), len(state_dict), skipped)


class MaskRCNNModel(nn.Module):
    """torchvision Mask R-CNN with anomaly-style inference outputs.

    Args:
        num_classes: Number of classes *including* background.
        backbone: torchvision ResNet/ResNeXt name, e.g. ``"resnet50"`` or
            ``"resnext101_64x4d"``.
        pretrained: ``"coco"`` (full detector weights, ``resnet50`` only),
            ``"imagenet"`` (backbone only) or ``"none"``.
        trainable_backbone_layers: Number of ResNet stages (from the top) left
            trainable; 3 matches MMDetection's ``frozen_stages=1``.
        min_size / max_size: Bounds of torchvision's internal resize. The
            defaults leave 1024x1280 images at native resolution.
        anchor_sizes / aspect_ratios: RPN anchors, one size tuple per FPN level.
        box_score_thresh: Detections below this score are discarded, which
            also sets the floor of the non-zero anomaly-map values.
        box_nms_thresh / box_detections_per_img: Standard RoI-head test settings.
    """

    def __init__(
        self,
        num_classes: int = 8,
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
    ) -> None:
        super().__init__()
        if pretrained not in {"coco", "imagenet", "none"}:
            msg = f"pretrained must be 'coco', 'imagenet' or 'none', got {pretrained!r}"
            raise ValueError(msg)
        if pretrained == "coco" and backbone != "resnet50":
            msg = f"COCO detector weights only exist for resnet50, not {backbone!r}; use pretrained='imagenet'."
            raise ValueError(msg)

        backbone_weights = get_model_weights(backbone).DEFAULT if pretrained == "imagenet" else None
        fpn_backbone = resnet_fpn_backbone(
            backbone_name=backbone,
            weights=backbone_weights,
            trainable_layers=trainable_backbone_layers,
        )
        anchor_sizes = tuple(tuple(sizes) for sizes in anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=(tuple(aspect_ratios),) * len(anchor_sizes),
        )
        self.detector = TorchvisionMaskRCNN(
            fpn_backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            rpn_anchor_generator=anchor_generator,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
        )
        if pretrained == "coco":
            coco_state = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True)
            _load_matching_weights(self.detector, coco_state)

    def compute_losses(self, images: torch.Tensor, targets: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Return Mask R-CNN's five loss terms for a batch.

        torchvision only returns losses in train mode, so this switches mode
        for the forward pass and restores it afterwards; that makes it usable
        from a validation loop (wrapped in ``torch.no_grad`` by the caller).
        """
        was_training = self.detector.training
        self.detector.train()
        try:
            return self.detector(list(images), targets)
        finally:
            self.detector.train(was_training)

    def detect(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return raw per-image detections: ``boxes``, ``labels``, ``scores``, soft ``masks`` ``[N, 1, H, W]``."""
        was_training = self.detector.training
        self.detector.eval()
        try:
            return self.detector(list(images))
        finally:
            self.detector.train(was_training)

    def forward(self, images: torch.Tensor) -> InferenceBatch:
        """Detect, then collapse to ``anomaly_map`` / ``pred_score``.

        ``pred_mask`` and ``pred_label`` are left unset: anomalib's
        post-processor derives them from thresholds fitted on the validation
        split, exactly as it does for the one-class models.
        """
        anomaly_map = detections_to_anomaly_map(self.detect(images), tuple(images.shape[-2:]))
        return InferenceBatch(pred_score=anomaly_map.amax(dim=(-2, -1)), anomaly_map=anomaly_map)
