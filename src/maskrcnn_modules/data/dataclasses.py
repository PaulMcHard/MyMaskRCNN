"""anomalib image dataclasses extended with instance annotations.

anomalib's ``ImageItem`` / ``ImageBatch`` carry a binary ``gt_mask`` only,
which is all a one-class model needs. Mask R-CNN trains on *instances*, so
these subclasses add per-image boxes, class ids and instance masks while
staying drop-in compatible with anomalib's callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from anomalib.data import ImageBatch, ImageItem

INSTANCE_FIELDS = ("gt_boxes", "gt_classes", "gt_instance_masks")


@dataclass
class InstanceItem(ImageItem):
    """One image plus its instance annotations.

    Attributes:
        gt_boxes: ``[N, 4]`` float boxes in absolute ``xyxy`` pixels.
        gt_classes: ``[N]`` int64 class ids (1-based; 0 is background).
        gt_instance_masks: ``[N, H, W]`` uint8 masks, one per instance.
    """

    gt_boxes: torch.Tensor | None = None
    gt_classes: torch.Tensor | None = None
    gt_instance_masks: torch.Tensor | None = None


@dataclass
class InstanceBatch(ImageBatch):
    """A batch of :class:`InstanceItem`.

    The instance fields are per-image *lists*: the number of instances varies
    between images, so they cannot be stacked like ``image`` and ``gt_mask``.
    """

    item_class: ClassVar[type[InstanceItem]] = InstanceItem

    gt_boxes: list[torch.Tensor] | None = None
    gt_classes: list[torch.Tensor] | None = None
    gt_instance_masks: list[torch.Tensor] | None = None

    @classmethod
    def collate(cls, items: list[InstanceItem]) -> InstanceBatch:
        """Stack the standard fields and keep the instance fields as lists."""
        instance_values = {name: [getattr(item, name) for item in items] for name in INSTANCE_FIELDS}
        # The base collate stacks every non-None field with ``default_collate``,
        # which fails on ragged tensors, so hide the instance fields from it.
        for item in items:
            for name in INSTANCE_FIELDS:
                setattr(item, name, None)
        batch = super().collate(items)
        for name, values in instance_values.items():
            setattr(batch, name, values)
        return batch

    def targets(self) -> list[dict[str, torch.Tensor]]:
        """Return the batch's annotations in torchvision detection format."""
        return [
            {"boxes": boxes, "labels": classes, "masks": masks}
            for boxes, classes, masks in zip(self.gt_boxes, self.gt_classes, self.gt_instance_masks, strict=True)
        ]
