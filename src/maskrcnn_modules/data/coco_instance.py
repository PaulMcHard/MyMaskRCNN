"""COCO instance-segmentation data as an anomalib dataset / datamodule.

anomalib's stock datamodules follow the one-class convention: normal images
in train, abnormal images only in val/test, and a binary mask as the only
annotation. A supervised detector needs the opposite on both counts, so this
module supplies:

- :class:`CocoInstanceDataset` -- yields :class:`InstanceItem` (image, binary
  ``gt_mask``, ``gt_label`` *and* per-instance boxes / classes / masks).
- :class:`CocoInstanceDataModule` -- takes explicit train/val/test annotation
  files and bypasses anomalib's automatic split logic.

The annotation files are those written by ``scripts/build_coco_annotations.py``:
masks are COCO RLE (polygons are also accepted), images carry ``part`` and
``is_good``. Images with ``is_good: true`` are *normal* samples; any other
image without (remaining) annotations is dropped, since its defects would
otherwise be scored as background.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import LabelName, Split, TestSplitMode, ValSplitMode, read_image
from pandas import DataFrame
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, Image, Mask

from maskrcnn_modules.data.dataclasses import InstanceBatch, InstanceItem

log = logging.getLogger(__name__)


def build_train_augmentations(
    crop_size: int = 1024,
    resize_range: tuple[int, int] = (1024, 1434),
) -> v2.Compose:
    """Training augmentation ported from the MMDetection config in ``train_tb.py``.

    Multi-scale resize (shorter edge drawn from ``resize_range``; the default
    is 1.0x-1.4x of the 1024 px native shorter edge), a square random crop,
    and horizontal / vertical flips. Applied jointly to the image, boxes and
    masks. Degenerate instances left by the crop are removed by the dataset.
    """
    return v2.Compose([
        v2.RandomResize(min_size=resize_range[0], max_size=resize_range[1], antialias=True),
        v2.RandomCrop(crop_size, pad_if_needed=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ])


def segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:  # noqa: ANN401
    """Decode one COCO ``segmentation`` to a ``[H, W]`` uint8 mask.

    Accepts compressed RLE (``counts`` as a string, as written by
    ``build_coco_annotations.py``), uncompressed RLE, or a polygon list.
    """
    if isinstance(segmentation, dict) and isinstance(segmentation["counts"], str):
        return mask_utils.decode({"size": segmentation["size"], "counts": segmentation["counts"].encode("ascii")})
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    return mask_utils.decode(rle)


class CocoInstanceDataset(AnomalibDataset):
    """One split of a COCO instance-segmentation dataset.

    Args:
        ann_file: COCO JSON for this split.
        root: Directory the COCO ``file_name`` values are relative to.
        split: Which split this is; stored in the ``samples`` DataFrame.
        ignore_classes: Category names to treat as background, as
            SuperDefect's ``dataset.ignore_classes`` does: their instances
            are removed from the targets and from ``gt_mask``. A defect image
            left with no instances is dropped.
        parts: Keep only images whose ``part`` is in this list
            (case-insensitive). ``None`` keeps every part.
        augmentations: torchvision v2 transform applied jointly to the image,
            boxes and masks.
    """

    def __init__(
        self,
        ann_file: str | Path,
        root: str | Path,
        split: Split | str,
        ignore_classes: list[str] | None = None,
        parts: list[str] | None = None,
        augmentations: v2.Transform | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        self.root = Path(root)
        self.split = Split(split)

        with Path(ann_file).open() as f:
            coco = json.load(f)
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        unknown = set(ignore_classes or []) - set(self.categories.values())
        if unknown:
            msg = f"ignore_classes {sorted(unknown)} are not categories of {ann_file}: {sorted(self.categories.values())}"
            raise ValueError(msg)
        ignored_ids = {cat_id for cat_id, name in self.categories.items() if name in set(ignore_classes or [])}

        annotations_by_image: dict[int, list[dict]] = {}
        num_ignored = 0
        for ann in coco["annotations"]:
            if ann["category_id"] in ignored_ids:
                num_ignored += 1
                continue
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        keep_parts = {part.lower() for part in parts} if parts else None
        rows = []
        # Keyed by image path rather than row index, so the lookup survives
        # the re-sorting / subsampling anomalib applies to ``samples``.
        self._annotations: dict[str, list[dict]] = {}
        num_dropped = 0
        for img in coco["images"]:
            part = str(img.get("part", "unknown"))
            if keep_parts is not None and part.lower() not in keep_parts:
                continue
            is_good = bool(img.get("is_good", False))
            anns = annotations_by_image.get(img["id"], [])
            if not is_good and not anns:
                num_dropped += 1
                continue
            image_path = str(self.root / img["file_name"].replace("\\", "/"))
            self._annotations[image_path] = [] if is_good else anns
            rows.append({
                "image_path": image_path,
                "split": self.split.value,
                "label_index": int(LabelName.NORMAL if is_good else LabelName.ABNORMAL),
                "mask_path": "",
                "part": part,
            })

        if num_ignored:
            log.info("%s: %d instance(s) of %s treated as background.", Path(ann_file).name, num_ignored, ignore_classes)
        if num_dropped:
            log.warning(
                "%s: dropped %d defect image(s) with no (remaining) annotations.", Path(ann_file).name, num_dropped,
            )
        if not rows:
            msg = f"No usable images in {ann_file} (parts={parts})."
            raise ValueError(msg)
        missing = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
        if missing:
            msg = (
                f"{len(missing)} of {len(rows)} images listed in {ann_file} were not found, e.g. {missing[0]}. "
                f"Check root={self.root}."
            )
            raise FileNotFoundError(msg)

        self.samples = DataFrame(rows)

    @property
    def task(self) -> TaskType:
        """Always segmentation: every sample has a pixel mask (all zeros for good images)."""
        return TaskType.SEGMENTATION

    @property
    def collate_fn(self):  # noqa: ANN201
        """Collate into :class:`InstanceBatch`, keeping instance fields as per-image lists."""
        return InstanceBatch.collate

    def __getitem__(self, index: int) -> InstanceItem:
        """Load one image with its binary mask and instance annotations."""
        sample = self.samples.iloc[index]
        image_path = sample.image_path
        anns = self._annotations[image_path]

        image = Image(read_image(image_path, as_tensor=True))
        height, width = image.shape[-2:]

        masks = np.zeros((len(anns), height, width), dtype=np.uint8)
        for i, ann in enumerate(anns):
            masks[i] = segmentation_to_mask(ann["segmentation"], height, width)

        # COCO boxes are [x, y, w, h]; torchvision detection models want xyxy.
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        target = {
            "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
            "classes": torch.tensor([ann["category_id"] for ann in anns], dtype=torch.int64),
        }
        if anns:
            target["masks"] = Mask(torch.from_numpy(masks))

        if self.augmentations:
            image, target = self.augmentations(image, target)
        height, width = image.shape[-2:]

        boxes = target["boxes"].as_subclass(torch.Tensor)
        classes = target["classes"]
        masks = target["masks"].as_subclass(torch.Tensor) if anns else torch.zeros((0, height, width), dtype=torch.uint8)
        gt_mask = masks.bool().any(dim=0) if anns else torch.zeros((height, width), dtype=torch.bool)

        # A crop can push an instance (partly) out of frame: drop instances
        # whose box collapsed or whose mask is empty, as torchvision rejects them.
        has_extent = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        valid = has_extent & masks.flatten(1).bool().any(dim=1)

        return InstanceItem(
            image=image,
            gt_mask=Mask(gt_mask),
            gt_label=torch.tensor(int(sample.label_index)),
            image_path=image_path,
            gt_boxes=boxes[valid],
            gt_classes=classes[valid],
            gt_instance_masks=masks[valid],
        )


class CocoInstanceDataModule(AnomalibDataModule):
    """Datamodule over explicit train / val / test COCO annotation files.

    Unlike anomalib's stock datamodules, the three splits are given, not
    derived: abnormal images are present in training, and normal images in
    val/test stay where the annotation files put them.

    Args:
        root: Directory the COCO ``file_name`` values are relative to (the
            ``3d-adam-full-masked`` root for files from
            ``build_coco_annotations.py``).
        train_ann_file / val_ann_file / test_ann_file: COCO JSON per split.
        ignore_classes: Category names to treat as background in every
            split; see :class:`CocoInstanceDataset`.
        name: Dataset name, used in the results directory.
        category: Category label, used in the results directory.
        train_batch_size / eval_batch_size / num_workers: Dataloader settings.
        train_augment: Apply :func:`build_train_augmentations` to training data.
        crop_size / resize_range: Parameters of the training augmentation.
        test_parts: Restrict the *test* split to these parts. Validation
            stays pooled so thresholds remain shared across parts.
        seed: Unused by the explicit splits; accepted for config parity.
    """

    def __init__(
        self,
        root: str | Path,
        train_ann_file: str | Path,
        val_ann_file: str | Path,
        test_ann_file: str | Path,
        ignore_classes: list[str] | None = None,
        name: str = "adam3d",
        category: str = "unified",
        train_batch_size: int = 4,
        eval_batch_size: int = 2,
        num_workers: int = 8,
        train_augment: bool = True,
        crop_size: int = 1024,
        resize_range: tuple[int, int] = (1024, 1434),
        test_parts: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=build_train_augmentations(crop_size, tuple(resize_range)) if train_augment else None,
            val_split_mode=ValSplitMode.FROM_DIR,
            test_split_mode=TestSplitMode.FROM_DIR,
            seed=seed,
        )
        self.root = Path(root)
        self.train_ann_file = Path(train_ann_file)
        self.val_ann_file = Path(val_ann_file)
        self.test_ann_file = Path(test_ann_file)
        self.ignore_classes = list(ignore_classes) if ignore_classes else None
        self.test_parts = list(test_parts) if test_parts else None
        self._name = name
        self._category = category

    @property
    def name(self) -> str:
        """Dataset name (anomalib's default would be the class name)."""
        return self._name

    def _setup(self, _stage: str | None = None) -> None:
        shared = {"root": self.root, "ignore_classes": self.ignore_classes}
        self.train_data = CocoInstanceDataset(self.train_ann_file, split=Split.TRAIN, **shared)
        self.val_data = CocoInstanceDataset(self.val_ann_file, split=Split.VAL, **shared)
        self.test_data = CocoInstanceDataset(self.test_ann_file, split=Split.TEST, parts=self.test_parts, **shared)

    def _dataloader(self, dataset: CocoInstanceDataset, batch_size: int, shuffle: bool) -> DataLoader:
        """Like anomalib's loaders, but keeps workers alive and pins host memory.

        Without ``persistent_workers`` every validation pass restarts the
        workers, and on Windows each restart re-imports torch and anomalib in
        a fresh process.
        """
        return DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_data, self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_data, self.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_data, self.eval_batch_size, shuffle=False)

    def _create_test_split(self) -> None:
        """Keep the test split as given.

        anomalib's default would separate the normal test images and, for some
        split modes, never add them back.
        """

    def _create_val_split(self) -> None:
        """Keep the validation split as given."""
