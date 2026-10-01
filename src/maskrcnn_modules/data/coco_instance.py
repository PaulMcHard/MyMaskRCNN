"""COCO instance-segmentation data as an anomalib dataset / datamodule.

anomalib's stock datamodules follow the one-class convention: normal images
in train, abnormal images only in val/test, and a binary mask as the only
annotation. A supervised detector needs the opposite on both counts, so this
module supplies:

- :class:`CocoInstanceDataset` -- yields :class:`InstanceItem` (image, binary
  ``gt_mask``, ``gt_label`` *and* per-instance boxes / classes / masks).
- :class:`CocoInstanceDataModule` -- takes explicit train/val/test annotation
  files and bypasses anomalib's automatic split logic.

Images without annotations are treated as follows: entries tagged
``"source": "good"`` (added by ``scripts/add_good_images.py``) are *normal*
samples; any other unannotated image is a defect image whose masks are
missing, and is dropped so it can't be scored as normal.
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
from PIL import Image as PILImage
from pycocotools import mask as mask_utils
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, Image, Mask

from maskrcnn_modules.data.dataclasses import InstanceBatch, InstanceItem

log = logging.getLogger(__name__)

MASK_SOURCES = ("polygon", "png")


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


def polygons_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:  # noqa: ANN401
    """Rasterise one COCO ``segmentation`` (polygon list or RLE) to a ``[H, W]`` uint8 mask."""
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    return mask_utils.decode(rle)


class CocoInstanceDataset(AnomalibDataset):
    """One split of a COCO instance-segmentation dataset.

    Args:
        ann_file: COCO JSON for this split.
        root: Directory the COCO ``file_name`` values are relative to.
        split: Which split this is; stored in the ``samples`` DataFrame.
        good_root: Directory that ``"source": "good"`` entries are relative
            to. Defaults to ``root``.
        mask_source: Where masks come from. ``"png"`` reads the original
            per-instance mask PNGs found next to the image
            (``../<mask_dir>/<stem>_*_defect_*.png``): each annotation takes
            the PNG with the same bounding box and area, and ``gt_mask`` is
            the union of *all* PNGs, which is what the one-class methods are
            scored against. ``"polygon"`` rasterises the COCO polygons
            instead, and is also the per-instance fallback when no PNG
            matches. Prefer ``"png"``: the polygons are contours traced
            through pixel centres, and on 3D-ADAM's small defects they
            rasterise to about 90% of the true mask area.
        mask_dir: Name of the sibling directory holding the mask PNGs.
        parts: Keep only images whose ``model_id`` is in this list
            (case-insensitive). ``None`` keeps every part.
        augmentations: torchvision v2 transform applied jointly to the image,
            boxes and masks.
    """

    def __init__(
        self,
        ann_file: str | Path,
        root: str | Path,
        split: Split | str,
        good_root: str | Path | None = None,
        mask_source: str = "png",
        mask_dir: str = "ground_truth",
        parts: list[str] | None = None,
        augmentations: v2.Transform | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        if mask_source not in MASK_SOURCES:
            msg = f"mask_source must be one of {MASK_SOURCES}, got {mask_source!r}"
            raise ValueError(msg)

        self.root = Path(root)
        self.good_root = Path(good_root) if good_root else self.root
        self.split = Split(split)
        self.mask_source = mask_source
        self.mask_dir = mask_dir
        self._warned_missing_png = False

        with Path(ann_file).open() as f:
            coco = json.load(f)
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        annotations_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        keep_parts = {part.lower() for part in parts} if parts else None
        rows = []
        # Keyed by image path rather than row index, so the lookup survives
        # the re-sorting / subsampling anomalib applies to ``samples``.
        self._annotations: dict[str, list[dict]] = {}
        num_dropped = 0
        for img in coco["images"]:
            part = str(img.get("model_id", "unknown")).lower()
            if keep_parts is not None and part not in keep_parts:
                continue
            is_good = img.get("source") == "good"
            anns = annotations_by_image.get(img["id"], [])
            if not is_good and not anns:
                num_dropped += 1
                continue
            base = self.good_root if is_good else self.root
            image_path = str(base / img["file_name"].replace("\\", "/"))
            self._annotations[image_path] = [] if is_good else anns
            rows.append({
                "image_path": image_path,
                "split": self.split.value,
                "label_index": int(LabelName.NORMAL if is_good else LabelName.ABNORMAL),
                "mask_path": "",
                "part": part,
            })

        if num_dropped:
            log.warning(
                "%s: dropped %d defect image(s) with no annotations (missing masks).", Path(ann_file).name, num_dropped,
            )
        if not rows:
            msg = f"No usable images in {ann_file} (parts={parts})."
            raise ValueError(msg)
        missing = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
        if missing:
            msg = (
                f"{len(missing)} of {len(rows)} images listed in {ann_file} were not found, e.g. {missing[0]}. "
                f"Check root={self.root} and good_root={self.good_root}."
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

    def _read_png_masks(self, image_path: Path) -> dict[tuple[int, ...], np.ndarray]:
        """Read one image's per-instance mask PNGs, keyed by ``(x, y, w, h, area)``.

        The key is what ``convert_coco.py`` stored as each annotation's
        ``bbox`` and ``area``, so it identifies the PNG an annotation came from.
        """
        masks = {}
        for mask_file in sorted((image_path.parent.parent / self.mask_dir).glob(f"{image_path.stem}_*_defect_*.png")):
            mask = np.array(PILImage.open(mask_file).convert("L")) > 0
            if not mask.any():
                continue
            rows, cols = np.where(mask)
            key = (cols.min(), rows.min(), cols.max() - cols.min() + 1, rows.max() - rows.min() + 1, mask.sum())
            masks[tuple(int(value) for value in key)] = mask.astype(np.uint8)
        return masks

    def __getitem__(self, index: int) -> InstanceItem:
        """Load one image with its binary mask and instance annotations."""
        sample = self.samples.iloc[index]
        image_path = sample.image_path
        anns = self._annotations[image_path]

        image = Image(read_image(image_path, as_tensor=True))
        height, width = image.shape[-2:]

        png_masks = self._read_png_masks(Path(image_path)) if self.mask_source == "png" and anns else {}
        instance_masks = np.zeros((len(anns), height, width), dtype=np.uint8)
        num_from_polygons = 0
        for i, ann in enumerate(anns):
            mask = png_masks.get((*(int(value) for value in ann["bbox"]), int(ann["area"])))
            if mask is None:
                num_from_polygons += 1
                mask = polygons_to_mask(ann["segmentation"], height, width)
            instance_masks[i] = mask
        if self.mask_source == "png" and num_from_polygons and not self._warned_missing_png:
            log.warning(
                "%d of %d instances of %s have no matching mask PNG; using rasterised polygons for them.",
                num_from_polygons, len(anns), image_path,
            )
            self._warned_missing_png = True

        # Union of every PNG (not just the annotated ones) so the binary mask
        # equals the one the one-class methods are scored against.
        semantic = np.any(list(png_masks.values()), axis=0) if png_masks else instance_masks.any(axis=0)
        semantic = semantic.astype(np.uint8)

        # COCO boxes are [x, y, w, h]; torchvision detection models want xyxy.
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        target = {
            "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
            "classes": torch.tensor([ann["category_id"] for ann in anns], dtype=torch.int64),
            "semantic": Mask(torch.from_numpy(semantic)[None]),
        }
        if anns:
            target["masks"] = Mask(torch.from_numpy(instance_masks))

        if self.augmentations:
            image, target = self.augmentations(image, target)
        height, width = image.shape[-2:]

        boxes = target["boxes"].as_subclass(torch.Tensor)
        classes = target["classes"]
        masks = target["masks"].as_subclass(torch.Tensor) if anns else torch.zeros((0, height, width), dtype=torch.uint8)

        # A crop can push an instance (partly) out of frame: drop instances
        # whose box collapsed or whose mask is empty, as torchvision rejects them.
        has_extent = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        valid = has_extent & masks.flatten(1).bool().any(dim=1)

        return InstanceItem(
            image=image,
            gt_mask=Mask(target["semantic"].as_subclass(torch.Tensor)[0]),
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
        root: Directory the COCO ``file_name`` values are relative to.
        train_ann_file / val_ann_file / test_ann_file: COCO JSON per split.
        good_root: Directory that ``"source": "good"`` entries are relative
            to. Defaults to ``root``.
        name: Dataset name, used in the results directory.
        category: Category label, used in the results directory.
        train_batch_size / eval_batch_size / num_workers: Dataloader settings.
        train_augment: Apply :func:`build_train_augmentations` to training data.
        crop_size / resize_range: Parameters of the training augmentation.
        mask_source: ``"png"`` or ``"polygon"``; see :class:`CocoInstanceDataset`.
        mask_dir: Name of the directory holding per-instance mask PNGs.
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
        good_root: str | Path | None = None,
        name: str = "adam3d",
        category: str = "unified",
        train_batch_size: int = 4,
        eval_batch_size: int = 2,
        num_workers: int = 8,
        train_augment: bool = True,
        crop_size: int = 1024,
        resize_range: tuple[int, int] = (1024, 1434),
        mask_source: str = "png",
        mask_dir: str = "ground_truth",
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
        self.good_root = Path(good_root) if good_root else None
        self.train_ann_file = Path(train_ann_file)
        self.val_ann_file = Path(val_ann_file)
        self.test_ann_file = Path(test_ann_file)
        self.mask_source = mask_source
        self.mask_dir = mask_dir
        self.test_parts = list(test_parts) if test_parts else None
        self._name = name
        self._category = category

    @property
    def name(self) -> str:
        """Dataset name (anomalib's default would be the class name)."""
        return self._name

    def _setup(self, _stage: str | None = None) -> None:
        shared = {
            "root": self.root,
            "good_root": self.good_root,
            "mask_source": self.mask_source,
            "mask_dir": self.mask_dir,
        }
        self.train_data = CocoInstanceDataset(self.train_ann_file, split=Split.TRAIN, **shared)
        self.val_data = CocoInstanceDataset(self.val_ann_file, split=Split.VAL, **shared)
        self.test_data = CocoInstanceDataset(self.test_ann_file, split=Split.TEST, parts=self.test_parts, **shared)

    def _create_test_split(self) -> None:
        """Keep the test split as given.

        anomalib's default would separate the normal test images and, for some
        split modes, never add them back.
        """

    def _create_val_split(self) -> None:
        """Keep the validation split as given."""
