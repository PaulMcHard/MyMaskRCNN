"""Shared fixtures: a tiny synthetic COCO dataset in the 3D-ADAM on-disk layout."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 80
PARTS = ("1m1", "tapa3m1")
GOOD_FOLDERS = {"1m1": "1M1_A_Good", "tapa3m1": "Tapa3M1_A_Good"}
CATEGORIES = ["bulge", "crack", "hole", "under_extrusion", "over_extrusion", "cut", "scratch"]
IMAGES_PER_PART = {"train": 3, "val": 2, "test": 2}


def _load_script(name: str):
    """Import a module from ``scripts/`` (not a package) by file path."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rng.integers(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


@pytest.fixture(scope="session")
def coco_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build the dataset once per session and return its paths.

    Layout mirrors the real data: ``<root>/<part>/<defect>/rgb`` images with
    per-instance PNGs in ``ground_truth``, Windows-style ``file_name`` values,
    one defect image with no annotations in the test split, and a separate
    root of good images that ``scripts/add_good_images.py`` samples from.
    """
    base = tmp_path_factory.mktemp("adam3d")
    root, good_root, ann_dir = base / "supervised", base / "full_masked", base / "annotations"
    ann_dir.mkdir()
    rng = np.random.default_rng(0)

    categories = [{"id": i + 1, "name": name, "supercategory": "defect"} for i, name in enumerate(CATEGORIES)]
    image_id = annotation_id = 0
    for split, per_part in IMAGES_PER_PART.items():
        images, annotations = [], []
        for part in PARTS:
            for index in range(per_part):
                name = f"{part.upper()}_A_Bulge_Nano_{split}_{index:03d}_image"
                _write_image(root / part / "bulge" / "rgb" / f"{name}.png", rng)
                images.append({
                    "id": image_id,
                    "file_name": f"{part}\\bulge\\rgb\\{name}.png",
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                    "model_id": part,
                    "defect_type": "bulge",
                })
                for instance, (x0, y0) in enumerate([(8, 8), (44, 30)]):
                    w, h = 16, 12
                    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                    mask[y0 : y0 + h, x0 : x0 + w] = 255
                    mask_path = root / part / "bulge" / "ground_truth" / f"{name}_bulge_defect_{instance + 1}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask).save(mask_path)
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1 + instance,
                        # Like cv2.findContours in convert_coco.py: vertices at the centres of
                        # the boundary pixels, so the polygon rasterises smaller than the PNG.
                        "segmentation": [[x0, y0, x0 + w - 1, y0, x0 + w - 1, y0 + h - 1, x0, y0 + h - 1]],
                        "area": w * h,
                        "bbox": [x0, y0, w, h],
                        "iscrowd": 0,
                    })
                    annotation_id += 1
                image_id += 1

        if split == "test":
            # A defect image whose masks are missing: must be dropped, not scored as normal.
            name = "1M1_A_Crack_Nano_unannotated_image"
            _write_image(root / "1m1" / "crack" / "rgb" / f"{name}.png", rng)
            images.append({
                "id": image_id,
                "file_name": f"1m1\\crack\\rgb\\{name}.png",
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "model_id": "1m1",
                "defect_type": "crack",
            })
            image_id += 1

        with (ann_dir / f"annotations_{split}.json").open("w") as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    for part, folder in GOOD_FOLDERS.items():
        for index in range(6):
            _write_image(good_root / folder / "MechMind-Nano" / "images" / f"{folder}_Nano_{index:03d}_image.png", rng)

    written = _load_script("add_good_images").add_good_images(ann_dir, good_root)

    return {
        "root": root,
        "good_root": good_root,
        "ann_dir": ann_dir,
        "train_ann_file": ann_dir / "annotations_train.json",
        "val_ann_file": written["val"],
        "test_ann_file": written["test"],
    }


@pytest.fixture
def datamodule_kwargs(coco_dataset: dict) -> dict:
    """Init args for a ``CocoInstanceDataModule`` over the synthetic dataset."""
    return {
        "root": coco_dataset["root"],
        "good_root": coco_dataset["good_root"],
        "train_ann_file": coco_dataset["train_ann_file"],
        "val_ann_file": coco_dataset["val_ann_file"],
        "test_ann_file": coco_dataset["test_ann_file"],
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "num_workers": 0,
        "crop_size": 64,
        "resize_range": (64, 80),
    }
