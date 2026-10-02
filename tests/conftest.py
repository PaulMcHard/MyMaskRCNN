"""Shared fixtures: a tiny synthetic dataset in the ``3d-adam-full-masked`` layout."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 80
CAMERA = "MechMind-Nano"

# folder -> (split, number of images, defect type of the first instance)
DEFECT_FOLDERS = {
    "1M1_A_Bulge": ("train", 3, "bulge"),
    "Tapa3M1_A_Hole": ("train", 3, "hole"),
    "1M1_B_Bulge": ("val", 2, "bulge"),
    "Tapa3M1_B_Hole": ("val", 2, "hole"),
    "1M1_C_Bulge": ("test", 2, "bulge"),
    "Tapa3M1_C_UnderExtrusion": ("test", 2, "under_extrusion"),
}
GOOD_FOLDERS = ("1M1_A_Good", "1M1_B_Good", "Tapa3M1_A_Good", "Tapa3M1_B_Good")
# Every defect image has two instances: the folder's own type here, and a burr
# there -- so a label taken from the folder name would be wrong for half of them.
FIRST_INSTANCE = (8, 8, 16, 12)   # x, y, w, h
SECOND_INSTANCE = (44, 30, 16, 12)
# Edge cases: an extra test image with no mask PNGs, and an empty mask PNG.
UNMASKED_IMAGE = "1M1_C_Bulge/MechMind-Nano/images/1M1_C_Bulge_Nano_002_image.png"
EMPTY_MASK_IMAGE = "1M1_A_Bulge_Nano_000_image"


def load_script(name: str):
    """Import a module from ``scripts/`` (not a package) by file path."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rectangle_mask(x: int, y: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _image_path(root: Path, folder: str, index: int) -> Path:
    return root / folder / CAMERA / "images" / f"{folder}_Nano_{index:03d}_image.png"


@pytest.fixture(scope="session")
def full_masked_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Images and per-instance mask PNGs laid out like ``3d-adam-full-masked``."""
    root = tmp_path_factory.mktemp("full_masked")
    rng = np.random.default_rng(0)

    def write_image(path: Path) -> None:
        _write_png(path, rng.integers(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))

    for folder, (_, num_images, defect_type) in DEFECT_FOLDERS.items():
        for index in range(num_images):
            image_path = _image_path(root, folder, index)
            write_image(image_path)
            masks_dir = image_path.parents[1] / "defect_masks"
            for n, (box, kind) in enumerate([(FIRST_INSTANCE, defect_type), (SECOND_INSTANCE, "burr")], start=1):
                _write_png(masks_dir / f"{image_path.stem}_{kind}_defect_{n}.png", rectangle_mask(*box).astype(np.uint8) * 255)
    write_image(root / UNMASKED_IMAGE)
    _write_png(
        root / "1M1_A_Bulge" / CAMERA / "defect_masks" / f"{EMPTY_MASK_IMAGE}_bulge_defect_3.png",
        np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8),
    )
    for folder in GOOD_FOLDERS:
        for index in range(6):
            write_image(_image_path(root, folder, index))
    return root


@pytest.fixture(scope="session")
def split_images(full_masked_root: Path) -> dict:
    """A hand-made specimen split standing in for SuperDefect's."""
    split = {"train": [], "val": [], "test": [full_masked_root / UNMASKED_IMAGE]}
    for folder, (name, num_images, _) in DEFECT_FOLDERS.items():
        split[name].extend(_image_path(full_masked_root, folder, index) for index in range(num_images))
    return split


@pytest.fixture(scope="session")
def built_coco(full_masked_root: Path, split_images: dict) -> tuple[dict, dict]:
    """``(cocos, summary)`` from ``build_coco_splits`` on the synthetic dataset."""
    return load_script("build_coco_annotations").build_coco_splits(full_masked_root, split_images, seed=0)


@pytest.fixture(scope="session")
def coco_dataset(full_masked_root: Path, built_coco: tuple[dict, dict], tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The built annotation files written to disk, plus the dataset root."""
    cocos, _ = built_coco
    ann_dir = tmp_path_factory.mktemp("annotations")
    paths = {}
    for split, coco in cocos.items():
        paths[f"{split}_ann_file"] = ann_dir / f"annotations_{split}.json"
        with paths[f"{split}_ann_file"].open("w") as f:
            json.dump(coco, f)
    return {"root": full_masked_root, **paths}


@pytest.fixture
def datamodule_kwargs(coco_dataset: dict) -> dict:
    """Init args for a ``CocoInstanceDataModule`` over the synthetic dataset."""
    return {
        "root": coco_dataset["root"],
        "train_ann_file": coco_dataset["train_ann_file"],
        "val_ann_file": coco_dataset["val_ann_file"],
        "test_ann_file": coco_dataset["test_ann_file"],
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "num_workers": 0,
        "crop_size": 64,
        "resize_range": (64, 80),
    }
