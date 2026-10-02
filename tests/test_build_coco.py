import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

from tests.conftest import (
    EMPTY_MASK_IMAGE,
    FIRST_INSTANCE,
    SECOND_INSTANCE,
    UNMASKED_IMAGE,
    load_script,
    rectangle_mask,
)

build_coco = load_script("build_coco_annotations")


def _names(coco):
    return {cat["id"]: cat["name"] for cat in coco["categories"]}


def test_labels_come_from_mask_file_names_not_folders(built_coco):
    cocos, _ = built_coco
    for coco in cocos.values():
        names = _names(coco)
        folder_of = {img["id"]: img["folder"] for img in coco["images"]}
        for ann in coco["annotations"]:
            token = re.match(r".+_image_(.+)_defect_\d+\.png$", Path(ann["mask_file"]).name).group(1)
            assert names[ann["category_id"]] == token
        # A Bulge folder's images also hold burrs, labelled as burr.
        bulge_labels = Counter(names[a["category_id"]] for a in coco["annotations"] if "Bulge" in folder_of[a["image_id"]])
        assert set(bulge_labels) == {"bulge", "burr"}


def test_underscored_defect_type_is_parsed(built_coco):
    cocos, _ = built_coco
    names = _names(cocos["test"])

    assert "under_extrusion" in {names[a["category_id"]] for a in cocos["test"]["annotations"]}


def test_masks_are_exact_rle_with_matching_box_and_area(built_coco):
    cocos, _ = built_coco
    for ann in cocos["train"]["annotations"]:
        rle = dict(ann["segmentation"], counts=ann["segmentation"]["counts"].encode("ascii"))
        decoded = mask_utils.decode(rle).astype(bool)
        box = FIRST_INSTANCE if ann["mask_file"].endswith("_defect_1.png") else SECOND_INSTANCE

        assert np.array_equal(decoded, rectangle_mask(*box))
        assert ann["bbox"] == list(box)
        assert ann["area"] == box[2] * box[3]


def test_image_without_masks_is_excluded_and_listed(built_coco):
    cocos, summary = built_coco

    assert UNMASKED_IMAGE not in {img["file_name"] for img in cocos["test"]["images"]}
    assert cocos["test"]["info"]["excluded_images_without_masks"] == [UNMASKED_IMAGE]
    assert summary["excluded"]["test"] == [UNMASKED_IMAGE]


def test_empty_mask_png_is_skipped(built_coco):
    cocos, summary = built_coco
    image = next(img for img in cocos["train"]["images"] if img["file_name"].endswith(f"{EMPTY_MASK_IMAGE}.png"))

    assert summary["num_empty_masks"] == 1
    assert sum(ann["image_id"] == image["id"] for ann in cocos["train"]["annotations"]) == 2


def test_image_metadata_identifies_the_specimen(built_coco):
    cocos, _ = built_coco
    image = next(img for img in cocos["test"]["images"] if img["folder"] == "Tapa3M1_C_UnderExtrusion")

    assert (image["part"], image["location"], image["defect_group"]) == ("Tapa3M1", "C", "UnderExtrusion")
    assert (image["camera"], image["width"], image["height"], image["is_good"]) == ("MechMind-Nano", 80, 64, False)
    assert image["pose"] in {"000", "001"}


def test_good_images_match_defect_counts_per_part_and_folders_stay_in_one_split(built_coco):
    cocos, _ = built_coco
    folder_splits = defaultdict(set)
    for split, coco in cocos.items():
        for img in coco["images"]:
            folder_splits[img["folder"]].add(split)
        defects = Counter(img["part"] for img in coco["images"] if not img["is_good"])
        goods = Counter(img["part"] for img in coco["images"] if img["is_good"])
        if split == "train":
            assert not goods  # --train-good-ratio defaults to 0
        else:
            assert goods == defects

    assert all(len(splits) == 1 for splits in folder_splits.values())


def test_ids_are_unique_across_files(built_coco):
    cocos, _ = built_coco
    image_ids = [img["id"] for coco in cocos.values() for img in coco["images"]]
    annotation_ids = [ann["id"] for coco in cocos.values() for ann in coco["annotations"]]

    assert len(image_ids) == len(set(image_ids))
    assert len(annotation_ids) == len(set(annotation_ids))


def test_unknown_defect_type_is_an_error(tmp_path):
    image = tmp_path / "1M1_A_Bulge" / "MechMind-Nano" / "images" / "1M1_A_Bulge_Nano_000_image.png"
    image.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
    masks = image.parents[1] / "defect_masks"
    masks.mkdir()
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(masks / f"{image.stem}_wobble_defect_1.png")

    with pytest.raises(ValueError, match="Unknown defect type 'wobble'"):
        build_coco.build_coco_splits(tmp_path, {"train": [image]})


def test_superdefect_split_runs_the_configured_split(tmp_path, monkeypatch):
    """The split comes from SuperDefect's own function, called with the config's settings."""
    fake_repo = tmp_path / "SuperDefect"
    (fake_repo / "data").mkdir(parents=True)
    (fake_repo / "data" / "__init__.py").write_text("")
    (fake_repo / "data" / "adam3d.py").write_text(
        "from types import SimpleNamespace\n"
        "CALLS = []\n"
        "def build_split_index(**kwargs):\n"
        "    CALLS.append(kwargs)\n"
        "    record = SimpleNamespace(image_path=kwargs['dataset_root'] / 'x.png')\n"
        "    return {'train': [record], 'val': [], 'test': []}, None\n"
    )
    (fake_repo / "configs").mkdir()
    dataset = {
        "kind": "adam3d", "root": "D:/Data/somewhere", "cameras": ["MechMind-Nano"], "part_ids": ["1M1", "3M1"],
        "split_strategy": "per_folder", "train_ratio": 0.7, "val_ratio": 0.15, "split_seed": 42,
    }
    (fake_repo / "configs" / "default.yaml").write_text(yaml.safe_dump({"dataset": dataset}))
    for name in [m for m in sys.modules if m == "data" or m.startswith("data.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))

    split, provenance = build_coco.superdefect_split(fake_repo, "configs/default.yaml", dataset_root=tmp_path)

    calls = sys.modules["data.adam3d"].CALLS
    assert calls == [{
        "dataset_root": tmp_path, "cameras": ["MechMind-Nano"], "split_strategy": "per_folder",
        "train_ratio": 0.7, "val_ratio": 0.15, "seed": 42, "part_ids": ["1M1", "3M1"],
    }]
    assert split == {"train": [tmp_path / "x.png"], "val": [], "test": []}
    assert provenance["split_source"]["superdefect_config"] == "configs/default.yaml"
    assert provenance["split_source"]["part_ids"] == ["1M1", "3M1"]
