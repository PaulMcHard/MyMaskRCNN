"""Build COCO instance annotations for 3D-ADAM straight from ``3d-adam-full-masked``.

Replaces the ``convert_coco.py`` -> ``split.py`` pipeline, which (a) labelled
every instance with its *folder's* defect group instead of its own type,
(b) stored masks as contour polygons that rasterise ~10% too small, (c) skipped
the Combination / Marked / Warped folders, and (d) split by image, putting
views of the same physical specimen in train and test.

Here:

- **Split**: taken from SuperDefect's own ``build_split_index`` with the
  settings of a SuperDefect config (default ``configs/default.yaml``), so the
  train / val / test images are exactly SuperDefect's. It holds out whole
  specimens (one ``<part>_<location>_<group>`` folder = one printed part
  scanned from ~16 poses), stratified by defect group.
- **Labels**: one annotation per ``defect_masks/<stem>_<type>_defect_<n>.png``,
  labelled with ``<type>`` from the file name.
- **Masks**: COCO RLE encoded from the PNG, so they are exact; ``bbox`` and
  ``area`` are computed from the same mask. ``mask_file`` records the PNG.
- **Good images**: whole ``<part>_<location>_Good`` folders are assigned to
  one split each (so a clean specimen never appears in two splits), and
  images are sampled from them per part: ``--good-ratio`` good images per
  defect image in val and test, ``--train-good-ratio`` in train (default 0).
- **Excluded**: defect images with no mask PNGs (their defects are
  unannotated, so they can be neither positives nor negatives) are left out
  and listed in each file's ``info``. Empty mask PNGs are skipped.

Usage:
    python scripts/build_coco_annotations.py \\
        --superdefect-root ../SuperDefect \\
        --dataset-root D:/Data/3d-adam-full-masked \\
        --output-dir adam3d/superdefect_default
"""

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

# Every defect type that occurs in the mask file names. Ids are fixed so
# that annotation files built at different times stay compatible.
DEFECT_TYPES = (
    "bulge",
    "burr",
    "crack",
    "cut",
    "gap",
    "hole",
    "mark",
    "over_extrusion",
    "scratch",
    "under_extrusion",
    "warp",
)
CATEGORY_IDS = {name: index + 1 for index, name in enumerate(DEFECT_TYPES)}

SPLITS = ("train", "val", "test")
# Test is served first when good images run short, then val, then train.
GOOD_FILL_ORDER = ("test", "val", "train")

_FOLDER_RE = re.compile(r"^(?P<part>.+)_(?P<location>[A-Za-z0-9]+)_(?P<group>[A-Za-z]+)$")
_POSE_RE = re.compile(r"_(\d{3})_image\.png$")


def parse_folder(name):
    """Split ``<part>_<location>_<group>`` (part ids may contain underscores)."""
    match = _FOLDER_RE.match(name)
    return (match["part"], match["location"], match["group"]) if match else None


def image_metadata(dataset_root, image_path, is_good):
    """COCO ``images`` fields for one image, minus the id."""
    folder = image_path.parents[2].name
    part, location, group = parse_folder(folder)
    with Image.open(image_path) as image:
        width, height = image.size
    pose = _POSE_RE.search(image_path.name)
    return {
        "file_name": image_path.relative_to(dataset_root).as_posix(),
        "width": width,
        "height": height,
        "part": part,
        "location": location,
        "defect_group": group,
        "folder": folder,
        "camera": image_path.parents[1].name,
        "pose": pose.group(1) if pose else None,
        "is_good": is_good,
    }


def instance_annotations(dataset_root, image_path):
    """Read one image's mask PNGs.

    Returns ``(annotations without ids, number of empty PNGs skipped)``.
    Raises on a defect type outside :data:`DEFECT_TYPES`, so nothing is
    dropped silently.
    """
    stem = image_path.stem
    mask_name_re = re.compile(rf"^{re.escape(stem)}_(?P<type>.+)_defect_(?P<n>\d+)\.png$")
    masks_dir = image_path.parents[1] / "defect_masks"
    annotations, num_empty = [], 0
    for mask_path in sorted(masks_dir.glob(f"{stem}_*_defect_*.png")):
        match = mask_name_re.match(mask_path.name)
        if match is None:
            continue
        defect_type = match["type"].lower()
        if defect_type not in CATEGORY_IDS:
            msg = f"Unknown defect type {defect_type!r} in {mask_path}; add it to DEFECT_TYPES."
            raise ValueError(msg)
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        if not mask.any():
            num_empty += 1
            continue
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations.append({
            "category_id": CATEGORY_IDS[defect_type],
            "segmentation": rle,
            "area": int(mask.sum()),
            "bbox": [int(v) for v in mask_utils.toBbox(rle)],
            "iscrowd": 0,
            "mask_file": mask_path.relative_to(dataset_root).as_posix(),
        })
    return annotations, num_empty


def good_folders_by_part(dataset_root, cameras, parts):
    """``{part: [(folder name, [image paths])]}`` for the Good folders of ``parts``."""
    folders = defaultdict(list)
    for folder in sorted(dataset_root.iterdir()):
        parsed = parse_folder(folder.name) if folder.is_dir() else None
        if parsed is None or parsed[2] != "Good" or parsed[0] not in parts:
            continue
        images = sorted(p for camera in cameras for p in (folder / camera / "images").glob("*.png"))
        if images:
            folders[parsed[0]].append((folder.name, images))
    return folders


def select_good_images(good_folders, defect_counts, ratios, seed):
    """Choose good images per split, assigning each Good folder to one split.

    Args:
        good_folders: ``{part: [(folder, [image paths])]}``.
        defect_counts: ``{split: {part: number of defect images}}``.
        ratios: ``{split: good images per defect image}``.
        seed: Seed for the folder and image shuffles.

    Returns:
        ``({split: [image paths]}, [warning strings])``.
    """
    rng = random.Random(seed)
    selected = {split: [] for split in SPLITS}
    warnings = []
    for part in sorted(good_folders):
        pool = list(good_folders[part])
        rng.shuffle(pool)
        for split in GOOD_FILL_ORDER:
            wanted = round(defect_counts[split].get(part, 0) * ratios.get(split, 0.0))
            if wanted == 0:
                continue
            taken = []
            while pool and len(taken) < wanted:
                taken.extend(pool.pop(0)[1])
            if len(taken) < wanted:
                warnings.append(f"{split}/{part}: wanted {wanted} good images, only {len(taken)} available")
            rng.shuffle(taken)
            selected[split].extend(sorted(taken[:wanted]))
    return selected, warnings


def build_coco_splits(dataset_root, split_images, good_ratio=1.0, train_good_ratio=0.0, seed=42, cameras=None, info=None):
    """Build one COCO dict per split.

    Args:
        dataset_root: Root of ``3d-adam-full-masked``.
        split_images: ``{split: [defect image paths]}`` -- the split to apply.
        good_ratio: Good images per defect image in val and test.
        train_good_ratio: Good images per defect image in train.
        seed: Seed for good-image selection.
        cameras: Cameras to draw good images from (default: those of the defect images).
        info: Extra provenance merged into every file's ``info``.

    Returns:
        ``({split: coco dict}, summary dict)``.
    """
    dataset_root = Path(dataset_root)
    cameras = cameras or sorted({Path(p).parents[1].name for paths in split_images.values() for p in paths})
    categories = [{"id": CATEGORY_IDS[name], "name": name, "supercategory": "defect"} for name in DEFECT_TYPES]

    image_id = annotation_id = 1
    cocos, excluded, num_empty_masks = {}, {}, 0
    defect_counts = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        images, annotations, excluded[split] = [], [], []
        for image_path in sorted(Path(p) for p in split_images.get(split, [])):
            instances, num_empty = instance_annotations(dataset_root, image_path)
            num_empty_masks += num_empty
            if not instances:
                excluded[split].append(image_path.relative_to(dataset_root).as_posix())
                continue
            entry = image_metadata(dataset_root, image_path, is_good=False)
            images.append({"id": image_id, **entry})
            defect_counts[split][entry["part"]] += 1
            for instance in instances:
                annotations.append({"id": annotation_id, "image_id": image_id, **instance})
                annotation_id += 1
            image_id += 1
        cocos[split] = {"images": images, "annotations": annotations, "categories": categories}

    parts = {part for counts in defect_counts.values() for part in counts}
    ratios = {"train": train_good_ratio, "val": good_ratio, "test": good_ratio}
    good, good_warnings = select_good_images(
        good_folders_by_part(dataset_root, cameras, parts), defect_counts, ratios, seed,
    )
    for split in SPLITS:
        for image_path in good[split]:
            cocos[split]["images"].append({"id": image_id, **image_metadata(dataset_root, image_path, is_good=True)})
            image_id += 1

    created = datetime.now().strftime("%Y-%m-%d")
    for split in SPLITS:
        cocos[split] = {
            "info": {
                "description": f"3D-ADAM instance segmentation, {split} split",
                "date_created": created,
                "generator": "scripts/build_coco_annotations.py",
                "good_images": {"ratio_val_test": good_ratio, "ratio_train": train_good_ratio, "seed": seed},
                "excluded_images_without_masks": excluded[split],
                **(info or {}),
            },
            **cocos[split],
        }

    summary = {
        "defect_counts": defect_counts,
        "excluded": excluded,
        "num_empty_masks": num_empty_masks,
        "good_warnings": good_warnings,
    }
    return cocos, summary


def superdefect_split(superdefect_root, config_path, dataset_root=None):
    """Run SuperDefect's own ``build_split_index`` with a SuperDefect config's settings.

    Returns ``({split: [image paths]}, provenance dict)``.
    """
    superdefect_root = Path(superdefect_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = superdefect_root / config_path
    with config_path.open() as f:
        dataset_cfg = yaml.safe_load(f)["dataset"]
    if dataset_cfg.get("kind", "adam3d") != "adam3d":
        msg = f"{config_path} is not a 3D-ADAM config (dataset.kind={dataset_cfg.get('kind')!r})"
        raise ValueError(msg)

    sys.path.insert(0, str(superdefect_root))
    from data.adam3d import build_split_index  # noqa: PLC0415  (SuperDefect's package)

    root = Path(dataset_root or dataset_cfg["root"])
    settings = {
        "dataset_root": root.as_posix(),
        "cameras": list(dataset_cfg["cameras"]),
        "split_strategy": dataset_cfg["split_strategy"],
        "train_ratio": dataset_cfg["train_ratio"],
        "val_ratio": dataset_cfg["val_ratio"],
        "seed": dataset_cfg["split_seed"],
        "part_ids": list(dataset_cfg.get("part_ids") or []),
    }
    splits, _ = build_split_index(
        dataset_root=root,
        cameras=settings["cameras"],
        split_strategy=settings["split_strategy"],
        train_ratio=settings["train_ratio"],
        val_ratio=settings["val_ratio"],
        seed=settings["seed"],
        part_ids=settings["part_ids"] or None,
    )

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(superdefect_root), *args], capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    provenance = {
        "split_source": {
            "superdefect_config": config_path.relative_to(superdefect_root).as_posix(),
            "superdefect_commit": git("rev-parse", "HEAD"),
            "superdefect_dirty": bool(status) if status is not None else None,
            **settings,
        },
    }
    return {split: [record.image_path for record in splits[split]] for split in SPLITS}, provenance


def main():
    parser = argparse.ArgumentParser(description="Build 3D-ADAM COCO annotations on SuperDefect's split")
    parser.add_argument("--superdefect-root", type=str, default="../SuperDefect",
                        help="Path to the SuperDefect repository (default: ../SuperDefect)")
    parser.add_argument("--superdefect-config", type=str, default="configs/default.yaml",
                        help="SuperDefect config whose dataset split to reproduce (default: configs/default.yaml)")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Root of 3d-adam-full-masked (default: dataset.root from the SuperDefect config)")
    parser.add_argument("--output-dir", type=str, default="adam3d/superdefect_default",
                        help="Where to write annotations_{train,val,test}.json")
    parser.add_argument("--good-ratio", type=float, default=1.0,
                        help="Good images per defect image in val and test (default: 1.0)")
    parser.add_argument("--train-good-ratio", type=float, default=0.0,
                        help="Good images per defect image in train (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for good-image selection (default: 42)")

    args = parser.parse_args()

    split_images, provenance = superdefect_split(args.superdefect_root, args.superdefect_config, args.dataset_root)
    dataset_root = Path(provenance["split_source"]["dataset_root"])
    if provenance["split_source"]["superdefect_dirty"]:
        print("Warning: the SuperDefect working tree has uncommitted changes; the recorded commit may not reproduce this split.")

    cocos, summary = build_coco_splits(
        dataset_root,
        split_images,
        good_ratio=args.good_ratio,
        train_good_ratio=args.train_good_ratio,
        seed=args.seed,
        cameras=provenance["split_source"]["cameras"],
        info=provenance,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {cat["id"]: cat["name"] for cat in cocos["train"]["categories"]}
    print(f"\nSplit: SuperDefect {provenance['split_source']['superdefect_config']} "
          f"@ {provenance['split_source']['superdefect_commit']}")
    for split in SPLITS:
        coco = cocos[split]
        output_file = output_dir / f"annotations_{split}.json"
        with output_file.open("w") as f:
            json.dump(coco, f)
        n_good = sum(img["is_good"] for img in coco["images"])
        parts = sorted({img["part"] for img in coco["images"] if not img["is_good"]})
        classes = Counter(names[a["category_id"]] for a in coco["annotations"])
        print(f"\n{split}: {len(coco['images']) - n_good} defect images + {n_good} good images, "
              f"{len(coco['annotations'])} instances, {len(parts)} parts -> {output_file}")
        print(f"  parts: {parts}")
        print(f"  instances: {dict(sorted(classes.items()))}")
        if summary["excluded"][split]:
            print(f"  excluded (no mask PNGs): {len(summary['excluded'][split])} images")
    print(f"\nEmpty mask PNGs skipped: {summary['num_empty_masks']}")
    for warning in summary["good_warnings"]:
        print(f"Warning: {warning}")

    return 0


if __name__ == "__main__":
    exit(main())
