"""Add 3D-ADAM ``Good`` images to the COCO splits as defect-free negatives.

The COCO splits in ``adam3d/`` hold defect images only, so image-level metrics
(AUROC, F1Max, false-alarm rate) have no negatives to be computed against.
This script samples good images per part and writes new annotation files next
to the originals, leaving the originals untouched:

    annotations_val.json   ->  annotations_val_good.json
    annotations_test.json  ->  annotations_test_good.json

Good images are read from the ``3d-adam-full-masked`` layout:

    <good-root>/<part>_<location>_Good/<camera>/images/<name>.png

Each added image entry carries no annotations and is tagged
``"defect_type": "good", "source": "good"``; its ``file_name`` is relative to
``--good-root`` (the datamodule resolves ``source == "good"`` against its
``good_root``).

Sampling is stratified by part, seeded, and disjoint across splits.

Usage:
    python scripts/add_good_images.py adam3d \\
        --good-root "D:/Data/3d-adam-full-masked"
    python scripts/add_good_images.py adam3d \\
        --good-root "D:/Data/3d-adam-full-masked" --ratio 0.5 --train-ratio 0.25
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

_GOOD_FOLDER_RE = re.compile(r"^(?P<part>[^_]+)_(?P<loc>[A-Za-z0-9]+)_Good$")

# Test is filled first so that, when a part runs short of good images, the
# held-out test split is the last one to be short-changed.
_SPLIT_ORDER = ("test", "val", "train")


def find_good_images(good_root, camera):
    """Return ``{part_id_lowercase: [image paths relative to good_root]}``."""
    pool = defaultdict(list)
    for folder in sorted(good_root.iterdir()):
        if not folder.is_dir():
            continue
        match = _GOOD_FOLDER_RE.match(folder.name)
        if match is None:
            continue
        images_dir = folder / camera / "images"
        if not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.glob("*.png")):
            pool[match["part"].lower()].append(image_path.relative_to(good_root).as_posix())
    return pool


def count_defect_images_per_part(coco_data):
    """Count annotated images per part (unannotated defect images are ignored)."""
    annotated_ids = {ann["image_id"] for ann in coco_data["annotations"]}
    counts = defaultdict(int)
    for img in coco_data["images"]:
        if img["id"] in annotated_ids:
            counts[str(img.get("model_id", "unknown")).lower()] += 1
    return counts


def add_good_images(annotations_dir, good_root, camera="MechMind-Nano", ratios=None, seed=42, suffix="_good"):
    """Write ``annotations_<split><suffix>.json`` for every split with a ratio above zero.

    Args:
        annotations_dir: Directory holding ``annotations_{train,val,test}.json``.
        good_root: Root of the ``3d-adam-full-masked`` layout.
        camera: Camera sub-folder to take good images from.
        ratios: ``{split: good images per defect image}``.
        seed: Random seed for the per-part shuffle.
        suffix: Appended to the split name in the output file name.

    Returns:
        ``{split: output path}`` for the files written.
    """
    annotations_dir = Path(annotations_dir)
    good_root = Path(good_root)
    ratios = ratios or {"test": 1.0, "val": 1.0, "train": 0.0}

    splits = {}
    for split in _SPLIT_ORDER:
        with open(annotations_dir / f"annotations_{split}.json", "r") as f:
            splits[split] = json.load(f)

    pool = find_good_images(good_root, camera)
    print(f"Found {sum(len(v) for v in pool.values())} good images across {len(pool)} parts in {good_root}")

    rng = random.Random(seed)
    for part in sorted(pool):
        rng.shuffle(pool[part])

    next_image_id = max(img["id"] for data in splits.values() for img in data["images"]) + 1

    written = {}
    for split in _SPLIT_ORDER:
        ratio = ratios.get(split, 0.0)
        if ratio <= 0:
            continue

        coco_data = splits[split]
        defect_counts = count_defect_images_per_part(coco_data)

        added = []
        for part in sorted(defect_counts):
            wanted = round(defect_counts[part] * ratio)
            available = pool.get(part, [])
            if len(available) < wanted:
                print(f"  Warning: {split}/{part}: wanted {wanted} good images, only {len(available)} left")
            taken, pool[part] = available[:wanted], available[wanted:]
            for file_name in taken:
                with Image.open(good_root / file_name) as image:
                    width, height = image.size
                added.append({
                    "id": next_image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "model_id": part,
                    "defect_type": "good",
                    "source": "good",
                })
                next_image_id += 1

        output = dict(coco_data)
        output["images"] = coco_data["images"] + added
        output_file = annotations_dir / f"annotations_{split}{suffix}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        written[split] = output_file

        print(f"{split}: {sum(defect_counts.values())} defect images + {len(added)} good images -> {output_file}")

    return written


def main():
    parser = argparse.ArgumentParser(description="Add 3D-ADAM Good images to the COCO splits as negatives")
    parser.add_argument("annotations_dir", type=str,
                        help="Directory containing annotations_{train,val,test}.json")
    parser.add_argument("--good-root", type=str, required=True,
                        help="Root of the 3d-adam-full-masked layout (<part>_<loc>_Good/<camera>/images)")
    parser.add_argument("--camera", type=str, default="MechMind-Nano",
                        help="Camera sub-folder to take good images from")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Good images per defect image in val and test (default: 1.0)")
    parser.add_argument("--train-ratio", type=float, default=0.0,
                        help="Good images per defect image in train (default: 0, none)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--suffix", type=str, default="_good",
                        help="Suffix of the output files (default: _good)")

    args = parser.parse_args()

    add_good_images(
        args.annotations_dir,
        args.good_root,
        camera=args.camera,
        ratios={"test": args.ratio, "val": args.ratio, "train": args.train_ratio},
        seed=args.seed,
        suffix=args.suffix,
    )

    return 0


if __name__ == "__main__":
    exit(main())
