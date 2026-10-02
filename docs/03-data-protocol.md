# Data protocol

## Annotation files

| File | Contents |
|---|---|
| `adam3d/superdefect_default/annotations_train.json` | 871 defect images, 4646 instances |
| `adam3d/superdefect_default/annotations_val.json` | 168 defect + 168 good images, 870 instances |
| `adam3d/superdefect_default/annotations_test.json` | 216 defect + 216 good images, 1334 instances |

All three are built from `3d-adam-full-masked` (MechMind-Nano camera) by `scripts/build_coco_annotations.py`:

```bash
python scripts/build_coco_annotations.py \
    --superdefect-root ../SuperDefect \
    --dataset-root D:/Data/3d-adam-full-masked \
    --output-dir adam3d/superdefect_default
```

Each file's `info` block records how it was made: the SuperDefect config and commit the split came from, the split settings, the good-image settings, and the images left out.

The older `adam3d/annotations*.json` files (from `convert_coco.py` and an image-level split) are kept for the MMDetection scripts but are not used by the new pipeline. The next section explains why they were replaced.

## Why the earlier annotations were replaced

An audit against every MechMind-Nano mask PNG found:

| Problem | Evidence |
|---|---|
| Wrong labels | `convert_coco.py` labelled every instance with its folder's defect group. 569 of 6755 instances (8.4%) were mislabelled: 307 burr, 152 scratch (all labelled cut), 71 warp, 24 mark and 15 others. |
| Incomplete classes | The masks use 11 defect types; the old files had 7 categories, one of them (`scratch`) with no instances. |
| Missing images | 104 defect images in 6 folders were absent: the Combination, Marked and Warped groups, and both Gripper parts. |
| Leaky split | A folder is one printed specimen scanned from about 16 viewpoints. 78 of 79 folders had images in more than one split, so test images showed defects the model had trained on. |
| Shrunken masks | Contour polygons cover about 91% of the true mask area (per-instance IoU 0.85 against the PNGs). |
| Unmasked images | 36 defect images have no mask PNGs in the source and were included as defect images with no annotations. |

Box and area values were correct: every old annotation matched its source PNG exactly.

## Split

The split is SuperDefect's own. The generator imports `build_split_index` from SuperDefect's `data/adam3d.py` and calls it with the dataset settings of SuperDefect's `configs/default.yaml`:

| Setting | Value |
|---|---|
| `cameras` | `[MechMind-Nano]` |
| `part_ids` | the 20 parts listed in `default.yaml` (not `Gripper_Closed`, `Gripper_Open`, `TapaTBB`) |
| `split_strategy` | `per_folder` |
| `train_ratio` / `val_ratio` | 0.70 / 0.15 |
| `split_seed` | 42 |
| SuperDefect commit | `bdfb866` (clean working tree) |

`per_folder` holds out whole specimens and stratifies by defect group: within each group (Bulge, Cut, Hole, ...), folders are shuffled and divided 70/15/15. Groups with a single folder (Combination, Marked, Warped) go to train.

Verified properties of the generated files:

- The defect images in each split are exactly the images SuperDefect's split assigns there, minus the 36 without masks (25 train, 11 val, 0 test).
- None of the 105 specimen folders, defect or good, spans two splits.
- Train covers 18 parts. Val covers 10 parts, all also in train. Test covers 9 parts, 2 of which (`1M1`, `3M1`) have no training images. That comes from stratifying by defect group, not by part.

Because the test images equal SuperDefect's `default.yaml` test images, per-image results can be paired with SuperDefect runs that use that config. The good images are an addition: SuperDefect's own test split has none.

## Classes

Twelve classes: background plus every defect type found in the mask file names. Ids are fixed in the generator.

| Id | Type | Train | Val | Test |
|---:|---|---:|---:|---:|
| 1 | bulge | 1084 | 284 | 305 |
| 2 | burr | 255 | 38 | 30 |
| 3 | crack | 218 | 11 | 51 |
| 4 | cut | 1240 | 137 | 376 |
| 5 | gap | 2 | 0 | 0 |
| 6 | hole | 887 | 234 | 370 |
| 7 | mark | 90 | 0 | 14 |
| 8 | over_extrusion | 353 | 77 | 131 |
| 9 | scratch | 119 | 33 | 0 |
| 10 | under_extrusion | 315 | 53 | 57 |
| 11 | warp | 83 | 3 | 0 |

Each annotation is one mask PNG, labelled with the type in its file name (`<image>_<type>_defect_<n>.png`). A mask with several disjoint pieces stays one instance, as in the source.

Specimen-level splitting leaves some types absent from evaluation: test has no gap, scratch or warp, and val no gap or mark. Their per-class AP cannot be measured.

### Treating classes as background

`CocoInstanceDataModule(ignore_classes=[...])` removes the named types from training targets and from the evaluation ground truth in every split, as SuperDefect's `dataset.ignore_classes` does. A defect image left with no instances is dropped. The configs default to `[]`.

This matters for paired comparisons with SuperDefect, whose `default.yaml` ground truth differs from the full annotations in two ways:

| Difference | Cause | Effect on SuperDefect's ground truth |
|---|---|---|
| `ignore_classes: [gap, mark, roughess, scratch, warp]` | Config | Those types are background. The equivalent here is `ignore_classes: [gap, mark, scratch, warp]`; no roughess masks exist in the Nano images. |
| `under_extrusion` and `over_extrusion` masks are never read | Bug: SuperDefect's mask-name regex `([a-zA-Z]+)` rejects the underscore | Extrusion defects are background in both training and evaluation |

Same images do not mean same ground truth. For a strictly paired comparison, fix SuperDefect's regex and rerun it, and use the same `ignore_classes` in both. Adding the extrusion types to `ignore_classes` here would match SuperDefect's current behaviour, but it would copy the bug.

## Masks

Masks are stored as COCO RLE encoded from the PNGs, so they are pixel-exact; `bbox` and `area` are computed from the same mask. All 6850 instances were checked against their PNG after generation. Each annotation also records its PNG path in `mask_file`.

The binary `gt_mask` used by the pixel metrics is the union of an image's instances. Because every non-empty mask PNG is an annotation, this equals the OR of all its mask PNGs, which is how `SuperDefectExperiments/scripts/prepare_adam3d.py` builds the masks the one-class methods are scored against.

## Good images

Image-level AUROC, F1Max and false-alarm rate need negatives, and SuperDefect's split has none. The generator adds good images to val and test:

- Source: `<part>_<location>_Good/MechMind-Nano/images/` for the parts in each split.
- Amount: one good image per defect image, per part (`--good-ratio 1.0`). Train gets none by default (`--train-good-ratio 0`).
- Folders, not just images, are kept apart: each Good folder is a clean specimen and is assigned to one split only.
- Selection is seeded (`--seed 42`). Test is filled first, then val.

## Excluded images

36 defect images have no mask PNGs in the source (for example, 1M3_A1_Crack has 17 images but masks for only some of them). They are left out of the files and listed under `info.excluded_images_without_masks`. They contain defects, so they can be used neither as negatives nor as unannotated positives. None are in SuperDefect's test split.

## Image metadata

| Field | Example |
|---|---|
| `file_name` | `1M1_A_Bulge/MechMind-Nano/images/1M1_A_Bulge_Nano_000_image.png`, relative to the dataset root |
| `part`, `location`, `defect_group` | `1M1`, `A`, `Bulge` |
| `folder` | `1M1_A_Bulge`, the specimen |
| `camera`, `pose` | `MechMind-Nano`, `000` |
| `is_good` | `false` |

`part` drives the per-part test rows in `metrics.csv`.

## Resolution

Images are not downscaled. torchvision's internal transform is configured with `min_size=1024`, `max_size=1280`. With a median defect area of about 300 px, downscaling would remove a large share of the signal.

## Open questions

- **Good images in training.** A detector that never sees a clean part may fire more often on clean parts. One ablation with `--train-good-ratio` above zero would show whether this affects the false-alarm rate.
- **Per-part sample size.** The 9 test parts average 24 defect images each, unevenly. Individual per-part rows are noisy; the `mean` row is the headline, as in `SuperDefectExperiments`.
- **Part-held-out evaluation.** Not built. It would measure generalisation to unseen part types, but its test set would no longer match SuperDefect's, so results could not be paired.
