# Data protocol

## Current state of the annotations

Measured on `adam3d/annotations*.json`:

| Split | Images | Instances | Images without annotations |
|---|---:|---:|---:|
| all | 1299 | 6755 | 36 |
| train | 909 | 4748 | 26 |
| val | 194 | 1048 | 5 |
| test | 196 | 959 | 5 |

- Every image is 1280x1024 (MechMind-Nano).
- Every image is a defect image. There are no good parts in any split.
- All 21 parts appear in train, val and test. The split is by image, not by part.
- Instances per class: cut 1943, bulge 1710, hole 1571, over_extrusion 623, under_extrusion 526, crack 382, scratch 0.
- Median instance area is 308 px (10th percentile 132 px, 90th percentile 1811 px).
- `file_name` values use Windows backslashes.

## Decisions

### Test set

Mask R-CNN is scored on its own held-out test split. A supervised model cannot be tested on the anomalib per-part test sets used by the other methods, because those contain every abnormal image, including the ones Mask R-CNN trains on.

Consequence: results are comparable with the other methods as unpaired numbers. Paired tests (Wilcoxon on matched images) against those runs are not valid.

### Good images

Image-level AUROC, F1Max and false-alarm rate need negatives. Good images are added to the val and test splits:

- Source: `<part>_<loc>_Good/MechMind-Nano/images/*.png` in the `3d-adam-full-masked` layout. 2009 images across 27 parts are usable. The `Gripper_*` folders are skipped; those parts are not in the COCO set.
- Selection: stratified by part, seeded, disjoint between splits.
- Amount: one good image per defect image in each split by default (`--ratio 1.0`).
- Training: no good images by default (`--train-ratio 0`). See open questions.

`scripts/add_good_images.py` writes new annotation files (`annotations_val_good.json`, `annotations_test_good.json`) and leaves the originals untouched. Both files have been generated with the defaults:

| Split | Defect images (annotated) | Good images added |
|---|---:|---:|
| val | 189 | 189 |
| test | 191 | 191 |

Every part received exactly as many good images as it has defect images, and no good image appears in both splits.

Added entries look like:

```json
{"id": 100000, "file_name": "1M1_A_Good/MechMind-Nano/images/1M1_A_Good_Nano_000_image.png",
 "width": 1280, "height": 1024, "model_id": "1m1", "defect_type": "good", "source": "good"}
```

`source: "good"` tells the datamodule to resolve the path against `good_root` instead of `root`.

### Unannotated defect images

The 36 images without annotations are crack and under-extrusion images whose masks are missing. They are not good parts. If kept, they would be scored as normal at image level and as all-background at pixel level. They are dropped from every split.

### Ground-truth masks

The COCO polygons are not an accurate copy of the masks. `convert_coco.py` traces each mask with `cv2.findContours`, which puts polygon vertices at the centres of the boundary pixels. Rasterising that polygon loses roughly half a pixel all round, and 3D-ADAM defects are small enough for that to matter.

Measured on 91 instances from 27 test images (parts `1m1` and `tapa3m1`):

| Comparison | Result |
|---|---|
| Rasterised polygon area / true mask area | mean 0.906 (10th percentile 0.855) |
| Per-instance IoU, polygon vs original PNG | mean 0.852 (minimum 0.590) |
| Annotations matched to their source PNG by box and area | 91 of 91 |

So a model trained and scored on polygons learns masks about 9% too small. The earlier MMDetection runs had this bias, although their mask mAP was self-consistent because predictions and ground truth shared it.

The dataset therefore reads masks from the original PNGs (`mask_source: png`, the default for every split):

| Output | Construction |
|---|---|
| Instance masks (training targets, mAP) | For each annotation, the PNG in `ground_truth/` with the same bounding box and pixel area. `convert_coco.py` wrote both values straight from the PNG, so the match is exact. |
| Binary `gt_mask` (pixel metrics) | OR of all `ground_truth/<stem>_*_defect_*.png` for the image. This is the construction used by `prepare_adam3d.py::_compose_defect_mask` in `SuperDefectExperiments`, so it matches what the other methods are scored against. |

An annotation without a matching PNG falls back to its rasterised polygon, and the dataset logs a warning once. `mask_source: polygon` forces polygons everywhere.

### Classes

Eight classes are used: background plus the seven categories in the COCO file. `scratch` is kept so category ids stay aligned with the annotation file, even though it has no instances.

### Resolution

Images are not downscaled. torchvision's internal transform is configured with `min_size=1024`, `max_size=1280`. With a median defect area of 308 px, downscaling would remove a large share of the signal.

## Layout expected on disk

```
<root>/<model_id>/<defect_type>/rgb/<name>.png
<root>/<model_id>/<defect_type>/ground_truth/<name>_<defect>_defect_<n>.png
<good_root>/<part>_<loc>_Good/MechMind-Nano/images/<name>.png
```

`root` is the dataset root the COCO `file_name` values are relative to. `good_root` defaults to `root` when not given.

## Open questions

- **Good images in training.** A detector that never sees a clean part may fire more often on clean parts than one that does. One ablation with `--train-ratio` above zero would settle whether this matters for the false-alarm rate.
- **Part-held-out split.** `split.py` supports splitting by part, but the committed split is by image. The current numbers measure performance on parts seen in training. A part-held-out split would be a separate experiment.
- **Per-part sample size.** The test split has about 9 defect images and 9 good images per part, so individual per-part rows are noisy. The `mean` row is the headline number, as in `SuperDefectExperiments`.
