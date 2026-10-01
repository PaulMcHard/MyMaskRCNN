# Architecture

## Layout

```
main.py                                  YAML-driven runner
configs/experiments/*.yaml               experiment configs
scripts/add_good_images.py               adds good images to the val/test annotation files
src/maskrcnn_modules/
  maskrcnn/torch_model.py                torchvision Mask R-CNN + detections -> anomaly map
  maskrcnn/lightning_model.py            MaskRCNN(AnomalibModule)
  maskrcnn/post_processing.py            DetectionPostProcessor, SparseF1AdaptiveThreshold
  data/dataclasses.py                    InstanceItem, InstanceBatch
  data/coco_instance.py                  CocoInstanceDataset, CocoInstanceDataModule
  shared/metrics.py, reporting.py, stats.py    vendored from SuperDefectExperiments
tests/
```

The package is named `maskrcnn_modules`, mirroring `anomaly_modules` in `SuperDefectExperiments`, so both can be installed in the same environment.

## Flow of one experiment

```
fit                     supervised training on defect images
  every N steps         validation: val loss, val mAP, thresholds refitted
validate (best ckpt)    load best weights, fit image + pixel thresholds on pooled val
for each part:
  test                  16 evaluator metrics + mAP on that part's test images
  predict               per-image rows -> <part>/per_image_metrics.csv
[test pooled]           optional: all test images in one pass
write metrics.csv       per-part rows, mean row, optional pooled row
```

Entry point: `main.py::run_experiment`. It follows `run_multi_class` in `SuperDefectExperiments/main.py`: one model for all categories, tested category by category.

## Model

### `MaskRCNNModel` (`torch_model.py`)

Wraps `torchvision.models.detection.MaskRCNN`.

| Setting | Value | Reason |
|---|---|---|
| Backbone | ResNet-50 FPN (default) or any torchvision ResNet/ResNeXt | `resnet50` has COCO detector weights; `resnext101_64x4d` matches the earlier MMDetection config |
| Pretraining | `coco`, `imagenet` or `none` | With `coco`, every tensor whose shape matches is loaded; the RPN head and the box/mask predictors are re-initialised because anchors and classes differ |
| Trainable backbone stages | 3 | Same as MMDetection `frozen_stages=1` |
| `min_size` / `max_size` | 1024 / 1280 | Native resolution, no downscaling |
| Anchor sizes | (8,16,32) ... (128,256,512), ratios 0.5/1/2 | Same as MMDetection `scales=[2,4,8]`; torchvision's default starts at 32 px, above the median defect size |
| Score threshold, NMS, detections per image | 0.05, 0.5, 100 | Same as the MMDetection config |

Three entry points:

- `compute_losses(images, targets)` returns the five Mask R-CNN losses.
- `detect(images)` returns raw detections (boxes, labels, scores, soft masks).
- `forward(images)` returns an anomalib `InferenceBatch` with `anomaly_map` and `pred_score`.

### Detections to anomaly outputs

`detections_to_anomaly_map` takes, for each pixel, the maximum over detections of `score x soft_mask`. Pixels outside all detections are zero. The predicted class is ignored. `pred_score` is the map's maximum.

`pred_mask` and `pred_label` are not produced by the model. The post-processor derives them from thresholds fitted on validation, as it does for the one-class methods.

### `MaskRCNN` (`lightning_model.py`)

| Hook | Behaviour |
|---|---|
| `training_step` | Sums the five losses; logs each term and the total under `train/` |
| `validation_step` | Optionally logs the losses under `val/`; runs detection; updates val mAP; writes `anomaly_map` and `pred_score` into the batch |
| `test_step` | Runs detection; updates test mAP; writes `anomaly_map` and `pred_score` into the batch |
| `on_test_start` | Raises if no threshold has been fitted |
| `configure_optimizers` | SGD (lr 0.01, momentum 0.9, weight decay 1e-3), linear warm-up then cosine decay, stepped per iteration |
| `configure_evaluator` | The vendored 16-metric evaluator |
| `configure_post_processor` | `DetectionPostProcessor` |
| `trainer_arguments` | `gradient_clip_val` (default 35), no sanity validation |
| `learning_type` | `ONE_CLASS` (see [01-anomalib-supervised-decision.md](01-anomalib-supervised-decision.md)) |

Points that are easy to get wrong:

- **The batch is updated in place.** anomalib's `Evaluator` reads the batch object passed to the hook, not the step's return value. The post-processor reads the return value. Updating in place and returning the same object satisfies both.
- **mAP lives in the module.** COCO mAP needs per-instance predictions, which the evaluator never sees, so it is computed with torchmetrics `MeanAveragePrecision` inside the module.
- **No pre-processor.** torchvision's Mask R-CNN normalises and resizes internally. Adding anomalib's default pre-processor would resize images to 256x256.

### `DetectionPostProcessor` (`post_processing.py`)

anomalib's `PostProcessor` with one metric swapped: the pixel threshold is fitted by `SparseF1AdaptiveThreshold`, which gives the same threshold as anomalib's `F1AdaptiveThreshold` while storing only non-zero-score pixels. Reasons and the equality test are in [04-metrics.md](04-metrics.md).

## Data

### `InstanceItem` / `InstanceBatch` (`dataclasses.py`)

Subclasses of anomalib's `ImageItem` / `ImageBatch` with three extra fields:

| Field | Item | Batch |
|---|---|---|
| `gt_boxes` | `[N, 4]` xyxy | list of `[N_i, 4]` |
| `gt_classes` | `[N]` int64, 1-based | list of `[N_i]` |
| `gt_instance_masks` | `[N, H, W]` uint8 | list of `[N_i, H, W]` |

The batch fields are lists because the number of instances differs per image. `InstanceBatch.collate` hides them from anomalib's `default_collate` and reattaches them. `InstanceBatch.targets()` returns them in torchvision's detection format.

Everything anomalib's callbacks expect (`image`, `gt_mask`, `gt_label`, `image_path`, the prediction fields, `update()`, `items`) is inherited unchanged.

### `CocoInstanceDataset` (`coco_instance.py`)

An `AnomalibDataset` built from one COCO JSON.

- Builds the `samples` DataFrame anomalib requires (`image_path`, `split`, `label_index`, `mask_path`) plus a `part` column.
- Normalises backslashes in `file_name`.
- Treats entries tagged `source: good` as normal images, resolved against `good_root`.
- Drops other images that have no annotations.
- Loads instance masks and the binary mask as described in [03-data-protocol.md](03-data-protocol.md).
- Applies torchvision v2 transforms jointly to the image, boxes, instance masks and binary mask, then removes instances a crop has pushed out of frame.
- Keeps annotations keyed by image path, so the lookup survives anomalib's re-sorting and subsampling of `samples`.

### `CocoInstanceDataModule` (`coco_instance.py`)

An `AnomalibDataModule` over three explicit annotation files.

- `_create_test_split` and `_create_val_split` are overridden to do nothing. anomalib's defaults assume abnormal images are absent from training and, for some split modes, drop the normal test images.
- `test_parts` restricts only the test split. Validation stays pooled so the thresholds remain shared.
- Training augmentation (on by default) is ported from `train_tb.py`: shorter edge resized to 1024-1434 px, 1024x1024 random crop, horizontal and vertical flips.

## Runner (`main.py`)

Same config shape and helper names as `SuperDefectExperiments/main.py` (`instantiate`, `build_wandb_logger`, `build_engine`, `_save_per_image_metrics`, `log_summary_to_wandb`).

```bash
python main.py --config configs/experiments/maskrcnn_adam3d.yaml
python main.py --config configs/experiments/maskrcnn_adam3d.yaml --parts 1m1 tapa3m1
python main.py --config configs/experiments/maskrcnn_adam3d.yaml --ckpt-path results/maskrcnn_adam3d/weights/best.ckpt
```

| Flag | Effect |
|---|---|
| `--parts` | Test only these parts. Default: every part with annotated images in the test file. |
| `--ckpt-path` | Skip training; fit thresholds for this checkpoint and test it. |

Outputs under `<results_dir>/<experiment name>/`:

| File | Contents |
|---|---|
| `metrics.csv` | One row per part, `mean`, and `pooled` when enabled |
| `<part>/per_image_metrics.csv` | One row per test image: `image_path`, `gt_label`, `pred_label`, `pred_score`, `correct`, `pixel_iou`, `pixel_dice`, `pixel_ap` |
| `resolved_config.yaml` | The config as run |
| `weights/best.ckpt`, `weights/last.ckpt` | Checkpoints; thresholds are stored inside them |

anomalib's `Engine` also writes its own workspace under `<results_dir>/MaskRCNN/adam3d/unified/`, including the visualizer's images when the visualizer is enabled.

## Configs

| Config | Purpose |
|---|---|
| `maskrcnn_adam3d.yaml` | ResNet-50 FPN, COCO-pretrained, 20000 steps at batch 4 |
| `maskrcnn_x101_adam3d.yaml` | ResNeXt-101 64x4d FPN, ImageNet-pretrained backbone, 40000 steps at batch 2 |
| `maskrcnn_adam3d_smoke.yaml` | Two parts, 50 steps, no wandb; checks the pipeline on real data |

All three expect `data.init_args.root` to point at the supervised dataset layout and `good_root` at `3d-adam-full-masked`.
