# Implementation plan and status

## Steps

| # | Step | Status |
|---|---|---|
| 1 | Planning docs in `docs/` | Done |
| 2 | Package skeleton (`pyproject.toml`) and API check against anomalib 2.5.0 | Done |
| 3 | Vendor `metrics.py`, `reporting.py`, `stats.py` and their tests | Done |
| 4 | Data: instance dataclasses, COCO dataset and datamodule | Done |
| 4b | Rebuild the annotations from `3d-adam-full-masked` on SuperDefect's `default.yaml` split (`build_coco_annotations.py`) | Done |
| 5 | Model: torchvision Mask R-CNN, anomaly-map mapping, Lightning module, post-processor | Done |
| 6 | Runner, configs, wandb logging | Done |
| 7 | Tests on a synthetic dataset | Done |
| 8 | Full training run on the training machine | **Not done** |
| 9 | Parity check against MMDetection retrained on the new annotations | **Not done** (needs step 8) |
| 10 | Online wandb upload to `SuperDefectExperiments` | **Not done** (needs `wandb login`) |

## API check against anomalib 2.5.0

The design was first read from anomalib's `main` branch. Each assumption was then checked against the installed 2.5.0:

| Assumption | Result |
|---|---|
| `ValSplitMode.FROM_DIR` and `TestSplitMode.FROM_DIR` exist | Yes |
| `ImageItem` / `ImageBatch` can be subclassed with extra dataclass fields | Yes; `update()`, `items` and device transfer all work |
| `BatchIterateMixin.collate` stacks every non-None field | Yes, hence the custom collate |
| `PostProcessor` reads the step's return value | Yes |
| `Evaluator` reads the batch passed to the hook, not the return value | Yes, hence the in-place update |
| `Engine.test` fits thresholds automatically | **No.** Only for zero/few-shot models. An explicit `validate` is required, as `SuperDefectExperiments/main.py` also does |
| `model.trainer_arguments` can be overridden from YAML | **No.** The model's values win, so `gradient_clip_val` is a model argument |
| `AnomalibDataModule._create_test_split` keeps normal test images | **Not always.** With `TestSplitMode.NONE` they are dropped, so the split hooks are overridden |

## What has been verified

All of this ran on the development workstation, on CPU unless stated.

| Check | Result |
|---|---|
| `pytest -q` | 75 tests pass (27 vendored, 48 new) |
| Fit, validate, test, predict through anomalib `Engine` on synthetic data | Passes; all 16 evaluator metrics and 6 mAP values are produced |
| Checkpoint save, reload with `--ckpt-path`, thresholds restored | Passes |
| `SparseF1AdaptiveThreshold` equals anomalib's `F1AdaptiveThreshold` | Equal in 13 cases: sparse maps, dense maps, heavily tied scores, both missing-class fallbacks |
| `build_coco_annotations.py` on the real data | Defect images per split equal SuperDefect's `default.yaml` split minus 36 mask-less images; all 6850 instances decode to their PNG exactly with the label from the mask name; no specimen folder spans two splits; good images per part equal defect images per part in val and test |
| `main.py` with the smoke config on the new annotation files (cut to 2 steps, part `1M1`) | Training, threshold fit and the per-part test complete in 13 minutes on CPU; `metrics.csv` has all 22 columns with no NaN; `per_image_metrics.csv` has 80 rows (40 defect, 40 good) |
| Smoke config on the Quadro P2000 (5 GB) | Completes in 7.8 minutes; training runs at 0.61 s per image, so the training loop is GPU-bound. Batch sizes above 2 do not fit in the free GPU memory |
| wandb logging, offline mode | Training run and summary run both written |

The real-image checks read `3d-adam-full-masked` directly from this workstation.

## What has not been verified

- **No real training run.** Nothing here shows the model learns; the 2-step run only shows the pipeline executes. Metric values from it are meaningless.
- **GPUs other than the P2000.** Batch 4 at 1024x1024 crops needs about 6.7 GB; it has not been run on a larger card.
- **Dataset path on the training machine.** Configs point `data.init_args.root` at `D:/Data/3d-adam-full-masked`. Change it if the dataset lives elsewhere.
- **Online wandb.** See [05-wandb.md](05-wandb.md).
- **Parity with MMDetection.** torchvision's Mask R-CNN is a different implementation. Until step 9 is done, do not assume the new baseline is as strong as the old one.

## Next steps on the training machine

```bash
pip install -e .[dev,wandb]
pytest -q

# Set data.init_args.root in the configs first, if 3d-adam-full-masked is not at D:/Data.
python main.py --config configs/experiments/maskrcnn_adam3d_smoke.yaml
wandb login
python main.py --config configs/experiments/maskrcnn_adam3d.yaml
```

The earlier MMDetection mAP is no longer a like-for-like reference: the labels, masks, class set and split have all changed, and the old split leaked. Expect lower numbers than before. A fair parity check would retrain the MMDetection config on the new annotation files; `maskrcnn_x101_adam3d.yaml` keeps the old backbone for that comparison.

## Open risks

| Risk | Mitigation |
|---|---|
| The port underperforms MMDetection | Retrain the MMDetection config on the new annotation files and compare; try the ResNeXt-101 config; tune `lr` and `trainable_backbone_layers` |
| GPU out of memory at batch 4 | Lower `train_batch_size` and raise `max_steps` in proportion; or set `trainer.precision: 16-mixed` |
| Validation every 500 steps is slow (336 images, two forward passes each when `log_val_loss` is on) | Raise `val_check_interval`, set `log_val_loss: false`, or set `trainer.limit_val_batches` |
| Pooled test row runs out of RAM | Leave `test_pooled: false`; the `mean` row is the headline |
| Score ties at zero distort pixel AUROC / AUPRO / AP relative to dense-map methods | Documented in [04-metrics.md](04-metrics.md); lower `box_score_thresh` for an ablation |
| Detector over-fires on good parts because it never trains on them | Rebuild the annotations with `--train-good-ratio` above 0 |
| Rare classes are absent from evaluation (no gap, scratch or warp in test) | Inherent to the specimen split; report per-class AP only for classes present |
| SuperDefect's ground truth differs from ours (its `ignore_classes` and its extrusion regex bug) | See [03-data-protocol.md](03-data-protocol.md); fix the regex and align `ignore_classes` before pairing |
| Vendored metrics drift from `SuperDefectExperiments` | Re-copy the three files when the source changes; provenance is in each file header |

## Possible follow-ups

- Part-held-out split, to measure generalisation to unseen parts. Its test set would not pair with SuperDefect's.
- Register the module in `SuperDefectExperiments` configs via `class_path: maskrcnn_modules.maskrcnn.MaskRCNN`, so one runner drives every method.
- Per-class pixel metrics with the vendored `MeanIoU` / `MacroDice`, using the predicted classes that the anomaly map currently discards.
