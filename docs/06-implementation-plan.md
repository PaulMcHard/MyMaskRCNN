# Implementation plan and status

## Steps

| # | Step | Status |
|---|---|---|
| 1 | Planning docs in `docs/` | Done |
| 2 | Package skeleton (`pyproject.toml`) and API check against anomalib 2.5.0 | Done |
| 3 | Vendor `metrics.py`, `reporting.py`, `stats.py` and their tests | Done |
| 4 | Data: `add_good_images.py`, instance dataclasses, COCO dataset and datamodule | Done |
| 5 | Model: torchvision Mask R-CNN, anomaly-map mapping, Lightning module, post-processor | Done |
| 6 | Runner, configs, wandb logging | Done |
| 7 | Tests on a synthetic dataset | Done |
| 8 | Full training run on the training machine | **Not done** |
| 9 | Compare validation mask mAP with the earlier MMDetection runs | **Not done** (needs step 8) |
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

All of this ran on the development workstation, on CPU.

| Check | Result |
|---|---|
| `pytest -q` | 63 tests pass (27 vendored, 36 new) |
| Fit, validate, test, predict through anomalib `Engine` on synthetic data | Passes; all 16 evaluator metrics and 6 mAP values are produced |
| Checkpoint save, reload with `--ckpt-path`, thresholds restored | Passes |
| `SparseF1AdaptiveThreshold` equals anomalib's `F1AdaptiveThreshold` | Equal in 13 cases: sparse maps, dense maps, heavily tied scores, both missing-class fallbacks |
| `add_good_images.py` on the real data | 189 val and 191 test good images added; every part matched; no overlap between splits |
| Dataset on real images (27 defect images, 91 instances, parts `1m1` and `tapa3m1`) | Every instance matched to its PNG; boxes enclose masks before and after augmentation |
| `main.py` on real images at 1024x1280 with COCO-pretrained weights | 2 training steps, threshold fit and a per-part test complete in about 4 minutes; `metrics.csv` has all 22 columns |
| wandb logging, offline mode | Training run and summary run both written |

The real-image checks used a temporary copy of those two parts rebuilt from `3d-adam-full-masked`, because the supervised dataset root is not on this workstation.

## What has not been verified

- **No real training run.** Nothing here shows the model learns; the 2-step run only shows the pipeline executes. Metric values from it are meaningless.
- **GPU execution.** All runs were on CPU. The code has no device-specific branches, but GPU memory use at batch 4 and 1024x1024 crops is unmeasured.
- **The supervised dataset root.** Configs point `data.init_args.root` at `./datasets/adam3d_supervised`, a placeholder. It must be set to the directory the COCO `file_name` values are relative to, and that directory must contain the `ground_truth/` PNGs.
- **Online wandb.** See [05-wandb.md](05-wandb.md).
- **Parity with MMDetection.** torchvision's Mask R-CNN is a different implementation. Until step 9 is done, do not assume the new baseline is as strong as the old one.

## Next steps on the training machine

```bash
pip install -e .[dev,wandb]
pytest -q

# Set root and good_root in the configs first.
python main.py --config configs/experiments/maskrcnn_adam3d_smoke.yaml
wandb login
python main.py --config configs/experiments/maskrcnn_adam3d.yaml
```

Then compare `val/segm_mAP` with the earlier MMDetection runs. Expect some difference in either direction: the ground-truth masks changed from polygons to PNGs, and the default backbone changed from ResNeXt-101 to a COCO-pretrained ResNet-50. `maskrcnn_x101_adam3d.yaml` keeps the old backbone for a closer comparison.

## Open risks

| Risk | Mitigation |
|---|---|
| The port underperforms the MMDetection baseline | Compare `val/segm_mAP`; try the ResNeXt-101 config; tune `lr` and `trainable_backbone_layers` |
| GPU out of memory at batch 4 | Lower `train_batch_size` and raise `max_steps` in proportion; or set `trainer.precision: 16-mixed` |
| Validation every 500 steps is slow (378 images, two forward passes each when `log_val_loss` is on) | Raise `val_check_interval`, set `log_val_loss: false`, or set `trainer.limit_val_batches` |
| Pooled test row runs out of RAM | Leave `test_pooled: false`; the `mean` row is the headline |
| Score ties at zero distort pixel AUROC / AUPRO / AP relative to dense-map methods | Documented in [04-metrics.md](04-metrics.md); lower `box_score_thresh` for an ablation |
| Detector over-fires on good parts because it never trains on them | Rerun `add_good_images.py` with `--train-ratio` above 0 and point `train_ann_file` at `annotations_train_good.json` |
| Vendored metrics drift from `SuperDefectExperiments` | Re-copy the three files when the source changes; provenance is in each file header |

## Possible follow-ups

- Part-held-out split, to measure generalisation to unseen parts (`split.py` already supports it).
- Register the module in `SuperDefectExperiments` configs via `class_path: maskrcnn_modules.maskrcnn.MaskRCNN`, so one runner drives every method.
- Per-class pixel metrics with the vendored `MeanIoU` / `MacroDice`, using the predicted classes that the anomaly map currently discards.
