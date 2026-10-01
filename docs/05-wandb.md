# Weights & Biases tracking

## Where runs go

Runs are logged to the existing `SuperDefectExperiments` project so Mask R-CNN appears beside the other methods. The conventions match `SuperDefectExperiments/main.py`.

| Setting | Value |
|---|---|
| project | `SuperDefectExperiments` |
| entity | `null` (your wandb default) |
| group | experiment name, e.g. `maskrcnn_adam3d` |

Configured in the experiment YAML:

```yaml
logging:
  wandb:
    project: SuperDefectExperiments
    entity: null
    group: maskrcnn_adam3d
```

Remove the `logging.wandb` block to run without wandb.

## Runs per experiment

| Run name | `job_type` | Contents |
|---|---|---|
| `<group>-unified` | `train` | Training curves, validation curves, and `mean_<metric>` summary keys |
| `summary` | `summary` | A `metrics_summary` table (one row per part, plus `mean`, plus `pooled` when enabled) and `mean_<metric>` summary keys |

`unified` is the name the reference repo uses for a single model trained across all categories.

When `main.py` is given `--ckpt-path`, it skips training and the first run has `job_type` `eval`.

## What is logged

Training, every optimisation step:

| Key | Meaning |
|---|---|
| `train/loss` | Sum of the five loss terms |
| `train/loss_classifier`, `train/loss_box_reg`, `train/loss_mask` | RoI head losses |
| `train/loss_objectness`, `train/loss_rpn_box_reg` | RPN losses |
| `lr-SGD` | Learning rate, from `LearningRateMonitor` |

Validation, every `val_check_interval` steps:

| Key | Meaning |
|---|---|
| `val/bbox_mAP`, `val/segm_mAP` and `_50` / `_75` variants | COCO mAP on the validation split |
| `val/loss` and per-term `val/loss_*` | Loss on the validation split (when `log_val_loss: true`) |

Test, once per part:

| Key | Meaning |
|---|---|
| `image_*`, `pixel_*` | The 16 evaluator metrics |
| `bbox_mAP`, `segm_mAP` and `_50` / `_75` variants | COCO mAP on that part's test images |

Each part logs the same keys into the training run, so those charts show one point per part in test order. Read per-part results from the `metrics_summary` table in the `summary` run, or from `metrics.csv`.

The x-axis is Lightning's global step, so curves line up with `max_steps` and `val_check_interval` in the config. This avoids the step problem of mmengine's `WandbVisBackend`, which logs without a step.

The full resolved config is stored as the run config, including `supervision_regime: fully_supervised`.

## Checkpoints

Checkpoints are saved to `<results_dir>/<experiment>/weights/` by `ModelCheckpoint`: `best.ckpt` (highest `val/segm_mAP`) and `last.ckpt`. They are not uploaded as wandb artifacts by default, matching the reference repo. Set `log_model: true` under `logging.wandb` to upload them.

## Offline and CI use

```bash
WANDB_MODE=offline python main.py --config configs/experiments/maskrcnn_adam3d_smoke.yaml
wandb sync wandb/offline-run-*        # upload later
```

The smoke config ships without a `logging.wandb` block, and the test suite never touches wandb.

Only offline mode has been exercised so far: a run on synthetic data produced both the training run and the summary run on disk. Uploading to the `SuperDefectExperiments` project needs `wandb login` on the training machine and has not been tried.

## Relation to TensorBoard

The MMDetection scripts logged to TensorBoard. The new pipeline logs to wandb only. wandb keeps a local copy of each run under `<results_dir>/wandb/`, and `metrics.csv` is always written to `<results_dir>/<experiment>/` regardless of the logger.
