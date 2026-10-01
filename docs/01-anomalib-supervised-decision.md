# Does anomalib support supervised models?

## Short answer

Anomalib is built around one-class learning, but that is a convention of its datasets and model zoo. The framework itself does not prevent a supervised model. A supervised Mask R-CNN can be wrapped as an anomalib module if we supply two things anomalib does not have: a batch type that carries instance annotations, and a datamodule that puts defect images in the training set.

## What is fixed in anomalib

- **Learning types.** `anomalib.LearningType` has three members: `ONE_CLASS`, `ZERO_SHOT` and `FEW_SHOT`. There is no supervised member.
- **Data splits.** The stock datamodules (`Folder`, `MVTec`, `Visa`, ...) put normal images in train and abnormal images in val/test.
- **Batch schema.** `ImageBatch` carries `image`, `gt_label`, a binary `gt_mask`, and the prediction fields. It has no boxes, instance masks or class labels.
- **Stated scope.** The SuperSimpleNet model docs say: "This implementation supports both unsupervised and supervised setting, but Anomalib currently supports only unsupervised learning."

## What is open

- An `AnomalibModule` is a `LightningModule` with four optional attachments: a pre-processor, a post-processor, an evaluator and a visualizer. `training_step` is free to use `gt_mask` or any other field.
- `SuperDefectExperiments` already relies on this. Its `SuperDefect` wrapper trains a pixel-classification loss against `gt_mask` and declares `LearningType.ONE_CLASS`.
- SuperSimpleNet in anomalib has a `supervised` flag and also returns `ONE_CLASS`, noting this "is subject to change in the future when support for supervised training is introduced."
- Metrics are duck-typed. `AnomalibMetric.update()` reads its fields with `getattr`, so any object with the right attributes can be scored.

`learning_type` only affects two things: `Engine.fit()` skips training for zero/few-shot models, and the default post-processor is only provided for `ONE_CLASS`. Declaring `ONE_CLASS` for a supervised model is therefore harmless and gives us the standard post-processor.

## Options considered

| Option | What it is | Why chosen or not |
|---|---|---|
| A. Metrics bridge | Keep MMDetection training. Convert predictions to anomaly fields in a separate eval step and score them with the shared evaluator. | Smallest change and keeps existing checkpoints. Not chosen: the model stays outside the shared `Engine` / config / logger pipeline. |
| B. Anomalib wrapper around the MMDetection model | Wrap the mmdet detector in an `AnomalibModule`. | Not chosen. Requires porting the mmengine training loop and co-installing `mmcv<2.2.0` with anomalib's newer torch/lightning requirements, which likely means building mmcv from source. |
| **C. Anomalib wrapper around torchvision Mask R-CNN** | Replace mmdet with `torchvision.models.detection.MaskRCNN` inside an `AnomalibModule`. | **Chosen.** No mmcv dependency, installs into the existing `sdx` environment, and runs through the same `Engine`, evaluator and wandb logger as the other methods. |

## What the chosen route costs

- **A different implementation.** torchvision's Mask R-CNN is not the MMDetection one. Existing checkpoints and results are not carried over; the baseline must be retrained.
- **A sanity check is needed.** After the first full run, validation mask mAP should be compared with the earlier MMDetection numbers. A large drop means the port needs tuning before its metrics are used.
- **Custom data plumbing.** Instance annotations need a batch dataclass and datamodule of our own (see [02-architecture.md](02-architecture.md)).

## What the wrapper gives us

- The 16-metric suite from the shared evaluator, computed by the same code path as every other method.
- Thresholds chosen the same way as the other methods: anomalib's `PostProcessor` fits F1-adaptive image and pixel thresholds on the validation split.
- wandb logging through Lightning's `WandbLogger` with no custom hook.
- A model that can be referenced by `class_path` from a YAML experiment config, in the same style as `SuperDefectExperiments`.
