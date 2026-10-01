# Planning docs: anomalib integration

These docs plan two changes to this repository:

1. Evaluate Mask R-CNN with the same metric suite that `SuperDefectExperiments` reports.
2. Track training and evaluation in Weights & Biases.

Both are delivered by wrapping a torchvision Mask R-CNN as an anomalib module, so the model trains and tests through the same `Engine`, evaluator and logger as the other methods.

| Doc | Contents |
|---|---|
| [01-anomalib-supervised-decision.md](01-anomalib-supervised-decision.md) | Whether anomalib supports supervised models, the options considered, and the route chosen |
| [02-architecture.md](02-architecture.md) | Package layout, the Lightning module, batch dataclasses, datamodule, and how detections become anomaly outputs |
| [03-data-protocol.md](03-data-protocol.md) | Splits, adding good images, unannotated images, ground-truth mask source |
| [04-metrics.md](04-metrics.md) | The metric suite, where each input comes from, vendoring provenance, and caveats when reading the numbers |
| [05-wandb.md](05-wandb.md) | Run structure, logged keys, and how to run offline |
| [06-implementation-plan.md](06-implementation-plan.md) | Build steps, verification, status, and open risks |

The MMDetection scripts at the repository root (`train.py`, `train_tb.py`, `validate.py`, `test.py`, `verification.py`) are the earlier baseline. They are left in place and are not used by the new pipeline.
