# Metrics

## The suite

The evaluator is `maskrcnn_modules.shared.metrics.configure_evaluator()`, vendored unchanged from `SuperDefectExperiments`. It reports 16 metrics at test time.

| Metric | Level | Reads | Threshold |
|---|---|---|---|
| `image_AUROC` | image | `pred_score`, `gt_label` | none |
| `image_F1Score` | image | `pred_label`, `gt_label` | operating point |
| `image_F1Max` | image | `pred_score`, `gt_label` | best over all |
| `image_FalseAlarmRate` | image | `pred_label`, `gt_label` | operating point |
| `pixel_AUROC` | pixel | `anomaly_map`, `gt_mask` | none |
| `pixel_F1Score` | pixel | `pred_mask`, `gt_mask` | operating point |
| `pixel_F1Max` | pixel | `anomaly_map`, `gt_mask` | best over all |
| `pixel_AUPRO` | pixel | `anomaly_map`, `gt_mask` | none (FPR up to 0.3) |
| `pixel_AP` | pixel | `anomaly_map`, `gt_mask` | none |
| `pixel_IoU` | pixel | `pred_mask`, `gt_mask` | operating point |
| `pixel_Dice` | pixel | `pred_mask`, `gt_mask` | operating point |
| `pixel_FalseAlarmRate` | pixel | `pred_mask`, `gt_mask` | operating point |
| `pixel_BFScore1/2/3/5` | pixel | `pred_mask`, `gt_mask` | operating point, boundary tolerance 1/2/3/5 px |

Definitions and interpretation are in `SuperDefectExperiments/docs/metrics_explainer.md`.

Detection metrics are reported alongside, computed in the Lightning module with torchmetrics `MeanAveragePrecision`:

| Metric | Meaning |
|---|---|
| `bbox_mAP`, `bbox_mAP_50`, `bbox_mAP_75` | COCO box mAP, averaged over IoU 0.50:0.95, at 0.50, at 0.75 |
| `segm_mAP`, `segm_mAP_50`, `segm_mAP_75` | COCO mask mAP, same thresholds |

These keep continuity with the earlier MMDetection runs, which reported `coco/bbox_mAP` and `coco/segm_mAP`. They are not identical to those numbers: the ground-truth instance masks now come from the original PNGs, not from the COCO polygons (see [03-data-protocol.md](03-data-protocol.md)).

## Where each input comes from

Mask R-CNN returns, per image, a set of detections with a box, a class, a confidence score and a soft mask in [0, 1] at image resolution.

| Field | Construction |
|---|---|
| `anomaly_map` | For each pixel, the maximum over detections of `score x soft_mask`. Zero where nothing is detected. Class is ignored. |
| `pred_score` | Maximum of `anomaly_map`, which equals the strongest detection's peak. Zero when there are no detections. |
| `pred_mask` | `anomaly_map` above the pixel threshold (set by the post-processor). |
| `pred_label` | `pred_score` above the image threshold (set by the post-processor). |
| `gt_mask` | Union of ground-truth defect masks (see [03-data-protocol.md](03-data-protocol.md)). |
| `gt_label` | 1 for defect images, 0 for good images. |

## Operating point

The thresholds are not hand-picked. anomalib's `PostProcessor` fits them on the validation split: the image threshold maximises image F1 on (`pred_score`, `gt_label`), the pixel threshold maximises pixel F1 on (`anomaly_map`, `gt_mask`). The same procedure is used for every other method in `SuperDefectExperiments`.

One model and one pair of thresholds serve every part. Thresholds are fitted on the pooled validation split, then each part is tested separately.

The post-processor also min-max normalises scores around the threshold, then clamps to [0, 1]. Thresholded metrics are unaffected. Ranking metrics can lose a little resolution at the top: scores far enough above the threshold all clamp to 1, and a detector's confident scores sit close together near 1. This is the stock behaviour for every method. To switch it off for an ablation:

```yaml
model:
  init_args:
    post_processor:
      class_path: maskrcnn_modules.maskrcnn.DetectionPostProcessor
      init_args:
        enable_normalization: false
```

Thresholds must be fitted with the same weights that are tested. The runner calls `validate` on the best checkpoint before `test`, and the module refuses to test when no threshold exists.

### The one deviation from stock: how the pixel threshold is stored

anomalib's pixel threshold metric keeps every validation pixel and sorts them. Measured here, that costs about 1 GB of RAM per 20 images at 1024x1280, so roughly 19 GB for the 378 pooled validation images, on every validation loop.

`DetectionPostProcessor` replaces only that metric with `SparseF1AdaptiveThreshold`. A detector's anomaly map is exactly zero outside its detections, so the metric stores non-zero-score pixels individually and counts the rest. The precision-recall curve, F1 formula, tie-breaking and missing-class fallbacks are the same. `tests/test_post_processing.py` checks the threshold is equal to anomalib's on sparse maps, dense maps, heavily tied scores, and both fallback cases.

## Rows in `metrics.csv`

| Row | Meaning |
|---|---|
| one per part | The shared model and thresholds, tested on that part's defect and good images |
| `mean` | Mean of the per-part rows. This is the headline, as in `SuperDefectExperiments`. |
| `pooled` | All test images scored in one pass. Off by default (`experiment.test_pooled`). |

The `pooled` row is off by default because the four pixel curve metrics (AUROC, F1Max, AUPRO, AP) are the stock anomalib ones and hold every test pixel in memory. For the 382 test images that is roughly 30 GB of RAM (about 19 GB to sort one metric plus about 2 GB of stored pixels for each of the four), extrapolated from the 20-image measurement above. A per-part test holds about 18 images and needs about 1 GB.

## Provenance of the vendored code

| File here | Source | Commit |
|---|---|---|
| `src/maskrcnn_modules/shared/metrics.py` | `SuperDefectExperiments/src/anomaly_modules/shared/metrics.py` | `11c1c57` |
| `src/maskrcnn_modules/shared/reporting.py` | `.../shared/reporting.py` | `11c1c57` |
| `src/maskrcnn_modules/shared/stats.py` | `.../shared/stats.py` | `11c1c57` |

The only edits are the import path inside `reporting.py` and a provenance header on each file. The matching tests are copied too. When the source changes, re-copy rather than editing in place, so the two repos do not drift.

The vendored module also contains `MacroDice`, `MeanIoU`, `InstanceIoU` and `InstanceDice`. They return dicts, so they are not part of the evaluator. They are available for offline analysis.

## Caveats when reading the numbers

- **Ties at zero.** A detector scores most pixels exactly 0. Anomaly-map methods give every pixel a distinct score. Pixel AUROC, AUPRO and AP for Mask R-CNN therefore have a large tie mass at the low end, and the curves are only informative above the detection score floor (`box_score_thresh`, default 0.05). Lowering the floor and raising `box_detections_per_img` densifies the map at the cost of memory.
- **Not image-paired.** The test images differ from the anomalib per-part test sets. Compare as unpaired results.
- **Per-part rows are noisy.** About 9 defect and 9 good test images per part. Use the `mean` row as the headline.
- **Seen parts.** Every part is in the training set.
- **AUPRO cost.** AUPRO runs connected components on CPU over every test image at 1280x1024. Expect it to dominate test time.
- **`scratch` has no instances.** It contributes nothing to mAP.
- **Class-agnostic.** The anomaly fields ignore the predicted defect class. Per-class quality is only visible in mAP.
