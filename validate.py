import os
import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Register custom dataset
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset

@DATASETS.register_module()
class Adam3DDefectDataset(CocoDataset):
    """Custom dataset for 3D-ADAM defect detection."""
    
    METAINFO = {
        'classes': ('bulge', 'crack', 'hole', 'under_extrusion', 
                   'over_extrusion', 'cut', 'scratch'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                   (106, 0, 228), (0, 60, 100), (0, 80, 100)]
    }

def clean_config_for_validation(cfg):
    """Remove training-specific hooks and settings from config."""
    # Remove custom hooks that are only needed for training
    if hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = []
    
    # Simplify default hooks for validation
    if hasattr(cfg, 'default_hooks'):
        # Keep only essential hooks for validation
        essential_hooks = ['timer', 'logger', 'param_scheduler', 'sampler_seed']
        cfg.default_hooks = {
            k: v for k, v in cfg.default_hooks.items() 
            if k in essential_hooks
        }
    
    return cfg

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
import mmcv


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_predictions_to_ground_truth(pred_bboxes, pred_labels, pred_scores, 
                                     gt_bboxes, gt_labels, iou_threshold=0.5):
    """
    Match predictions to ground truth instances using Hungarian matching.
    
    Returns:
        matched_gt: List of matched GT indices for each prediction
        matched_pred: List of matched prediction indices for each GT
        tp_count: Number of true positives
        fp_count: Number of false positives  
        fn_count: Number of false negatives
    """
    num_preds = len(pred_bboxes)
    num_gts = len(gt_bboxes)
    
    if num_preds == 0 and num_gts == 0:
        return [], [], 0, 0, 0
    
    if num_preds == 0:
        return [], [], 0, 0, num_gts
    
    if num_gts == 0:
        return [None] * num_preds, [], 0, num_preds, 0
    
    # Compute IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    for i, pred_bbox in enumerate(pred_bboxes):
        for j, gt_bbox in enumerate(gt_bboxes):
            if pred_labels[i] == gt_labels[j]:  # Same class
                iou_matrix[i, j] = compute_iou(pred_bbox, gt_bbox)
    
    # Greedy matching: sort by score and match each prediction
    matched_gt = [None] * num_preds
    matched_pred = [None] * num_gts
    used_gt = set()
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_scores)
    
    for pred_idx in sorted_indices:
        best_iou = iou_threshold
        best_gt_idx = None
        
        for gt_idx in range(num_gts):
            if gt_idx not in used_gt and iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt_idx = gt_idx
        
        if best_gt_idx is not None:
            matched_gt[pred_idx] = best_gt_idx
            matched_pred[best_gt_idx] = pred_idx
            used_gt.add(best_gt_idx)
    
    # Count TP, FP, FN
    tp_count = len(used_gt)
    fp_count = num_preds - tp_count
    fn_count = num_gts - tp_count
    
    return matched_gt, matched_pred, tp_count, fp_count, fn_count


def visualize_predictions(image_path, predictions, ground_truth, output_path, class_names, conf_threshold=0.5):
    """Create side-by-side visualization of predictions and ground truth."""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot ground truth
    ax_gt = axes[0]
    ax_gt.imshow(image)
    ax_gt.set_title('Ground Truth', fontsize=16, fontweight='bold')
    ax_gt.axis('off')
    
    if ground_truth is not None:
        for ann in ground_truth.get('annotations', []):
            # Draw bbox
            bbox = ann['bbox']
            # Convert from [x, y, w, h] to rectangle
            rect = mpatches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax_gt.add_patch(rect)
            
            # Add label
            cat_id = ann['category_id'] - 1
            label = class_names[cat_id] if cat_id < len(class_names) else 'unknown'
            ax_gt.text(bbox[0], bbox[1] - 5, label,
                      bbox=dict(facecolor='yellow', alpha=0.7),
                      fontsize=10, color='black')
    
    # Plot predictions
    ax_pred = axes[1]
    ax_pred.imshow(image)
    ax_pred.set_title('Predictions', fontsize=16, fontweight='bold')
    ax_pred.axis('off')
    
    if predictions is not None and hasattr(predictions, 'pred_instances'):
        pred_instances = predictions.pred_instances
        
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        
        valid_idx = scores >= conf_threshold
        scores = scores[valid_idx]
        labels = labels[valid_idx]
        bboxes = bboxes[valid_idx]
        
        # Draw predictions
        for i, (label, score, bbox) in enumerate(zip(labels, scores, bboxes)):
            color = plt.cm.Set3(label / len(class_names))
            
            # Draw bbox
            rect = mpatches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax_pred.add_patch(rect)
            
            # Add label with confidence
            class_name = class_names[label] if label < len(class_names) else 'unknown'
            text = f'{class_name}: {score:.2f}'
            ax_pred.text(bbox[0], bbox[1] - 5, text,
                        bbox=dict(facecolor=color, alpha=0.7),
                        fontsize=10, color='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def validate_model(config_path, checkpoint_path, split='val', output_dir='validation_results',
                   visualize=False, vis_samples=10, conf_threshold=0.5, iou_threshold=0.5):
    """
    Validate trained Mask R-CNN model with instance-level tracking.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        split: Dataset split to evaluate ('train', 'val', or 'test')
        output_dir: Directory to save results
        visualize: Whether to generate visualizations
        vis_samples: Number of samples to visualize
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to ground truth
    """
    
    print("=" * 80)
    print("Mask R-CNN Model Validation with Instance-Level Tracking")
    print("=" * 80)
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Load config
    print(f"\nLoading config from {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Clean config for validation
    cfg = clean_config_for_validation(cfg)

    # Update config for validation
    cfg.work_dir = str(output_dir)
    
    # Build model
    print(f"Loading model from {checkpoint_path}")
    model = init_detector(cfg, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Get class names
    class_names = model.dataset_meta['classes']
    print(f"\nClasses: {class_names}")
    
    # Load annotations for the split
    ann_file = None
    if split == 'train':
        ann_file = Path(cfg.train_dataloader.dataset.ann_file)
    elif split == 'val':
        ann_file = Path(cfg.val_dataloader.dataset.ann_file)
    elif split == 'test':
        ann_file = Path(cfg.test_dataloader.dataset.ann_file)
    
    print(f"\nLoading {split} annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Number of images: {len(coco_data['images'])}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")
    
    # Group annotations by image
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Initialize tracking
    print("\n" + "-" * 80)
    print("Running inference with instance-level tracking...")
    print("-" * 80)
    
    all_predictions = []
    instance_tracking = defaultdict(lambda: {
        'total_instances': 0,
        'detected_instances': 0,
        'missed_instances': 0,
        'false_detections': 0
    })
    
    per_image_results = []
    
    for idx, img_info in enumerate(tqdm(coco_data['images'], desc="Processing images")):
        img_path = Path(cfg.val_dataloader.dataset.data_root) / img_info['file_name']
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Run inference
        result = inference_detector(model, str(img_path))
        
        # Get predictions
        pred_instances = result.pred_instances
        
        # Filter by confidence
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        
        valid_idx = scores >= conf_threshold
        pred_scores = scores[valid_idx]
        pred_labels = labels[valid_idx]
        pred_bboxes = bboxes[valid_idx]
        
        # Get ground truth
        gt_anns = image_annotations[img_info['id']]
        gt_labels = np.array([ann['category_id'] - 1 for ann in gt_anns])
        gt_bboxes = []
        for ann in gt_anns:
            bbox = ann['bbox']  # [x, y, w, h]
            gt_bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # Convert to [x1, y1, x2, y2]
        gt_bboxes = np.array(gt_bboxes) if gt_bboxes else np.zeros((0, 4))
        
        # Match predictions to ground truth
        matched_gt, matched_pred, tp, fp, fn = match_predictions_to_ground_truth(
            pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_threshold
        )
        
        # Track per-class instance-level metrics
        for cat_id in range(len(class_names)):
            cat_name = class_names[cat_id]
            
            # Count GT instances of this class
            gt_count = int((gt_labels == cat_id).sum())
            instance_tracking[cat_name]['total_instances'] += gt_count
            
            # Count detected instances (TP for this class)
            detected = 0
            for i, label in enumerate(pred_labels):
                if label == cat_id and matched_gt[i] is not None:
                    detected += 1
            
            instance_tracking[cat_name]['detected_instances'] += detected
            
            # Count missed instances (FN for this class)
            missed = gt_count - detected
            instance_tracking[cat_name]['missed_instances'] += missed
            
            # Count false detections (FP for this class)
            false_dets = 0
            for i, label in enumerate(pred_labels):
                if label == cat_id and matched_gt[i] is None:
                    false_dets += 1
            instance_tracking[cat_name]['false_detections'] += false_dets
        
        # Store per-image results (convert numpy types)
        img_result = {
            'image_id': int(img_info['id']),
            'file_name': str(img_info['file_name']),
            'model_id': str(img_info.get('model_id', 'unknown')),
            'num_gt_instances': len(gt_anns),
            'num_predictions': int(len(pred_scores)),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'detection_rate': float(tp / len(gt_anns)) if len(gt_anns) > 0 else None,
            'precision': float(tp / len(pred_scores)) if len(pred_scores) > 0 else None
        }
        per_image_results.append(img_result)
        
        # Store predictions (convert numpy types)
        img_predictions = {
            'image_id': int(img_info['id']),
            'file_name': str(img_info['file_name']),
            'predictions': {
                'labels': [int(x) for x in pred_labels],
                'scores': [float(x) for x in pred_scores],
                'bboxes': [[float(c) for c in bbox] for bbox in pred_bboxes],
                'num_detections': int(len(pred_scores))
            },
            'ground_truth': {
                'labels': [int(x) for x in gt_labels],
                'bboxes': [[float(c) for c in bbox] for bbox in gt_bboxes],
                'num_instances': len(gt_anns)
            },
            'matching': {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }
        }
        all_predictions.append(img_predictions)
        
        # Visualize sample predictions
        if visualize and idx < vis_samples:
            vis_path = vis_dir / f"{Path(img_info['file_name']).stem}_comparison.png"
            gt_data = {
                'annotations': gt_anns,
                'categories': coco_data['categories']
            }
            visualize_predictions(
                img_path, result, gt_data, vis_path, 
                class_names, conf_threshold
            )
    
    # Compute final instance-level metrics
    print("\n" + "=" * 80)
    print("INSTANCE-LEVEL DETECTION METRICS")
    print("=" * 80)
    
    per_class_metrics = {}
    
    for cat_name, tracking in instance_tracking.items():
        total = tracking['total_instances']
        detected = tracking['detected_instances']
        missed = tracking['missed_instances']
        false_dets = tracking['false_detections']
        
        detection_rate = float(detected / total) if total > 0 else 0.0
        precision = float(detected / (detected + false_dets)) if (detected + false_dets) > 0 else 0.0
        recall = detection_rate  # Same as detection rate
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[cat_name] = {
            'total_instances': int(total),
            'detected_instances': int(detected),
            'missed_instances': int(missed),
            'false_detections': int(false_dets),
            'detection_rate': detection_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"\n{cat_name}:")
        print(f"  Total GT Instances: {total}")
        print(f"  Detected Instances: {detected}")
        print(f"  Missed Instances: {missed}")
        print(f"  False Detections: {false_dets}")
        print(f"  *** DETECTION RATE: {detection_rate:.2%} ***")
        print(f"      (Found {detected} out of {total} instances)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Compute overall metrics
    total_instances = sum(t['total_instances'] for t in instance_tracking.values())
    total_detected = sum(t['detected_instances'] for t in instance_tracking.values())
    total_missed = sum(t['missed_instances'] for t in instance_tracking.values())
    total_false = sum(t['false_detections'] for t in instance_tracking.values())
    
    overall_detection_rate = float(total_detected / total_instances) if total_instances > 0 else 0.0
    overall_precision = float(total_detected / (total_detected + total_false)) if (total_detected + total_false) > 0 else 0.0
    overall_recall = overall_detection_rate
    overall_f1 = float(2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("OVERALL INSTANCE-LEVEL PERFORMANCE")
    print("=" * 80)
    print(f"Total GT Instances: {total_instances}")
    print(f"Detected Instances: {total_detected}")
    print(f"Missed Instances: {total_missed}")
    print(f"False Detections: {total_false}")
    print(f"\n*** OVERALL DETECTION RATE: {overall_detection_rate:.2%} ***")
    print(f"    (Successfully found {total_detected} out of {total_instances} defect instances)")
    print(f"\nPrecision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    
    # Run official COCO evaluation
    print("\n" + "=" * 80)
    print("Running Official COCO Evaluation")
    print("=" * 80)
    
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    
    print("\nCOCO Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save all results with proper type conversion
    results = {
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'split': split,
        'confidence_threshold': float(conf_threshold),
        'iou_threshold': float(iou_threshold),
        'overall_instance_metrics': {
            'total_gt_instances': int(total_instances),
            'detected_instances': int(total_detected),
            'missed_instances': int(total_missed),
            'false_detections': int(total_false),
            'detection_rate': overall_detection_rate,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1
        },
        'per_class_instance_metrics': per_class_metrics,
        'coco_metrics': {k: float(v) if v != -1.0 else -1.0 for k, v in metrics.items()},
        'per_image_results': per_image_results,
        'detailed_predictions': all_predictions
    }
    
    # Convert all numpy types to native Python types
    results = convert_to_json_serializable(results)
    
    results_file = output_dir / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    summary_file = output_dir / 'summary_report.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("INSTANCE-LEVEL DETECTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset Split: {split}\n")
        f.write(f"Confidence Threshold: {conf_threshold}\n")
        f.write(f"IoU Threshold: {iou_threshold}\n\n")
        
        f.write("OVERALL RESULTS:\n")
        f.write(f"  Total defect instances in {split} set: {total_instances}\n")
        f.write(f"  Successfully detected: {total_detected}\n")
        f.write(f"  Missed: {total_missed}\n")
        f.write(f"  False detections: {total_false}\n")
        f.write(f"  DETECTION RATE: {overall_detection_rate:.2%}\n\n")
        
        f.write("PER-CLASS DETECTION RATES:\n")
        for cat_name, metrics in sorted(per_class_metrics.items()):
            f.write(f"\n  {cat_name}:\n")
            f.write(f"    Instances: {metrics['total_instances']}\n")
            f.write(f"    Detected: {metrics['detected_instances']}\n")
            f.write(f"    Detection Rate: {metrics['detection_rate']:.2%}\n")
    
    print(f"\n" + "=" * 80)
    print(f"Results saved to: {results_file}")
    print(f"Summary report: {summary_file}")
    if visualize:
        print(f"Visualizations saved to: {vis_dir}")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate trained Mask R-CNN model with instance-level tracking')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--vis-samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching predictions to GT')
    
    args = parser.parse_args()
    
    validate_model(
        args.config,
        args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        visualize=args.visualize,
        vis_samples=args.vis_samples,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )


if __name__ == '__main__':
    main()