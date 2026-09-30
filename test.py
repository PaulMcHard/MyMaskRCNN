import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import torch

# Register custom dataset BEFORE importing MMDet APIs
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

# Now import MMDet APIs
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained MaskRCNN')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_output',
                        help='Output directory for results')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Build model
    print("Loading model...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get class names
    classes = model.dataset_meta['classes']
    print(f"Classes: {classes}")
    
    # Process image(s)
    image_path = Path(args.image)
    
    if image_path.is_file():
        image_files = [image_path]
    elif image_path.is_dir():
        image_files = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg'))
    else:
        print(f"Error: {image_path} is not a valid file or directory")
        return
    
    print(f"\nProcessing {len(image_files)} images...")
    
    # Initialize visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")
        
        # Run inference
        result = inference_detector(model, str(img_file))
        
        # Get predictions
        pred_instances = result.pred_instances
        
        # Filter by score
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        
        # Check if masks exist
        if hasattr(pred_instances, 'masks'):
            masks = pred_instances.masks.cpu().numpy()
        else:
            masks = None
        
        valid_idx = scores >= args.score_threshold
        scores = scores[valid_idx]
        labels = labels[valid_idx]
        bboxes = bboxes[valid_idx]
        if masks is not None:
            masks = masks[valid_idx]
        
        num_detections = len(scores)
        print(f"  Detected {num_detections} defects")
        
        # Print detection details
        for i, (label, score) in enumerate(zip(labels, scores)):
            class_name = classes[label]
            print(f"    {i+1}. {class_name} (confidence: {score:.3f})")
        
        # Visualize
        img = mmcv.imread(str(img_file))
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        visualizer.add_datasample(
            img_file.stem,
            img,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            out_file=str(output_dir / f"result_{img_file.name}"),
            pred_score_thr=args.score_threshold
        )
        
        print(f"  Saved to: {output_dir / f'result_{img_file.name}'}")
    
    print(f"\nInference complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()