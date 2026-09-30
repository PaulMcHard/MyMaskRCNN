import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def check_coco_annotations(annotation_file):
    """Check for annotation issues."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nChecking: {annotation_file}")
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    
    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # Check for issues
    issues = []
    for img_id, anns in img_to_anns.items():
        # Each annotation should have exactly one segmentation
        for ann in anns:
            if not ann.get('segmentation'):
                issues.append(f"Image {img_id}, Ann {ann['id']}: Missing segmentation")
            elif len(ann['segmentation']) != 1:
                issues.append(f"Image {img_id}, Ann {ann['id']}: Multiple segmentations ({len(ann['segmentation'])})")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✓ No issues found")
    
    defect_sizes = []
    for ann in data['annotations']:
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]  # width * height
        defect_sizes.append(area)
    
    print(f"Median defect area: {np.median(defect_sizes)}")
    print(f"25th percentile: {np.percentile(defect_sizes, 25)}")
    print(f"75th percentile: {np.percentile(defect_sizes, 75)}")
    
    return data, issues

# Check your annotations
check_coco_annotations('coco_unseen/annotations_train.json')
check_coco_annotations('coco_unseen/annotations_val.json')