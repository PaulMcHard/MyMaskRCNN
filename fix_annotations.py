import json
from pathlib import Path
import copy

def fix_coco_annotations(annotation_file, output_file=None):
    """
    Fix COCO annotations by splitting multi-segmentation annotations into separate instances.
    
    Args:
        annotation_file: Path to input COCO JSON file
        output_file: Path to output fixed JSON file (if None, overwrites input)
    """
    if output_file is None:
        output_file = annotation_file.replace('.json', '_fixed.json')
    
    print(f"\nFixing: {annotation_file}")
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    original_count = len(data['annotations'])
    print(f"Original annotations: {original_count}")
    
    # Process annotations
    fixed_annotations = []
    next_ann_id = max([ann['id'] for ann in data['annotations']]) + 1
    issues_fixed = 0
    
    for ann in data['annotations']:
        segmentation = ann['segmentation']
        
        if len(segmentation) == 1:
            # Single segmentation - keep as is
            fixed_annotations.append(ann)
        else:
            # Multiple segmentations - split into separate annotations
            issues_fixed += 1
            
            for seg_idx, seg in enumerate(segmentation):
                new_ann = copy.deepcopy(ann)
                new_ann['segmentation'] = [seg]  # Single segmentation
                
                # Update annotation ID for all but the first
                if seg_idx > 0:
                    new_ann['id'] = next_ann_id
                    next_ann_id += 1
                
                # Recalculate bbox and area for this specific segmentation
                # (optional but recommended for accuracy)
                import numpy as np
                points = np.array(seg).reshape(-1, 2)
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                
                new_ann['bbox'] = [
                    float(x_min), 
                    float(y_min), 
                    float(x_max - x_min), 
                    float(y_max - y_min)
                ]
                
                # Approximate area using bounding box
                new_ann['area'] = float((x_max - x_min) * (y_max - y_min))
                
                fixed_annotations.append(new_ann)
    
    # Update data
    data['annotations'] = fixed_annotations
    
    print(f"Fixed annotations: {len(fixed_annotations)}")
    print(f"Issues fixed: {issues_fixed}")
    print(f"New annotations created: {len(fixed_annotations) - original_count}")
    
    # Save fixed annotations
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to: {output_file}")
    
    return data


def verify_annotations(annotation_file):
    """Verify that all annotations have single segmentations."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nVerifying: {annotation_file}")
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    
    issues = []
    for ann in data['annotations']:
        if len(ann['segmentation']) != 1:
            issues.append(f"Ann {ann['id']}: {len(ann['segmentation'])} segmentations")
    
    if issues:
        print(f"⚠️  Still have {len(issues)} issues!")
        for issue in issues[:5]:
            print(f"  - {issue}")
    else:
        print("✓ All annotations have single segmentations")
    
    return len(issues) == 0


if __name__ == '__main__':
    # Fix training annotations
    fix_coco_annotations(
        'coco_unseen/annotations_train.json',
        'coco_unseen/annotations_train_fixed.json'
    )
    
    # Verify fix
    verify_annotations('coco_unseen/annotations_train_fixed.json')
    
    print("\n" + "="*80 + "\n")
    
    # Fix validation annotations
    fix_coco_annotations(
        'coco_unseen/annotations_val.json',
        'coco_unseen/annotations_val_fixed.json'
    )
    
    # Verify fix
    verify_annotations('coco_unseen/annotations_val_fixed.json')
    
    print("\n" + "="*80)
    print("✓ Done! Use the '_fixed.json' files for training")
    print("\nUpdate your training command to use:")
    print("  --annotations-dir coco_unseen")
    print("\nAnd rename the files:")
    print("  mv coco_unseen/annotations_train_fixed.json coco_unseen/annotations_train.json")
    print("  mv coco_unseen/annotations_val_fixed.json coco_unseen/annotations_val.json")