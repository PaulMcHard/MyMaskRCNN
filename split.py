import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

def create_train_val_split_by_parts(annotations_file, val_parts=None, test_parts=None, seed=42):
    """
    Split COCO annotations into train/val/test sets based on part categories.
    Validation and test sets contain completely unseen part models.
    
    Args:
        annotations_file: Path to COCO annotations JSON
        val_parts: List of part model IDs for validation (e.g., ['1m2', '1m3'])
        test_parts: List of part model IDs for testing (e.g., ['1m4'])
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Extract part model ID from each image
    # Images have 'model_id' field added during COCO conversion
    image_to_part = {}
    part_to_images = defaultdict(list)
    
    for img in coco_data['images']:
        # Get model_id from image metadata
        model_id = img.get('model_id', 'unknown')
        image_to_part[img['id']] = model_id
        part_to_images[model_id].append(img['id'])
    
    # Get all unique part models
    all_parts = sorted(part_to_images.keys())
    print(f"\nFound {len(all_parts)} unique part models:")
    for part in all_parts:
        print(f"  {part}: {len(part_to_images[part])} images")
    
    # Determine validation and test parts
    if val_parts is None:
        # Auto-select: use ~20% of parts for validation
        num_val_parts = max(1, int(len(all_parts) * 0.2))
        shuffled_parts = all_parts.copy()
        random.shuffle(shuffled_parts)
        val_parts = shuffled_parts[:num_val_parts]
        remaining_parts = shuffled_parts[num_val_parts:]
    else:
        val_parts = [p.lower() for p in val_parts]
        remaining_parts = [p for p in all_parts if p not in val_parts]
    
    if test_parts is None:
        # Auto-select: use ~10% of remaining parts for testing
        num_test_parts = max(1, int(len(remaining_parts) * 0.125))  # 10% of total
        random.shuffle(remaining_parts)
        test_parts = remaining_parts[:num_test_parts]
        train_parts = remaining_parts[num_test_parts:]
    else:
        test_parts = [p.lower() for p in test_parts]
        train_parts = [p for p in remaining_parts if p not in test_parts]
    
    print(f"\nPart distribution:")
    print(f"  Training parts: {train_parts}")
    print(f"  Validation parts: {val_parts}")
    print(f"  Test parts: {test_parts}")
    
    # Create image ID sets for each split
    train_ids = set()
    val_ids = set()
    test_ids = set()
    
    for part, images in part_to_images.items():
        if part in train_parts:
            train_ids.update(images)
        elif part in val_parts:
            val_ids.update(images)
        elif part in test_parts:
            test_ids.update(images)
    
    total_images = len(coco_data['images'])
    print(f"\nImage distribution:")
    print(f"  Total images: {total_images}")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/total_images*100:.1f}%)")
    print(f"  Val: {len(val_ids)} ({len(val_ids)/total_images*100:.1f}%)")
    print(f"  Test: {len(test_ids)} ({len(test_ids)/total_images*100:.1f}%)")
    
    # Create split datasets
    def create_split(split_name, image_id_set):
        split_data = {
            "info": coco_data.get("info", {
                "description": f"3D-ADAM {split_name.capitalize()} Split - Part-based",
                "url": "",
                "version": "1.0",
                "year": 2024,
                "contributor": "3D-ADAM",
                "date_created": ""
            }),
            "licenses": coco_data.get("licenses", [{
                "id": 1,
                "name": "Unknown",
                "url": ""
            }]),
            "images": [],
            "annotations": [],
            "categories": coco_data["categories"]
        }
        
        # Filter images
        for img in coco_data['images']:
            if img['id'] in image_id_set:
                split_data['images'].append(img)
        
        # Filter annotations
        for ann in coco_data['annotations']:
            if ann['image_id'] in image_id_set:
                split_data['annotations'].append(ann)
        
        return split_data
    
    # Create splits
    train_data = create_split('train', train_ids)
    val_data = create_split('val', val_ids)
    test_data = create_split('test', test_ids)
    
    # Count annotations per split
    train_ann_count = len(train_data['annotations'])
    val_ann_count = len(val_data['annotations'])
    test_ann_count = len(test_data['annotations'])
    
    print(f"\nAnnotation counts:")
    print(f"  Train: {train_ann_count}")
    print(f"  Val: {val_ann_count}")
    print(f"  Test: {test_ann_count}")
    
    # Count annotations per category per split
    def count_categories(data, split_name):
        cat_counts = defaultdict(int)
        for ann in data['annotations']:
            cat_id = ann['category_id']
            cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
            cat_counts[cat_name] += 1
        
        print(f"\n{split_name} category distribution:")
        for cat_name, count in sorted(cat_counts.items()):
            print(f"  {cat_name}: {count}")
    
    # Count parts per split
    def count_parts_in_split(data, split_name):
        part_counts = defaultdict(int)
        for img in data['images']:
            model_id = img.get('model_id', 'unknown')
            part_counts[model_id] += 1
        
        print(f"\n{split_name} part distribution:")
        for part, count in sorted(part_counts.items()):
            print(f"  {part}: {count} images")
    
    count_categories(train_data, "Train")
    count_parts_in_split(train_data, "Train")
    
    count_categories(val_data, "Val")
    count_parts_in_split(val_data, "Val")
    
    count_categories(test_data, "Test")
    count_parts_in_split(test_data, "Test")
    
    # Save splits
    output_dir = Path(annotations_file).parent
    
    train_file = output_dir / 'annotations_train.json'
    val_file = output_dir / 'annotations_val.json'
    test_file = output_dir / 'annotations_test.json'
    
    print(f"\nSaving splits...")
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  Train: {train_file}")
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  Val: {val_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  Test: {test_file}")
    
    # Save split configuration for reproducibility
    split_config = {
        'train_parts': train_parts,
        'val_parts': val_parts,
        'test_parts': test_parts,
        'seed': seed
    }
    config_file = output_dir / 'split_config.json'
    with open(config_file, 'w') as f:
        json.dump(split_config, f, indent=2)
    print(f"  Split config: {config_file}")
    
    print("\nSplit creation completed!")
    print("\n⚠️  IMPORTANT: Validation and test sets contain UNSEEN part models!")
    print("This evaluates generalization to new parts, not just new defect instances.")


def main():
    parser = argparse.ArgumentParser(
        description='Create train/val/test split from COCO annotations with part-based splitting'
    )
    parser.add_argument('annotations_file', type=str, 
                        help='Path to COCO annotations JSON file')
    parser.add_argument('--val-parts', type=str, nargs='+', default=None,
                        help='Part model IDs for validation (e.g., 1m2 1m3). '
                             'If not specified, automatically selects ~20%% of parts.')
    parser.add_argument('--test-parts', type=str, nargs='+', default=None,
                        help='Part model IDs for testing (e.g., 1m4). '
                             'If not specified, automatically selects ~10%% of parts.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    create_train_val_split_by_parts(
        args.annotations_file,
        val_parts=args.val_parts,
        test_parts=args.test_parts,
        seed=args.seed
    )
    
    return 0


if __name__ == '__main__':
    exit(main())