import json
from collections import defaultdict

def analyze_dataset_parts(annotations_file):
    """Analyze parts in the dataset to help decide splits."""
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Collect statistics per part
    part_stats = defaultdict(lambda: {
        'images': 0,
        'annotations': 0,
        'defect_types': defaultdict(int)
    })
    
    # Count images per part
    for img in data['images']:
        model_id = img.get('model_id', 'unknown')
        part_stats[model_id]['images'] += 1
    
    # Count annotations per part and defect type
    img_id_to_part = {img['id']: img.get('model_id', 'unknown') 
                      for img in data['images']}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        model_id = img_id_to_part.get(img_id, 'unknown')
        cat_id = ann['category_id']
        cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
        
        part_stats[model_id]['annotations'] += 1
        part_stats[model_id]['defect_types'][cat_name] += 1
    
    # Print analysis
    print("=" * 80)
    print("Dataset Part Analysis")
    print("=" * 80)
    
    for part in sorted(part_stats.keys()):
        stats = part_stats[part]
        print(f"\nPart: {part}")
        print(f"  Images: {stats['images']}")
        print(f"  Annotations: {stats['annotations']}")
        print(f"  Defect types:")
        for defect, count in sorted(stats['defect_types'].items()):
            print(f"    {defect}: {count}")
    
    print("\n" + "=" * 80)
    print("Suggested Split Strategy:")
    print("=" * 80)
    
    parts = sorted(part_stats.keys())
    num_parts = len(parts)
    
    if num_parts < 3:
        print("⚠️  Warning: Less than 3 parts detected!")
        print("Recommend using random split instead of part-based split.")
    else:
        num_val = max(1, int(num_parts * 0.2))
        num_test = max(1, int(num_parts * 0.1))
        
        print(f"\nTotal parts: {num_parts}")
        print(f"Suggested validation parts: {num_val} (~20%)")
        print(f"Suggested test parts: {num_test} (~10%)")
        print(f"\nExample command:")
        print(f"python create_train_val_split.py {annotations_file} \\")
        if num_val > 0:
            val_example = ' '.join(parts[:num_val])
            print(f"    --val-parts {val_example} \\")
        if num_test > 0:
            test_example = ' '.join(parts[num_val:num_val+num_test])
            print(f"    --test-parts {test_example}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_parts.py <annotations_file>")
        sys.exit(1)
    
    analyze_dataset_parts(sys.argv[1])