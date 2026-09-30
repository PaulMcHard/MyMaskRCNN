import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import cv2
from tqdm import tqdm
import argparse
from datetime import datetime

class COCODatasetGenerator:
    """Convert 3D-ADAM dataset to COCO format for Mask RCNN."""
    
    def __init__(self, dataset_root, output_path):
        self.dataset_root = Path(dataset_root)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Defect type to category ID mapping
        self.defect_types = [
            'bulge', 'crack', 'hole', 'under_extrusion', 
            'over_extrusion', 'cut', 'scratch'
        ]
        self.category_map = {defect: idx + 1 for idx, defect in enumerate(self.defect_types)}
        
        # COCO format structure with required metadata
        self.coco_output = {
            "info": {
                "description": "3D-ADAM Supervised Defect Detection Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "3D-ADAM",
                "date_created": datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.image_id = 0
        self.annotation_id = 0
    
    def create_categories(self):
        """Create COCO categories from defect types."""
        for defect_type, cat_id in self.category_map.items():
            self.coco_output["categories"].append({
                "id": cat_id,
                "name": defect_type,
                "supercategory": "defect"
            })
    
    def parse_filename(self, filename):
        """
        Parse filename to extract metadata.
        Example: 1M1_A_Bulge_Nano_000_image_bulge_defect_1.png
        Returns: (model, variant, defect_type, image_num, defect_num)
        """
        parts = filename.stem.split('_')
        
        # Find defect type in filename
        defect_type = None
        for dt in self.defect_types:
            if dt in filename.stem.lower():
                defect_type = dt
                break
        
        # Extract image number
        for i, part in enumerate(parts):
            if part == 'Nano' and i + 1 < len(parts):
                try:
                    image_num = int(parts[i + 1])
                    break
                except ValueError:
                    continue
        
        # Extract defect instance number if present
        defect_num = None
        if 'defect' in parts:
            defect_idx = parts.index('defect')
            if defect_idx + 1 < len(parts):
                try:
                    defect_num = int(parts[defect_idx + 1])
                except ValueError:
                    pass
        
        return defect_type, image_num, defect_num
    
    def mask_to_polygon(self, mask):
        """Convert binary mask to polygon coordinates."""
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            # Simplify contour
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # At least 3 points
                polygons.append(contour)
        
        return polygons
    
    def calculate_bbox(self, mask):
        """Calculate bounding box from mask."""
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return None
        
        ymin, ymax = pos[0].min(), pos[0].max()
        xmin, xmax = pos[1].min(), pos[1].max()
        
        width = int(xmax - xmin + 1)
        height = int(ymax - ymin + 1)
        
        return [int(xmin), int(ymin), width, height]
    
    def calculate_area(self, mask):
        """Calculate area from mask."""
        return int(np.sum(mask > 0))
    
    def process_image(self, model_id, defect_dir, rgb_file):
        """Process a single RGB image and its corresponding masks."""
        # Convert to Path object if string
        if isinstance(rgb_file, str):
            rgb_file = Path(rgb_file)
        
        rgb_path = defect_dir / 'rgb' / rgb_file.name
        binary_mask_path = defect_dir / 'binary_masks' / rgb_file.name
        
        if not rgb_path.exists():
            print(f"Warning: RGB image not found: {rgb_path}")
            return
        
        if not binary_mask_path.exists():
            print(f"Warning: Binary mask not found: {binary_mask_path}")
            return
        
        # Load RGB image to get dimensions
        try:
            rgb_image = Image.open(rgb_path)
            width, height = rgb_image.size
        except Exception as e:
            print(f"Error loading image {rgb_path}: {e}")
            return
        
        # Get defect type from directory
        defect_type = defect_dir.name.lower()
        
        if defect_type not in self.category_map:
            print(f"Warning: Unknown defect type '{defect_type}', skipping")
            return
        
        # Add image entry
        relative_path = str(rgb_path.relative_to(self.dataset_root)).replace('\\', '/')
        image_entry = {
            "id": self.image_id,
            "file_name": relative_path,
            "width": width,
            "height": height,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": "",
            "model_id": model_id,
            "defect_type": defect_type
        }
        self.coco_output["images"].append(image_entry)
        
        # Find all corresponding defect masks
        ground_truth_dir = defect_dir / 'ground_truth'
        if not ground_truth_dir.exists():
            # No defects for this image (might be good sample)
            self.image_id += 1
            return
        
        # Parse filename to get base name
        base_name = rgb_file.stem
        
        # Find all defect instance masks for this image
        defect_masks = list(ground_truth_dir.glob(f"{base_name}_*_defect_*.png"))
        
        if len(defect_masks) == 0:
            print(f"Warning: No defect masks found for {base_name}")
        
        for mask_file in defect_masks:
            self.process_mask(mask_file, defect_type, self.image_id, width, height)
        
        self.image_id += 1
    
    def process_mask(self, mask_path, defect_type, image_id, img_width, img_height):
        """Process a single defect mask and create annotation."""
        try:
            # Load mask
            mask = np.array(Image.open(mask_path))
            
            # Convert to binary if needed
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)
            
            # Skip if mask is empty
            if mask.sum() == 0:
                print(f"Warning: Empty mask {mask_path}")
                return
            
            # Calculate bbox and area
            bbox = self.calculate_bbox(mask)
            if bbox is None:
                print(f"Warning: Could not calculate bbox for {mask_path}")
                return
            
            area = self.calculate_area(mask)
            
            # Convert mask to polygon
            segmentation = self.mask_to_polygon(mask)
            if not segmentation:
                print(f"Warning: Could not create polygon for {mask_path}")
                return
            
            # Create annotation
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": self.category_map[defect_type],
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            
            self.coco_output["annotations"].append(annotation)
            self.annotation_id += 1
        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")
    
    def process_dataset(self):
        """Process entire dataset."""
        print("Creating COCO categories...")
        self.create_categories()
        
        print("Processing images and masks...")
        
        # Iterate through all model directories
        for model_dir in sorted(self.dataset_root.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model_id = model_dir.name
            print(f"\nProcessing model: {model_id}")
            
            # Iterate through defect type directories
            for defect_dir in sorted(model_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                
                defect_type = defect_dir.name
                rgb_dir = defect_dir / 'rgb'
                
                if not rgb_dir.exists():
                    print(f"  Warning: No RGB directory for {defect_type}")
                    continue
                
                print(f"  Processing {defect_type}...")
                
                # Process all RGB images
                rgb_files = sorted(rgb_dir.glob('*.png'))
                if len(rgb_files) == 0:
                    print(f"    Warning: No RGB images found in {rgb_dir}")
                    continue
                
                for rgb_file in tqdm(rgb_files, desc=f"    {defect_type}"):
                    self.process_image(model_id, defect_dir, rgb_file)
        
        # Save COCO JSON
        output_file = self.output_path / 'annotations.json'
        with open(output_file, 'w') as f:
            json.dump(self.coco_output, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"COCO annotation file created: {output_file}")
        print(f"Total images: {len(self.coco_output['images'])}")
        print(f"Total annotations: {len(self.coco_output['annotations'])}")
        print(f"Categories: {len(self.coco_output['categories'])}")
        print(f"{'='*60}")
        
        # Print category statistics
        category_counts = defaultdict(int)
        for ann in self.coco_output['annotations']:
            cat_id = ann['category_id']
            cat_name = next(c['name'] for c in self.coco_output['categories'] if c['id'] == cat_id)
            category_counts[cat_name] += 1
        
        print("\nDefect instance counts:")
        for cat_name, count in sorted(category_counts.items()):
            print(f"  {cat_name}: {count}")
        
        # Return success status
        return len(self.coco_output['images']) > 0


def main():
    parser = argparse.ArgumentParser(description='Convert 3D-ADAM dataset to COCO format')
    parser.add_argument('dataset_root', type=str, help='Root directory of the dataset')
    parser.add_argument('--output', type=str, default='coco_annotations', 
                        help='Output directory for COCO annotations')
    
    args = parser.parse_args()
    
    generator = COCODatasetGenerator(args.dataset_root, args.output)
    success = generator.process_dataset()
    
    if not success:
        print("\nError: No images were processed!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())