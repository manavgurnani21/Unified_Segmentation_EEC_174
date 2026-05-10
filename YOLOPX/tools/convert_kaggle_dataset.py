import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    print("Starting BDD100K Kaggle Dataset Conversion...")
    
    # Paths based on the workspace structure
    workspace_root = Path('/home/hades/Delete Later/Unified_Segmentation_EEC_174')
    
    source_json = workspace_root / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    source_masks_dir = workspace_root / 'bdd100k_seg/bdd100k/seg/labels/val'
    
    target_labels_dir = workspace_root / 'data/bdd100k/labels/val'
    target_seg_dir = workspace_root / 'data/bdd100k/bdd_seg_gt/val'
    target_lane_dir = workspace_root / 'data/bdd100k/bdd_lane_gt/val'
    
    # Ensure target directories exist
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    target_seg_dir.mkdir(parents=True, exist_ok=True)
    target_lane_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading monolithic JSON: {source_json}")
    with open(source_json, 'r') as f:
        data = json.load(f)
        
    print(f"Processing {len(data)} images...")
    
    for item in tqdm(data):
        img_name = item['name']
        img_id = img_name.replace('.jpg', '')
        
        # 1. Write the split JSON
        # YOLOPX expects older BDD100K format: {'frames': [{'objects': [...]}]}
        yolopx_format = {
            "frames": [
                {
                    "objects": item.get('labels', [])
                }
            ]
        }
        
        label_out_path = target_labels_dir / f"{img_id}.json"
        with open(label_out_path, 'w') as f:
            json.dump(yolopx_format, f)
            
        # 2. Convert the Semantic Mask to Drivable Area Mask
        src_mask_path = source_masks_dir / f"{img_id}_train_id.png"
        seg_out_path = target_seg_dir / f"{img_id}.png"
        lane_out_path = target_lane_dir / f"{img_id}.png"
        
        if src_mask_path.exists():
            mask = cv2.imread(str(src_mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Semantic TrainIds: 0 is road. Map 0 -> 1 (drivable), else -> 0
            drivable = np.zeros_like(mask)
            drivable[mask == 0] = 1
            
            # Save Drivable mask
            cv2.imwrite(str(seg_out_path), drivable)
            
            # Save Blank Lane mask (since authentic lane lines are missing)
            blank_lane = np.zeros_like(mask)
            cv2.imwrite(str(lane_out_path), blank_lane)
        else:
            # If for some reason the mask is missing, create blank defaults
            blank = np.zeros((720, 1280), dtype=np.uint8)
            cv2.imwrite(str(seg_out_path), blank)
            cv2.imwrite(str(lane_out_path), blank)
            
    print("Dataset conversion completed successfully!")

if __name__ == '__main__':
    main()
