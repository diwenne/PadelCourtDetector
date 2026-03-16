#!/usr/bin/env python3
"""
Prepare padel court keypoints dataset for training.
Converts YAML annotations to TennisCourtDetector JSON format.

Usage (on VM):
    python prepare_padel_dataset.py --input_dir ~/court-keypoints-dataset --output_dir ./data
"""

import os
import json
import yaml
import shutil
import argparse
from pathlib import Path
from PIL import Image
import random

# Keypoint order: tol, tor, point_7, point_9
KEYPOINT_NAMES = ['tol', 'tor', 'point_7', 'point_9']


def parse_yaml_keypoints(yaml_path):
    """Parse keypoints from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    keypoints = []
    for name in KEYPOINT_NAMES:
        if name in data:
            # YAML format: name: [x, y]
            coords = data[name]
            keypoints.append([int(coords[0]), int(coords[1])])
        else:
            # Missing keypoint - mark as invalid
            keypoints.append([-1, -1])
    
    return keypoints


def get_image_size(image_path):
    """Get image dimensions."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def process_dataset(input_dir, output_dir, train_ratio=0.9):
    """Process all samples and create train/val splits."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    image_sizes = set()
    
    # Find all sample folders
    sample_folders = [f for f in input_path.iterdir() if f.is_dir()]
    print(f"Found {len(sample_folders)} sample folders")
    
    for i, sample_dir in enumerate(sample_folders):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(sample_folders)}...")
        
        yaml_path = sample_dir / 'court-keypoints.yaml'
        
        # Check for available frames
        frame_paths = list(sample_dir.glob('frame_*.jpg'))
        
        if not yaml_path.exists() or not frame_paths:
            continue
        
        try:
            keypoints = parse_yaml_keypoints(yaml_path)
            
            # Check if all keypoints are valid
            if any(kp[0] < 0 or kp[1] < 0 for kp in keypoints):
                continue
            
            # Process each frame
            for frame_path in frame_paths:
                # Create unique ID
                sample_id = f"{sample_dir.name}_{frame_path.stem}"
                
                # Get image size
                img_size = get_image_size(frame_path)
                image_sizes.add(img_size)
                
                # Copy image
                dest_path = images_dir / f"{sample_id}.jpg"
                if not dest_path.exists():
                    shutil.copy2(frame_path, dest_path)
                
                samples.append({
                    'id': sample_id,
                    'kps': keypoints,
                    'size': list(img_size)
                })
        except Exception as e:
            print(f"Error processing {sample_dir.name}: {e}")
            continue
    
    print(f"\nTotal valid samples: {len(samples)}")
    print(f"Image sizes found: {image_sizes}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * train_ratio)
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]
    
    # Save JSON files
    train_path = output_path / 'data_train.json'
    val_path = output_path / 'data_val.json'
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    
    print(f"\nSaved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")
    
    return train_data, val_data, image_sizes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to downloaded court-keypoints-dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Train/val split ratio')
    args = parser.parse_args()
    
    train_data, val_data, sizes = process_dataset(
        args.input_dir, 
        args.output_dir,
        args.train_ratio
    )
    
    print("\n=== Dataset Ready ===")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Image sizes: {sizes}")
