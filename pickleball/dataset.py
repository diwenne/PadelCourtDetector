"""
Pickleball dataset loader for court keypoints.
Reads from ../pickleball_dataset/annotations_custom.json and filters for annotated images.
"""
import os
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from utils import draw_umich_gaussian

class PickleballDataset(Dataset):
    """
    Dataset loader for Pickleball court keypoints.
    Filters for annotated images (4 points) from the external pickleball_dataset repo.
    """
    def __init__(self, mode='train', input_height=1088, input_width=1920, scale=2, hp_radius=55):
        super().__init__()
        self.mode = mode
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = int(input_height/scale)
        self.output_width = int(input_width/scale)
        self.scale = scale
        self.hp_radius = hp_radius
        self.num_joints = 4 # tol, tor, bol, bor

        # Base directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'pickleball'))
        self.path_images = os.path.join(self.base_dir, 'images')
        self.path_annotations = os.path.join(self.base_dir, 'annotations_filtered.json')

        if not os.path.exists(self.path_annotations):
            raise FileNotFoundError(f"Annotations file not found: {self.path_annotations}")

        with open(self.path_annotations, 'r') as f:
            all_ann = json.load(f)

        # Filter for entries that have at least some annotations
        # We'll use any image that has at least one valid keypoint
        self.data = []
        for img_name, ann in all_ann.items():
            if ann and any(k in ann and ann[k] is not None for k in ['tol', 'tor', 'bol', 'bor']):
                self.data.append({
                    'id': img_name,
                    'ann': ann
                })

        # Split into train/val (90/10)
        np.random.seed(42)
        np.random.shuffle(self.data)
        split_idx = int(0.9 * len(self.data))
        if mode == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        print(f"Pickleball Dataset: {mode} split, {len(self.data)} images")

    def __getitem__(self, index):
        sample = self.data[index]
        img_name = sample['id']
        ann = sample['ann']
        
        # We need a predictable order for the heatmap channels
        kp_order = ['tol', 'tor', 'bol', 'bor']
        
        img_path = os.path.join(self.path_images, img_name)
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for common extensions if not specified
            img_path_jpg = os.path.join(self.path_images, img_name + '.jpg')
            img = cv2.imread(img_path_jpg)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
        
        orig_h, orig_w = img.shape[:2]
        
        # Resize to output resolution
        img_resized = cv2.resize(img, (self.output_width, self.output_height))
        inp = (img_resized.astype(np.float32) / 255.)
        inp = np.rollaxis(inp, 2, 0)

        # 6-channel heatmap: tol, tor, bol, bor, tom, bom
        hm_hp = np.zeros((6, self.output_height, self.output_width), dtype=np.float32)
        
        scaled_kps = []
        for i, kp_name in enumerate(kp_order):
            pt = ann.get(kp_name)
            if pt is not None:
                # Scale from normalized [0, 1] to output resolution
                x_scaled = int(pt[0] * self.output_width)
                y_scaled = int(pt[1] * self.output_height)
                scaled_kps.append((x_scaled, y_scaled))
                
                if 0 <= x_scaled < self.output_width and 0 <= y_scaled < self.output_height:
                    draw_umich_gaussian(hm_hp[i], (x_scaled, y_scaled), self.hp_radius)
            else:
                scaled_kps.append(None)

        # Draw midpoints if both endpoints exist
        # tom: midpoint of tol(0) and tor(1)
        if scaled_kps[0] is not None and scaled_kps[1] is not None:
            x_top = int((scaled_kps[0][0] + scaled_kps[1][0]) / 2)
            y_top = int((scaled_kps[0][1] + scaled_kps[1][1]) / 2)
            if 0 <= x_top < self.output_width and 0 <= y_top < self.output_height:
                draw_umich_gaussian(hm_hp[4], (x_top, y_top), self.hp_radius)

        # bom: midpoint of bol(2) and bor(3)
        if scaled_kps[2] is not None and scaled_kps[3] is not None:
            x_bot = int((scaled_kps[2][0] + scaled_kps[3][0]) / 2)
            y_bot = int((scaled_kps[2][1] + scaled_kps[3][1]) / 2)
            if 0 <= x_bot < self.output_width and 0 <= y_bot < self.output_height:
                draw_umich_gaussian(hm_hp[5], (x_bot, y_bot), self.hp_radius)

        # For scaled_kps to return, we convert to a numpy array, replacing None with [-1, -1]
        kps_array = np.array([pt if pt is not None else [-1, -1] for pt in scaled_kps], dtype=int)
        
        return inp, hm_hp, kps_array, img_name

    def __len__(self):
        return len(self.data)
