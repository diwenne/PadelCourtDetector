"""
PyTorch Dataset for padel court keypoint detection.

Loads images and annotations from ./data/ directory. For each image, generates
6-channel ground-truth heatmaps (one per keypoint) with 2D Gaussian blobs.

Data format (data_train.json / data_val.json):
    [
        {
            "id": "UUID_frame_01",        # Image filename (without extension)
            "kps": [[x,y], ...],           # 4 corner keypoints: [tol, tor, point_7, point_9]
            "size": [width, height],       # Original image dimensions
            "bottom_t": [x, y]             # Optional precomputed bottom-T coordinate
        }
    ]

Heatmap channels:
    0: tol       — Top-left (far-side left corner)
    1: tor       — Top-right (far-side right corner)
    2: point_7   — Bottom-left (near-side left corner)
    3: point_9   — Bottom-right (near-side right corner)
    4: tom       — Top-middle (dynamically computed as midpoint of tol + tor)
    5: bottom_t  — Bottom-middle T-junction (precomputed or midpoint of point_7 + point_9)

The dataset automatically handles:
    - Image resizing to output dimensions (input_size / scale)
    - Keypoint coordinate scaling from original to output resolution
    - Gaussian heatmap generation with configurable radius
    - Dynamic T-anchor computation from corner coordinates
    - Support for both .jpg and .png image formats
    - Automatic v4 dataset fallback (data_train_v4.json if available)
"""
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import json
from utils import draw_umich_gaussian, line_intersection, is_point_in_image

class PadelDataset(Dataset):
    """PyTorch Dataset for padel court keypoint detection.
    
    Produces 6-channel heatmaps: 4 annotated corners + 2 dynamically computed
    T-junction anchor points (tom, bottom_t).
    
    The dataset supports automatic version fallback: if data_{mode}_v4.json exists,
    it is used instead of data_{mode}.json.
    """
    
    def __init__(self, mode, input_height=1088, input_width=1920, scale=2, hp_radius=55):
        """Initialize the padel court dataset.
        
        Args:
            mode:         'train' or 'val'.
            input_height: Full input image height before scaling (default: 1088).
                          Must be divisible by 8 (3 pooling layers at 2× each).
            input_width:  Full input image width before scaling (default: 1920).
            scale:        Downscale factor for model input/output (default: 2).
                          Output dims = input_dims / scale = 960×544.
            hp_radius:    Gaussian heatmap radius in pixels at output resolution
                          (default: 55). Controls the spread of the target Gaussian.
                          Sigma = diameter / 6 = (2*55+1) / 6 ≈ 18.5px.
        """
        self.mode = mode
        assert mode in ['train', 'val'], 'incorrect mode'
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = int(input_height/scale)
        self.output_width = int(input_width/scale)
        self.num_joints = 4  # tol, tor, point_7, point_9
        self.hp_radius = hp_radius
        self.scale = scale

        self.path_dataset = './data'
        self.path_images = os.path.join(self.path_dataset, 'images')
        json_path_v4 = os.path.join(self.path_dataset, 'data_{}_v4.json'.format(mode))
        json_path_old = os.path.join(self.path_dataset, 'data_{}.json'.format(mode))
        with open(json_path_v4 if os.path.exists(json_path_v4) else json_path_old, 'r') as f:
            self.data = json.load(f)
        print('mode = {}, len = {}'.format(mode, len(self.data)))

    def filter_data(self):
        new_data = []
        for i in range(len(self.data)):
            max_elems = np.array(self.data[i]['kps']).max(axis=0)
            min_elems = np.array(self.data[i]['kps']).min(axis=0)
            if max_elems[0] < self.input_width and min_elems[0] > 0 and max_elems[1] < self.input_height and \
                    min_elems[1] > 0:
                new_data.append(self.data[i])
        return new_data

    def __getitem__(self, index):
        """Load one sample: image + 6-channel heatmap + scaled keypoints.
        
        Returns:
            tuple: (inp, hm_hp, scaled_kps, img_name)
                - inp:        (3, out_H, out_W) float32 array, normalized [0, 1], CHW.
                - hm_hp:      (6, out_H, out_W) float32 heatmap array with Gaussians.
                - scaled_kps: (4, 2) int array of [x, y] keypoints at output resolution.
                - img_name:   str, image identifier (without extension).
        
        Scaling logic:
            Original image coords → output resolution coords using:
                scale_x = output_width / original_width
                scale_y = output_height / original_height
        
        T-anchor computation:
            - tom (channel 4): midpoint of scaled tol and tor.
            - bottom_t (channel 5): uses precomputed 'bottom_t' field from
              annotation if available, otherwise falls back to midpoint of
              scaled point_7 and point_9.
        """
        # Support both .jpg and .png
        img_name = self.data[index]['id']
        kps = self.data[index]['kps']
        
        # Try jpg first, then png
        img_path = os.path.join(self.path_images, img_name + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.path_images, img_name + '.png')
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Resize to output dimensions
        img = cv2.resize(img, (self.output_width, self.output_height))
        inp = (img.astype(np.float32) / 255.)
        inp = np.rollaxis(inp, 2, 0)

        # Create heatmaps for 4 keypoints + 2 T points (Top/Bottom anchors)
        hm_hp = np.zeros((self.num_joints+2, self.output_height, self.output_width), dtype=np.float32)
        draw_gaussian = draw_umich_gaussian
        # Scale keypoints from original image size to output size
        orig_width = self.data[index].get('size', [self.input_width, self.input_height])[0]
        orig_height = self.data[index].get('size', [self.input_width, self.input_height])[1]
        
        scale_x = self.output_width / orig_width
        scale_y = self.output_height / orig_height

        scaled_kps = []
        for i in range(len(kps)):
            x_scaled = int(kps[i][0] * scale_x)
            y_scaled = int(kps[i][1] * scale_y)
            scaled_kps.append([x_scaled, y_scaled])
            
            if 0 <= x_scaled <= self.output_width and 0 <= y_scaled <= self.output_height:
                draw_gaussian(hm_hp[i], (x_scaled, y_scaled), self.hp_radius)

        # Draw Tom and Bottom T anchors
        try:
            # Tom: Average of tol (0) and tor (1)
            x_top = int((scaled_kps[0][0] + scaled_kps[1][0]) / 2)
            y_top = int((scaled_kps[0][1] + scaled_kps[1][1]) / 2)
            if 0 <= x_top <= self.output_width and 0 <= y_top <= self.output_height:
                draw_gaussian(hm_hp[self.num_joints], (x_top, y_top), self.hp_radius)

            # Bottom T: Average of point_7 (2) and point_9 (3), or precomputed geometry
            if 'bottom_t' in self.data[index] and self.data[index]['bottom_t'] is not None:
                x_bot = int(self.data[index]['bottom_t'][0] * scale_x)
                y_bot = int(self.data[index]['bottom_t'][1] * scale_y)
            else:
                x_bot = int((scaled_kps[2][0] + scaled_kps[3][0]) / 2)
                y_bot = int((scaled_kps[2][1] + scaled_kps[3][1]) / 2)
            if 0 <= x_bot <= self.output_width and 0 <= y_bot <= self.output_height:
                draw_gaussian(hm_hp[self.num_joints + 1], (x_bot, y_bot), self.hp_radius)
        except:
            pass  # Skip if coordinates fail
        
        return inp, hm_hp, np.array(scaled_kps, dtype=int), img_name

    def __len__(self):
        return len(self.data)
