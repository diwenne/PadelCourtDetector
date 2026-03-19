from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import json
from utils import draw_umich_gaussian, line_intersection, is_point_in_image

class PadelDataset(Dataset):
    """Dataset for padel court keypoint detection (4 keypoints)."""
    
    def __init__(self, mode, input_height=1088, input_width=1920, scale=2, hp_radius=55):
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
        with open(os.path.join(self.path_dataset, 'data_{}.json'.format(mode)), 'r') as f:
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

        # Draw Top T and Bottom T anchors
        try:
            # Top T: Average of tol (0) and tor (1)
            x_top = int((scaled_kps[0][0] + scaled_kps[1][0]) / 2)
            y_top = int((scaled_kps[0][1] + scaled_kps[1][1]) / 2)
            if 0 <= x_top <= self.output_width and 0 <= y_top <= self.output_height:
                draw_gaussian(hm_hp[self.num_joints], (x_top, y_top), self.hp_radius)

            # Bottom T: Average of point_7 (2) and point_9 (3)
            x_bot = int((scaled_kps[2][0] + scaled_kps[3][0]) / 2)
            y_bot = int((scaled_kps[2][1] + scaled_kps[3][1]) / 2)
            if 0 <= x_bot <= self.output_width and 0 <= y_bot <= self.output_height:
                draw_gaussian(hm_hp[self.num_joints + 1], (x_bot, y_bot), self.hp_radius)
        except:
            pass  # Skip if coordinates fail
        
        return inp, hm_hp, np.array(scaled_kps, dtype=int), img_name

    def __len__(self):
        return len(self.data)
