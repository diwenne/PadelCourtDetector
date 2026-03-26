import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os
import cv2
import sys
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

sys.path.append('camera_keypoints_monitor')
from src.compute_t_intersection import find_t_intersection

def process_item(item):
    img_name = item['id']
    kps = item['kps']
    
    path_images = './data/images'
    img_path = os.path.join(path_images, img_name + '.jpg')
    if not os.path.exists(img_path):
        img_path = os.path.join(path_images, img_name + '.png')
        
    # Estimate is average of point 7 and point 9
    try:
        est_x = (kps[2][0] + kps[3][0]) / 2.0
        est_y = (kps[2][1] + kps[3][1]) / 2.0
        estimate = (est_x, est_y)
        
        corrected = find_t_intersection(Path(img_path), estimate, debug=False)
        item['bottom_t'] = corrected if corrected else estimate
    except Exception as e:
        item['bottom_t'] = None
    
    return item

def recompute_json(mode, output_file):
    with open(f'./data/data_{mode}.json', 'r') as f:
        data = json.load(f)
        
    print(f"Processing {len(data)} items for {mode}...")
    
    with Pool(14) as p: # Leave 2 cores free
        # Process in chunks and display progress
        results = list(tqdm(p.imap(process_item, data), total=len(data)))
        
    with open(output_file, 'w') as f:
        json.dump(results, f)
        
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    recompute_json('train', './data/data_train_v4.json')
