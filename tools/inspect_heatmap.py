import torch
import cv2
import numpy as np
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracknet import BallTrackerNet

img_path = 'imgs/padel_frame2.png'
pt_model_path = 'exps/padel_v4/model_best.pt'

img = cv2.imread(img_path)
h, w = img.shape[:2]

out_w, out_h = 960, 544
img_resized = cv2.resize(img, (out_w, out_h))
inp = (img_resized.astype(np.float32) / 255.)
inp = np.rollaxis(inp, 2, 0)
inp_pt = torch.tensor(inp).unsqueeze(0).float()

model = BallTrackerNet(out_channels=6)
ckpt = torch.load(pt_model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

with torch.no_grad():
    out = model(inp_pt)
pred = torch.sigmoid(out).numpy()[0]

print("--- Raw Max Sigmoid Peak Values ---")
kp_names = ['tol', 'tor', 'point_7', 'point_9', 'tom', 'bottom_t']
for ch in range(6):
    hm = pred[ch]
    max_val = np.max(hm)
    min_val = np.min(hm)
    # Find argmax
    y_max, x_max = np.unravel_index(np.argmax(hm), hm.shape)
    
    # Coordinates mapped back
    x_orig = int(x_max * w / out_w)
    y_orig = int(y_max * h / out_h)
    
    print(f"{kp_names[ch]}: Max={max_val:.4f}, Min={min_val:.4f} at mapped: ({x_orig}, {y_orig})")

# Let's also test raw argmax thresholding without HoughCircles
print("\n--- Argmax coordinates (No HoughCircles) ---")
for ch in range(6):
    hm = pred[ch]
    y_max, x_max = np.unravel_index(np.argmax(hm), hm.shape)
    x_orig = int(x_max * w / out_w)
    y_orig = int(y_max * h / out_h)
    print(f"  {kp_names[ch]}: ({x_orig}, {y_orig}) [Peak value: {hm[y_max, x_max]:.4f}]")
