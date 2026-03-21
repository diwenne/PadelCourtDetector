import torch
import cv2
import numpy as np
import os
import sys
import onnxruntime as ort

# Add root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tracknet import BallTrackerNet
from postprocess import postprocess

img_path = 'imgs/padel_frame2.png'
pt_model_path = 'exps/padel_v4/model_best.pt'
onnx_model_path = 'exps/padel_v4/model_best.onnx'

img = cv2.imread(img_path)
h, w = img.shape[:2]

# Preprocess
out_w, out_h = 960, 544
img_resized = cv2.resize(img, (out_w, out_h))
inp = (img_resized.astype(np.float32) / 255.)
inp = np.rollaxis(inp, 2, 0)

# --- 1. PyTorch ---
print("--- PyTorch Inference ---")
model = BallTrackerNet(out_channels=6)
ckpt = torch.load(pt_model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

inp_pt = torch.tensor(inp).unsqueeze(0).float()
with torch.no_grad():
    out_pt = model(inp_pt)
pred_pt = torch.sigmoid(out_pt).numpy()[0]

pt_kps = []
for ch in range(6):
    hm = (pred_pt[ch] * 255).astype(np.uint8)
    x, y = postprocess(hm, scale=1)
    if x is not None:
        pt_kps.append((int(x * w / out_w), int(y * h / out_h)))
    else:
        pt_kps.append(None)
print("PyTorch coordinates:", pt_kps)

# --- 2. ONNX ---
print("\n--- ONNX Inference ---")
sess = ort.InferenceSession(onnx_model_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

inp_onnx = np.expand_dims(inp, axis=0).astype(np.float32)
out_onnx = sess.run([output_name], {input_name: inp_onnx})[0]
pred_onnx = 1 / (1 + np.exp(-out_onnx[0]))

onnx_kps = []
for ch in range(6):
    hm = (pred_onnx[ch] * 255).astype(np.uint8)
    x, y = postprocess(hm, scale=1)
    if x is not None:
        onnx_kps.append((int(x * w / out_w), int(y * h / out_h)))
    else:
        onnx_kps.append(None)
print("ONNX coordinates:   ", onnx_kps)
print("\nScale Factors Used:")
print(f"Original: {w}x{h}, Network input: {out_w}x{out_h}")
print(f"Postprocess with scale=1")
print(f"Mapping x = x_out * {w} / {out_w}")
print(f"Mapping y = y_out * {h} / {out_h}")

# Check keypoint lists consistency
kp_names = ['tol', 'tor', 'point_7', 'point_9', 'tom', 'bottom_t']
print("\nDiscrepancy Check:")
for i, name in enumerate(kp_names):
    p_pt = pt_kps[i]
    p_on = onnx_kps[i]
    if p_pt != p_on:
        print(f"  {name}: PyTorch={p_pt} vs ONNX={p_on} DIFF!")
    else:
        print(f"  {name}: PyTorch={p_pt} vs ONNX={p_on} MATCH")
