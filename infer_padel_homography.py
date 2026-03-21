import torch
import cv2
import numpy as np
import os
import argparse
from tracknet import BallTrackerNet
from postprocess import postprocess
from homography_padel import compute_homography, warp_point_to_image

# Keypoint names for visualization
KEYPOINT_NAMES = ['tol', 'tor', 'point_7', 'point_9', 'tom', 'bottom_t']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 200, 255), (200, 0, 255)]  # BGR

def inference(model, img, device, input_size=(960, 544)):
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, input_size)
    inp = (img_resized.astype(np.float32) / 255.)
    inp = np.rollaxis(inp, 2, 0)
    inp = torch.tensor(inp).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        out = model(inp)
    out = torch.sigmoid(out).cpu().numpy()[0]
    
    keypoints = []
    for i in range(min(6, out.shape[0])):
        hm = (out[i] * 255).astype(np.uint8)
        kp = postprocess(hm, scale=1)
        if kp[0] is not None:
            x = int(kp[0] * w / input_size[0])
            y = int(kp[1] * h / input_size[1])
            keypoints.append((x, y))
        else:
            keypoints.append(None)
    
    return keypoints

def draw_keypoints(img, keypoints, radius=8, thickness=2):
    result = img.copy()
    for i, kp in enumerate(keypoints):
        if kp is not None:
            x, y = kp
            color = COLORS[i % len(COLORS)]
            cv2.circle(result, (x, y), radius, color, -1)
            cv2.putText(result, KEYPOINT_NAMES[i], (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    if all(kp is not None for kp in keypoints[:4]):
        cv2.line(result, keypoints[0], keypoints[1], (255, 255, 255), 2)
        cv2.line(result, keypoints[0], keypoints[2], (255, 255, 255), 2)
        cv2.line(result, keypoints[1], keypoints[3], (255, 255, 255), 2)
        cv2.line(result, keypoints[2], keypoints[3], (255, 255, 255), 2)
    return result

if __name__ == '__main__':
    model_path = 'exps/padel_v4/model_best.pt'
    input_path = 'imgs/padel_frame2.png'
    output_path = 'results/output_padel_frame2_v4_homography.png'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = BallTrackerNet(out_channels=6)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model = model.to(device)
    model.eval()
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error loading {input_path}")
        exit(1)
        
    kps = inference(model, img, device)
    print(f"Detected keypoints: {kps}")
    
    # Apply Homography if any point is missing
    if None in kps and sum(p is not None for p in kps) >= 4:
        print("Applying homography to infer missing points...")
        H = compute_homography(kps)
        if H is not None:
            output_w, output_h = 500, 1000
            dst_coords = [
                (0, output_h), (output_w, output_h), (0, 0), (output_w, 0),
                (output_w // 2, output_h), (output_w // 2, 0)
            ]
            for i in range(len(kps)):
                if kps[i] is None:
                    inferred = warp_point_to_image(dst_coords[i], H)
                    if inferred is not None:
                        kps[i] = inferred
                        print(f"Inferred {KEYPOINT_NAMES[i]}: {inferred}")
                        
    result = draw_keypoints(img, kps)
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")
