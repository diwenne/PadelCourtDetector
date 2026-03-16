"""
Inference script for padel court keypoint detection.

Usage:
    python infer_padel.py --model_path exps/padel_v1/model_best.pt --input_path image.jpg --output_path output.png
"""
import torch
import cv2
import numpy as np
import argparse
from tracknet import BallTrackerNet
from postprocess import postprocess

# Keypoint names for visualization
KEYPOINT_NAMES = ['tol', 'tor', 'point_7', 'point_9']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR


def inference(model, img, device, input_size=(960, 540)):
    """Run inference on single image."""
    h, w = img.shape[:2]
    
    # Resize to model input
    img_resized = cv2.resize(img, input_size)
    inp = (img_resized.astype(np.float32) / 255.)
    inp = np.rollaxis(inp, 2, 0)
    inp = torch.tensor(inp).unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        out = model(inp)
    out = torch.sigmoid(out).cpu().numpy()[0]
    
    # Extract keypoints from heatmaps
    keypoints = []
    for i in range(4):  # 4 keypoints
        hm = out[i]
        kp = postprocess(hm)
        if kp[0] is not None:
            # Scale back to original image size
            x = int(kp[0] * w / input_size[0])
            y = int(kp[1] * h / input_size[1])
            keypoints.append((x, y))
        else:
            keypoints.append(None)
    
    return keypoints


def draw_keypoints(img, keypoints, radius=8, thickness=2):
    """Draw keypoints on image."""
    result = img.copy()
    
    for i, kp in enumerate(keypoints):
        if kp is not None:
            x, y = kp
            color = COLORS[i % len(COLORS)]
            cv2.circle(result, (x, y), radius, color, -1)
            cv2.putText(result, KEYPOINT_NAMES[i], (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    
    # Draw lines between keypoints if all are detected
    if all(kp is not None for kp in keypoints):
        # Top line: tol-tor
        cv2.line(result, keypoints[0], keypoints[1], (255, 255, 255), 2)
        # Left line: tol-point_7
        cv2.line(result, keypoints[0], keypoints[2], (255, 255, 255), 2)
        # Right line: tor-point_9
        cv2.line(result, keypoints[1], keypoints[3], (255, 255, 255), 2)
        # Bottom line: point_7-point_9
        cv2.line(result, keypoints[2], keypoints[3], (255, 255, 255), 2)
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_path', type=str, required=True, help='Input image path')
    parser.add_argument('--output_path', type=str, default='output_padel.png', help='Output path')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = BallTrackerNet(out_channels=5)
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, best_acc={ckpt.get('best_accuracy', '?')})")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    
    # Load image
    img = cv2.imread(args.input_path)
    if img is None:
        print(f"Error: Could not load image {args.input_path}")
        exit(1)
    
    # Run inference
    keypoints = inference(model, img, device)
    print(f"Detected keypoints: {dict(zip(KEYPOINT_NAMES, keypoints))}")
    
    # Draw and save
    result = draw_keypoints(img, keypoints)
    cv2.imwrite(args.output_path, result)
    print(f"Result saved to {args.output_path}")
