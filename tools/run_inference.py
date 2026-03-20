import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Run padel court keypoint inference on N validation images.

Usage:
    python run_inference.py --num_samples 10
    python run_inference.py --model_path exps/padel_v1/model_last.pt --num_samples 5
"""
import torch
import cv2
import numpy as np
import json
import os
import argparse
from tracknet import BallTrackerNet
from postprocess import postprocess

# Keypoint config
KP_NAMES = ['tol', 'tor', 'point_7', 'point_9', 'top_t', 'bottom_t']
PRED_COLORS = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (0, 220, 220), (220, 50, 220), (200, 200, 50)]
GT_COLOR = (0, 200, 255)  # cyan-ish for ground truth

# Must match train_padel.py
INPUT_W, INPUT_H = 1920, 1088
SCALE = 2
OUT_W, OUT_H = INPUT_W // SCALE, INPUT_H // SCALE  # 960 x 544


def load_model(model_path, device):
    model = BallTrackerNet(out_channels=6)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, img, device):
    """Run inference on a single image. Returns list of (x, y) in original image coords."""
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (OUT_W, OUT_H))
    inp = (img_resized.astype(np.float32) / 255.)
    inp = np.rollaxis(inp, 2, 0)
    inp = torch.tensor(inp).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(inp)
    pred = torch.sigmoid(out).cpu().numpy()[0]

    keypoints = []
    for ch in range(min(6, pred.shape[0])):
        hm = (pred[ch] * 255).astype(np.uint8)
        x_out, y_out = postprocess(hm, scale=1)
        if x_out is not None:
            x_orig = int(x_out * w / OUT_W)
            y_orig = int(y_out * h / OUT_H)
            keypoints.append((x_orig, y_orig))
        else:
            keypoints.append(None)
    return keypoints


def draw_result(img, pred_kps, gt_kps=None):
    """Draw predicted (colored) and GT (cyan outline) keypoints on image."""
    result = img.copy()

    # Draw predicted keypoints
    for i, kp in enumerate(pred_kps):
        if kp is not None:
            color = PRED_COLORS[i % len(PRED_COLORS)]
            cv2.circle(result, kp, 12, color, -1)
            cv2.circle(result, kp, 12, (255, 255, 255), 2)
            label = KP_NAMES[i] if i < len(KP_NAMES) else f'kp{i}'
            cv2.putText(result, label, (kp[0] + 16, kp[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Draw GT keypoints (if provided)
    if gt_kps:
        for i, kp in enumerate(gt_kps):
            cv2.circle(result, (kp[0], kp[1]), 14, GT_COLOR, 3)

    # Draw lines between predicted court corners (if all 4 detected)
    corners = pred_kps[:4]
    if all(c is not None for c in corners):
        cv2.line(result, corners[0], corners[1], (255, 255, 255), 2)  # top
        cv2.line(result, corners[0], corners[2], (255, 255, 255), 2)  # left
        cv2.line(result, corners[1], corners[3], (255, 255, 255), 2)  # right
        cv2.line(result, corners[2], corners[3], (255, 255, 255), 2)  # bottom

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='exps/padel_v1/model_last.pt')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")

    model = load_model(args.model_path, device)

    # Load val data
    with open(os.path.join(args.data_dir, 'data_val.json')) as f:
        val_data = json.load(f)
    print(f"Val set: {len(val_data)} images")

    # Pick evenly spaced samples for diversity
    step = max(1, len(val_data) // args.num_samples)
    indices = list(range(0, len(val_data), step))[:args.num_samples]

    from scipy.spatial import distance
    errors = []

    for i, idx in enumerate(indices):
        sample = val_data[idx]
        img_name = sample['id']
        gt_kps = sample['kps']  # [[x,y], [x,y], [x,y], [x,y]]

        # Load image
        img_path = os.path.join(args.data_dir, 'images', img_name + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(args.data_dir, 'images', img_name + '.png')
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{i+1}] SKIP — could not load {img_name}")
            continue

        # Run inference
        pred_kps = run_inference(model, img, device)

        # Calculate errors
        sample_errors = []
        for j in range(4):
            if pred_kps[j] is not None:
                err = distance.euclidean(pred_kps[j], gt_kps[j])
                sample_errors.append(err)
                errors.append(err)

        # Draw and save
        result = draw_result(img, pred_kps, gt_kps)

        # Add info text
        err_str = ', '.join([f'{e:.1f}px' for e in sample_errors])
        cv2.putText(result, f'Model: {os.path.basename(args.model_path)}', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, f'Errors: {err_str}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out_path = os.path.join(args.output_dir, f'sample_{i+1:02d}.png')
        cv2.imwrite(out_path, result)
        avg = np.mean(sample_errors) if sample_errors else float('nan')
        print(f"  [{i+1}/{args.num_samples}] {img_name} — avg error: {avg:.1f}px → {out_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Mean error: {np.mean(errors):.2f}px")
    print(f"Median error: {np.median(errors):.2f}px")
    print(f"Results saved to: {args.output_dir}/")
