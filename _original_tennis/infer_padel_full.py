"""
Full inference pipeline for padel court detection with homography.

Usage:
    python infer_padel_full.py --model_path exps/padel_v1/model_last.pt --input_path image.jpg
"""
import torch
import cv2
import numpy as np
import argparse
from tracknet import BallTrackerNet
from court_reference_padel import PadelCourtReference
from homography_padel import compute_homography, warp_image_to_court

# Keypoint names and colors for visualization
KEYPOINT_NAMES = ['tol', 'tor', 'point_7', 'point_9']
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR: red, green, blue, yellow


def load_model(model_path, device='cuda'):
    """Load trained model from checkpoint."""
    model = BallTrackerNet(out_channels=5)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint format and direct state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model (direct state_dict)")
    
    model = model.to(device)
    model.eval()
    return model


def detect_keypoints(model, img, device, input_size=(960, 540), threshold=0.5):
    """
    Run keypoint detection on image.
    
    Returns:
        List of 4 (x, y) tuples or None for each keypoint
    """
    h, w = img.shape[:2]
    
    # Preprocess
    img_resized = cv2.resize(img, input_size)
    inp = (img_resized.astype(np.float32) / 255.)
    inp = np.rollaxis(inp, 2, 0)
    inp = torch.tensor(inp).unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        out = model(inp)
    out = torch.sigmoid(out).cpu().numpy()[0]
    
    # Extract keypoints using argmax
    keypoints = []
    confidences = []
    
    for i in range(4):
        hm = out[i]
        max_val = hm.max()
        confidences.append(max_val)
        
        if max_val > threshold:
            y_out, x_out = np.unravel_index(np.argmax(hm), hm.shape)
            x = int(x_out * w / input_size[0])
            y = int(y_out * h / input_size[1])
            keypoints.append((x, y))
        else:
            keypoints.append(None)
    
    return keypoints, confidences


def draw_court_overlay(img, keypoints, confidences=None):
    """Draw detected keypoints and court lines on image."""
    result = img.copy()
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp is not None:
            x, y = kp
            color = COLORS[i]
            cv2.circle(result, (x, y), 12, color, -1)
            cv2.circle(result, (x, y), 14, (255, 255, 255), 2)
            
            label = KEYPOINT_NAMES[i]
            if confidences is not None:
                label += f" ({confidences[i]:.2f})"
            cv2.putText(result, label, (x + 18, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw court lines if all keypoints detected
    if all(kp is not None for kp in keypoints):
        # Top line (tol -> tor)
        cv2.line(result, keypoints[0], keypoints[1], (255, 255, 255), 3)
        # Left line (tol -> point_7)
        cv2.line(result, keypoints[0], keypoints[2], (255, 255, 255), 3)
        # Right line (tor -> point_9)
        cv2.line(result, keypoints[1], keypoints[3], (255, 255, 255), 3)
        # Bottom line (point_7 -> point_9)
        cv2.line(result, keypoints[2], keypoints[3], (255, 255, 255), 3)
        
        # Diagonals (faint)
        cv2.line(result, keypoints[0], keypoints[3], (128, 128, 128), 1)
        cv2.line(result, keypoints[1], keypoints[2], (128, 128, 128), 1)
    
    return result


def create_side_by_side(original, warped, court_ref):
    """Create side-by-side visualization with original and warped court."""
    # Get court reference overlay
    court_overlay = court_ref.build_court_reference()
    
    # Blend warped image with court reference lines
    if warped is not None:
        # Resize warped to match court overlay
        warped_resized = cv2.resize(warped, (court_overlay.shape[1], court_overlay.shape[0]))
        # Blend
        combined = cv2.addWeighted(warped_resized, 0.7, court_overlay, 0.3, 0)
    else:
        combined = court_overlay
    
    # Scale to match heights
    h1 = original.shape[0]
    h2 = combined.shape[0]
    
    if h2 != h1:
        scale = h1 / h2
        new_w = int(combined.shape[1] * scale)
        combined = cv2.resize(combined, (new_w, h1))
    
    # Side by side
    result = np.hstack([original, combined])
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Padel court detection with homography')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--output_path', type=str, default='result_padel.png',
                       help='Output overlay image path')
    parser.add_argument('--output_warped', type=str, default='result_warped.png',
                       help='Output warped court image path')
    parser.add_argument('--output_combined', type=str, default='result_combined.png',
                       help='Output side-by-side comparison')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load image
    img = cv2.imread(args.input_path)
    if img is None:
        print(f"Error: Could not load image {args.input_path}")
        return
    print(f"Loaded image: {img.shape[1]}x{img.shape[0]}")
    
    # Detect keypoints
    keypoints, confidences = detect_keypoints(model, img, device, threshold=args.threshold)
    
    # Print results
    print("\nDetected keypoints:")
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        status = f"({kp[0]}, {kp[1]})" if kp else "NOT DETECTED"
        print(f"  {KEYPOINT_NAMES[i]}: {status} (conf: {conf:.4f})")
    
    # Draw overlay
    overlay = draw_court_overlay(img, keypoints, confidences)
    cv2.imwrite(args.output_path, overlay)
    print(f"\nOverlay saved to: {args.output_path}")
    
    # Compute homography and warp
    court_ref = PadelCourtReference()
    
    if all(kp is not None for kp in keypoints):
        warped = warp_image_to_court(img, keypoints, court_ref)
        if warped is not None:
            cv2.imwrite(args.output_warped, warped)
            print(f"Warped court saved to: {args.output_warped}")
            
            # Create combined visualization
            combined = create_side_by_side(overlay, warped, court_ref)
            cv2.imwrite(args.output_combined, combined)
            print(f"Combined view saved to: {args.output_combined}")
        else:
            print("Warning: Homography computation failed")
    else:
        print("Warning: Not all keypoints detected, skipping homography")


if __name__ == '__main__':
    main()
