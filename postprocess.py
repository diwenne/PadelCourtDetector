"""
Heatmap postprocessing: extract (x, y) keypoint coordinates from predicted heatmaps.

Primary method (`postprocess`):
    1. Binary threshold at `low_thresh` (default 155/255)
    2. Detect circles via cv2.HoughCircles on the thresholded heatmap
    3. Return the center of the first detected circle, scaled by `scale`

Secondary method (`refine_kps`):
    1. Crop a region around the detected keypoint
    2. Detect line segments with HoughLinesP
    3. Find the intersection of the two dominant lines
    4. Return the intersection as the refined keypoint position

Used by:
    - base_validator.py (validation metrics)
    - predictor.py (production ONNX inference)
    - infer_padel.py (standalone PyTorch inference)
    - run_inference.py (batch validation inference)
"""
import cv2
import numpy as np
from scipy.spatial import distance
from utils import line_intersection



def postprocess(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    """Extract keypoint (x, y) from a single-channel heatmap using HoughCircles.
    
    Pipeline:
        1. Binary threshold the heatmap at `low_thresh` (values below become 0).
        2. Run cv2.HoughCircles to detect circular blobs.
        3. Return the center of the first (strongest) detected circle.
    
    Args:
        heatmap:    (H, W) uint8 array, values 0-255. Typically obtained by
                    multiplying sigmoid output by 255 and casting to uint8.
        scale:      Multiply output coordinates by this factor (default: 2).
                    Set to 1 when the heatmap is already at the desired resolution.
                    Set to 2 when mapping half-res heatmap coords to full-res.
        low_thresh: Binary threshold cutoff, 0-255 (default: 155).
        min_radius: Minimum circle radius for HoughCircles (default: 10).
        max_radius: Maximum circle radius for HoughCircles (default: 30).
    
    Returns:
        tuple: (x, y) in scaled coordinates, or (None, None) if no circle detected.
    """
    x_pred, y_pred = None, None
    ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred


def refine_kps(img, x_ct, y_ct, crop_size=40):
    """Refine a keypoint position by finding line intersections in a crop.
    
    Crops a region around the initial keypoint, detects lines via Hough
    transform, merges nearby lines, and returns the intersection of the
    two dominant lines as the refined position.
    
    Args:
        img:       Full input image (BGR, for line detection).
        x_ct:      Initial keypoint row (y-axis in image convention).
        y_ct:      Initial keypoint column (x-axis in image convention).
        crop_size: Half-size of the crop region in pixels (default: 40).
    
    Returns:
        tuple: (refined_y, refined_x) — note the swapped order matches
               the original codebase convention.
    """
    refined_x_ct, refined_y_ct = x_ct, y_ct
    
    img_height, img_width = img.shape[:2]
    x_min = max(x_ct-crop_size, 0)
    x_max = min(img_height, x_ct+crop_size)
    y_min = max(y_ct-crop_size, 0)
    y_max = min(img_width, y_ct+crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    if img_crop.size == 0:
        return refined_y_ct, refined_x_ct
    lines = detect_lines(img_crop)
    
    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if new_x_ct > 0 and new_x_ct < img_crop.shape[0] and new_y_ct > 0 and new_y_ct < img_crop.shape[1]:
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct                    
    return refined_y_ct, refined_x_ct



def detect_lines(image):
    """Detect line segments in an image crop using probabilistic Hough transform.
    
    Args:
        image: BGR image crop.
    
    Returns:
        list: Line segments as [x1, y1, x2, y2] arrays. May be empty.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines) 
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines


def merge_lines(lines):
    """Merge nearby parallel lines into single representative lines.
    
    Two lines are merged if both endpoints are within 20px of each other.
    Merged line coordinates are averaged.
    
    Args:
        lines: List of [x1, y1, x2, y2] line segments.
    
    Returns:
        list: Reduced list of merged line segments.
    """
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)],
                                        dtype=np.int32)
                        mask[i + j + 1] = False
            new_lines.append(line)  
    return new_lines       

