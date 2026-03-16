"""
Postprocessing for padel court keypoint detection.

Two refinement techniques (from README):
1. Refine keypoints using classical CV - detect line intersections
2. Use homography to project reference court points -> refined positions
"""
import cv2
import numpy as np
from scipy.spatial import distance


# =============================================================================
# Padel Court Reference for Homography Refinement
# =============================================================================

class PadelCourtRef:
    """Reference court corners (normalized 0-1). Order: tol, tor, point_7, point_9"""
    def __init__(self):
        self.key_points = [
            (0.0, 0.0),   # tol - far-left
            (1.0, 0.0),   # tor - far-right 
            (0.0, 1.0),   # point_7 - near-left
            (1.0, 1.0)    # point_9 - near-right
        ]


# =============================================================================
# 1. Refine keypoints using classical CV (line intersection detection)
# =============================================================================

def detect_lines(image):
    """Detect lines in image crop using Hough transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    if lines is None:
        return []
    lines = np.squeeze(lines)
    if len(lines.shape) == 1 and len(lines) == 4:
        lines = [lines]
    elif len(lines.shape) == 1:
        lines = []
    return lines


def merge_lines(lines):
    """Merge nearby lines into single lines."""
    if len(lines) == 0:
        return []
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
                        line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)], dtype=np.int32)
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines


def line_intersection(line1, line2):
    """Find intersection of two lines."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
    return (px, py)


def refine_kps(img, y_ct, x_ct, crop_size=40):
    """Refine keypoint by finding line intersection in crop around point."""
    refined_x, refined_y = x_ct, y_ct
    h, w = img.shape[:2]
    x_min = max(y_ct - crop_size, 0)
    x_max = min(h, y_ct + crop_size)
    y_min = max(x_ct - crop_size, 0)
    y_max = min(w, x_ct + crop_size)
    
    crop = img[x_min:x_max, y_min:y_max]
    if crop.size == 0:
        return refined_x, refined_y
    
    lines = detect_lines(crop)
    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x, new_y = int(inters[0]), int(inters[1])
                if 0 < new_x < crop.shape[1] and 0 < new_y < crop.shape[0]:
                    refined_x = y_min + new_x
                    refined_y = x_min + new_y
    return refined_x, refined_y


# =============================================================================
# 2. Homography-based keypoint refinement
# =============================================================================

def get_homography_refined_kps(detected_kps, img_shape):
    """
    Use homography to refine keypoint positions.
    Projects reference court corners through homography for accurate positions.
    """
    valid_kps = [kp for kp in detected_kps if kp is not None and kp[0] is not None]
    if len(valid_kps) < 4:
        return detected_kps
    
    court_ref = PadelCourtRef()
    ref_pts = np.array(court_ref.key_points, dtype=np.float32)
    det_pts = np.array(detected_kps, dtype=np.float32)
    
    H, _ = cv2.findHomography(ref_pts, det_pts)
    if H is None:
        return detected_kps
    
    ref_pts_homog = ref_pts.reshape(-1, 1, 2)
    refined_pts = cv2.perspectiveTransform(ref_pts_homog, H)
    
    refined_kps = []
    h, w = img_shape[:2]
    for pt in refined_pts:
        x, y = pt[0]
        if 0 <= x < w and 0 <= y < h:
            refined_kps.append((float(x), float(y)))
        else:
            refined_kps.append(None)
    return refined_kps


# =============================================================================
# 3. Full pipeline
# =============================================================================

def postprocess_keypoints(model_output, original_img, input_size=(960, 540),
                          use_refine_kps=True, use_homography=True, threshold=0.5):
    """
    Full postprocessing: extract keypoints, refine with CV, refine with homography.
    Returns list of 4 (x, y) tuples in original image coordinates.
    """
    if hasattr(model_output, 'cpu'):
        out = model_output.cpu().numpy()
    else:
        out = model_output
    if len(out.shape) == 4:
        out = out[0]
    
    h_orig, w_orig = original_img.shape[:2]
    inp_w, inp_h = input_size
    
    keypoints = []
    
    # Extract from heatmaps
    for i in range(min(4, out.shape[0])):
        hm = out[i]
        if hm.max() > threshold:
            y_out, x_out = np.unravel_index(np.argmax(hm), hm.shape)
            hm_h, hm_w = hm.shape
            x = int(x_out * w_orig / hm_w)
            y = int(y_out * h_orig / hm_h)
            keypoints.append((x, y))
        else:
            keypoints.append(None)
    
    # Refine with CV
    if use_refine_kps:
        refined = []
        for kp in keypoints:
            if kp is not None:
                x_ref, y_ref = refine_kps(original_img, kp[1], kp[0])
                refined.append((x_ref, y_ref))
            else:
                refined.append(None)
        keypoints = refined
    
    # Refine with homography
    if use_homography:
        keypoints = get_homography_refined_kps(keypoints, original_img.shape)
    
    return keypoints


if __name__ == '__main__':
    print("Postprocessing module for padel keypoints")
