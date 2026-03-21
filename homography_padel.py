"""
Homography computation for padel court keypoint detection.

Maps detected image keypoints to a canonical top-down court view (500×1000 px)
using perspective transformation. Supports graceful degradation: if any keypoint
is missing but ≥4 are detected, homography is still computed and the missing
point(s) can be recovered via inverse projection.

Point mapping (image → top-down court):
    Index 0: tol       → (0, 1000)       bottom-left
    Index 1: tor       → (500, 1000)     bottom-right
    Index 2: point_7   → (0, 0)          top-left
    Index 3: point_9   → (500, 0)        top-right
    Index 4: tom       → (250, 1000)     bottom-center
    Index 5: bottom_t  → (250, 0)        top-center

Homography requires a minimum of 4 point correspondences to solve the 3×3
matrix (8 degrees of freedom). With 5-6 points, RANSAC is used for
outlier robustness and least-squares refinement.

Key functions:
    - compute_homography: Build H matrix from detected → court point pairs
    - warp_image_to_court: Produce a top-down warped court image
    - warp_point_to_court: Transform image coords → court coords via H
    - warp_point_to_image: Transform court coords → image coords via H⁻¹
      (used to recover missing keypoints)

Used by:
    - predictor.py (production API: infer missing keypoints)
    - tools/infer_padel_homography.py (visualization)
"""
import numpy as np
import cv2


def compute_homography(detected_points, output_size=(500, 1000)):
    """
    Compute homography matrix from detected keypoints to top-down rectangular view.
    Can compute from ANY 4 or more valid keypoints (corners + anchors).
    
    Args:
        detected_points: List of up to 6 (x, y) tuples:
                        [tol, tor, point_7, point_9, tom, bottom_t]
                        Any can be None if not detected.
        output_size: (width, height) of output image
    
    Returns:
        Homography matrix (3x3), status, or (None, None) if not enough points
    """
    output_w, output_h = output_size
    
    # Destination points in top-down pixels
    dst_coords = np.array([
        [0, output_h],           # 0: tol (bottom-left)
        [output_w, output_h],    # 1: tor (bottom-right)
        [0, 0],                  # 2: point_7 (top-left)
        [output_w, 0],           # 3: point_9 (top-right)
        [output_w // 2, output_h], # 4: tom (bottom-center)
        [output_w // 2, 0]       # 5: bottom_t (top-center)
    ], dtype=np.float32)

    src_pts = []
    dst_pts = []
    
    for i in range(min(len(detected_points), len(dst_coords))):
        if detected_points[i] is not None:
            src_pts.append(detected_points[i])
            dst_pts.append(dst_coords[i])
            
    if len(src_pts) < 4:
        # Cannot solve 3x3 homography matrix with < 4 point correspondences
        print(f"Homography failed: only {len(src_pts)} points available (need at least 4)")
        return None

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    # Compute homography with RANSAC for outlier robustness if we have surplus points
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H



def warp_image_to_court(img, detected_corners, output_size=(500, 1000), draw_lines=True):
    """Warp input image to a top-down court view using detected keypoints.
    
    Applies perspective warp via the computed homography matrix. Optionally
    overlays court lines (boundary, net, service lines, center line) on the
    warped output for visualization.
    
    Args:
        img:               Input image (BGR, any size).
        detected_corners:  List of up to 6 (x, y) tuples (see compute_homography).
        output_size:       (width, height) of top-down output (default: 500×1000).
        draw_lines:        Whether to overlay court line markings (default: True).
    
    Returns:
        np.ndarray: Warped BGR image of shape (height, width, 3), or None
                    if homography computation failed.
    
    Court line positions (official FIP regulations):
        - Net line: center of court (50% of height)
        - Service lines: 6.95m from net = 34.75% of court length
        - Center service line: vertical, between service lines
    """
    H = compute_homography(detected_corners, output_size)
    if H is None:
        return None
    
    output_w, output_h = output_size
    warped = cv2.warpPerspective(img, H, (output_w, output_h))
    
    if draw_lines:
        # Draw court boundary
        cv2.rectangle(warped, (0, 0), (output_w-1, output_h-1), (255, 255, 255), 3)
        # Net line (middle)
        cv2.line(warped, (0, output_h//2), (output_w, output_h//2), (255, 255, 255), 2)
        # Service lines (6.95m from net on 20m court = 34.75%)
        service_dist = int(output_h * 0.3475)
        cv2.line(warped, (0, output_h//2 - service_dist), 
                (output_w, output_h//2 - service_dist), (255, 255, 255), 2)
        cv2.line(warped, (0, output_h//2 + service_dist), 
                (output_w, output_h//2 + service_dist), (255, 255, 255), 2)
        # Center service line
        cv2.line(warped, (output_w//2, output_h//2 - service_dist),
                (output_w//2, output_h//2 + service_dist), (255, 255, 255), 2)
    
    return warped



def warp_point_to_court(point, H):
    """Transform a single point from image coordinates to court coordinates.
    
    Applies the homography matrix H to project an image-space point
    into the canonical top-down court coordinate system.
    
    Args:
        point: (x, y) in image pixel coordinates.
        H:     3×3 homography matrix (image → court).
    
    Returns:
        tuple: (x_court, y_court) in court coordinates, or None if H or point is None.
    """
    if H is None or point is None:
        return None
    pt = np.array([[point[0], point[1], 1]], dtype=np.float32).T
    result = H @ pt
    result = result / result[2]
    return (result[0, 0], result[1, 0])



def warp_point_to_image(point, H):
    """Transform a point from court coordinates back to image coordinates.
    
    Computes the inverse homography H⁻¹ and applies it to map a known
    court-space position back to image pixel coordinates. This is the key
    function for recovering missing keypoints.
    
    Example: If point_7 is not detected, pass its known court position
    (0, 0) and the H matrix computed from the remaining 5 points to get
    the estimated image position of point_7.
    
    Args:
        point: (x, y) in court coordinates (e.g., (0, 0) for point_7).
        H:     3×3 homography matrix (image → court). H⁻¹ is computed internally.
    
    Returns:
        tuple: (x_image, y_image) as integers, or None if H or point is None.
    """
    if H is None or point is None:
        return None
    H_inv = np.linalg.inv(H)
    pt = np.array([[point[0], point[1], 1]], dtype=np.float32).T
    result = H_inv @ pt
    result = result / result[2]
    return (int(result[0, 0]), int(result[1, 0]))
