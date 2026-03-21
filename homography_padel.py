"""
Homography computation for padel court using 4 corner keypoints.
Computes perspective transform from detected corners to standard court view.
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
    """
    Warp input image to top-down court view using detected corners.
    
    Args:
        img: Input image (BGR)
        detected_corners: List of 4 (x, y) tuples
        output_size: (width, height) of output, defaults to 500x1000
        draw_lines: Whether to draw court lines on output
    
    Returns:
        Warped image showing top-down court view, or None if homography failed
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
    """Transform a point from image coordinates to court coordinates."""
    if H is None or point is None:
        return None
    pt = np.array([[point[0], point[1], 1]], dtype=np.float32).T
    result = H @ pt
    result = result / result[2]
    return (result[0, 0], result[1, 0])


def warp_point_to_image(point, H):
    """Transform a point from court coordinates to image coordinates."""
    if H is None or point is None:
        return None
    H_inv = np.linalg.inv(H)
    pt = np.array([[point[0], point[1], 1]], dtype=np.float32).T
    result = H_inv @ pt
    result = result / result[2]
    return (int(result[0, 0]), int(result[1, 0]))
