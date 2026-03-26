"""
Generic homography computation for court keypoint detection.
Maps detected image keypoints to a canonical top-down court view.
"""
import numpy as np
import cv2

def compute_homography(detected_points, court_ref, output_size=(500, 1000)):
    """
    Compute homography matrix from detected keypoints to top-down rectangular view.
    
    Args:
        detected_points: List of (x, y) tuples or None.
        court_ref: CourtReference object.
        output_size: (width, height) for mapping in pixels.
    """
    output_w, output_h = output_size
    
    # Reference points are in mm, we map them to pixels (0..output_w, 0..output_h)
    ref_points = np.array(court_ref.key_points, dtype=np.float32)
    
    # Normalize ref points to [0, 1] then scale to pixels
    dst_coords = np.zeros_like(ref_points)
    dst_coords[:, 0] = ref_points[:, 0] / court_ref.court_width * output_w
    dst_coords[:, 1] = ref_points[:, 1] / court_ref.court_length * output_h

    src_pts = []
    dst_pts = []
    
    for i in range(min(len(detected_points), len(dst_coords))):
        if detected_points[i] is not None:
            src_pts.append(detected_points[i])
            dst_pts.append(dst_coords[i])
            
    if len(src_pts) < 4:
        return None

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warp_point_to_image(point_in_court_pixels, H):
    """
    Transform a point in top-down court pixels back to image coordinates.
    """
    if H is None or point_in_court_pixels is None:
        return None
    H_inv = np.linalg.inv(H)
    pt = np.array([[point_in_court_pixels[0], point_in_court_pixels[1], 1]], dtype=np.float32).T
    result = H_inv @ pt
    result = result / result[2]
    return (int(result[0, 0]), int(result[1, 0]))
