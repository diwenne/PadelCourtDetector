#!/usr/bin/env python3
"""
Aggressive approach: Try multiple strategies to find the top service line.
"""
import sys
import math
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from compute_t_intersection import (
    read_frame,
    detect_lights_off,
    CourtLinePixelDetector,
    CLPDParams,
    CourtLineCandidateDetector,
    CLCDParams,
    pick_horizontal_and_vertical,
    resize_for_display,
    Line,
)
from src.utils import ImagePoint


def enhance_image_for_lines(frame: np.ndarray) -> np.ndarray:
    """Enhance the image to make white lines more visible."""
    # Convert to LAB color space
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    # Apply CLAHE to L channel
    clahe = cv.CLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge and convert back
    enhanced_lab = cv.merge([l_enhanced, a, b])
    enhanced = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

    return enhanced


def find_all_horizontal_segments(
    lines: List[Line],
    w: int,
    h: int,
    min_angle: float = -15.0,
    max_angle: float = 15.0,
) -> List[Tuple[Line, float]]:
    """
    Find ALL horizontal line segments, even short ones.
    Returns list of (line, y_center) tuples.
    """
    h_segments = []

    for ln in lines:
        ang_rad = ln.angle_abs()
        ang_deg = math.degrees(ang_rad)

        # Normalize angle to -180 to 180
        if ang_deg > 90:
            ang_deg = ang_deg - 180

        # Check if nearly horizontal
        if min_angle <= ang_deg <= max_angle:
            # Get y position at image center
            y_pos = ln.evaluate_by_x(w / 2.0)
            if 0 <= y_pos < h:
                h_segments.append((ln, y_pos))

    # Sort by y position
    h_segments.sort(key=lambda x: x[1])
    return h_segments


def cluster_lines_by_y(
    segments: List[Tuple[Line, float]],
    tolerance: float = 20.0
) -> List[List[Tuple[Line, float]]]:
    """
    Cluster line segments that are close in y-coordinate.
    """
    if not segments:
        return []

    clusters = []
    current_cluster = [segments[0]]

    for i in range(1, len(segments)):
        prev_y = segments[i-1][1]
        curr_y = segments[i][1]

        if abs(curr_y - prev_y) <= tolerance:
            current_cluster.append(segments[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [segments[i]]

    clusters.append(current_cluster)
    return clusters


def main():
    image_path = Path(__file__).parent.parent.parent / "hard-example.png"

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return 1

    print(f"Processing: {image_path.name}")

    # Read image
    frame = read_frame(str(image_path))
    H, W = frame.shape[:2]
    print(f"Image size: {W}x{H}")

    # Estimates
    bottom_estimate = (1330, 1250)
    top_estimate = (1330, 650)

    print("\n" + "="*60)
    print("STRATEGY 1: Enhanced image with relaxed parameters")
    print("="*60)

    # Enhance image
    enhanced = enhance_image_for_lines(frame)

    # More sensitive line pixel detection
    sensitive_params = CLPDParams(
        threshold=70,  # Lower from 80
        diffThreshold=15,  # Lower from 20
        t=8,
        gradientKernelSize=3,
        kernelSize=21
    )

    px = CourtLinePixelDetector(params=sensitive_params, debug=False).run(enhanced)

    # More aggressive Hough parameters
    aggressive_hough = CLCDParams(
        houghThreshold=30,  # Lower from 50
        refinementIterations=50
    )

    # Search entire image
    cand_detector = CourtLineCandidateDetector(
        estimate_t_intersection=(W//2, H//2),
        params=aggressive_hough,
        debug=False
    )
    cand_detector.line_max_distance_from_estimate = max(W, H)
    cand_detector.min_lines_to_consider = 50  # Consider more lines

    all_lines = cand_detector.run(px, enhanced)
    print(f"Found {len(all_lines)} total line candidates")

    # Find center vertical (strict)
    print("\nFinding center vertical line...")
    cand_bottom = CourtLineCandidateDetector(
        estimate_t_intersection=bottom_estimate,
        debug=False
    ).run(px, frame)

    _, v_lines = pick_horizontal_and_vertical(cand_bottom, W, H)

    if not v_lines:
        print("✗ Could not find center vertical line!")
        return 1

    center_v = v_lines[0]
    center_x = center_v.evaluate_by_y(H/2)
    print(f"✓ Center line at x ≈ {center_x:.1f}")

    # Find ALL horizontal segments
    print("\nFinding all horizontal segments...")
    h_segments = find_all_horizontal_segments(all_lines, W, H)
    print(f"Found {len(h_segments)} horizontal segments")

    # Cluster by y-position
    print("\nClustering segments by y-position...")
    clusters = cluster_lines_by_y(h_segments, tolerance=30)
    print(f"Found {len(clusters)} clusters:")

    for i, cluster in enumerate(clusters):
        y_avg = np.mean([y for _, y in cluster])
        print(f"  Cluster {i+1}: {len(cluster):2d} segments at y ≈ {y_avg:.1f}")

    # Find intersections with center line for each cluster
    print("\nFinding T-intersections from clusters...")
    cluster_intersections = []

    for cluster in clusters:
        # Take the longest line in the cluster
        best_line = max(cluster, key=lambda x: x[0].spans_image_extent(W, H, "x"))
        line, y_pos = best_line

        inter = line.intersection(center_v)
        if inter and 0 <= inter[0] < W and 0 <= inter[1] < H:
            cluster_intersections.append((inter[0], inter[1], line, center_v))
            print(f"  Cluster at y ≈ {y_pos:.1f} -> T at ({inter[0]:.1f}, {inter[1]:.1f})")

    # Identify top and bottom T
    tom = None
    bottom_t = None

    # Define regions
    # Net is around y=484, so top service line should be between 520-800
    # Bottom service line should be between 1000-1400

    top_region = (520, 800)
    bottom_region = (1000, 1400)

    print(f"\nLooking for top T in region y={top_region[0]}-{top_region[1]}")
    print(f"Looking for bottom T in region y={bottom_region[0]}-{bottom_region[1]}")

    for x, y, h_line, v_line in cluster_intersections:
        if top_region[0] <= y <= top_region[1]:
            if tom is None or abs(y - top_estimate[1]) < abs(tom[1] - top_estimate[1]):
                tom = (x, y, h_line, v_line)

        if bottom_region[0] <= y <= bottom_region[1]:
            if bottom_t is None or abs(y - bottom_estimate[1]) < abs(bottom_t[1] - bottom_estimate[1]):
                bottom_t = (x, y, h_line, v_line)

    # Results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)

    if bottom_t:
        x, y = bottom_t[0], bottom_t[1]
        print(f"✓ BOTTOM T: ({x:.1f}, {y:.1f})")
    else:
        print("✗ No bottom T found in expected region")

    if tom:
        x, y = tom[0], tom[1]
        print(f"✓ TOP T:    ({x:.1f}, {y:.1f})")
    else:
        print("✗ No top T found in expected region")

    print("="*60)

    # Visualize
    vis = frame.copy()

    # Draw center vertical (thick yellow)
    lnN = center_v.normalized()
    p1 = (lnN.p - 4000 * lnN.v).astype(int)
    p2 = (lnN.p + 4000 * lnN.v).astype(int)
    cv.line(vis, tuple(p1), tuple(p2), (0, 255, 255), 3, cv.LINE_AA)

    # Draw all cluster lines (thin green)
    for cluster in clusters:
        for line, y_pos in cluster:
            lnN = line.normalized()
            p1 = (lnN.p - 2000 * lnN.v).astype(int)
            p2 = (lnN.p + 2000 * lnN.v).astype(int)
            cv.line(vis, tuple(p1), tuple(p2), (0, 200, 0), 1, cv.LINE_AA)

    # Draw region boundaries
    cv.line(vis, (0, top_region[0]), (W, top_region[0]), (255, 100, 0), 2, cv.LINE_AA)
    cv.line(vis, (0, top_region[1]), (W, top_region[1]), (255, 100, 0), 2, cv.LINE_AA)
    cv.line(vis, (0, bottom_region[0]), (W, bottom_region[0]), (255, 100, 0), 2, cv.LINE_AA)
    cv.line(vis, (0, bottom_region[1]), (W, bottom_region[1]), (255, 100, 0), 2, cv.LINE_AA)

    # Draw estimates (cyan)
    cv.circle(vis, top_estimate, 8, (255, 255, 0), 2, cv.LINE_AA)
    cv.circle(vis, bottom_estimate, 8, (255, 255, 0), 2, cv.LINE_AA)

    # Draw detected T's
    if bottom_t:
        x, y = int(bottom_t[0]), int(bottom_t[1])
        cv.circle(vis, (x, y), 15, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(vis, f"Bottom T ({x}, {y})", (x + 20, y + 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

    if tom:
        x, y = int(tom[0]), int(tom[1])
        cv.circle(vis, (x, y), 15, (255, 0, 0), -1, cv.LINE_AA)
        cv.putText(vis, f"Top T ({x}, {y})", (x + 20, y - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv.LINE_AA)

    # Show and save
    cv.imshow("Aggressive Detection", resize_for_display(vis))
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_aggressive.png"
    cv.imwrite(str(output_path), vis)
    print(f"\n✓ Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
