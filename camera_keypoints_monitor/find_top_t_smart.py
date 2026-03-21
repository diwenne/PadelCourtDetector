#!/usr/bin/env python3
"""
Smart approach: Find center line first, then look for horizontal segments near it.
"""
import sys
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from compute_t_intersection import (
    read_frame,
    detect_lights_off,
    CourtLinePixelDetector,
    CourtLineCandidateDetector,
    pick_horizontal_and_vertical,
    resize_for_display,
    Line,
)
from src.utils import ImagePoint


def find_horizontal_segments_near_vertical(
    lines: List[Line],
    vertical_line: Line,
    w: int,
    h: int,
    max_distance: float = 100.0,
) -> List[Line]:
    """
    Find horizontal line segments that pass near the vertical line.

    This catches service lines even if they're broken by shadows.
    """
    h_lines = []
    cx = w / 2.0

    for ln in lines:
        ang = abs(math.degrees(ln.angle_abs()))

        # Must be nearly horizontal
        if not (ang < 10.0 or ang > 170.0):
            continue

        # Check if this line passes near the vertical line
        # Compute distance from line to vertical line at image center
        inter = ln.intersection(vertical_line)
        if inter is None:
            continue

        x_inter, y_inter = inter

        # Must intersect within the image bounds
        if not (0 <= x_inter < w and 0 <= y_inter < h):
            continue

        # Must be reasonably close to center x
        if abs(x_inter - cx) > max_distance:
            continue

        h_lines.append((ln, y_inter))

    # Sort by y-position (top to bottom)
    h_lines.sort(key=lambda x: x[1])
    return [ln for ln, y in h_lines]


def main():
    # Path to the image
    image_path = Path(__file__).parent.parent.parent / "hard-example.png"

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return 1

    print(f"Processing image: {image_path}")

    # Read and process
    frame = read_frame(str(image_path))
    if detect_lights_off(frame, debug=False):
        print("Lights might be off")
        return 1

    H, W = frame.shape[:2]
    print(f"Image dimensions: {W}x{H}")

    # Estimates
    bottom_estimate = (1330, 1250)
    top_estimate = (1330, 650)

    # 1) Detect line pixels
    print("\n1. Detecting line pixels...")
    px = CourtLinePixelDetector(debug=False).run(frame)

    # 2) Find line candidates near bottom estimate (this works well)
    print("2. Finding line candidates near bottom estimate...")
    cand = CourtLineCandidateDetector(
        estimate_t_intersection=bottom_estimate,
        debug=False
    ).run(px, frame)

    # 3) Find center vertical line with STRICT constraints
    print("3. Finding center vertical line (strict)...")
    h_strict, v_strict = pick_horizontal_and_vertical(cand, W, H)

    if not v_strict:
        print("   ✗ Could not find center vertical line!")
        return 1

    center_v = v_strict[0]  # The center court line
    print(f"   ✓ Found center vertical line at x ≈ {center_v.evaluate_by_y(H/2):.1f}")

    # 4) Now search ENTIRE image for all line candidates
    print("4. Searching entire image for all line candidates...")
    center_estimate = (W // 2, H // 2)
    cand_detector_full = CourtLineCandidateDetector(
        estimate_t_intersection=center_estimate,
        debug=False
    )
    cand_detector_full.line_max_distance_from_estimate = max(W, H)
    all_cand = cand_detector_full.run(px, frame)

    # 5) Find horizontal segments that pass near the center vertical
    print("5. Finding horizontal segments near center line...")
    h_near_center = find_horizontal_segments_near_vertical(
        all_cand, center_v, W, H, max_distance=100
    )
    print(f"   Found {len(h_near_center)} horizontal segments")

    if h_near_center:
        print("\n   Horizontal line positions:")
        for i, hln in enumerate(h_near_center):
            y_pos = hln.evaluate_by_x(W / 2.0)
            inter = hln.intersection(center_v)
            if inter:
                print(f"     Line {i+1}: y = {y_pos:.1f}, intersects at ({inter[0]:.1f}, {inter[1]:.1f})")

    # 6) Find top and bottom T by choosing lines closest to estimates
    print("\n6. Finding top and bottom T-intersections...")

    tom = None
    bottom_t = None

    for hln in h_near_center:
        inter = hln.intersection(center_v)
        if not inter:
            continue

        x, y = inter

        # Classify as top or bottom based on position
        if y < H / 2:  # Top half
            # Is this closer to top estimate than current best?
            dist = math.hypot(x - top_estimate[0], y - top_estimate[1])
            if tom is None or dist < math.hypot(tom[0] - top_estimate[0], tom[1] - top_estimate[1]):
                tom = (x, y, hln, center_v)
        else:  # Bottom half
            # Is this closer to bottom estimate than current best?
            dist = math.hypot(x - bottom_estimate[0], y - bottom_estimate[1])
            if bottom_t is None or dist < math.hypot(bottom_t[0] - bottom_estimate[0], bottom_t[1] - bottom_estimate[1]):
                bottom_t = (x, y, hln, center_v)

    # Display results
    print("\n" + "="*60)
    if bottom_t:
        x, y, _, _ = bottom_t
        dist = math.hypot(x - bottom_estimate[0], y - bottom_estimate[1])
        print(f"✓ BOTTOM T: ({x:.1f}, {y:.1f}) - distance from estimate: {dist:.1f}px")
    else:
        print("✗ No bottom T-intersection found")

    if tom:
        x, y, _, _ = tom
        dist = math.hypot(x - top_estimate[0], y - top_estimate[1])
        print(f"✓ TOP T:    ({x:.1f}, {y:.1f}) - distance from estimate: {dist:.1f}px")
    else:
        print("✗ No top T-intersection found")
    print("="*60)

    # Visualize
    vis = frame.copy()

    # Draw center vertical line (yellow, thick)
    lnN = center_v.normalized()
    p1 = (lnN.p - 4000 * lnN.v).astype(int)
    p2 = (lnN.p + 4000 * lnN.v).astype(int)
    cv.line(vis, tuple(p1), tuple(p2), (0, 255, 255), 3, cv.LINE_AA)

    # Draw horizontal lines near center (green)
    for hln in h_near_center:
        lnN = hln.normalized()
        p1 = (lnN.p - 4000 * lnN.v).astype(int)
        p2 = (lnN.p + 4000 * lnN.v).astype(int)
        cv.line(vis, tuple(p1), tuple(p2), (0, 255, 0), 2, cv.LINE_AA)

    # Draw estimates (cyan circles)
    cv.circle(vis, top_estimate, 8, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(vis, "Top estimate", (top_estimate[0] + 15, top_estimate[1] - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA)

    cv.circle(vis, bottom_estimate, 8, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(vis, "Bottom estimate", (bottom_estimate[0] + 15, bottom_estimate[1] - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA)

    # Draw bottom T (red)
    if bottom_t:
        x, y = int(bottom_t[0]), int(bottom_t[1])
        cv.circle(vis, (x, y), 12, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(vis, f"Bottom T ({x}, {y})", (x + 15, y + 35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

    # Draw top T (blue)
    if tom:
        x, y = int(tom[0]), int(tom[1])
        cv.circle(vis, (x, y), 12, (255, 0, 0), -1, cv.LINE_AA)
        cv.putText(vis, f"Top T ({x}, {y})", (x + 15, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)

    # Show and save
    cv.imshow("Smart T Detection", resize_for_display(vis))
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_smart_t.png"
    cv.imwrite(str(output_path), vis)
    print(f"\n✓ Saved visualization to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
