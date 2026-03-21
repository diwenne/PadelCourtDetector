#!/usr/bin/env python3
"""
Extended version to find top service line with relaxed constraints.
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
    resize_for_display,
    Line,
)
from src.utils import ImagePoint


def pick_horizontal_and_vertical_relaxed(
    lines: List[Line], w: int, h: int,
    min_horizontal_span: float = 0.5,  # Relaxed to 50% of width
    min_vertical_span: float = 0.99,   # Keep vertical at 99%
) -> Tuple[List[Line], List[Line]]:
    """
    Pick horizontal and vertical lines with relaxed constraints for horizontals.

    This allows detection of service lines that may be partially obscured.
    """
    h_lines, v_lines = [], []

    for ln in lines:
        ang = abs(math.degrees(ln.angle_abs()))

        if ang < 10.0 or ang > 170.0:  # near-horizontal
            span = ln.spans_image_extent(w, h, along="x")
            if span >= min_horizontal_span * w:
                h_lines.append((ln, span))  # Store with span for sorting

        elif 80.0 <= ang <= 100.0:     # near-vertical
            span = ln.spans_image_extent(w, h, along="y")
            if span >= min_vertical_span * h:
                v_lines.append(ln)

    # Sort horizontals by y-position (top to bottom)
    cx = w / 2.0
    h_lines.sort(key=lambda x: x[0].evaluate_by_x(cx))
    h_lines = [ln for ln, span in h_lines]  # Remove span from result

    # Sort verticals by x-position (left to right)
    cy = h / 2.0
    v_lines.sort(key=lambda L: L.evaluate_by_y(cy))

    return h_lines, v_lines


def find_top_and_bottom_t(
    h_lines: List[Line],
    v_lines: List[Line],
    w: int,
    h: int,
    bottom_estimate: Optional[ImagePoint] = None,
    top_estimate: Optional[ImagePoint] = None,
) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Find both top and bottom T-intersections.

    Returns (tom, bottom_t) where each is (x, y, h_line, v_line) or None
    """
    if not h_lines or not v_lines:
        return None, None

    # Find center vertical line (usually only one)
    if not v_lines:
        return None, None
    center_v = v_lines[0]  # Assume first/only vertical is center line

    # Find intersections with all horizontal lines
    intersections = []
    for h_ln in h_lines:
        inter = h_ln.intersection(center_v)
        if inter is None:
            continue

        x, y = inter
        # Score by perpendicularity
        vh = h_ln.normalized().v
        vv = center_v.normalized().v
        cos_t = np.clip(abs(np.dot(vh, vv)), -1, 1)
        angle_diff = abs(90.0 - math.degrees(math.acos(cos_t)))

        intersections.append((x, y, h_ln, center_v, angle_diff))

    if not intersections:
        return None, None

    # Separate by top/bottom half
    mid_y = h / 2.0
    top_candidates = [t for t in intersections if t[1] < mid_y]
    bottom_candidates = [t for t in intersections if t[1] >= mid_y]

    # For each half, prefer the one closest to estimate (if provided)
    tom = None
    if top_candidates:
        if top_estimate:
            # Sort by distance to top estimate
            top_candidates.sort(key=lambda t: math.hypot(t[0] - top_estimate[0], t[1] - top_estimate[1]))
        else:
            # Sort by angle quality
            top_candidates.sort(key=lambda t: t[4])
        tom = top_candidates[0][:4]  # (x, y, h_line, v_line)

    bottom_t = None
    if bottom_candidates:
        if bottom_estimate:
            # Sort by distance to bottom estimate
            bottom_candidates.sort(key=lambda t: math.hypot(t[0] - bottom_estimate[0], t[1] - bottom_estimate[1]))
        else:
            # Sort by angle quality
            bottom_candidates.sort(key=lambda t: t[4])
        bottom_t = bottom_candidates[0][:4]  # (x, y, h_line, v_line)

    return tom, bottom_t


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
    top_estimate = (1330, 650)  # Estimate for top service line

    # 1) Detect line pixels
    print("Detecting line pixels...")
    px = CourtLinePixelDetector(debug=False).run(frame)

    # 2) Find line candidates (use center of image to search whole court)
    print("Finding line candidates...")
    # Use image center to avoid filtering out top or bottom lines
    center_estimate = (W // 2, H // 2)
    cand_detector = CourtLineCandidateDetector(
        estimate_t_intersection=center_estimate,
        debug=False
    )
    # Increase search radius to cover whole court
    cand_detector.line_max_distance_from_estimate = max(W, H)  # Search entire image
    cand = cand_detector.run(px, frame)

    # 3) Separate horizontal and vertical lines (RELAXED constraints)
    print("Separating horizontal and vertical lines (relaxed)...")
    h_lines, v_lines = pick_horizontal_and_vertical_relaxed(cand, W, H)
    print(f"Found {len(h_lines)} horizontal lines and {len(v_lines)} vertical lines")

    # Print y-positions of all horizontal lines
    if h_lines:
        print("\nHorizontal line positions (y at center):")
        for i, hln in enumerate(h_lines):
            y_pos = hln.evaluate_by_x(W / 2.0)
            print(f"  Line {i+1}: y = {y_pos:.1f}")

    # 4) Find top and bottom T
    print("\nFinding top and bottom T-intersections...")
    tom, bottom_t = find_top_and_bottom_t(
        h_lines, v_lines, W, H,
        bottom_estimate=bottom_estimate,
        top_estimate=top_estimate
    )

    # Display results
    print("\n" + "="*60)
    if bottom_t:
        x, y, _, _ = bottom_t
        print(f"✓ BOTTOM T-intersection: ({x:.1f}, {y:.1f})")
    else:
        print("✗ No bottom T-intersection found")

    if tom:
        x, y, _, _ = tom
        print(f"✓ TOP T-intersection:    ({x:.1f}, {y:.1f})")
    else:
        print("✗ No top T-intersection found")
    print("="*60)

    # Visualize
    vis = frame.copy()

    # Draw all horizontal lines (green with different shades)
    for i, hln in enumerate(h_lines):
        lnN = hln.normalized()
        p1 = (lnN.p - 4000 * lnN.v).astype(int)
        p2 = (lnN.p + 4000 * lnN.v).astype(int)
        # Vary color intensity based on position
        intensity = 255 if i == 0 else 150
        cv.line(vis, tuple(p1), tuple(p2), (0, intensity, 0), 2, cv.LINE_AA)

    # Draw vertical lines (yellow)
    for vln in v_lines:
        lnN = vln.normalized()
        p1 = (lnN.p - 4000 * lnN.v).astype(int)
        p2 = (lnN.p + 4000 * lnN.v).astype(int)
        cv.line(vis, tuple(p1), tuple(p2), (0, 255, 255), 2, cv.LINE_AA)

    # Draw bottom T (red)
    if bottom_t:
        x, y = int(bottom_t[0]), int(bottom_t[1])
        cv.circle(vis, (x, y), 12, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(vis, f"Bottom T ({x}, {y})", (x + 15, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

    # Draw top T (blue)
    if tom:
        x, y = int(tom[0]), int(tom[1])
        cv.circle(vis, (x, y), 12, (255, 0, 0), -1, cv.LINE_AA)
        cv.putText(vis, f"Top T ({x}, {y})", (x + 15, y + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)

    # Show and save
    cv.imshow("Top & Bottom Service Lines", resize_for_display(vis))
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_top_bottom_t.png"
    cv.imwrite(str(output_path), vis)
    print(f"\n✓ Saved visualization to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
