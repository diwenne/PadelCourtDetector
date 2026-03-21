#!/usr/bin/env python3
"""
Extension to find multiple T-intersections (top and bottom service lines).
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


def find_multiple_t_intersections(
    h_lines: List[Line],
    v_lines: List[Line],
    w: int,
    h: int,
    estimate: Optional[ImagePoint] = None,
    distance_bias_deg: float = 5.0,
    top_n: int = 2,
) -> List[Tuple[float, float, Line, Line, float]]:
    """
    Find multiple T-intersections, sorted by score.

    Returns list of (x, y, h_line, v_line, score) tuples.
    """
    if not h_lines or not v_lines:
        return []

    diag = math.hypot(w, h)
    candidates = []

    for hln in h_lines:
        vh = hln.normalized().v
        for vln in v_lines:
            vv = vln.normalized().v
            cos_t = np.clip(abs(np.dot(vh, vv)), -1, 1)
            angle_diff = abs(90.0 - math.degrees(math.acos(cos_t)))

            inter = hln.intersection(vln)
            if inter is None:
                continue

            # Soft prior toward estimate (if provided)
            prior = 0.0
            if estimate is not None and diag > 0:
                ex, ey = estimate
                ix, iy = inter
                prior = distance_bias_deg * (math.hypot(ix - ex, iy - ey) / diag)

            score = angle_diff + prior
            candidates.append((inter[0], inter[1], hln, vln, score))

    # Sort by score (best first) and return top N
    candidates.sort(key=lambda x: x[4])
    return candidates[:top_n]


def filter_top_and_bottom_t(
    intersections: List[Tuple[float, float, Line, Line, float]],
    h: int,
) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Separate top and bottom T-intersections based on y-coordinate.

    Returns (tom, bottom_t) where each is (x, y, h_line, v_line, score) or None
    """
    if not intersections:
        return None, None

    # Split by image center
    top_candidates = [t for t in intersections if t[1] < h / 2]
    bottom_candidates = [t for t in intersections if t[1] >= h / 2]

    # Get best from each half
    tom = top_candidates[0] if top_candidates else None
    bottom_t = bottom_candidates[0] if bottom_candidates else None

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

    # Estimate for bottom T (you can provide top estimate too)
    bottom_estimate = (1330, 1250)

    # 1) Detect line pixels
    print("Detecting line pixels...")
    px = CourtLinePixelDetector(debug=False).run(frame)

    # 2) Find line candidates
    print("Finding line candidates...")
    cand = CourtLineCandidateDetector(
        estimate_t_intersection=bottom_estimate,
        debug=False
    ).run(px, frame)

    # 3) Separate horizontal and vertical lines
    print("Separating horizontal and vertical lines...")
    h_lines, v_lines = pick_horizontal_and_vertical(cand, W, H)
    print(f"Found {len(h_lines)} horizontal lines and {len(v_lines)} vertical lines")

    # 4) Find multiple T-intersections
    print("\nFinding multiple T-intersections...")
    intersections = find_multiple_t_intersections(
        h_lines, v_lines, W, H,
        estimate=bottom_estimate,
        top_n=10  # Get top 10 candidates
    )

    # 5) Separate top and bottom
    tom, bottom_t = filter_top_and_bottom_t(intersections, H)

    # Display results
    print("\n" + "="*60)
    if bottom_t:
        x, y, _, _, score = bottom_t
        print(f"✓ BOTTOM T-intersection: ({x:.1f}, {y:.1f}) [score: {score:.2f}]")
    else:
        print("✗ No bottom T-intersection found")

    if tom:
        x, y, _, _, score = tom
        print(f"✓ TOP T-intersection:    ({x:.1f}, {y:.1f}) [score: {score:.2f}]")
    else:
        print("✗ No top T-intersection found")
    print("="*60)

    # Visualize
    vis = frame.copy()

    # Draw horizontal lines (green)
    for hln in h_lines:
        lnN = hln.normalized()
        p1 = (lnN.p - 4000 * lnN.v).astype(int)
        p2 = (lnN.p + 4000 * lnN.v).astype(int)
        cv.line(vis, tuple(p1), tuple(p2), (0, 255, 0), 2, cv.LINE_AA)

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
        cv.putText(vis, "Bottom T", (x + 15, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

    # Draw top T (blue)
    if tom:
        x, y = int(tom[0]), int(tom[1])
        cv.circle(vis, (x, y), 12, (255, 0, 0), -1, cv.LINE_AA)
        cv.putText(vis, "Top T", (x + 15, y + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv.LINE_AA)

    # Show and save
    cv.imshow("Court Lines with Top & Bottom T", resize_for_display(vis))
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_both_t.png"
    cv.imwrite(str(output_path), vis)
    print(f"\n✓ Saved visualization to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
