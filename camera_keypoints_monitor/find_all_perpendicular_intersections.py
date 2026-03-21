#!/usr/bin/env python3
"""
Find all pairs of lines intersecting at ~90 degrees and mark them.
"""
import sys
import math
from pathlib import Path
from typing import List, Tuple

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


def find_all_perpendicular_intersections(
    lines: List[Line],
    w: int,
    h: int,
    angle_tolerance: float = 10.0,  # Degrees away from 90°
) -> List[Tuple[float, float, float, Line, Line]]:
    """
    Find all pairs of lines that intersect at approximately 90 degrees.

    Returns list of (x, y, angle_error, line1, line2) tuples.
    """
    intersections = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]

            # Check perpendicularity
            v1 = line1.normalized().v
            v2 = line2.normalized().v

            cos_angle = np.clip(abs(np.dot(v1, v2)), 0, 1)
            angle_deg = math.degrees(math.acos(cos_angle))
            angle_error = abs(90.0 - angle_deg)

            if angle_error > angle_tolerance:
                continue

            # Find intersection point
            inter = line1.intersection(line2)
            if inter is None:
                continue

            x, y = inter

            # Check if intersection is within image bounds
            if not (0 <= x < w and 0 <= y < h):
                continue

            intersections.append((x, y, angle_error, line1, line2))

    # Sort by angle quality (best perpendicularity first)
    intersections.sort(key=lambda x: x[2])

    return intersections


def main():
    image_path = Path(__file__).parent.parent.parent / "hard-example.png"

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return 1

    print(f"Processing: {image_path.name}")

    # Read image
    frame = read_frame(str(image_path))
    if detect_lights_off(frame, debug=False):
        print("Lights might be off")
        return 1

    H, W = frame.shape[:2]
    print(f"Image size: {W}x{H}\n")

    # Use the existing method to detect line pixels
    print("Step 1: Detecting line pixels...")
    px = CourtLinePixelDetector(debug=False).run(frame)

    # Use existing method to find line candidates
    # Search entire image by using center estimate and large radius
    print("Step 2: Finding line candidates (searching entire image)...")
    center_estimate = (W // 2, H // 2)
    cand_detector = CourtLineCandidateDetector(
        estimate_t_intersection=center_estimate,
        debug=False
    )
    cand_detector.line_max_distance_from_estimate = max(W, H)

    all_lines = cand_detector.run(px, frame)
    print(f"Found {len(all_lines)} line candidates")

    # Find all perpendicular intersections
    print("\nStep 3: Finding all perpendicular intersections (angle tolerance = 10°)...")
    intersections = find_all_perpendicular_intersections(all_lines, W, H, angle_tolerance=10.0)

    print(f"Found {len(intersections)} perpendicular intersections\n")

    # Print all intersections
    print("All perpendicular intersections:")
    print("-" * 60)
    for i, (x, y, angle_err, _, _) in enumerate(intersections, 1):
        print(f"{i:3d}. ({x:7.1f}, {y:7.1f})  angle_error={angle_err:5.2f}°")
    print("-" * 60)

    # Visualize
    vis = frame.copy()

    # Draw all detected lines (thin gray)
    for ln in all_lines:
        lnN = ln.normalized()
        p1 = (lnN.p - 2000 * lnN.v).astype(int)
        p2 = (lnN.p + 2000 * lnN.v).astype(int)
        cv.line(vis, tuple(p1), tuple(p2), (100, 100, 100), 1, cv.LINE_AA)

    # Draw all intersection points
    # Use different colors based on position
    for i, (x, y, angle_err, line1, line2) in enumerate(intersections, 1):
        xi, yi = int(round(x)), int(round(y))

        # Color based on vertical position
        if y < H / 3:
            color = (255, 0, 0)  # Blue for top
        elif y < 2 * H / 3:
            color = (0, 255, 255)  # Yellow for middle
        else:
            color = (0, 0, 255)  # Red for bottom

        # Draw circle
        radius = 10
        cv.circle(vis, (xi, yi), radius, color, -1, cv.LINE_AA)
        cv.circle(vis, (xi, yi), radius + 2, (255, 255, 255), 2, cv.LINE_AA)

        # Draw number
        label = str(i)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        # Get text size for background
        (text_width, text_height), baseline = cv.getTextSize(label, font, font_scale, thickness)

        # Put text with background
        text_x = xi + radius + 5
        text_y = yi + 5

        # Draw background rectangle
        cv.rectangle(vis,
                    (text_x - 2, text_y - text_height - 2),
                    (text_x + text_width + 2, text_y + baseline + 2),
                    (0, 0, 0), -1)

        # Draw text
        cv.putText(vis, label, (text_x, text_y),
                  font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    # Add legend
    legend_y = 50
    cv.putText(vis, "Blue = Top third", (20, legend_y),
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv.LINE_AA)
    cv.putText(vis, "Yellow = Middle third", (20, legend_y + 40),
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(vis, "Red = Bottom third", (20, legend_y + 80),
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

    cv.putText(vis, f"Total: {len(intersections)} intersections", (20, legend_y + 140),
              cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    # Show and save
    cv.imshow("All Perpendicular Intersections", resize_for_display(vis))
    print("\nPress any key to close...")
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_all_intersections.png"
    cv.imwrite(str(output_path), vis)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
