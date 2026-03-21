#!/usr/bin/env python3
"""
Targeted approach: Focus specifically on the top service line region.
"""
import sys
import math
from pathlib import Path

import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from compute_t_intersection import (
    read_frame,
    CourtLinePixelDetector,
    CLPDParams,
    resize_for_display,
)


def find_service_line_in_region(px, frame, y_min, y_max, x_center, x_tolerance=300):
    """
    Look for horizontal line segments in a specific region.
    """
    H, W = frame.shape[:2]

    # Create ROI mask
    roi = px[y_min:y_max, max(0, x_center-x_tolerance):min(W, x_center+x_tolerance)]

    print(f"  Searching region: y=[{y_min},{y_max}], x=[{x_center-x_tolerance},{x_center+x_tolerance}]")
    print(f"  ROI size: {roi.shape}")
    print(f"  White pixels in ROI: {np.sum(roi > 0)}")

    # Apply Hough in this ROI
    lines = cv.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=50,
        maxLineGap=20
    )

    if lines is None or len(lines) == 0:
        print(f"  No lines found in ROI")
        return None

    print(f"  Found {len(lines)} lines in ROI")

    # Convert back to full image coordinates
    x_offset = max(0, x_center - x_tolerance)
    y_offset = y_min

    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1 += x_offset
        x2 += x_offset
        y1 += y_offset
        y2 += y_offset

        # Check if horizontal
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 10:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        print(f"    Line: ({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f}), angle={angle:.1f}°, len={math.hypot(dx,dy):.0f}")

        # Much more relaxed angle constraint
        if angle > 25 and angle < 155:
            continue

        y_mid = (y1 + y2) / 2
        length = math.hypot(dx, dy)
        segments.append((x1, y1, x2, y2, y_mid, length))

    if not segments:
        print(f"  No horizontal segments after filtering")
        return None

    # Find the dominant y-position (weighted by length)
    y_weighted = sum(s[4] * s[5] for s in segments) / sum(s[5] for s in segments)
    print(f"  Weighted average y: {y_weighted:.1f} from {len(segments)} segments")

    return segments, y_weighted


def main():
    image_path = Path(__file__).parent.parent.parent / "hard-example.png"
    frame = read_frame(str(image_path))
    H, W = frame.shape[:2]
    print(f"Image: {W}x{H}\n")

    # Detect line pixels with moderate sensitivity
    params = CLPDParams(
        threshold=70,
        diffThreshold=15,
        t=8,
        gradientKernelSize=3,
        kernelSize=21
    )

    print("Detecting line pixels...")
    px = CourtLinePixelDetector(params=params, debug=False).run(frame)

    # Known good values
    center_x = 1330
    bottom_y_expected = 1280

    # Define search regions more precisely
    # Net is at y≈480
    # Top service line should be around y=600-750
    # Bottom service line should be around y=1220-1320

    print("\nSearching for TOP service line...")
    top_result = find_service_line_in_region(
        px, frame,
        y_min=580,
        y_max=780,
        x_center=center_x,
        x_tolerance=400
    )

    print("\nSearching for BOTTOM service line...")
    bottom_result = find_service_line_in_region(
        px, frame,
        y_min=1200,
        y_max=1350,
        x_center=center_x,
        x_tolerance=400
    )

    # Results
    print("\n" + "="*60)
    tom = None
    bottom_t = None

    if bottom_result:
        segments, y = bottom_result
        bottom_t = (center_x, y)
        print(f"✓ BOTTOM T: ({center_x}, {int(y)})")
    else:
        print("✗ No bottom service line found")

    if top_result:
        segments, y = top_result
        tom = (center_x, y)
        print(f"✓ TOP T:    ({center_x}, {int(y)})")
    else:
        print("✗ No top service line found")
    print("="*60)

    # Visualize
    vis = frame.copy()

    # Draw center line
    cv.line(vis, (center_x, 0), (center_x, H), (0, 255, 255), 2)

    # Draw search regions
    cv.rectangle(vis, (center_x-400, 580), (center_x+400, 780), (255, 100, 0), 2)
    cv.rectangle(vis, (center_x-400, 1200), (center_x+400, 1350), (255, 100, 0), 2)

    # Draw detected segments
    if top_result:
        segments, y_avg = top_result
        for seg in segments:
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            cv.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw T
        x, y = int(tom[0]), int(tom[1])
        cv.circle(vis, (x, y), 15, (255, 0, 0), -1)
        cv.putText(vis, f"Top T ({x}, {y})", (x + 20, y - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if bottom_result:
        segments, y_avg = bottom_result
        for seg in segments:
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw T
        x, y = int(bottom_t[0]), int(bottom_t[1])
        cv.circle(vis, (x, y), 15, (0, 0, 255), -1)
        cv.putText(vis, f"Bottom T ({x}, {y})", (x + 20, y + 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv.imshow("Targeted Detection", resize_for_display(vis))
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_targeted.png"
    cv.imwrite(str(output_path), vis)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
