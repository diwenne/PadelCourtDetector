#!/usr/bin/env python3
"""
Use morphological operations to connect broken line segments.
"""
import sys
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


def main():
    image_path = Path(__file__).parent.parent.parent / "hard-example.png"
    frame = read_frame(str(image_path))
    H, W = frame.shape[:2]

    print(f"Image: {W}x{H}\n")

    # More sensitive detection
    params = CLPDParams(
        threshold=65,
        diffThreshold=12,
        t=8,
        gradientKernelSize=3,
        kernelSize=21
    )

    print("Step 1: Detecting line pixels...")
    px = CourtLinePixelDetector(params=params, debug=False).run(frame)

    print("Step 2: Applying morphological closing to connect broken lines...")

    # Horizontal closing - connect horizontally broken lines
    kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (60, 1))  # Wide horizontal kernel
    closed_h = cv.morphologyEx(px, cv.MORPH_CLOSE, kernel_h)

    # Vertical closing - connect vertically broken lines
    kernel_v = cv.getStructuringElement(cv.MORPH_RECT, (1, 60))  # Tall vertical kernel
    closed_v = cv.morphologyEx(px, cv.MORPH_CLOSE, kernel_v)

    # Combine
    closed = cv.bitwise_or(closed_h, closed_v)

    print("Step 3: Running Hough on closed image...")
    lines = cv.HoughLinesP(
        closed,
        rho=1,
        theta=np.pi/180,
        threshold=100,  # Higher threshold for longer lines
        minLineLength=200,  # Longer minimum length
        maxLineGap=50  # Allow larger gaps
    )

    if lines is None:
        print("No lines found!")
        return 1

    print(f"Found {len(lines)} lines")

    # Separate horizontal and vertical
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 10:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        length = np.hypot(dx, dy)
        y_mid = (y1 + y2) / 2
        x_mid = (x1 + x2) / 2

        if angle < 15 or angle > 165:  # Horizontal
            h_lines.append((x1, y1, x2, y2, y_mid, length))
        elif 75 < angle < 105:  # Vertical
            v_lines.append((x1, y1, x2, y2, x_mid, length))

    print(f"  Horizontal: {len(h_lines)}")
    print(f"  Vertical: {len(v_lines)}")

    # Sort and display
    h_lines.sort(key=lambda x: x[4])  # Sort by y
    v_lines.sort(key=lambda x: x[4])  # Sort by x

    print("\nHorizontal lines (sorted by y):")
    for i, (x1, y1, x2, y2, y_mid, length) in enumerate(h_lines, 1):
        print(f"  {i}: y={y_mid:7.1f}, length={length:6.1f}, x=[{min(x1,x2):7.1f}, {max(x1,x2):7.1f}]")

    print("\nVertical lines (sorted by x):")
    for i, (x1, y1, x2, y2, x_mid, length) in enumerate(v_lines, 1):
        print(f"  {i}: x={x_mid:7.1f}, length={length:6.1f}, y=[{min(y1,y2):7.1f}, {max(y1,y2):7.1f}]")

    # Find center vertical (closest to x=1330)
    center_x_target = 1330
    if v_lines:
        v_lines_sorted = sorted(v_lines, key=lambda x: abs(x[4] - center_x_target))
        best_v = v_lines_sorted[0]
        center_x = best_v[4]
        print(f"\n✓ Center vertical at x={center_x:.1f}")
    else:
        center_x = center_x_target
        print(f"\n✗ No center vertical found, using estimate x={center_x}")

    # Look for service lines
    top_y = None
    bottom_y = None

    for x1, y1, x2, y2, y_mid, length in h_lines:
        # Check if this line crosses center
        x_min, x_max = min(x1, x2), max(x1, x2)
        if not (x_min <= center_x <= x_max):
            continue

        if 600 <= y_mid <= 800:  # Top service line region
            if top_y is None or abs(y_mid - 680) < abs(top_y - 680):
                top_y = y_mid

        if 1200 <= y_mid <= 1350:  # Bottom service line region
            if bottom_y is None or abs(y_mid - 1280) < abs(bottom_y - 1280):
                bottom_y = y_mid

    print("\n" + "="*60)
    if bottom_y:
        print(f"✓ BOTTOM T: ({center_x:.0f}, {bottom_y:.0f})")
    else:
        print("✗ No bottom service line found")

    if top_y:
        print(f"✓ TOP T:    ({center_x:.0f}, {top_y:.0f})")
    else:
        print("✗ No top service line found")
    print("="*60)

    # Visualize
    # Show before/after morphology
    vis_before = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    vis_before = cv.cvtColor(vis_before, cv.COLOR_GRAY2BGR)
    vis_before[px > 0] = [0, 0, 255]

    vis_after = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    vis_after = cv.cvtColor(vis_after, cv.COLOR_GRAY2BGR)
    vis_after[closed > 0] = [0, 255, 0]

    cv.imshow("Before Morphology (red)", resize_for_display(vis_before))
    cv.imshow("After Morphology (green)", resize_for_display(vis_after))

    # Draw detected lines
    vis_lines = frame.copy()

    # Draw horizontal lines
    for x1, y1, x2, y2, y_mid, length in h_lines:
        if 600 <= y_mid <= 800:
            color = (255, 0, 0)  # Blue for top
            thickness = 3
        elif 1200 <= y_mid <= 1350:
            color = (0, 0, 255)  # Red for bottom
            thickness = 3
        else:
            color = (0, 200, 0)  # Green for others
            thickness = 1

        cv.line(vis_lines, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    # Draw vertical lines
    for x1, y1, x2, y2, x_mid, length in v_lines:
        cv.line(vis_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    # Draw T intersections
    if bottom_y:
        cv.circle(vis_lines, (int(center_x), int(bottom_y)), 15, (0, 0, 255), -1)
        cv.putText(vis_lines, f"Bottom T", (int(center_x) + 20, int(bottom_y) + 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if top_y:
        cv.circle(vis_lines, (int(center_x), int(top_y)), 15, (255, 0, 0), -1)
        cv.putText(vis_lines, f"Top T", (int(center_x) + 20, int(top_y) - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv.imshow("Detected Lines", resize_for_display(vis_lines))

    print("\nPress any key to close...")
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_morphology.png"
    cv.imwrite(str(output_path), vis_lines)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
