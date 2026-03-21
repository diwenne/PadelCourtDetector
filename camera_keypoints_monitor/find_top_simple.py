#!/usr/bin/env python3
"""
Simple aggressive approach: Lower thresholds, collect all segments.
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
    CourtLinePixelDetector,
    CLPDParams,
    resize_for_display,
)
from src.utils import ImagePoint


def main():
    image_path = Path(__file__).parent.parent.parent / "hard-example.png"
    frame = read_frame(str(image_path))
    H, W = frame.shape[:2]
    print(f"Image: {W}x{H}")

    # More sensitive line detection
    params = CLPDParams(
        threshold=65,        # Lower from 80
        diffThreshold=12,    # Lower from 20
        t=8,
        gradientKernelSize=3,
        kernelSize=21
    )

    print("\nDetecting line pixels with lower thresholds...")
    px = CourtLinePixelDetector(params=params, debug=False).run(frame)

    print("Running Hough with aggressive parameters...")
    # Very aggressive Hough
    lines = cv.HoughLinesP(
        px,
        rho=1,
        theta=np.pi/180,
        threshold=25,      # Very low threshold
        minLineLength=30,  # Very short lines OK
        maxLineGap=15      # Allow bigger gaps
    )

    if lines is None:
        print("No lines found!")
        return 1

    print(f"Found {len(lines)} line segments")

    # Collect horizontal segments
    h_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if horizontal
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))

        # Normalize to -90 to 90
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        if abs(angle) < 15:  # Within 15 degrees of horizontal
            y_mid = (y1 + y2) / 2
            length = math.hypot(dx, dy)
            h_segments.append((x1, y1, x2, y2, y_mid, length))

    print(f"Found {len(h_segments)} horizontal segments")

    # Group by y position
    h_segments.sort(key=lambda x: x[4])  # Sort by y_mid

    # Cluster segments
    clusters = []
    if h_segments:
        current = [h_segments[0]]
        for seg in h_segments[1:]:
            if abs(seg[4] - current[-1][4]) < 30:
                current.append(seg)
            else:
                clusters.append(current)
                current = [seg]
        clusters.append(current)

    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        y_avg = np.mean([s[4] for s in cluster])
        total_len = sum([s[5] for s in cluster])
        print(f"  Cluster {i+1}: {len(cluster):3d} segments at y={y_avg:6.1f}, total_length={total_len:.0f}px")

    # Find center x
    center_x = W / 2.0

    # Look for top T in range 550-800, bottom T in range 1100-1400
    top_range = (550, 800)
    bottom_range = (1100, 1400)

    top_cluster = None
    bottom_cluster = None

    for cluster in clusters:
        y_avg = np.mean([s[4] for s in cluster])

        if top_range[0] <= y_avg <= top_range[1]:
            if top_cluster is None:
                top_cluster = (cluster, y_avg)

        if bottom_range[0] <= y_avg <= bottom_range[1]:
            if bottom_cluster is None or abs(y_avg - 1250) < abs(bottom_cluster[1] - 1250):
                bottom_cluster = (cluster, y_avg)

    print("\n" + "="*60)
    if bottom_cluster:
        y = bottom_cluster[1]
        print(f"✓ BOTTOM service line: y ≈ {y:.1f}")
    else:
        print("✗ No bottom service line found")

    if top_cluster:
        y = top_cluster[1]
        print(f"✓ TOP service line: y ≈ {y:.1f}")
    else:
        print("✗ No top service line found")
    print("="*60)

    # Visualize
    vis = frame.copy()

    # Draw all horizontal segments (faint green)
    for seg in h_segments:
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        cv.line(vis, (x1, y1), (x2, y2), (0, 150, 0), 1)

    # Draw range boundaries
    cv.line(vis, (0, top_range[0]), (W, top_range[0]), (255, 100, 0), 2)
    cv.line(vis, (0, top_range[1]), (W, top_range[1]), (255, 100, 0), 2)
    cv.line(vis, (0, bottom_range[0]), (W, bottom_range[0]), (255, 100, 0), 2)
    cv.line(vis, (0, bottom_range[1]), (W, bottom_range[1]), (255, 100, 0), 2)

    # Highlight detected clusters
    if top_cluster:
        cluster, y_avg = top_cluster
        for seg in cluster:
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            cv.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw T
        x, y = int(center_x), int(y_avg)
        cv.circle(vis, (x, y), 15, (255, 0, 0), -1)
        cv.putText(vis, f"Top T ({x}, {y})", (x + 20, y - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if bottom_cluster:
        cluster, y_avg = bottom_cluster
        for seg in cluster:
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw T
        x, y = int(center_x), int(y_avg)
        cv.circle(vis, (x, y), 15, (0, 0, 255), -1)
        cv.putText(vis, f"Bottom T ({x}, {y})", (x + 20, y + 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw center line estimate
    cv.line(vis, (int(center_x), 0), (int(center_x), H), (0, 255, 255), 2)

    cv.imshow("Simple Aggressive", resize_for_display(vis))
    cv.waitKey(0)
    cv.destroyAllWindows()

    output_path = image_path.parent / f"{image_path.stem}_simple_aggressive.png"
    cv.imwrite(str(output_path), vis)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
