#!/usr/bin/env python3
"""
Visualize the line pixel detection to see what's being found.
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

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

    # Try multiple threshold levels
    thresholds = [80, 70, 60, 50]

    for thresh in thresholds:
        print(f"\nTrying threshold={thresh}...")

        params = CLPDParams(
            threshold=thresh,
            diffThreshold=15,
            t=8,
            gradientKernelSize=3,
            kernelSize=21
        )

        px = CourtLinePixelDetector(params=params, debug=False).run(frame)

        # Check top service line region
        top_roi = px[580:780, 930:1730]
        top_pixels = np.sum(top_roi > 0)

        # Check bottom service line region
        bottom_roi = px[1200:1350, 930:1730]
        bottom_pixels = np.sum(bottom_roi > 0)

        print(f"  Top region (y=580-780):    {top_pixels:5d} white pixels")
        print(f"  Bottom region (y=1200-1350): {bottom_pixels:5d} white pixels")

        # Visualize
        vis = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

        # Overlay white pixels in red
        vis[px > 0] = [0, 0, 255]

        # Draw ROI rectangles
        cv.rectangle(vis, (930, 580), (1730, 780), (0, 255, 255), 2)
        cv.rectangle(vis, (930, 1200), (1730, 1350), (0, 255, 255), 2)

        # Add text
        cv.putText(vis, f"Threshold={thresh}", (50, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv.putText(vis, f"Top ROI: {top_pixels} px", (950, 650),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv.putText(vis, f"Bottom ROI: {bottom_pixels} px", (950, 1320),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        cv.imshow(f"Line Pixels (thresh={thresh})", resize_for_display(vis))
        print("  Press any key to continue...")
        cv.waitKey(0)

    cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
