#!/usr/bin/env python3
"""
Test script to run T-intersection detection on a local image.
"""
import sys
from pathlib import Path

# Add src to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from compute_t_intersection import find_t_intersection
from image_extracter import add_points_to_image


def main():
    # Path to the image (relative to repo root)
    image_path = Path(__file__).parent.parent.parent / "hard-example.png"

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return 1

    print(f"Processing image: {image_path}")
    print(f"Image size: {image_path.stat().st_size / 1024:.1f} KB")

    # Estimate for T-intersection (x, y)
    # This is an initial guess - the algorithm will search around this area
    # For this image, the T should be at bottom-center of the court
    # Image is 2659x1651, so estimate center-bottom
    estimate_t = (1330, 1250)

    print(f"Estimated T-intersection: {estimate_t}")
    print("Running detection with debug mode enabled...")
    print("(Press any key in the debug windows to continue through each step)")

    # Run the detection with debug=True to see intermediate steps
    t_intersection = find_t_intersection(
        image_path,
        estimate_t,
        debug=True
    )

    if t_intersection:
        print(f"\n✓ SUCCESS: Found T-intersection at: {t_intersection}")

        # Create annotated output image
        output_path = image_path.parent / f"{image_path.stem}_detected.png"

        # Copy the image and add the detected point
        import shutil
        shutil.copy(image_path, output_path)

        points = {
            "T-intersection": t_intersection,
            "estimate": estimate_t
        }
        add_points_to_image(output_path, points)

        print(f"✓ Saved annotated image to: {output_path}")
        return 0
    else:
        print("\n✗ FAILED: Could not find T-intersection")
        print("Possible reasons:")
        print("  - Lights might be off (image too dark)")
        print("  - Court lines not clearly visible")
        print("  - Estimate is too far from actual T-intersection")
        print("  - Lines are too distorted or unclear")
        return 1


if __name__ == "__main__":
    sys.exit(main())
