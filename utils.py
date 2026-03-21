"""
Shared utility functions for heatmap generation and geometric operations.

Provides:
    - gaussian2D: Generate 2D Gaussian kernel
    - draw_umich_gaussian: Stamp a Gaussian blob onto a heatmap at (x, y)
    - gaussian_radius: Compute optimal Gaussian radius for a given bbox/overlap
    - line_intersection: Find intersection point of two line segments (uses SymPy)
    - is_point_in_image: Check if a point falls within image bounds

Used by:
    - dataset_padel.py (heatmap generation during training)
    - base_validator.py (point-in-image checks during validation)
    - postprocess.py (line intersection for keypoint refinement)
"""
import numpy as np
from sympy import Line
import sympy


def gaussian2D(shape, sigma=1):
    """Generate a 2D Gaussian kernel.
    
    Args:
        shape: (height, width) of the kernel. Should be odd numbers for symmetry.
        sigma: Standard deviation of the Gaussian distribution.
    
    Returns:
        np.ndarray: 2D Gaussian kernel normalized to peak value of 1.0.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D Gaussian blob on a heatmap at the specified center.
    
    Uses element-wise maximum to avoid overwriting stronger peaks from
    nearby keypoints. Safe for centers near image edges (auto-crops).
    
    Args:
        heatmap: (H, W) float32 array to draw on (modified in-place).
        center:  (x, y) pixel coordinates of the Gaussian center.
        radius:  Radius of the Gaussian in pixels. Determines diameter = 2*radius+1.
                 Sigma is computed as diameter/6.
        k:       Amplitude multiplier for the Gaussian peak (default: 1).
    
    Returns:
        np.ndarray: The modified heatmap (same reference as input).
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: 
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap



def gaussian_radius(det_size, min_overlap=0.7):
    """Compute optimal Gaussian radius for a given detection size.
    
    Based on the CornerNet paper formula. Finds the largest radius such that
    the IoU between the ground-truth box and a box centered at the Gaussian
    peak exceeds `min_overlap`.
    
    Args:
        det_size: (height, width) of the detection bounding box.
        min_overlap: Minimum IoU overlap threshold (default: 0.7).
    
    Returns:
        float: Optimal Gaussian radius in pixels.
    """
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def line_intersection(line1, line2):
    """Find the intersection point of two line segments using SymPy.
    
    Args:
        line1: (x1, y1, x2, y2) endpoints of first line segment.
        line2: (x3, y3, x4, y4) endpoints of second line segment.
    
    Returns:
        tuple or None: (x, y) intersection coordinates, or None if lines
                       are parallel or do not intersect.
    """
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point



def is_point_in_image(x, y, input_width=1280, input_height=720):
    """Check if a point (x, y) falls within image boundaries.
    
    Returns False if x or y is None/0/falsy.
    
    Args:
        x: Horizontal coordinate.
        y: Vertical coordinate.
        input_width:  Image width (default: 1280, padel uses 960).
        input_height: Image height (default: 720, padel uses 544).
    
    Returns:
        bool: True if point is within [0, width] x [0, height].
    """
    res = False
    if x and y:
        res = (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return res

