"""
Pickleball court reference model.
Court: 6.1m x 13.41m (20' x 44')
"""
import cv2
import numpy as np

class PickleballCourtReference:
    """
    Reference model for pickleball court geometry.
    All measurements in millimeters.
    """
    def __init__(self):
        # Court dimensions
        self.court_width = 6100   # 6.1 meters (20 feet)
        self.court_length = 13410 # 13.41 meters (44 feet)
        
        # Kitchen (Non-volley zone): 2.13m (7 feet) from net
        self.kitchen_dist = 2130
        
        # Key lines
        self.net_height_center = 864   # 34 inches
        self.net_height_ends = 914     # 36 inches
        
        # The corners and midpoints (6 points total)
        self.key_points = [
            (0, self.court_length),               # tol: top-left (far)
            (self.court_width, self.court_length),  # tor: top-right (far)
            (0, 0),                               # bol: bottom-left (near)
            (self.court_width, 0),                 # bor: bottom-right (near)
            (self.court_width // 2, self.court_length), # tom: top-middle (mid of tol/tor)
            (self.court_width // 2, 0)                  # bom: bottom-middle (mid of bol/bor)
        ]
        
        # For visualization
        self.viz_scale = 0.05
        self.line_width = 3

    def get_corners_array(self):
        return np.array(self.key_points[:4], dtype=np.float32)

    def build_court_reference(self, scale=None):
        """Create reference court image for visualization."""
        if scale is None:
            scale = self.viz_scale
            
        h = int(self.court_length * scale)
        w = int(self.court_width * scale)
        
        # Blueish background
        court = np.zeros((h, w, 3), dtype=np.uint8)
        court[:] = (120, 60, 20)
        
        def scale_point(pt):
            return (int(pt[0] * scale), int(pt[1] * scale))
        
        cv2.rectangle(court, (0, 0), (w-1, h-1), (255, 255, 255), self.line_width)
        
        # Draw net line
        cv2.line(court, scale_point((0, self.court_length // 2)), 
                 scale_point((self.court_width, self.court_length // 2)), (255, 255, 255), self.line_width)
                 
        # Draw kitchen lines
        cv2.line(court, scale_point((0, self.court_length // 2 - self.kitchen_dist)), 
                 scale_point((self.court_width, self.court_length // 2 - self.kitchen_dist)), (255, 255, 255), self.line_width)
        cv2.line(court, scale_point((0, self.court_length // 2 + self.kitchen_dist)), 
                 scale_point((self.court_width, self.court_length // 2 + self.kitchen_dist)), (255, 255, 255), self.line_width)
        
        # Draw center lines (runs from kitchen to baseline)
        cv2.line(court, scale_point((self.court_width // 2, 0)), 
                 scale_point((self.court_width // 2, self.court_length // 2 - self.kitchen_dist)), (255, 255, 255), self.line_width)
        cv2.line(court, scale_point((self.court_width // 2, self.court_length // 2 + self.kitchen_dist)), 
                 scale_point((self.court_width // 2, self.court_length)), (255, 255, 255), self.line_width)
                 
        return court
