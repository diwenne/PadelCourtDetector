"""
Padel court reference model with official dimensions.
Court: 10m x 20m (using 1000 units = 1m for precision)
"""
import cv2
import numpy as np


class PadelCourtReference:
    """
    Reference model for padel court geometry.
    All measurements in millimeters (10m = 10000mm).
    """
    def __init__(self):
        # Court dimensions (official FIP regulations)
        self.court_width = 10000   # 10 meters
        self.court_length = 20000  # 20 meters
        
        # Key lines
        self.service_line_dist = 6950  # 6.95m from net
        self.net_height_center = 880   # 0.88m
        self.net_height_ends = 920     # 0.92m
        
        # The 4 corners we detect (matching model output order)
        # Camera view: tol/tor at TOP of image (far), point_7/point_9 at BOTTOM (near)
        # Top-down view: point_7/point_9 at TOP, tol/tor at BOTTOM
        self.key_points = [
            (0, self.court_length),              # tol (index 0) -> bottom-left
            (self.court_width, self.court_length), # tor (index 1) -> bottom-right
            (0, 0),                              # point_7 (index 2) -> top-left
            (self.court_width, 0)                # point_9 (index 3) -> top-right
        ]
        
        # Service lines
        self.net_line = ((0, self.court_length // 2), 
                         (self.court_width, self.court_length // 2))
        self.service_top = ((0, self.court_length // 2 - self.service_line_dist),
                           (self.court_width, self.court_length // 2 - self.service_line_dist))
        self.service_bottom = ((0, self.court_length // 2 + self.service_line_dist),
                              (self.court_width, self.court_length // 2 + self.service_line_dist))
        self.center_line = ((self.court_width // 2, self.court_length // 2 - self.service_line_dist),
                           (self.court_width // 2, self.court_length // 2 + self.service_line_dist))

        # For visualization (scale down for reasonable image size)
        self.viz_scale = 0.05  # 10000mm -> 500px
        self.line_width = 3

    def get_corners_array(self):
        """Return corners as numpy array for homography computation."""
        return np.array(self.key_points, dtype=np.float32)

    def build_court_reference(self, scale=None):
        """
        Create reference court image for visualization.
        Returns image with white court on dark background.
        """
        if scale is None:
            scale = self.viz_scale
            
        h = int(self.court_length * scale)
        w = int(self.court_width * scale)
        
        # Dark green background (like real padel court)
        court = np.zeros((h, w, 3), dtype=np.uint8)
        court[:] = (34, 85, 51)  # Dark green in BGR
        
        def scale_point(pt):
            return (int(pt[0] * scale), int(pt[1] * scale))
        
        # Draw court boundary
        cv2.rectangle(court, (0, 0), (w-1, h-1), (255, 255, 255), self.line_width)
        
        # Draw net line
        cv2.line(court, scale_point(self.net_line[0]), 
                 scale_point(self.net_line[1]), (255, 255, 255), self.line_width)
        
        # Draw service lines
        cv2.line(court, scale_point(self.service_top[0]), 
                 scale_point(self.service_top[1]), (255, 255, 255), self.line_width)
        cv2.line(court, scale_point(self.service_bottom[0]), 
                 scale_point(self.service_bottom[1]), (255, 255, 255), self.line_width)
        
        # Draw center service line
        cv2.line(court, scale_point(self.center_line[0]), 
                 scale_point(self.center_line[1]), (255, 255, 255), self.line_width)
        
        return court


if __name__ == '__main__':
    # Test: create and display court reference
    court_ref = PadelCourtReference()
    court_img = court_ref.build_court_reference()
    cv2.imwrite('padel_court_reference.png', court_img)
    print(f"Court reference saved. Size: {court_img.shape}")
    print(f"Corner points: {court_ref.key_points}")
