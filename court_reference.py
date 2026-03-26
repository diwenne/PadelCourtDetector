"""
Court reference factory for different sports.
All measurements in millimeters (1000 = 1m).
"""
import cv2
import numpy as np

class CourtReference:
    """
    Base class for court reference geometry.
    """
    def __init__(self, sport_name, width, length):
        self.sport_name = sport_name
        self.court_width = width
        self.court_length = length
        self.key_points = []  # To be defined by subclasses
        self.viz_scale = 0.05
        self.line_width = 3

    def get_corners_array(self):
        """Return corners as numpy array for homography computation."""
        return np.array(self.key_points[:4], dtype=np.float32)

    def build_court_reference(self, scale=None):
        """Implement sport-specific drawing if needed."""
        raise NotImplementedError

class PadelCourtReference(CourtReference):
    def __init__(self):
        super().__init__('padel', 10000, 20000)
        self.service_line_dist = 6950
        
        # Camera order: tol, tor, point_7, point_9 (corners)
        # top of image (far) -> bottom of image (near)
        self.key_points = [
            (0, self.court_length),               # tol (0)
            (self.court_width, self.court_length),  # tor (1)
            (0, 0),                               # point_7 (2)
            (self.court_width, 0),                 # point_9 (3)
            (self.court_width // 2, self.court_length), # tom (4) - added for v4
            (self.court_width // 2, 0)                  # bottom_t (5)
        ]

    def build_court_reference(self, scale=None):
        if scale is None: scale = self.viz_scale
        h, w = int(self.court_length * scale), int(self.court_width * scale)
        court = np.zeros((h, w, 3), dtype=np.uint8)
        court[:] = (34, 85, 51)
        def s(pt): return (int(pt[0] * scale), int(pt[1] * scale))
        cv2.rectangle(court, (0, 0), (w-1, h-1), (255, 255, 255), self.line_width)
        # Net
        cv2.line(court, s((0, self.court_length//2)), s((self.court_width, self.court_length//2)), (255,255,255), self.line_width)
        # Services
        cv2.line(court, s((0, self.court_length//2 - self.service_line_dist)), s((self.court_width, self.court_length//2 - self.service_line_dist)), (255,255,255), self.line_width)
        cv2.line(court, s((0, self.court_length//2 + self.service_line_dist)), s((self.court_width, self.court_length//2 + self.service_line_dist)), (255,255,255), self.line_width)
        # Center line
        cv2.line(court, s((self.court_width//2, self.court_length//2 - self.service_line_dist)), s((self.court_width//2, self.court_length//2 + self.service_line_dist)), (255,255,255), self.line_width)
        return court

class PickleballCourtReference(CourtReference):
    """
    Pickleball court (20' x 44') or 6.1m x 13.41m.
    Kitchen (Non-volley zone): 7' (2.13m) from net.
    """
    def __init__(self):
        super().__init__('pickleball', 6100, 13410)
        self.kitchen_dist = 2130
        
        # Similar 6-keypoint structure
        self.key_points = [
            (0, self.court_length),               # tol (0)
            (self.court_width, self.court_length),  # tor (1)
            (0, 0),                               # point_7 (2)
            (self.court_width, 0),                 # point_9 (3)
            (self.court_width // 2, self.court_length), # tom (4)
            (self.court_width // 2, 0)                  # bottom_t (5)
        ]

    def build_court_reference(self, scale=None):
        if scale is None: scale = self.viz_scale
        h, w = int(self.court_length * scale), int(self.court_width * scale)
        court = np.zeros((h, w, 3), dtype=np.uint8)
        court[:] = (120, 60, 20) # Blue/Brown-ish or whatever
        def s(pt): return (int(pt[0] * scale), int(pt[1] * scale))
        cv2.rectangle(court, (0, 0), (w-1, h-1), (255, 255, 255), self.line_width)
        # Net
        cv2.line(court, s((0, self.court_length//2)), s((self.court_width, self.court_length//2)), (255,255,255), self.line_width)
        # Kitchen lines
        cv2.line(court, s((0, self.court_length//2 - self.kitchen_dist)), s((self.court_width, self.court_length//2 - self.kitchen_dist)), (255,255,255), self.line_width)
        cv2.line(court, s((0, self.court_length//2 + self.kitchen_dist)), s((self.court_width, self.court_length//2 + self.kitchen_dist)), (255,255,255), self.line_width)
        # Center line (runs from kitchen to baseline)
        cv2.line(court, s((self.court_width//2, 0)), s((self.court_width//2, self.court_length//2 - self.kitchen_dist)), (255,255,255), self.line_width)
        cv2.line(court, s((self.court_width//2, self.court_length//2 + self.kitchen_dist)), s((self.court_width//2, self.court_length)), (255,255,255), self.line_width)
        return court

def get_court_reference(sport):
    if sport == 'padel':
        return PadelCourtReference()
    elif sport == 'pickleball':
        return PickleballCourtReference()
    else:
        raise ValueError(f"Unknown sport: {sport}")
