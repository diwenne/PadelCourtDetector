from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Dict, List, Optional, Tuple

ImagePoint = Tuple[float, float]


@dataclass
class CourtKeypoints:
    tol: ImagePoint
    tor: ImagePoint
    point_7: ImagePoint
    point_9: ImagePoint

    @property
    def bottom_line_center(self) -> ImagePoint:
        """Gets rought area of t-intersection (between point 7 and 9)"""
        x_coord = (self.point_7[0] + self.point_9[0]) / 2
        y_coord = (self.point_7[1] + self.point_9[1]) / 2

        return (x_coord, y_coord)

    def to_dict(self) -> Dict[str, ImagePoint]:
        return {"tol": self.tol, "tor": self.tor, "point_7": self.point_7, "point_9": self.point_9}

@dataclass
class DeviceInfo:
    camera_id: str 
    camera_name: str
    court_keypoints: CourtKeypoints
    venue_id: str
    admin_ids: List[str]
    emails: List[str]
    should_send_email: bool
    t_intersection: Optional[ImagePoint] = None
    last_alert_ts: Optional[datetime] = None

    @property
    def bottom_line_center(self) -> ImagePoint:
        """The center of the bottom court keypoints between point7 and point9"""
        return self.court_keypoints.bottom_line_center

@dataclass
class VideoData:
    creator_identity_id: str 
    video_id: str
    updated_ts: datetime

def near_line(p: ImagePoint, a: ImagePoint, b: ImagePoint, tol: float = 30.0) -> bool:
    """
    Return True if point p lies roughly on the line through a and b (2D image coords).

    Args:
        p: The point to test (x, y).
        a: First point on the line (x, y).
        b: Second point on the line (x, y).
        tol: Max perpendicular distance allowed (in pixels).
    """
    (x0, y0), (x1, y1), (x, y) = a, b, p
    dx, dy = x1 - x0, y1 - y0
    if dx == dy == 0:  # degenerate case
        return sqrt((x - x0) ** 2 + (y - y0) ** 2) <= tol
    dist = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / sqrt(dx ** 2 + dy ** 2)
    return dist <= tol