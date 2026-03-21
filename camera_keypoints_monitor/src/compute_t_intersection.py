#!/usr/bin/env python3
"""
Detect the bottom 'T' intersection on a padel court image.

Usage:
  python detect_padel_t.py input_path [--out_xy coords.txt] [--out_img viz.png] [--debug]
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np
from src.utils import ImagePoint


# ---------------------------- Global parameters & helpers ----------------------------

class GlobalParameters:
    fgValue = 255
    bgValue = 0


def read_frame(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot open image file {path}")
        return img
    cap = cv.VideoCapture(path)  # treat as video
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open file {path}")
    idx = max(0, int(cap.get(cv.CAP_PROP_FRAME_COUNT)) // 2)
    cap.set(cv.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {idx} from {path}")
    return frame


# ------------------------------------- Line model ------------------------------------

@dataclass
class Line:
    """Line in point-direction form: p + t*v (v not necessarily normalized)."""
    p: np.ndarray  # (2,), float32
    v: np.ndarray  # (2,), float32

    @staticmethod
    def from_two_points(u: ImagePoint, v: ImagePoint) -> "Line":
        u_array = np.array(u, dtype=np.float32)
        v_array = np.array(v, dtype=np.float32)
        d = v_array - u_array
        if np.allclose(d, 0):
            d = np.array([1.0, 0.0], dtype=np.float32)
        return Line(u_array, d)

    def angle_abs(self) -> float:
        return abs(math.atan2(float(self.v[1]), float(self.v[0])))

    def normalized(self) -> "Line":
        n = np.linalg.norm(self.v)
        if n < 1e-6:
            return Line(self.p.copy(), np.array([1.0, 0.0], dtype=np.float32))
        return Line(self.p.copy(), self.v / n)

    def evaluate_by_x(self, x: float) -> float:
        vx = float(self.v[0])
        if abs(vx) < 1e-6:  # vertical
            return float("inf")
        t = (x - float(self.p[0])) / vx
        return float(self.p[1] + t * self.v[1])

    def evaluate_by_y(self, y: float) -> float:
        vy = float(self.v[1])
        if abs(vy) < 1e-6:  # horizontal
            return float("inf")
        t = (y - float(self.p[1])) / vy
        return float(self.p[0] + t * self.v[0])

    def distance_to_point(self, q: ImagePoint) -> float:
        qv = np.array(q, dtype=np.float32) - self.p
        v = self.v
        nrm2 = float(np.dot(v, v))
        if nrm2 < 1e-6:
            return float(np.linalg.norm(qv))
        proj = float(np.cross(np.array([v[0], v[1], 0.0]),
                              np.array([qv[0], qv[1], 0.0]))[2])
        return abs(proj) / math.sqrt(nrm2)

    def distance_to_point_absolute(self, q: ImagePoint) -> float:
        """If the projection falls outside of the segment p->p+v, use nearest endpoint."""
        a = self.p.astype(np.float32)
        b = (self.p + self.v).astype(np.float32)
        q_array = np.array(q, np.float32)
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-6:
            return float(np.linalg.norm(q_array - a))
        t = float(np.dot(q_array - a, ab)) / ab2
        if 0.0 <= t <= 1.0:
            return float(np.linalg.norm(q_array - (a + t * ab)))
        
        return min(float(np.linalg.norm(q_array - a)), float(np.linalg.norm(q_array - b)))


    def intersection(self, other: "Line") -> Optional[ImagePoint]:
        p1, v1 = self.p.astype(np.float64), self.v.astype(np.float64)
        p2, v2 = other.p.astype(np.float64), other.v.astype(np.float64)
        A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]], dtype=np.float64)
        b = p2 - p1
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return None
        t = np.linalg.solve(A, b)[0]
        pt = p1 + t * v1
        return float(pt[0]), float(pt[1])

    def spans_image_extent(self, w: int, h: int, along: str) -> float:
        """Length across image bounds along the major axis ('x' or 'y')."""
        intersections = []
        for x in (0.0, float(w - 1)):
            y = self.evaluate_by_x(x)
            if 0.0 <= y < h:
                intersections.append((x, y))
        for y in (0.0, float(h - 1)):
            x = self.evaluate_by_y(y)
            if 0.0 <= x < w:
                intersections.append((x, y))
        if len(intersections) < 2:
            return 0.0
        a, b = intersections[0], intersections[1]
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def is_duplicate(self, other: "Line", angle_tol_deg: float = 3.0, offset_tol_px: float = 8.0) -> bool:
        a1 = math.degrees(self.angle_abs())
        a2 = math.degrees(other.angle_abs())
        if min(abs(a1 - a2), 180 - abs(a1 - a2)) > angle_tol_deg:
            return False
        return other.distance_to_point(tuple(self.p)) < offset_tol_px


# -------------------------------- Pixel detector (white lines) --------------------------------

@dataclass
class CLPDParams:
    threshold: int = 80
    diffThreshold: int = 20
    t: int = 8
    gradientKernelSize: int = 3
    kernelSize: int = 21


class CourtLinePixelDetector:
    def __init__(self, params: CLPDParams = CLPDParams(), debug: bool = False):
        self.p = params
        self.debug = debug

    def get_luminance(self, frame: np.ndarray) -> np.ndarray:
        return cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)[:, :, 0]

    def detect_line_pixels(self, y: np.ndarray) -> np.ndarray:
        h, w = y.shape
        t = self.p.t
        out = np.full((h, w), GlobalParameters.bgValue, dtype=np.uint8)
        for yy in range(t, h - t):
            row, above, below = y[yy], y[yy - t], y[yy + t]
            for xx in range(t, w - t):
                val = int(row[xx])
                if val < self.p.threshold:
                    continue
                left, right = int(row[xx - t]), int(row[xx + t])
                top, bot = int(above[xx]), int(below[xx])
                tl, tr = int(y[yy - t, xx - t]), int(y[yy - t, xx + t])
                bl, br = int(y[yy + t, xx - t]), int(y[yy + t, xx + t])
                if ((val - left > self.p.diffThreshold) and (val - right > self.p.diffThreshold)) or \
                   ((val - top > self.p.diffThreshold) and (val - bot > self.p.diffThreshold)) or \
                   ((val - tl > self.p.diffThreshold) and (val - br > self.p.diffThreshold)) or \
                   ((val - tr > self.p.diffThreshold) and (val - bl > self.p.diffThreshold)):
                    out[yy, xx] = GlobalParameters.fgValue
        if self.debug:
            cv.imshow("line_pixels", out); cv.waitKey(1)
        return out

    def structure_tensor_filter(self, binary: np.ndarray, luminance: np.ndarray) -> np.ndarray:
        img = cv.GaussianBlur(luminance.astype(np.float32), (5, 5), 0)
        dx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=self.p.gradientKernelSize)
        dy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=self.p.gradientKernelSize)
        dx2, dxy, dy2 = dx * dx, dx * dy, dy * dy
        kernel = np.ones((self.p.kernelSize, self.p.kernelSize), dtype=np.float32)
        dx2, dxy, dy2 = cv.filter2D(dx2, -1, kernel), cv.filter2D(dxy, -1, kernel), cv.filter2D(dy2, -1, kernel)

        out = np.full_like(binary, GlobalParameters.bgValue)
        ys, xs = np.where(binary == GlobalParameters.fgValue)
        for y, x in zip(ys, xs):
            t = np.array([[dx2[y, x], dxy[y, x]], [dxy[y, x], dy2[y, x]]], dtype=np.float32)
            w, _ = np.linalg.eig(t)
            w = np.sort(w)[::-1]
            ratio = np.inf if w[1] <= 0 else w[0] / w[1]
            if ratio > 4.0:
                out[y, x] = GlobalParameters.fgValue
        if self.debug:
            cv.imshow("line_pixels_filtered", out); cv.waitKey(1)
        return out

    def run(self, frame: np.ndarray) -> np.ndarray:
        y = self.get_luminance(frame)
        raw = self.detect_line_pixels(y)
        return self.structure_tensor_filter(raw, y)

def detect_lights_off(bgr: np.ndarray, debug: bool=False) -> bool:
    # --- luminance ---
    y = cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)[:, :, 0]
    H, W = y.shape

    # --- define a court-focused ROI: bottom 55% height, center 60% width ---
    x0 = int(W * 0.20); x1 = int(W * 0.80)
    y0 = int(H * 0.45); y1 = int(H * 1.00)
    roi = y[y0:y1, x0:x1]

    # downsample to stabilize stats
    roi_small = cv.resize(roi, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

    # exposure & contrast
    mean_y  = float(roi_small.mean())
    p95_y   = float(np.quantile(roi_small, 0.95))   # friendlier typing than percentile
    std_y   = float(roi_small.std())

    # edges inside ROI (not whole image)
    edges = cv.Canny(roi_small, 50, 150, L2gradient=True)
    edge_density = float(edges.mean() / 255.0)

    # tiny fraction of very bright pixels (safety valve)
    bright_frac = float((roi_small > 200).mean())

    # --- decision logic ---
    # Primary: dark + low edges inside the ROI
    dark_env = (p95_y < 65 and mean_y < 35 and std_y < 22)
    low_edges = edge_density < 0.020
    very_few_highlights = bright_frac < 0.0008

    lights_off = (dark_env and low_edges and very_few_highlights)

    if debug:
        print(f"[lights/roi x={x0}:{x1} y={y0}:{y1}] "
              f"mean={mean_y:.1f} p95={p95_y:.1f} std={std_y:.1f} "
              f"edgeD={edge_density:.3f} bright={bright_frac:.4f} "
              f"=> lights_off={lights_off}")
    return lights_off

# ----------------------------- Candidate lines via Hough (no refine) -----------------------------

@dataclass
class CLCDParams:
    houghThreshold: int = 50
    refinementIterations: int = 50  # used as max dedupe convergence iters


class CourtLineCandidateDetector:
    def __init__(self, estimate_t_intersection: ImagePoint, params: CLCDParams = CLCDParams(), debug: bool = False):
        self.estimate_t_intersection = estimate_t_intersection
        self.line_max_distance_from_estimate = 150
        self.min_lines_to_consider = 20
        self.p = params
        self.debug = debug

    def extract_lines(self, binary: np.ndarray, rgb: np.ndarray) -> List[Line]:
        linesP = cv.HoughLinesP(binary, 1, np.pi / 180.0,
                                threshold=self.p.houghThreshold,
                                minLineLength=50, maxLineGap=10)
        lines: List[Line] = []
        if linesP is not None:
            for x1, y1, x2, y2 in linesP[:, 0, :]:
                lines.append(Line.from_two_points((x1, y1), (x2, y2)))
        if self.debug:
            vis = rgb.copy()
            for ln in lines:
                p1 = tuple(ln.p.astype(int))
                p2 = tuple((ln.p + ln.v).astype(int))
                cv.line(vis, p1, p2, (0, 0, 255), 2)
            cv.imshow("hough_lines", vis); cv.waitKey(1)
        return lines

    def dedupe(self, lines: List[Line], rgb: np.ndarray) -> List[Line]:
        n, taken, gid = len(lines), [-1]*len(lines), 0
        for i in range(n):
            if taken[i] >= 0: continue
            taken[i] = gid
            for j in range(i+1, n):
                if taken[j] >= 0: continue
                if lines[i].is_duplicate(lines[j]): taken[j] = gid
            gid += 1

        grouped = [[] for _ in range(gid)]
        for idx, g in enumerate(taken): grouped[g].append(lines[idx])

        out = []
        for grp in grouped:
            P = np.stack([g.p for g in grp], 0).astype(np.float32)
            V = np.stack([g.v for g in grp], 0).astype(np.float32)

            # Align directions
            ref = V[0] / (np.linalg.norm(V[0]) + 1e-9)
            V[(V @ ref) < 0] *= -1

            # Median of unit directions; re-normalize
            Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            p = np.median(P, axis=0)
            v = np.median(Vn, axis=0)
            v = v / (np.linalg.norm(v) + 1e-9)

            out.append(Line(p.astype(np.float32), v.astype(np.float32)))

        if self.debug:
            vis = rgb.copy()
            for ln in out:
                p1 = tuple(ln.p.astype(int))
                p2 = tuple((ln.p + 2000*ln.normalized().v).astype(int))
                cv.line(vis, p1, p2, (255,0,0), 2)
            cv.imshow("deduped_lines", vis); cv.waitKey(1)
        return out

    def filter_lines(self, lines: List[Line], rgb: np.ndarray) -> List[Line]:
        """Keep lines closest to the estimated T-intersection."""
        if len(lines) == 0:
            return lines

        tx, ty = self.estimate_t_intersection
        gate = self.line_max_distance_from_estimate
        scored = [(l.distance_to_point_absolute((tx, ty)), l) for l in lines]
        near = [s for s in scored if s[0] <= gate]
        if len(near) < self.min_lines_to_consider:
            near = sorted(scored, key=lambda x: x[0])[:self.min_lines_to_consider]

        if self.debug:
            vis = rgb.copy()
            cv.circle(vis, (int(tx), int(ty)), 6, (0, 255, 255), -1)
            cv.putText(vis, "T-int", (int(tx)+8, int(ty)-8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)
            for d, ln in near:
                p1 = tuple(ln.p.astype(int))
                p2 = tuple((ln.p + ln.v).astype(int))
                cv.line(vis, p1, p2, (0, 0, 255), 2)
                cv.putText(vis, f"{d:.1f}", (p1[0]+5, p1[1]-5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("labeled_lines", vis); cv.waitKey(1)

        return [s[1] for s in near]

    def run(self, binary: np.ndarray, rgb: np.ndarray) -> List[Line]:
        lines = self.extract_lines(binary, rgb)
        lines = self.filter_lines(lines, rgb)
        # Iterative dedupe-only convergence (uses refinementIterations as a max-iter cap)
        prev = len(lines)
        for i in range(self.p.refinementIterations):
            lines = self.dedupe(lines, rgb)
            cnt = len(lines)
            if prev == 0: break
            if i > 0 and abs(cnt - prev) / float(max(1, prev)) < 0.03: break
            prev = cnt
        return lines


# ----------------------------- Select court lines & compute T -----------------------------

def pick_horizontal_and_vertical(lines: List[Line], w: int, h: int
                                 ) -> Tuple[List[Line], List[Line]]:
    h_lines, v_lines = [], []
    for ln in lines:
        ang = abs(math.degrees(ln.angle_abs()))
        if ang < 10.0 or ang > 170.0:  # near-horizontal
            if ln.spans_image_extent(w, h, along="x") >= 0.99 * w:
                h_lines.append(ln)
        elif 80.0 <= ang <= 100.0:     # near-vertical
            if ln.spans_image_extent(w, h, along="y") >= 0.99 * h:
                v_lines.append(ln)
    cx, cy = w / 2.0, h / 2.0
    h_lines.sort(key=lambda L: L.evaluate_by_x(cx))     # top -> bottom
    v_lines.sort(key=lambda L: L.evaluate_by_y(cy))     # left -> right
    return h_lines, v_lines


def find_bottom_t_intersection(
    h_lines: List[Line], v_lines: List[Line], w: int, h: int,
    estimate: Optional[ImagePoint] = None, distance_bias_deg: float = 5.0
) -> Optional[Tuple[float, float, Line, Line]]:
    """
    distance_bias_deg: how many 'degrees' of penalty a full-diagonal miss adds.
                       Smaller -> angle dominates (recommended 2–8).
    """
    if not h_lines or not v_lines:
        return None

    diag = math.hypot(w, h)
    best = None
    best_score = float("inf")

    for hln in h_lines:
        vh = hln.normalized().v
        for vln in v_lines:
            vv = vln.normalized().v
            cos_t = np.clip(abs(np.dot(vh, vv)), -1, 1)
            angle_diff = abs(90.0 - math.degrees(math.acos(cos_t)))  # 0 = perfectly orthogonal

            inter = hln.intersection(vln)
            if inter is None:
                continue

            # soft prior toward the estimated T (optional)
            prior = 0.0
            if estimate is not None and diag > 0:
                ex, ey = estimate
                ix, iy = inter
                prior = distance_bias_deg * (math.hypot(ix - ex, iy - ey) / diag)

            score = angle_diff + prior
            if score < best_score:
                best_score = score
                best = (inter, hln, vln)

    if best is None:
        return None
    (x, y), h_best, v_best = best
    return x, y, h_best, v_best


def draw_viz(rgb: np.ndarray,
             h_line: Line,
             v_line: Line,
             pt: ImagePoint) -> np.ndarray:
    vis = rgb.copy()
    for ln, color in [(h_line, (0, 255, 0)), (v_line, (0, 255, 255))]:
        lnN = ln.normalized()
        p1 = (lnN.p - 4000 * lnN.v).astype(int)
        p2 = (lnN.p + 4000 * lnN.v).astype(int)
        cv.line(vis, tuple(p1), tuple(p2), color, 2, cv.LINE_AA)
    cv.circle(vis, (int(round(pt[0])), int(round(pt[1]))), 8, (0, 0, 255), -1, cv.LINE_AA)
    cv.putText(vis, f"T ({pt[0]:.1f}, {pt[1]:.1f})", (int(pt[0]) + 10, int(pt[1]) - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)
    return vis


# ---------------------------------------------- Main ----------------------------------------------

def find_t_intersection(
    image_path: Path,
    estimate_t_intersection: ImagePoint,
    debug: bool = False
) -> ImagePoint | None:
    frame = read_frame(str(image_path))
    if detect_lights_off(frame, debug):
        print("Lights might be off")
        return None 
    H, W = frame.shape[:2]

    # 1) line pixels
    px = CourtLinePixelDetector(debug=debug).run(frame)

    # 2) line candidates (no refine)
    cand = CourtLineCandidateDetector(
        estimate_t_intersection=estimate_t_intersection,
        debug=debug
    ).run(px, frame)

    # 3) choose court horizontals/verticals & compute bottom T
    h_lines, v_lines = pick_horizontal_and_vertical(cand, W, H)
    res = find_bottom_t_intersection(h_lines, v_lines, W, H, estimate_t_intersection)
    if res is None:
        return None

    x, y, h_line, v_line = res

    # --- sanity check: intersection must be in bottom half ---
    if y < H / 2:
        print(f"[warn] Rejected T-intersection at y={y:.1f} (top half of image)")
        return None

    return (round(x), round(y))