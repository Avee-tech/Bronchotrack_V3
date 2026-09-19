"""Small geometry / math helpers used across modules."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .types import BBox


def iou(a: BBox, b: BBox) -> float:
    """Standard axis-aligned IoU between two BBox in (x_c, y_c, h, a) form."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 1e-9:
        return 0.0
    return float(inter_area / union)


def contains(outer: BBox, inner: BBox, tol: float = 0.0) -> bool:
    """True if `inner` box lies (approximately) within `outer` box.

    Used by association.py to build the per-frame parent/child subgraph of
    detections purely from bounding-box inclusion, as described in the
    paper's "Intra-frame Association" step.
    """
    ox1, oy1, ox2, oy2 = outer.xyxy
    ix1, iy1, ix2, iy2 = inner.xyxy
    return (
        ix1 >= ox1 - tol
        and iy1 >= oy1 - tol
        and ix2 <= ox2 + tol
        and iy2 <= oy2 + tol
    )


def polygon_area(poly: Optional[np.ndarray]) -> float:
    """Shoelace-formula area of a simple polygon. `poly` is (N, 2) pixel
    coordinates (e.g. a YOLO-seg instance's `result.masks.xy[i]`). Returns
    0.0 for None/degenerate (<3 point) input rather than raising, so
    callers can use it directly on possibly-missing segmentation data."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def mask_equivalent_diameter(poly: Optional[np.ndarray]) -> Optional[float]:
    """Diameter of the circle with the same area as `poly` (2*sqrt(area/pi))
    -- a shape-agnostic size estimate for a segmentation mask, used as a
    strictly better substitute for a bounding box's (w+h)/2 when a real
    mask is available (a lumen opening is rarely circular or axis-aligned,
    so a box over/under-estimates its true size depending on orientation).
    Returns None if the polygon is missing or degenerate."""
    area = polygon_area(poly)
    if area <= 1e-9:
        return None
    return float(2.0 * np.sqrt(area / np.pi))


def polygon_longest_diameter(poly: Optional[np.ndarray]) -> Optional[float]:
    """The longest diameter of `poly` -- the maximum distance between any
    two of its own vertices (a Feret-diameter-style size estimate), as
    opposed to `mask_equivalent_diameter`'s same-area-circle estimate.

    Where `mask_equivalent_diameter` answers "how big is this opening on
    average", this answers "how wide is this opening at its widest point" --
    the more relevant number for the diameter:distance ("angular width")
    matching cue in `paper_exact.association`, since that cue is trying to
    approximate how large the lumen would *look* at a glance, which tracks
    the widest visible extent more than the area-equivalent size (a
    crescent-shaped or foreshortened opening can have a small area but
    still read as "wide" to the cue it's feeding).

    O(n^2) over the polygon's own vertices -- masks from a YOLO-seg model
    are small (tens of points), so this is cheap; no need for a convex-hull
    reduction at this scale. Returns None for a missing/degenerate (<2
    point) polygon.
    """
    if poly is None or len(poly) < 2:
        return None
    pts = np.asarray(poly, dtype=np.float64)
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return float(dists.max())


def polygon_opposing_center(poly: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """A robust center estimate for `poly`: for every vertex, pair it with
    whichever other vertex sits most nearly diametrically opposite it (by
    angle around a preliminary centroid), take the midpoint of each such
    pair, and average all those midpoints.

    This is deliberately not the plain vertex/area centroid -- a lumen
    opening's mask is often lopsided (partly occluded, foreshortened, or
    caught mid-transition between two branches), which drags a plain
    centroid toward whichever side has more boundary detail. Averaging
    "midpoint of opposite wall to opposite wall" across the whole boundary
    instead estimates "the middle of the open space", which is closer to
    where you'd actually point a crosshair at the lumen and is what the new
    diameter:distance geometry (and the live-video center dot) both use.

    Algorithm: seed a preliminary centroid c0 from the plain vertex mean;
    for each vertex p_i, find the vertex p_j (j != i) whose direction from
    c0 is closest (by circular/angular distance) to the opposite direction
    of p_i's own; average the midpoints (p_i + p_j) / 2 over all i.

    O(n^2) -- see `polygon_longest_diameter`'s docstring on why that's fine
    at mask-polygon scale. Returns None for a missing/degenerate (<3
    point) polygon.
    """
    if poly is None or len(poly) < 3:
        return None
    pts = np.asarray(poly, dtype=np.float64)
    c0 = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0])

    n = len(pts)
    midpoints = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        target = angles[i] + np.pi
        # circular angular distance from `target` to every other vertex's angle
        delta = np.abs((angles - target + np.pi) % (2 * np.pi) - np.pi)
        delta[i] = np.inf  # never pair a vertex with itself
        j = int(np.argmin(delta))
        midpoints[i] = (pts[i] + pts[j]) / 2.0

    return midpoints.mean(axis=0)


def polygon_contains_point(poly: Optional[np.ndarray], point: Tuple[float, float]) -> bool:
    """True if `point` (x, y) lies inside `poly` (an (N, 2) pixel-coordinate
    polygon), using OpenCV's point-in-polygon test. Returns False for
    None/degenerate input."""
    if poly is None or len(poly) < 3:
        return False
    import cv2

    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def rotate_2d(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate an (N, 2) array of points about the origin by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def project_to_tangent_plane(
    points_3d: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """Project 3D points onto the 2D plane orthogonal to `axis`, centered at
    `origin`. Used to flatten a branch's children from the 3D airway graph
    into the image-like 2D space where they can be Hungarian-matched against
    detected lumen bounding boxes (paper: "2D Projection & Rotation").

    Returns an (N, 2) array of coordinates in an arbitrary but consistent
    local 2D basis (u, v) spanning the tangent plane.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    # build an orthonormal basis (u, v) for the plane orthogonal to axis
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, axis)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis, u)

    rel = points_3d - origin[None, :]
    coords_u = rel @ u
    coords_v = rel @ v
    return np.stack([coords_u, coords_v], axis=1)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two vectors, robust to zero-length input."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_val))


def hungarian(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around scipy's linear_sum_assignment so callers don't need
    to import scipy directly (and so we have one place to swap algorithms)."""
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return linear_sum_assignment(cost_matrix)


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve a user-facing device spec ("cuda", "cuda:0", "cpu", or None)
    to a concrete torch device string, auto-detecting a CUDA GPU when
    `device` is None -- the single place LumenDetector/ReIDEmbedder/the CLI
    all go through, so "did it actually pick up my GPU" has one consistent
    answer everywhere rather than three separate auto-detect implementations
    that could disagree."""
    if device is not None:
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def describe_device(device: str) -> str:
    """Human-readable description of a resolved device string, e.g.
    "cuda:0 (NVIDIA GeForce RTX 4050 Laptop GPU)" or "cpu". Best-effort --
    falls back to just the device string if torch/CUDA introspection fails
    for any reason."""
    if not device.startswith("cuda"):
        return device
    try:
        import torch

        idx = 0
        if ":" in device:
            idx = int(device.split(":")[1])
        return f"{device} ({torch.cuda.get_device_name(idx)})"
    except Exception:
        return device
