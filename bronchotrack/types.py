"""Core data structures shared across the pipeline.

These mirror the notation used in the BronchoTrack paper as closely as
possible so the code and the paper can be read side by side:

    x_k = [x_c, y_c, h, a, xdot_c, ydot_c, hdot]   (tracker state, kalman.py)
    T   = {ID, {ind_start, ind_end}, {b_i, ..., b_j}}   (tracklet, below)
    G   = {G_l0: T_0, ...}                              (gallery, below)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BBox:
    """Axis-aligned bounding box in the paper's (x_c, y_c, h, a) parameterization.

    x_c, y_c : center of the box, in pixels
    h        : box height, in pixels
    a        : aspect ratio (width / height)
    """

    x_c: float
    y_c: float
    h: float
    a: float

    @property
    def w(self) -> float:
        return self.h * self.a

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        w = self.w
        return (
            self.x_c - w / 2.0,
            self.y_c - self.h / 2.0,
            self.x_c + w / 2.0,
            self.y_c + self.h / 2.0,
        )

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> "BBox":
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        return BBox(x_c=x1 + w / 2.0, y_c=y1 + h / 2.0, h=h, a=w / h)

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "BBox":
        h = max(h, 1e-6)
        return BBox(x_c=x + w / 2.0, y_c=y + h / 2.0, h=h, a=w / max(h, 1e-6))

    def as_vector(self) -> np.ndarray:
        return np.array([self.x_c, self.y_c, self.h, self.a], dtype=np.float64)


@dataclass
class Detection:
    """A single raw detection output by the lumen detector for one frame."""

    bbox: BBox
    confidence: float
    frame_idx: int
    class_id: int = 0
    embedding: Optional[np.ndarray] = None  # Re-ID feature vector, filled in later
    crop: Optional[np.ndarray] = None  # optional raw image crop, for Re-ID
    mask: Optional[np.ndarray] = None  # (N, 2) polygon, pixel coords -- YOLO-seg only, else None

    @property
    def is_high_conf(self) -> bool:
        # threshold applied by caller (tracker.py); default kept for convenience
        return self.confidence >= 0.5


@dataclass
class Tracklet:
    """T = {ID, {ind_start, ind_end}, {b_i, ..., b_j}}  (paper notation).

    Extended with the bookkeeping the tracker/association modules need:
    a Kalman filter instance, a cumulative Re-ID embedding (EMA), and the
    airway label currently assigned to this tracklet (if any).
    """

    track_id: int
    ind_start: int
    ind_end: int
    boxes: List[BBox] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    kalman_state: Optional[np.ndarray] = None  # filled by kalman.py
    kalman_covariance: Optional[np.ndarray] = None

    embedding: Optional[np.ndarray] = None  # e_i^t, EMA appearance feature

    mask: Optional[np.ndarray] = None  # (N, 2) polygon, pixel coords, latest observed frame
                                        # (YOLO-seg only, else None; NOT Kalman-smoothed --
                                        # frame-to-frame shape noise for the same lumen is
                                        # small enough that this isn't worth a shape filter)

    label: Optional[str] = None  # anatomical branch label currently assigned
    label_history: List[Tuple[int, str]] = field(default_factory=list)  # (frame, label)

    diameter_distance_match: Optional[bool] = None  # paper_exact only -- see association.py's
                                                      # module docstring ("virtual verification").
                                                      # True/False once a diameter:distance
                                                      # comparison has actually run for this
                                                      # tracklet's current label; None beforehand
                                                      # (e.g. unlabeled, or no radius data at all).

    time_since_update: int = 0
    hits: int = 0

    @property
    def age_frames(self) -> int:
        """Number of frames this tracklet has existed for (paper: 'track age')."""
        return self.ind_end - self.ind_start + 1

    @property
    def last_box(self) -> BBox:
        return self.boxes[-1]


@dataclass
class GalleryEntry:
    """G_{l_i}: T_i -- one visited-branch record in the gallery G."""

    label: str
    track_ids: List[int] = field(default_factory=list)
    first_seen_frame: Optional[int] = None
    last_seen_frame: Optional[int] = None
