"""Motion model for multi-lumen tracking (paper section 3, "Motion Model").

Implements the paper's exact state vector:

    x_k = [x_c, y_c, h, a, xdot_c, ydot_c, hdot]

i.e. a constant-velocity model over the box center and height, with the
aspect ratio `a` treated as (approximately) constant frame-to-frame -- no
velocity term for it, matching the paper.

This is a minimal, dependency-free (numpy only) linear Kalman filter so we
don't need to pull in filterpy; the math is the standard predict/update
recursion.
"""
from __future__ import annotations

import numpy as np

from .types import BBox

STATE_DIM = 7
MEAS_DIM = 4


def _build_matrices(dt: float = 1.0):
    F = np.eye(STATE_DIM)
    F[0, 4] = dt  # x_c += vxc * dt
    F[1, 5] = dt  # y_c += vyc * dt
    F[2, 6] = dt  # h   += vh  * dt

    H = np.zeros((MEAS_DIM, STATE_DIM))
    H[0, 0] = 1.0  # x_c
    H[1, 1] = 1.0  # y_c
    H[2, 2] = 1.0  # h
    H[3, 3] = 1.0  # a
    return F, H


class KalmanBoxFilter:
    """One Kalman filter instance per tracklet."""

    def __init__(
        self,
        init_bbox: BBox,
        dt: float = 1.0,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        velocity_uncertainty_scale: float = 10.0,
    ):
        self.dt = dt
        self.F, self.H = _build_matrices(dt)

        self.x = np.zeros(STATE_DIM)
        self.x[0] = init_bbox.x_c
        self.x[1] = init_bbox.y_c
        self.x[2] = init_bbox.h
        self.x[3] = init_bbox.a
        # velocities start at 0

        self.P = np.eye(STATE_DIM)
        self.P[4:, 4:] *= velocity_uncertainty_scale  # high initial velocity uncertainty

        self.Q = np.eye(STATE_DIM) * process_noise
        self.R = np.eye(MEAS_DIM) * measurement_noise

    def predict(self) -> BBox:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # guard against degenerate/negative height or aspect ratio after predict
        self.x[2] = max(self.x[2], 1e-3)
        self.x[3] = max(self.x[3], 1e-3)
        return self.as_bbox()

    def update(self, measurement: BBox) -> None:
        z = measurement.as_vector()
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K @ self.H) @ self.P

        self.x[2] = max(self.x[2], 1e-3)
        self.x[3] = max(self.x[3], 1e-3)

    def as_bbox(self) -> BBox:
        return BBox(x_c=self.x[0], y_c=self.x[1], h=self.x[2], a=self.x[3])

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()
