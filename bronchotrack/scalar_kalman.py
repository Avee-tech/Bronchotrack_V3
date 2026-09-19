"""A minimal constant-velocity Kalman filter over a single scalar value
(e.g. an angle in radians, a dimensionless size ratio, or a lumen's own
apparent diameter) plus its rate of change -- the same predict/update
recursion as ``kalman.KalmanBoxFilter`` (paper section 3's box-tracking
Kalman filter), just with a 2-dimensional state instead of 7. Not paper
content itself -- this is infrastructure the paper doesn't need (it only
Kalman-filters bounding boxes).

Originally lived inside ``paper_exact/`` (which still re-exports it from
there for backward compatibility -- see that module) since it was first
built for that package's roll-angle estimation and diameter:distance
ratio smoothing. Promoted to this top-level, package-shared location once
``fusion/`` needed the exact same predict/update recursion for its own
diameter-growth motion model and per-candidate confidence smoothing --
see each package's own docstrings for how each one is wired up.

Dependency-free (numpy only), deliberately kept separate from
``kalman.py`` rather than generalizing that one to arbitrary state size --
the box filter's fixed (x_c, y_c, h, a, ...) layout is the paper's own
notation and shouldn't be genericized away from it.
"""
from __future__ import annotations

import numpy as np

STATE_DIM = 2  # [value, d(value)/dt]


def _build_matrices(dt: float = 1.0):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    return F, H


class ScalarKalmanFilter:
    """One Kalman filter instance per smoothed scalar quantity (e.g. one
    per tracklet for the diameter:distance ratio, or one shared instance
    for the roll-angle estimate)."""

    def __init__(
        self,
        init_value: float,
        dt: float = 1.0,
        process_noise: float = 1e-3,
        measurement_noise: float = 5e-2,
        velocity_uncertainty_scale: float = 10.0,
    ):
        """
        process_noise : how much the true value is expected to drift
            frame-to-frame on its own (higher = filter trusts new
            measurements more / smooths less).
        measurement_noise : how noisy a single per-frame measurement is
            expected to be (higher = filter trusts its own prior more /
            smooths more).
        """
        self.dt = dt
        self.F, self.H = _build_matrices(dt)

        self.x = np.array([init_value, 0.0])
        self.P = np.eye(STATE_DIM)
        self.P[1, 1] *= velocity_uncertainty_scale

        self.Q = np.eye(STATE_DIM) * process_noise
        self.R = np.array([[measurement_noise]])

    def predict(self) -> float:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.value

    def update(self, measurement: float) -> float:
        z = np.array([measurement])
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain, (2,1)

        self.x = self.x + (K @ y)
        self.P = (np.eye(STATE_DIM) - K @ self.H) @ self.P
        return self.value

    @property
    def value(self) -> float:
        return float(self.x[0])

    @property
    def rate(self) -> float:
        return float(self.x[1])
