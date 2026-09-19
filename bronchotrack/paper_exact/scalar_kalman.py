"""Backward-compatible re-export.

``ScalarKalmanFilter`` was promoted to ``bronchotrack.scalar_kalman`` (a
package-shared location) once ``bronchotrack.fusion`` also needed it --
see that module's docstring for the full history. This shim keeps
``from .scalar_kalman import ScalarKalmanFilter`` working unchanged inside
``paper_exact/association.py`` and any external code that imported from
this path.
"""
from __future__ import annotations

from ..scalar_kalman import STATE_DIM, ScalarKalmanFilter  # noqa: F401

__all__ = ["ScalarKalmanFilter", "STATE_DIM"]
