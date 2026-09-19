"""
BronchoTrack pipeline
======================

A from-scratch, modular re-implementation of the tracking + branch-level
localization pipeline described in:

    Tian, S., Liao, X. et al. "BronchoTrack: Airway Lumen Tracking for
    Branch-Level Bronchoscopic Localization." (arXiv:2402.12763)

This package is deliberately split so you can drop in the two artifacts you
already have:

  * a trained YOLOv11 lumen-detection model (``.pt`` weights)  -> see
    :mod:`bronchotrack.detection`
  * a 3D airway graph built from a 3D Slicer segmentation/centerline
    (a JSON file of nodes/edges/labels/coordinates) -> see
    :mod:`bronchotrack.graph`

and get branch-level bronchoscope localization per video frame, following
the paper's three-stage design:

    1. Lumen Detection            -> ``detection.py``
    2. Multi-Lumen Tracking       -> ``kalman.py`` + ``reid.py`` +
                                      ``matching.py`` + ``tracker.py``
    3. Airway Association         -> ``association.py`` + ``localization.py``

``pipeline.py`` wires all of this together frame-by-frame, and ``cli.py``
exposes it as a command-line tool. ``motion_model.py`` adds an optional
graph-constrained Bayes filter (seeded at the trachea, predicting/updating
which branch the camera is currently in using the diameter-ratio matching
results) that runs alongside the paper's own vote-based localizer rather
than replacing it -- see its module docstring.

Loop closure (BronchoTrack-LC) is intentionally NOT implemented here (see
README "Not implemented" section), and roll-angle estimation has been
removed (candidate matching assumes zero roll -- see association.py module
docstring) -- everything else the paper describes for the base BronchoTrack
system, plus Re-ID, is.
"""

from .types import BBox, Detection, Tracklet, GalleryEntry
from .graph import AirwayGraph
from .graph_view import AirwayGraphView
from .motion_model import TreeMotionFilter
from .pipeline import BronchoTrackPipeline

__all__ = [
    "BBox",
    "Detection",
    "Tracklet",
    "GalleryEntry",
    "AirwayGraph",
    "AirwayGraphView",
    "TreeMotionFilter",
    "BronchoTrackPipeline",
]

__version__ = "0.1.0"
