"""Point-based branch identification -- "the bronchotrack version" (the
user's own naming), i.e. the paper's core angular-bearing matching idea
(paper section 4), re-derived fresh for this package rather than imported
from `paper_exact.association` (see `fusion/__init__.py` for why this
package doesn't import that module's logic).

Reduces each candidate branch and each detected lumen to a SINGLE POINT --
per the user's explicit instruction ("for the bronchotrack version use the
center of the segmentation") -- and scores candidates purely on whether
that point's direction from a reference position agrees between graph
space and image space:

    graph side:  `AirwayGraph.project_children_2d(ref_label)` -- each
                 child's overall midpoint, projected onto the 2D plane
                 orthogonal to the reference branch's own forward axis.
    image side:  `utils.polygon_opposing_center` (falling back to the
                 detection's plain box center when no mask is available)
                 -- NOT the plain vertex/area centroid; see that
                 function's own docstring for why. This is "the center of
                 the segmentation" the user asked this model to use.

Both sides reduce to one (u, v) or (x, y) point each; the match cost is
purely angular (cosine distance between the reference-relative offset
vectors), exactly as `paper_exact.association`'s own bearing cost -- this
is deliberately the SAME core idea, just without that module's roll-angle
correction (a genuine scope-reduction for this first version of the
fusion package, not an oversight -- see `fusion/__init__.py`).

This module's job stops at producing a (candidates x detections) cost
matrix; `fusion.association` is responsible for deciding what to do with
it (Hungarian assignment against the OTHER cue's cost matrix, blended --
see `fusion.kalman_fusion`).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..graph import AirwayGraph
from ..types import Tracklet
from ..utils import polygon_opposing_center

NEUTRAL_COST = 1.0  # no usable geometry -> maximally disagreeing, never picked over a real cue


def detection_point(t: Tracklet) -> np.ndarray:
    """A detection's identifying point for this cue: the average of
    opposing mask-boundary points (see `utils.polygon_opposing_center`)
    when a segmentation mask is available, falling back to the box center
    otherwise."""
    if t.mask is not None:
        c = polygon_opposing_center(t.mask)
        if c is not None:
            return c
    box = t.last_box
    return np.array([box.x_c, box.y_c])


def bearing_cost_matrix(
    graph: AirwayGraph,
    ref_label: str,
    candidate_labels: List[str],
    detections: List[Tracklet],
    reference_point: np.ndarray,
) -> Optional[np.ndarray]:
    """(len(candidate_labels), len(detections)) cost matrix, 0 = bearing
    perfectly aligned, 1 = opposite direction from `reference_point`.
    `reference_point` is the image-space point standing in for
    `ref_label`'s own bifurcation (typically the currently-labeled
    tracklet occupying `ref_label`'s own detection_point -- see
    `fusion.association`). Returns None if `ref_label` has no children
    projected (e.g. a leaf branch)."""
    proj = graph.project_children_2d(ref_label)
    proj = {l: proj[l] for l in candidate_labels if l in proj}
    if not proj or not detections:
        return None

    labels = list(proj.keys())
    graph_pts = np.stack([proj[l] for l in labels], axis=0)

    image_offsets = np.stack([detection_point(d) - reference_point for d in detections], axis=0)

    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n = np.where(n < 1e-9, 1.0, n)
        return v / n

    g = unit(graph_pts)
    d = unit(image_offsets)
    cost = 1.0 - g @ d.T  # (n_candidates_with_geometry, n_dets)

    if len(labels) == len(candidate_labels):
        return cost

    # pad rows for candidates with no graph geometry (e.g. missing centerline)
    full = np.full((len(candidate_labels), len(detections)), NEUTRAL_COST)
    row_of = {l: i for i, l in enumerate(labels)}
    for i, l in enumerate(candidate_labels):
        if l in row_of:
            full[i, :] = cost[row_of[l], :]
    return full
