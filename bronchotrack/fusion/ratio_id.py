"""Whole-mask branch identification -- "the ratio method" (the user's own
naming), explicitly required to "make use of the fact that we are using a
segmentation model instead of just point based" (the user's own words,
given after rejecting an earlier draft of this cue that just reused
`paper_exact.association`'s existing longest-diameter/opposing-center
cue as-is).

`fusion.bronchotrack_id` (the OTHER identification cue) already reduces
each mask to a single point (its opposing-boundary center) plus, for the
motion model, a single scalar size. This module instead keeps both size
and distance as measurements over the ENTIRE detected boundary, not a
point or a pair of extreme points:

  * Size: `utils.mask_equivalent_diameter` -- the diameter of the circle
    with the same AREA as the whole mask polygon (a shoelace-formula
    integral over every vertex), not `polygon_longest_diameter`'s max
    pairwise vertex distance (which depends on only TWO of the mask's
    points) and not a bounding box.

  * Distance: `mask_boundary_gap` (below) -- the nearest distance between
    ANY point on the candidate detection's own mask boundary and ANY
    point on the reference detection's mask boundary. This is a genuinely
    different quantity from "distance between the two centers"
    (`bronchotrack_id`'s / `paper_exact.association`'s convention): two
    large, close-together lumens and two small, far-apart lumens can have
    the same center-to-center distance while having very different
    boundary-to-boundary gaps, and it's the gap -- how much open airway
    wall actually separates the two openings -- that a real bronchoscopy
    frame's apparent "how far apart do these look" is more directly
    tracking, especially for large or irregularly-shaped masks where the
    center point is a poor stand-in for the whole shape's extent.

Graph-side counterpart
------------------------
The airway graph has no rendered surface to take a "mask boundary" from,
but it does know each opening's own true radius (`AirwayNode.radius_at`)
at any point along its centerline -- enough to model each opening as a
CIRCLE (center = its projected virtual-viewpoint position, radius = its
own apparent size) and compute the exact same "boundary gap" between two
circles in closed form:

    circle_boundary_gap(c1, r1, c2, r2) = |c1 - c2| - r1 - r2

This is the graph-side analogue of `mask_boundary_gap` that actually uses
BOTH openings' full extents (their radii), not just their center points --
consistent with the image-side computation using the whole mask rather
than a point, even though the graph side is closed-form rather than a
point-cloud nearest-neighbor search (there's no denser boundary
representation to search over without a full mesh, which a JSON-only
graph may not have).

The reference opening (the branch the scope is currently sitting in, at
its own bifurcation) is modeled as a circle of radius
`graph.get(ref_label).radius_at(1.0)` (its own true radius right at the
bifurcation) centered at the origin of `project_children_at_distance`'s
own coordinate system (that function's origin already IS the reference
branch's distal end). Each candidate child is modeled as a circle of
radius `graph.child_apparent_equivalent_diameter_at_distance(...) / 2`
(the foreshortening-corrected, AREA-based apparent radius -- see that
method's docstring) centered at its own projected virtual-viewpoint
position.

Cross-domain normalization (same fix as `paper_exact.association`, same
reason: image space is pixels, graph space is millimetres, related by an
unknown, unmeasured camera scale factor). Both domains' diameter:distance
ratios are normalized by their own mean before comparison -- see that
module's docstring's "That angular width is directly comparable..."
section for the derivation; identical reasoning applies here unchanged,
just with this module's own size/distance measurements substituted in.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..graph import AirwayGraph
from ..types import Tracklet
from ..utils import mask_equivalent_diameter

_LOG_RATIO_CAP = 2.0
_MIN_RATIO = 1e-3
_MIN_DIST = 1e-6
NEUTRAL_COST = 0.5  # no radius data for a candidate -> don't penalize, let the other cue decide


def mask_boundary_gap(poly_a: Optional[np.ndarray], poly_b: Optional[np.ndarray]) -> Optional[float]:
    """Nearest distance between any point on `poly_a`'s boundary and any
    point on `poly_b`'s boundary -- an O(n*m) pairwise search over every
    vertex of both polygons (masks are small, tens of points, so this is
    cheap; no need for a spatial index at this scale). Returns None if
    either polygon is missing or empty."""
    if poly_a is None or len(poly_a) == 0 or poly_b is None or len(poly_b) == 0:
        return None
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    diffs = a[:, None, :] - b[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return float(dists.min())


def circle_boundary_gap(center_a: np.ndarray, radius_a: float, center_b: np.ndarray, radius_b: float) -> float:
    """Gap between two circles' boundaries; clamped to `_MIN_DIST` rather
    than allowed to go to zero/negative (overlapping circles), which
    would blow up the ratio this feeds into."""
    center_dist = float(np.linalg.norm(np.asarray(center_a) - np.asarray(center_b)))
    return max(center_dist - radius_a - radius_b, _MIN_DIST)


def _reference_opening_radius(graph: AirwayGraph, ref_label: str) -> Optional[float]:
    node = graph.get(ref_label)
    r = node.radius_at(1.0)
    return r


def ratio_cost_matrix(
    graph: AirwayGraph,
    ref_label: str,
    candidate_labels: List[str],
    reference_detection: Tracklet,
    detections: List[Tracklet],
    virtual_advance_mm: float,
) -> Optional[np.ndarray]:
    """(len(candidate_labels), len(detections)) cost matrix, 0 = the two
    domains' peer-normalized diameter:boundary-gap ratios agree exactly.
    Returns None if no candidate has graph radius data at all, if
    `reference_detection` has no mask (this cue needs the reference's own
    full boundary too, unlike `bronchotrack_id` which only needs its
    center point), or if there are no detections to score at all (a
    reference-only frame -- nothing else currently visible)."""
    if not candidate_labels or not detections:
        return None
    if reference_detection.mask is None:
        return None

    ref_radius = _reference_opening_radius(graph, ref_label)
    if ref_radius is None:
        return None

    def _graph_diameter(label: str) -> float:
        v = graph.child_apparent_equivalent_diameter_at_distance(ref_label, label, virtual_advance_mm)
        return v if v is not None else np.nan

    graph_diam = np.array([_graph_diameter(l) for l in candidate_labels], dtype=np.float64)
    valid = ~np.isnan(graph_diam)
    if not valid.any():
        return None

    virtual_pts = graph.project_children_at_distance(ref_label, virtual_advance_mm)
    origin = np.zeros(2)
    graph_dist = np.array(
        [
            circle_boundary_gap(origin, ref_radius, virtual_pts.get(l, np.zeros(2)), graph_diam[i] / 2.0)
            if valid[i]
            else np.nan
            for i, l in enumerate(candidate_labels)
        ],
        dtype=np.float64,
    )
    ratio_graph = graph_diam / graph_dist
    mean_graph_ratio = np.nanmean(ratio_graph[valid])
    norm_ratio_graph = ratio_graph / max(mean_graph_ratio, _MIN_RATIO)

    def _image_diameter(d: Tracklet) -> float:
        v = mask_equivalent_diameter(d.mask)
        return v if v is not None else (d.last_box.w + d.last_box.h) / 2.0

    def _image_distance(d: Tracklet) -> float:
        v = mask_boundary_gap(d.mask, reference_detection.mask)
        if v is not None:
            return v
        return float(
            np.linalg.norm(
                np.array([d.last_box.x_c, d.last_box.y_c])
                - np.array([reference_detection.last_box.x_c, reference_detection.last_box.y_c])
            )
        )

    image_diam = np.array([_image_diameter(d) for d in detections], dtype=np.float64)
    image_dist = np.array([_image_distance(d) for d in detections], dtype=np.float64)
    image_dist = np.maximum(image_dist, _MIN_DIST)
    ratio_image = image_diam / image_dist
    mean_image_ratio = max(float(np.mean(ratio_image)), _MIN_RATIO)
    norm_ratio_image = ratio_image / mean_image_ratio

    log_graph = np.log(np.maximum(norm_ratio_graph, _MIN_RATIO))
    log_image = np.log(np.maximum(norm_ratio_image, _MIN_RATIO))
    diff = np.abs(log_graph[:, None] - log_image[None, :])
    cost = np.minimum(diff, _LOG_RATIO_CAP) / _LOG_RATIO_CAP

    cost[~valid, :] = NEUTRAL_COST
    return cost
