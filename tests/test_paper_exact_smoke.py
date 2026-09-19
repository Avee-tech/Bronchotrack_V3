"""Smoke / sanity tests for bronchotrack.paper_exact (the strict paper
reference build). Mirrors tests/test_pipeline_smoke.py's non-ML pipeline
smoke test, minus anything related to the motion model (doesn't exist
here), plus a dedicated test for the roll-angle estimation this package
adds that the main package doesn't have.

Runnable either with pytest (`pytest tests/`) or directly:
    python3 tests/test_paper_exact_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bronchotrack.graph import AirwayGraph, AirwayNode
from bronchotrack.tracker import MultiLumenTracker
from bronchotrack.types import BBox, Detection, Tracklet
from bronchotrack.utils import (
    iou,
    mask_equivalent_diameter,
    polygon_area,
    polygon_contains_point,
    polygon_longest_diameter,
    polygon_opposing_center,
)
from bronchotrack.paper_exact.association import AirwayAssociation
from bronchotrack.paper_exact.localization import Localizer
from bronchotrack.paper_exact.pipeline import BronchoTrackPipeline, FrameResult

EXAMPLE_GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "example_graph.json"
)


def test_localizer_has_no_stickiness():
    """Unlike the main package's Localizer, this one must report the raw
    per-frame winner immediately -- no history-based hysteresis."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    loc = Localizer(graph)
    assert not hasattr(loc, "smoothing_window")
    assert not hasattr(loc, "_apply_smoothing")


def test_association_has_no_diameter_cue_or_motion_extras():
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    assoc = AirwayAssociation(graph)
    for attr in ("diameter_weight", "_diameter_ratio_cost_matrix", "approach_trend", "flip_v"):
        assert not hasattr(assoc, attr), f"paper_exact.association still has {attr!r}"
    assert hasattr(assoc, "roll") and assoc.roll == 0.0


class _FakeTracklet:
    def __init__(self, track_id, x, y, age, label="LMB"):
        self.track_id = track_id
        self.label = label  # just needs to be non-None and in-graph
        self._age = age
        self.last_box = BBox(x_c=x, y_c=y, h=10.0, a=1.0)

    @property
    def age_frames(self):
        return self._age


def _rotate_pair_positions(mid, base_positions, theta):
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    out = []
    for x0, y0 in base_positions:
        v = np.array([x0, y0]) - mid
        new = mid + rot @ v
        out.append((float(new[0]), float(new[1])))
    return out


def test_roll_angle_estimation_converges_to_sustained_rotation():
    """Synthetic check for the roll-angle logic (Eq. 6/7), NOW Kalman-
    smoothed (see association.py module docstring point 2): if the two
    reference tracklets' image-plane positions rotate by a known angle and
    STAY there (a real, sustained camera roll), the filtered `roll`
    estimate should converge close to that angle after a few frames --
    not jump there in one step, since a Kalman filter blends each new
    measurement with its prior rather than trusting it outright."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    assoc = AirwayAssociation(graph)

    mid = np.array([100.0, 100.0])
    base = [(110.0, 100.0), (90.0, 100.0)]
    t1 = _FakeTracklet(1, *base[0], age=10)
    t2 = _FakeTracklet(2, *base[1], age=9)

    assoc._update_roll([t1, t2])
    assert assoc.roll == 0.0  # first sight of this pair: reference adopted, no measurement yet

    theta = np.radians(30.0)
    (x1, y1), (x2, y2) = _rotate_pair_positions(mid, base, theta)
    t1.last_box, t2.last_box = BBox(x1, y1, 10.0, 1.0), BBox(x2, y2, 10.0, 1.0)

    # one measurement should move the estimate toward theta, but a Kalman
    # filter blends with its prior rather than jumping straight there
    assoc._update_roll([t1, t2])
    assert 0.0 < assoc.roll < theta

    # holding the same (sustained) rotation for several more frames should
    # converge the filtered estimate close to the true angle
    for _ in range(15):
        assoc._update_roll([t1, t2])
    assert abs(assoc.roll - theta) < 0.02, f"expected roll~={theta:.4f}, got {assoc.roll:.4f}"


def test_roll_angle_kalman_filter_attenuates_measurement_noise():
    """The whole point of adding a Kalman filter here: a noisy per-frame
    roll measurement should produce a LESS noisy filtered `roll` series
    than the raw measurements would be on their own."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    assoc = AirwayAssociation(graph)

    mid = np.array([100.0, 100.0])
    base = [(110.0, 100.0), (90.0, 100.0)]
    t1 = _FakeTracklet(1, *base[0], age=10)
    t2 = _FakeTracklet(2, *base[1], age=9)
    assoc._update_roll([t1, t2])  # adopt reference pair

    rng = np.random.RandomState(0)
    true_theta = np.radians(15.0)
    filtered_series = []
    raw_series = []
    for _ in range(60):
        noisy_theta = true_theta + np.radians(rng.randn() * 4.0)  # ~4deg noise stddev
        (x1, y1), (x2, y2) = _rotate_pair_positions(mid, base, noisy_theta)
        t1.last_box, t2.last_box = BBox(x1, y1, 10.0, 1.0), BBox(x2, y2, 10.0, 1.0)
        assoc._update_roll([t1, t2])
        filtered_series.append(assoc.roll)
        raw_series.append(noisy_theta)

    # compare variance in the back half (after initial convergence) --
    # the filtered series should be visibly smoother than the raw signal
    filtered_var = np.var(filtered_series[20:])
    raw_var = np.var(raw_series[20:])
    assert filtered_var < raw_var * 0.5, (
        f"expected filtered variance well below raw variance, got "
        f"filtered={filtered_var:.6f} raw={raw_var:.6f}"
    )
    # and it should still be centered near the true angle, not just flat/stuck
    assert abs(np.mean(filtered_series[20:]) - true_theta) < 0.03


def _synthetic_bifurcation_graph():
    """Root with two children at the SAME bearing (directly opposite along
    x, symmetric -- see below) but different radii: 'big' (radius 4mm) and
    'small' (radius 1mm), used to check the diameter:distance cue can
    disambiguate candidates that angle-only matching can't."""
    root = AirwayNode(
        label="root", generation=0, parent=None,
        start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
    )
    big = AirwayNode(
        label="big", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([10.0, 5.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]]),
        radius=np.array([4.0, 4.0]),
    )
    small = AirwayNode(
        label="small", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([-10.0, 5.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [-10.0, 5.0, 0.0]]),
        radius=np.array([1.0, 1.0]),
    )
    return AirwayGraph([root, big, small])


def test_diameter_distance_cue_disambiguates_equal_bearing_candidates():
    """With angle-only matching, two candidates at symmetric bearings from
    the anchor are ambiguous by construction in this test (both offsets
    have identical magnitude in the projected plane, and the Hungarian
    solver has no size information to break the tie correctly). The
    diameter:distance cue should push a bigger detection to match 'big'
    and a smaller detection to match 'small'."""
    graph = _synthetic_bifurcation_graph()
    assoc = AirwayAssociation(graph, distance_diameter_weight=0.8)

    labels = ["big", "small"]
    ref_label = "root"
    virtual_pts = graph.project_children_at_distance(ref_label, assoc.virtual_advance_mm)
    graph_pts = np.stack([virtual_pts[l] for l in labels], axis=0)

    class FakeTracklet:
        def __init__(self, track_id, x, y, w, h):
            self.track_id = track_id
            self.label = None
            self.label_history = []
            self.last_box = BBox(x_c=x, y_c=y, h=h, a=w / h)
            self.mask = None
            self.diameter_distance_match = None

    anchor = FakeTracklet(0, x=0.0, y=0.0, w=20.0, h=20.0)  # at the origin -- image_offsets below are relative to it

    # detection A: large box (should match 'big'); detection B: small box
    # (should match 'small'), positioned so their offsets from the anchor
    # mirror the graph_pts' bearing pattern exactly (zero roll) with EQUAL
    # magnitude to each other -- so angle-only matching alone can't tell
    # them apart by distance either, only bearing direction + size must
    # do the disambiguating work.
    mag = np.linalg.norm(graph_pts, axis=1).mean()
    unit_big = graph_pts[0] / np.linalg.norm(graph_pts[0])
    unit_small = graph_pts[1] / np.linalg.norm(graph_pts[1])
    off_big, off_small = unit_big * mag, unit_small * mag

    det_big = FakeTracklet(1, x=off_big[0], y=off_big[1], w=40.0, h=40.0)
    det_small = FakeTracklet(2, x=off_small[0], y=off_small[1], w=10.0, h=10.0)
    detections = [det_big, det_small]

    result = assoc._diameter_distance_cost_matrix(labels, ref_label, anchor, detections)
    assert result is not None
    diam_cost, norm_ratio_graph, norm_ratio_image, valid = result
    assert valid.all()
    # 'big' candidate (row 0) should cost less against det_big (col 0) than
    # against det_small (col 1), and vice versa for 'small' (row 1).
    assert diam_cost[0, 0] < diam_cost[0, 1]
    assert diam_cost[1, 1] < diam_cost[1, 0]


def test_diameter_distance_cue_skips_gracefully_without_radius_data():
    """A graph with no radius data at all (like examples/example_graph.json)
    must make the cue return None so matching falls back to bearing-only,
    not crash."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    assoc = AirwayAssociation(graph, distance_diameter_weight=0.8)
    labels = graph.children(graph.root())

    class FakeTracklet:
        def __init__(self, track_id, x=0.0):
            self.track_id = track_id
            self.last_box = BBox(x_c=x, y_c=0.0, h=20.0, a=1.0)
            self.mask = None

    anchor = FakeTracklet(0, x=0.0)
    detections = [FakeTracklet(i + 1, x=float(i + 1) * 10.0) for i in range(len(labels))]
    result = assoc._diameter_distance_cost_matrix(labels, graph.root(), anchor, detections)
    assert result is None


def test_bootstrap_labels_get_diameter_distance_verified():
    """The carina bootstrap (_try_initialize) hands out labels directly,
    with no Hungarian matching step and no pre-existing anchor -- so
    without _verify_bootstrap_matches, these tracklets' diameter_distance_
    match would sit at its default None forever (they never become
    "unlabeled candidates" being matched again, the only other place the
    verdict gets set), and since viz.py only draws a dot once it's True,
    the very first, most load-bearing labels of the whole run would never
    be displayed at all no matter how correct they are."""
    graph = _synthetic_bifurcation_graph()
    assoc = AirwayAssociation(graph, distance_diameter_weight=0.8)

    ref_label = "root"
    virtual_pts = graph.project_children_at_distance(ref_label, assoc.virtual_advance_mm)
    labels = ["big", "small"]
    graph_pts = np.stack([virtual_pts[l] for l in labels], axis=0)

    class FakeTracklet:
        def __init__(self, track_id, x, y, w, h):
            self.track_id = track_id
            self.label = None
            self.label_history = []
            self.last_box = BBox(x_c=x, y_c=y, h=h, a=w / h)
            self.mask = None
            self.diameter_distance_match = None

    mag = np.linalg.norm(graph_pts, axis=1).mean()
    unit_big = graph_pts[0] / np.linalg.norm(graph_pts[0])
    unit_small = graph_pts[1] / np.linalg.norm(graph_pts[1])
    off_big, off_small = unit_big * mag, unit_small * mag

    scale = 3.0
    diam_big = 2.0 * graph.get("big").radius_at(1.0) * scale
    diam_small = 2.0 * graph.get("small").radius_at(1.0) * scale
    det_big = FakeTracklet(1, x=off_big[0], y=off_big[1], w=diam_big, h=diam_big)
    det_small = FakeTracklet(2, x=off_small[0], y=off_small[1], w=diam_small, h=diam_small)

    assert det_big.diameter_distance_match is None  # sanity: starts unset
    assert det_small.diameter_distance_match is None

    assoc._try_initialize([det_big, det_small], frame_idx=0)

    assert assoc.initialized
    assert det_big.label is not None and det_small.label is not None
    assert det_big.diameter_distance_match is not None, "bootstrap label never got verified"
    assert det_small.diameter_distance_match is not None, "bootstrap label never got verified"


def test_virtual_verification_match_flag():
    """AirwayAssociation._match_candidates should mark a matched detection's
    Tracklet.diameter_distance_match True when its real-image
    diameter:distance ratio agrees with the 3D model's virtual-viewpoint
    ratio within `virtual_match_threshold`, and False when it doesn't --
    the 'verify against the 3D model, output true/false' step."""
    graph = _synthetic_bifurcation_graph()
    assoc = AirwayAssociation(graph, distance_diameter_weight=0.8, virtual_match_threshold=0.75)

    ref_label = "root"
    virtual_pts = graph.project_children_at_distance(ref_label, assoc.virtual_advance_mm)
    labels = ["big", "small"]
    graph_pts = np.stack([virtual_pts[l] for l in labels], axis=0)

    class FakeTracklet:
        def __init__(self, track_id, x, y, w, h, label=None):
            self.track_id = track_id
            self.label = label
            self.label_history = []
            self.last_box = BBox(x_c=x, y_c=y, h=h, a=w / h)
            self.mask = None
            self.diameter_distance_match = None

    anchor = FakeTracklet(0, x=0.0, y=0.0, w=20.0, h=20.0, label="root")

    mag = np.linalg.norm(graph_pts, axis=1).mean()
    unit_big = graph_pts[0] / np.linalg.norm(graph_pts[0])
    unit_small = graph_pts[1] / np.linalg.norm(graph_pts[1])
    off_big, off_small = unit_big * mag, unit_small * mag

    # sized so ratio_image and ratio_graph agree closely -- same constant
    # scale factor applied to both, which peer-normalization cancels exactly
    scale = 3.0
    diam_big = 2.0 * graph.get("big").radius_at(1.0) * scale
    diam_small = 2.0 * graph.get("small").radius_at(1.0) * scale
    det_big = FakeTracklet(1, x=off_big[0], y=off_big[1], w=diam_big, h=diam_big)
    det_small = FakeTracklet(2, x=off_small[0], y=off_small[1], w=diam_small, h=diam_small)

    matched = assoc._match_candidates(anchor, labels, [det_big, det_small], used_labels=set(), frame_idx=0)
    assert matched == {1, 2}
    assert det_big.label == "big" and det_small.label == "small"
    assert det_big.diameter_distance_match is True
    assert det_small.diameter_distance_match is True

    # now distort det_small's own size (1.5x) enough to break the 75%
    # agreement threshold, without distorting it so much the pairing gets
    # rejected outright by max_match_cost -- should still get labeled by
    # bearing, but flagged as NOT verified by the 3D model.
    det_small_bad = FakeTracklet(2, x=off_small[0], y=off_small[1], w=diam_small * 1.5, h=diam_small * 1.5)
    matched2 = assoc._match_candidates(anchor, labels, [det_big, det_small_bad], used_labels=set(), frame_idx=1)
    assert 2 in matched2
    assert det_small_bad.label == "small"
    assert det_small_bad.diameter_distance_match is False


def test_apparent_diameter_foreshortening():
    """AirwayGraph.child_apparent_diameter_at_distance should read close to
    the branch's true diameter for a child that continues straight ahead
    of the parent's own axis (foreshortening factor ~= 1.0), and
    materially smaller for a child that peels off at a steep angle --
    using the PARENT's fixed forward axis (end_tangent) as the viewing
    direction, not a ray recomputed per child from the bifurcation to that
    child's own sampled point. That per-child-ray formulation was this
    method's original (buggy) implementation: for a roughly straight
    branch, a ray from the bifurcation to a point further down that same
    branch is itself nearly parallel to the branch's own tangent by
    construction, so it always reported foreshortening ~= 1.0 regardless
    of takeoff angle -- silently never correcting anything. This test
    pins down the fix by checking a steeply-angled child actually reads
    smaller than its true diameter."""
    root = AirwayNode(
        label="root", generation=0, parent=None,
        start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
    )
    # 'straight': continues directly ahead of the parent's own +y axis.
    straight = AirwayNode(
        label="straight", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, 10.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
        radius=np.array([3.0, 3.0]),
    )
    # 'sideways': peels off at 90 degrees from the parent's own axis.
    sideways = AirwayNode(
        label="sideways", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([10.0, 0.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        radius=np.array([3.0, 3.0]),
    )
    graph = AirwayGraph([root, straight, sideways])

    true_diam = 6.0  # 2 * radius (constant along both branches here)
    d_straight = graph.child_apparent_diameter_at_distance("root", "straight", 5.0)
    d_sideways = graph.child_apparent_diameter_at_distance("root", "sideways", 5.0)

    assert d_straight is not None and d_sideways is not None
    assert d_straight > 0.95 * true_diam, d_straight  # nearly face-on: ~= true diameter
    assert d_sideways < 0.35 * true_diam, d_sideways  # edge-on: sharply foreshortened
    assert d_straight > d_sideways * 2  # unambiguously distinguishes the two takeoff angles


def test_ratio_smoothing_attenuates_noise_and_updates_once_per_frame():
    """The diameter:distance cue's per-tracklet ratio_image is Kalman-
    smoothed (see association.py module docstring). Checks: (1) a noisy
    series of raw ratios produces a smoother filtered series, and (2)
    calling `_smoothed_ratio` twice for the same track_id + frame_idx
    (simulating a tracklet considered in both a 'children' and 'siblings'
    match this frame) does NOT double-update the filter -- both calls
    should return the identical value."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    assoc = AirwayAssociation(graph)

    rng = np.random.RandomState(0)
    true_ratio = 2.0
    filtered_series, raw_series = [], []
    for frame_idx in range(50):
        noisy = true_ratio + rng.randn() * 0.3
        v = assoc._smoothed_ratio(track_id=7, raw_ratio=noisy, frame_idx=frame_idx)
        filtered_series.append(v)
        raw_series.append(noisy)

    filtered_var = np.var(filtered_series[15:])
    raw_var = np.var(raw_series[15:])
    assert filtered_var < raw_var * 0.75, (
        f"expected filtered variance meaningfully below raw variance, got "
        f"filtered={filtered_var:.4f} raw={raw_var:.4f}"
    )
    assert abs(np.mean(filtered_series[15:]) - true_ratio) < 0.15

    # same-frame double call (children pass, then siblings pass) must be a no-op
    v_a = assoc._smoothed_ratio(track_id=7, raw_ratio=99.0, frame_idx=50)
    v_b = assoc._smoothed_ratio(track_id=7, raw_ratio=-50.0, frame_idx=50)
    assert v_a == v_b, "second call within the same frame must not re-update the filter"


def test_mask_geometry_helpers():
    """Sanity-check the shared polygon helpers (utils.py) that back
    segmentation support: a square's shoelace area is exact, its
    equivalent diameter is smaller than its own bounding-box side (a
    circle of that area fits inside the square), and point-in-polygon
    agrees with the obvious cases."""
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert abs(polygon_area(square) - 100.0) < 1e-6

    d = mask_equivalent_diameter(square)
    assert d is not None and 10.0 < d < 12.0  # circle of area 100 has diameter ~11.28

    assert polygon_contains_point(square, (5.0, 5.0)) is True
    assert polygon_contains_point(square, (50.0, 50.0)) is False
    assert mask_equivalent_diameter(None) is None
    assert polygon_contains_point(None, (0.0, 0.0)) is False


def test_detection_diameter_prefers_mask_over_box():
    """_detection_diameter should use the mask's own LONGEST diameter (max
    distance between any two of its boundary points) when a mask is
    present, and only fall back to max(box.w, box.h) when it's not.

    The box fixture here is deliberately NON-square (40 x 20) so the box
    fallback actually exercises max() rather than happening to agree with
    an average by coincidence -- max(w, h) = 40, but (w+h)/2 would be 30,
    so this also pins down that the fallback is the larger side, not the
    average."""
    # a thin diagonal sliver spanning the box corner to corner: its own
    # longest extent (corner to corner) is materially bigger than either
    # the box's larger side or its averaged (w+h)/2.
    diagonal_sliver = np.array([[0.0, 0.0], [40.0, 20.0], [38.0, 20.0], [-2.0, 0.0]])
    box = BBox.from_xyxy(0.0, 0.0, 40.0, 20.0)  # w=40, h=20 -- max=40, (w+h)/2=30

    class FakeTracklet:
        def __init__(self, last_box, mask):
            self.last_box = last_box
            self.mask = mask

    with_mask = FakeTracklet(box, diagonal_sliver)
    without_mask = FakeTracklet(box, None)

    d_mask = AirwayAssociation._detection_diameter(with_mask)
    d_box = AirwayAssociation._detection_diameter(without_mask)
    assert d_box == max(box.w, box.h) == 40.0
    assert d_box != (box.w + box.h) / 2.0, "box fallback should be the larger side, not the average"
    assert d_mask > d_box, (
        "longest-diameter should reflect the sliver's true corner-to-corner "
        "extent, which exceeds even the box's larger side"
    )


def test_detection_center_prefers_mask_over_box():
    """_detection_center should use the opposing-point-averaged mask center
    when a mask is present, and the box center otherwise."""
    box = BBox.from_xyxy(0.0, 0.0, 10.0, 10.0)  # box center = (5, 5)

    class FakeTracklet:
        def __init__(self, last_box, mask):
            self.last_box = last_box
            self.mask = mask

    # a mask badly off-center from its own bounding box -- a thin sliver
    # hugging one corner -- so mask-center and box-center clearly disagree.
    corner_sliver = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 3.0]])
    with_mask = FakeTracklet(box, corner_sliver)
    without_mask = FakeTracklet(box, None)

    c_mask = AirwayAssociation._detection_center(with_mask)
    c_box = AirwayAssociation._detection_center(without_mask)
    assert np.allclose(c_box, [5.0, 5.0])
    assert not np.allclose(c_mask, c_box, atol=0.5), "mask-derived center should differ from the box's for an off-center mask"
    # the mask-derived center should stay near the sliver itself, not out
    # at the box's own (unrelated) center
    assert c_mask[0] < 3.0 and c_mask[1] < 3.0


def test_longest_diameter_and_opposing_center():
    """New geometry helpers backing the diameter:distance cue's rework:
    longest diameter (max pairwise vertex distance) and the opposing-
    point-averaged center (robust to a lopsided polygon in a way a plain
    centroid isn't)."""
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert abs(polygon_longest_diameter(square) - 10.0 * np.sqrt(2)) < 1e-6
    center = polygon_opposing_center(square)
    assert center is not None
    assert np.allclose(center, [5.0, 5.0], atol=1e-6)

    # a lopsided L-shape: the "open space" center found by averaging
    # opposing boundary points should differ from the plain vertex
    # centroid, which gets dragged toward whichever arm has more detail.
    l_shape = np.array([[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]])
    plain_centroid = l_shape.mean(axis=0)
    opp_center = polygon_opposing_center(l_shape)
    assert opp_center is not None
    assert not np.allclose(opp_center, plain_centroid, atol=1.0)

    assert polygon_longest_diameter(None) is None
    assert polygon_opposing_center(None) is None
    assert polygon_longest_diameter(np.array([[0.0, 0.0]])) is None  # < 2 points
    assert polygon_opposing_center(np.array([[0.0, 0.0], [1.0, 1.0]])) is None  # < 3 points


def test_is_nested_uses_mask_when_available():
    """_is_nested should use point-in-polygon against the outer tracklet's
    mask when it has one, rather than box-in-box containment -- checked
    with a case where the two disagree: an L-shaped mask whose bounding
    box contains a point the mask itself does NOT cover."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    assoc = AirwayAssociation(graph)

    # L-shaped polygon: box-covers (80, 80) but the L-shape itself doesn't
    # -- bottom strip (x:0-100, y:0-40) plus left column (x:0-40, y:40-100),
    # leaving the (x:40-100, y:40-100) corner as an excluded notch.
    l_shape = np.array([[0, 0], [100, 0], [100, 40], [40, 40], [40, 100], [0, 100]])
    outer_box = BBox.from_xyxy(0, 0, 100, 100)

    class FakeTracklet:
        def __init__(self, last_box, mask):
            self.last_box = last_box
            self.mask = mask

    outer = FakeTracklet(outer_box, l_shape)
    inner_in_notch = FakeTracklet(BBox(x_c=80.0, y_c=80.0, h=4.0, a=1.0), None)  # inside the box, NOT the L
    inner_in_mask = FakeTracklet(BBox(x_c=20.0, y_c=20.0, h=4.0, a=1.0), None)  # inside both

    assert assoc._is_nested(outer, inner_in_notch) is False  # mask correctly excludes it
    assert assoc._is_nested(outer, inner_in_mask) is True

    # without a mask on `outer`, falls back to box containment (both points
    # are inside outer_box, so both should now read as nested)
    outer_no_mask = FakeTracklet(outer_box, None)
    assert assoc._is_nested(outer_no_mask, inner_in_notch) is True


def test_overlay_only_draws_confirmed_matches():
    """draw_overlay should only draw a dot for tracklets whose
    diameter_distance_match is True -- the design that lets a permissive
    detector threshold (more raw candidate detections admitted) be used
    without cluttering the video, by leaning on the Kalman-filtered
    virtual-model comparison to gate what's actually shown. `False`
    (visible + labeled but disagreeing with the 3D model) and `None`
    (cue disabled, or no verdict yet) must both be suppressed, not just
    unlabeled tracklets -- see viz.py's module docstring."""
    from bronchotrack.paper_exact.viz import draw_overlay

    class FakeTracklet:
        def __init__(self, track_id, x, y, label, diameter_distance_match):
            self.track_id = track_id
            self.time_since_update = 0
            self.mask = None
            self.last_box = BBox(x_c=x, y_c=y, h=20.0, a=1.0)
            self.label = label
            self.diameter_distance_match = diameter_distance_match

    class FakeResult:
        def __init__(self, tracklets):
            self.tracklets = tracklets
            self.location = "RUL"
            self.generation = 3

    confirmed = FakeTracklet(1, x=50, y=50, label="RUL", diameter_distance_match=True)
    rejected = FakeTracklet(2, x=150, y=50, label="RML", diameter_distance_match=False)
    unverified = FakeTracklet(3, x=250, y=50, label="LUL", diameter_distance_match=None)

    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    result = FakeResult([confirmed, rejected, unverified])
    out = draw_overlay(frame, result)

    # confirmed tracklet's dot pixel should differ from the untouched background
    assert not np.array_equal(out[50, 50], frame[50, 50])
    # rejected (False) and unverified (None) tracklets must leave their region untouched
    assert np.array_equal(out[50, 150], frame[50, 150])
    assert np.array_equal(out[50, 250], frame[50, 250])


def test_overlay_draws_full_mask_polygon_when_available():
    """When a confirmed tracklet has a segmentation polygon, draw_overlay
    must render the actual mask shape (translucent fill + outline) rather
    than just the fixed-radius center dot -- an explicit request to show
    the whole lumen outline instead of a marker. A tracklet with no mask
    must still fall back to the old dot marker (covered by
    test_overlay_only_draws_confirmed_matches above), so this test checks
    the two paths are actually different: a masked tracklet should paint
    pixels far outside where an 8px-radius dot could ever reach, since the
    polygon here is much larger than that."""
    from bronchotrack.paper_exact.viz import draw_overlay, _DOT_RADIUS

    class FakeTracklet:
        def __init__(self, track_id, mask, last_box, label, diameter_distance_match):
            self.track_id = track_id
            self.time_since_update = 0
            self.mask = mask
            self.last_box = last_box
            self.label = label
            self.diameter_distance_match = diameter_distance_match

    class FakeResult:
        def __init__(self, tracklets):
            self.tracklets = tracklets
            self.location = "RUL"
            self.generation = 3

    # a 40x40 square polygon centered on (150, 50) -- well larger than the
    # _DOT_RADIUS=8 fallback marker, so a pixel near its edge (but outside
    # any possible dot) is only touched if the actual polygon was drawn.
    square = np.array(
        [[130.0, 30.0], [170.0, 30.0], [170.0, 70.0], [130.0, 70.0]], dtype=np.float64
    )
    masked = FakeTracklet(
        1, mask=square, last_box=BBox(x_c=150, y_c=50, h=40.0, a=1.0),
        label="RUL", diameter_distance_match=True,
    )

    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    result = FakeResult([masked])
    out = draw_overlay(frame, result)

    # near the polygon's edge, far outside where an 8px dot at the center
    # could reach -- only the full mask outline/fill would touch this pixel
    edge_x, edge_y = 132, 50
    assert (edge_x - 150) ** 2 + (edge_y - 50) ** 2 > _DOT_RADIUS ** 2
    assert not np.array_equal(out[edge_y, edge_x], frame[edge_y, edge_x])

    # the fill should also be translucent, not opaque -- somewhere inside
    # the polygon but off the outline itself, the drawn color should be a
    # blend of the track color and the (black) background, not pure black
    # and not the fully-saturated track color either
    inside_x, inside_y = 150, 50
    blended = out[inside_y, inside_x]
    assert not np.array_equal(blended, frame[inside_y, inside_x])
    assert int(blended.max()) < 255


def test_json_log_exports_diameter_distance_match():
    """The per-frame JSON log (--out-json) must expose
    Tracklet.diameter_distance_match -- the same field viz.py's overlay
    gates the on-screen dot on (only True gets drawn). Without this in the
    log, nothing on disk distinguishes a tracklet that was actually
    confirmed against the 3D model from one that was merely tracked and
    labeled but suppressed from the video for disagreeing with it."""
    class FakeTracklet:
        def __init__(self, track_id, label, diameter_distance_match, conf=0.9):
            self.track_id = track_id
            self.label = label
            self.last_box = BBox(x_c=10.0, y_c=10.0, h=5.0, a=1.0)
            self.confidences = [conf]
            self.time_since_update = 0
            self.diameter_distance_match = diameter_distance_match

    confirmed = FakeTracklet(1, "RUL", True)
    rejected = FakeTracklet(2, "RML", False)
    unverified = FakeTracklet(3, "LUL", None)

    result = FrameResult(
        frame_idx=0,
        detections=[],
        tracklets=[confirmed, rejected, unverified],
        labels={},
        location="RUL",
        generation=3,
        location_is_live=True,
    )
    entry = BronchoTrackPipeline._frame_log_entry(result)
    by_id = {t["track_id"]: t for t in entry["tracklets"]}

    assert by_id[1]["diameter_distance_match"] is True
    assert by_id[2]["diameter_distance_match"] is False
    assert by_id[3]["diameter_distance_match"] is None


def test_json_log_exports_location_is_live():
    """The per-frame JSON log must also expose whether `location` came from
    a fresh Eq. 8 vote this frame or was carried forward with nothing
    currently visible to reconfirm it (Localizer.last_vote_was_live --
    see localization.py's "Live vs. carried-forward votes" docstring
    section). Without this, a consumer reading only `location` can't tell
    "still here, just reconfirmed" from "no current evidence, repeating
    our last guess" -- a real, silent accuracy risk, not a cosmetic one."""
    result_live = FrameResult(
        frame_idx=5, detections=[], tracklets=[], labels={},
        location="RUL", generation=3, location_is_live=True,
    )
    result_stale = FrameResult(
        frame_idx=6, detections=[], tracklets=[], labels={},
        location="RUL", generation=3, location_is_live=False,
    )
    result_unset = FrameResult(
        frame_idx=7, detections=[], tracklets=[], labels={},
        location=None, generation=None,
    )

    assert BronchoTrackPipeline._frame_log_entry(result_live)["location_is_live"] is True
    assert BronchoTrackPipeline._frame_log_entry(result_stale)["location_is_live"] is False
    assert BronchoTrackPipeline._frame_log_entry(result_unset)["location_is_live"] is None


def test_localizer_reports_live_vs_carried_forward():
    """Localizer.last_vote_was_live() must reflect, frame by frame, whether
    localize() produced a fresh vote or repeated the last known location
    because nothing labeled was currently visible -- the mechanism
    test_json_log_exports_location_is_live above exercises end-to-end via
    FrameResult/the JSON log."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    localizer = Localizer(graph)

    # nothing visible yet: no vote possible, and no prior location either
    assert localizer.localize([]) is None
    assert localizer.last_vote_was_live() is False

    labeled = Tracklet(track_id=1, ind_start=0, ind_end=0, label="RMB", time_since_update=0)
    winner = localizer.localize([labeled])
    assert winner == "RMB"
    assert localizer.last_vote_was_live() is True

    # nothing visible this frame: carries "RMB" forward, but flags it stale
    carried = localizer.localize([])
    assert carried == "RMB"
    assert localizer.last_vote_was_live() is False


def _carina_bootstrap(assoc, tracker, frame_idx, n_frames=3):
    """Shared setup for the re-acquisition tests below: run the standard
    carina bootstrap (two lumens, left and right) for `n_frames`, leaving
    `assoc` initialized with LMB/RMB labeled and `assoc._last_real_anchor`
    populated. Returns the next unused frame_idx."""
    for _ in range(n_frames):
        dets = [
            Detection(bbox=BBox.from_xywh(60, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=BBox.from_xywh(160, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
        ]
        active = tracker.update(dets, frame_idx, eligibility_fn=assoc.eligibility_fn)
        assoc.process_frame(active, frame_idx)
        frame_idx += 1
    return frame_idx


def test_reacquisition_relabels_same_lumen_after_total_anchor_loss():
    """The dead end this feature fixes: once every labeled anchor is
    fully DROPPED by the tracker (not just aged -- >max_time_since_update
    consecutive missed frames, so track_ids 1 and 2 are genuinely gone,
    not merely stale), a brand-new tracklet reappearing at essentially the
    same screen position as the lost anchor must be relabeled with that
    anchor's own label (self re-acquisition via IoU), rather than staying
    unlabeled forever -- confirmed via a real MultiLumenTracker (not a
    fake), so the tracker's own default max_time_since_update=30 is what
    actually drops the old tracklets here."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    tracker = MultiLumenTracker(use_reid=False)
    assoc = AirwayAssociation(graph, reacquire_max_gap_frames=90)

    frame_idx = _carina_bootstrap(assoc, tracker, 0)
    assert assoc._last_real_anchor is not None
    assert assoc._last_real_anchor.label == "LMB"

    # total loss: feed zero detections until the tracker has actually
    # dropped every tracklet (default max_time_since_update=30)
    for _ in range(35):
        active = tracker.update([], frame_idx, eligibility_fn=assoc.eligibility_fn)
        assoc.process_frame(active, frame_idx)
        frame_idx += 1
    assert len(tracker.tracklets) == 0  # confirms this is a genuine drop, not just aging

    # a NEW tracklet (fresh track_id) reappears at the same box the lost
    # LMB anchor was last seen at
    dets = [Detection(bbox=BBox.from_xywh(60, 90, 40, 60), confidence=0.9, frame_idx=frame_idx)]
    active = tracker.update(dets, frame_idx, eligibility_fn=assoc.eligibility_fn)
    assoc.process_frame(active, frame_idx)

    assert len(active) == 1
    assert active[0].track_id not in (1, 2)  # genuinely a new track, not the old one persisting
    assert active[0].label == "LMB"


def test_reacquisition_gives_up_past_max_gap_frames():
    """A gap longer than reacquire_max_gap_frames must NOT relabel --
    the frozen snapshot is considered too stale to trust past that point."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    tracker = MultiLumenTracker(use_reid=False)
    assoc = AirwayAssociation(graph, reacquire_max_gap_frames=90)

    frame_idx = _carina_bootstrap(assoc, tracker, 0)
    for _ in range(100):  # well past reacquire_max_gap_frames=90
        active = tracker.update([], frame_idx, eligibility_fn=assoc.eligibility_fn)
        assoc.process_frame(active, frame_idx)
        frame_idx += 1

    dets = [Detection(bbox=BBox.from_xywh(60, 90, 40, 60), confidence=0.9, frame_idx=frame_idx)]
    active = tracker.update(dets, frame_idx, eligibility_fn=assoc.eligibility_fn)
    assoc.process_frame(active, frame_idx)
    assert active[0].label is None


def test_reacquisition_disabled_when_max_gap_is_zero():
    """reacquire_max_gap_frames=0 must reproduce the original one-shot-
    anchor dead end exactly -- a deliberate escape hatch, not a bug."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    tracker = MultiLumenTracker(use_reid=False)
    assoc = AirwayAssociation(graph, reacquire_max_gap_frames=0)

    frame_idx = _carina_bootstrap(assoc, tracker, 0)
    for _ in range(35):
        active = tracker.update([], frame_idx, eligibility_fn=assoc.eligibility_fn)
        assoc.process_frame(active, frame_idx)
        frame_idx += 1

    dets = [Detection(bbox=BBox.from_xywh(60, 90, 40, 60), confidence=0.9, frame_idx=frame_idx)]
    active = tracker.update(dets, frame_idx, eligibility_fn=assoc.eligibility_fn)
    assoc.process_frame(active, frame_idx)
    assert active[0].label is None


def test_reacquisition_virtual_anchor_matches_new_child_when_self_match_fails():
    """The second tier: if nothing in the current frame resembles the lost
    anchor's own old box closely enough (IoU below threshold -- e.g. the
    scope kept advancing while it was gone), the frozen snapshot should
    still be usable as a virtual anchor to match a genuinely NEW child
    lumen against, using the same bearing geometry a live anchor would."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    tracker = MultiLumenTracker(use_reid=False)
    assoc = AirwayAssociation(graph, reacquire_max_gap_frames=90, reacquire_iou_threshold=0.3)

    frame_idx = _carina_bootstrap(assoc, tracker, 0)
    # _last_real_anchor always snapshots whichever anchor sorts first by
    # age (a tie at the carina, broken by tracker list order -- LMB here,
    # deterministically), so this test builds its child off LMB, not RMB
    assert assoc._last_real_anchor.label == "LMB"
    lmb_track = next(t for t in tracker.tracklets if t.label == "LMB")
    parent_bbox = lmb_track.last_box
    parent_center = np.array([parent_bbox.x_c, parent_bbox.y_c])
    proj = graph.project_children_2d("LMB")
    scale = 3.0
    lul_offset = proj["LUL"] * scale
    child_box = BBox(x_c=parent_center[0] + lul_offset[0], y_c=parent_center[1] + lul_offset[1], h=15, a=1.0)

    for _ in range(35):
        active = tracker.update([], frame_idx, eligibility_fn=assoc.eligibility_fn)
        assoc.process_frame(active, frame_idx)
        frame_idx += 1
    assert len(tracker.tracklets) == 0

    # a new lumen appears only at LUL's expected bearing off LMB's own
    # last-known position -- nothing overlaps LMB's own old box at all,
    # so self re-acquisition (tier 1) can't apply; only the virtual-anchor
    # path (tier 2) can label this
    dets = [Detection(bbox=child_box, confidence=0.9, frame_idx=frame_idx)]
    active = tracker.update(dets, frame_idx, eligibility_fn=assoc.eligibility_fn)
    assoc.process_frame(active, frame_idx)

    assert iou(parent_bbox, active[0].last_box) < 0.3  # confirms tier 1 genuinely couldn't apply
    assert active[0].label == "LUL"


def test_full_pipeline_smoke_no_ml():
    """Runs tracker + paper_exact association + paper_exact localizer
    together across synthetic frames (carina -> down one main bronchus).
    Only asserts the pipeline runs cleanly and produces internally
    consistent output."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    tracker = MultiLumenTracker(use_reid=False)
    association = AirwayAssociation(graph)
    localizer = Localizer(graph)

    frame_idx = 0

    # --- carina: two lumens, left and right ---
    for _ in range(3):
        dets = [
            Detection(bbox=BBox.from_xywh(60, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=BBox.from_xywh(160, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
        ]
        active = tracker.update(dets, frame_idx, eligibility_fn=association.eligibility_fn)
        association.process_frame(active, frame_idx)
        loc = localizer.localize(active)
        frame_idx += 1

    assert association.initialized
    labeled = [t for t in tracker.tracklets if t.label is not None]
    assert len(labeled) == 2
    assert {t.label for t in labeled} == {"LMB", "RMB"}
    assert loc == "trachea"  # two primary-level siblings visible -> vote for parent

    rmb_track = next(t for t in tracker.tracklets if t.label == "RMB")
    other_track = next(t for t in tracker.tracklets if t.label == "LMB")
    initial_box = rmb_track.last_box
    final_box = BBox.from_xyxy(140, 60, 240, 180)

    n_grow = 6
    for step in range(1, n_grow + 1):
        alpha = step / n_grow
        grown_box = BBox(
            x_c=initial_box.x_c + alpha * (final_box.x_c - initial_box.x_c),
            y_c=initial_box.y_c + alpha * (final_box.y_c - initial_box.y_c),
            h=initial_box.h + alpha * (final_box.h - initial_box.h),
            a=initial_box.a + alpha * (final_box.a - initial_box.a),
        )
        other_bbox = other_track.last_box
        dets = [
            Detection(bbox=grown_box, confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=other_bbox, confidence=0.9, frame_idx=frame_idx),
        ]
        active = tracker.update(dets, frame_idx, eligibility_fn=association.eligibility_fn)
        association.process_frame(active, frame_idx)
        loc = localizer.localize(active)
        frame_idx += 1

    parent_bbox = rmb_track.last_box
    parent_center = np.array([parent_bbox.x_c, parent_bbox.y_c])
    proj = graph.project_children_2d("RMB")

    scale = 3.0
    rul_offset = proj["RUL"] * scale
    bi_offset = proj["BI"] * scale
    child1 = BBox(x_c=parent_center[0] + rul_offset[0], y_c=parent_center[1] + rul_offset[1], h=15, a=1.0)
    child2 = BBox(x_c=parent_center[0] + bi_offset[0], y_c=parent_center[1] + bi_offset[1], h=15, a=1.0)

    for _ in range(5):
        other_bbox = other_track.last_box
        dets = [
            Detection(bbox=parent_bbox, confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=child1, confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=child2, confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=other_bbox, confidence=0.9, frame_idx=frame_idx),
        ]
        active = tracker.update(dets, frame_idx, eligibility_fn=association.eligibility_fn)
        association.process_frame(active, frame_idx)
        loc = localizer.localize(active)
        frame_idx += 1

    rmb_children_labels = set(graph.children("RMB"))
    nested_labels = {
        t.label for t in tracker.tracklets if t.label is not None and t.label in rmb_children_labels
    }
    assert nested_labels == rmb_children_labels, f"expected both of RMB's children to get labeled, got {nested_labels}"
    assert loc is not None and loc in graph.all_labels()

    print(
        "[paper_exact smoke] final location:", loc,
        "| all labels seen:", sorted(t.label for t in tracker.tracklets if t.label),
    )


def test_boxmot_adapter_never_spawns_track_from_low_confidence_detection():
    """Regression test for a real false-positive found while A/B testing
    BoxMOT's ByteTrack against the custom tracker on ModelV3_2.mp4: a
    spurious low-confidence duplicate detection recurring frame after frame
    must never be surfaced as its own Tracklet, matching tracker.py's own
    rule (unmatched low-confidence detections are discarded, never spawn a
    tracklet) -- see boxmot_adapter.py's module docstring for the full
    story of how BoxMOT's own `min_hits` alone did NOT prevent this."""
    try:
        from bronchotrack.paper_exact.boxmot_adapter import BoxMotByteTrackAdapter
    except ImportError:
        print("[skip] test_boxmot_adapter_never_spawns_track_from_low_confidence_detection: boxmot not installed")
        return

    tracker = BoxMotByteTrackAdapter(track_thresh=0.1, high_conf_thresh=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    real_box = BBox.from_xyxy(100, 100, 200, 200)
    spurious_box = BBox.from_xyxy(80, 60, 260, 240)

    active = []
    for frame_idx in range(6):
        dets = [
            Detection(bbox=real_box, confidence=0.69, frame_idx=frame_idx),
            Detection(bbox=spurious_box, confidence=0.21, frame_idx=frame_idx),
        ]
        active = tracker.update(dets, frame_idx, frame_bgr=frame)

    assert len(active) == 1, (
        f"expected only the high-confidence detection to ever be surfaced "
        f"as a Tracklet, got {len(active)}: "
        f"{[(t.track_id, t.confidences[-1]) for t in active]}"
    )
    assert abs(active[0].confidences[-1] - 0.69) < 1e-6


def _run_all():
    tests = [
        test_localizer_has_no_stickiness,
        test_association_has_no_diameter_cue_or_motion_extras,
        test_roll_angle_estimation_converges_to_sustained_rotation,
        test_roll_angle_kalman_filter_attenuates_measurement_noise,
        test_diameter_distance_cue_disambiguates_equal_bearing_candidates,
        test_diameter_distance_cue_skips_gracefully_without_radius_data,
        test_bootstrap_labels_get_diameter_distance_verified,
        test_virtual_verification_match_flag,
        test_apparent_diameter_foreshortening,
        test_ratio_smoothing_attenuates_noise_and_updates_once_per_frame,
        test_mask_geometry_helpers,
        test_detection_diameter_prefers_mask_over_box,
        test_detection_center_prefers_mask_over_box,
        test_longest_diameter_and_opposing_center,
        test_is_nested_uses_mask_when_available,
        test_overlay_only_draws_confirmed_matches,
        test_overlay_draws_full_mask_polygon_when_available,
        test_json_log_exports_diameter_distance_match,
        test_reacquisition_relabels_same_lumen_after_total_anchor_loss,
        test_reacquisition_gives_up_past_max_gap_frames,
        test_reacquisition_disabled_when_max_gap_is_zero,
        test_reacquisition_virtual_anchor_matches_new_child_when_self_match_fails,
        test_full_pipeline_smoke_no_ml,
        test_boxmot_adapter_never_spawns_track_from_low_confidence_detection,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
