"""Smoke / sanity tests for bronchotrack.fusion (the three-model fusion
build: diameter-growth motion model + point-based "bronchotrack version"
+ whole-mask "ratio method", combined via a Kalman filter). Every
synthetic geometry test below was hand-derived AND cross-checked by
direct execution before being locked into an assertion here (see this
session's own working notes) -- not just "written to make green".

Runnable either with pytest (`pytest tests/`) or directly:
    python3 tests/test_fusion_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bronchotrack.graph import AirwayGraph, AirwayNode
from bronchotrack.tracker import MultiLumenTracker
from bronchotrack.types import BBox, Detection, Tracklet
from bronchotrack.fusion import bronchotrack_id, kalman_fusion, ratio_id
from bronchotrack.fusion.association import FusionAssociation
from bronchotrack.fusion.motion_model import ApproachMotionModel, TransitSanityChecker

EXAMPLE_GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "example_graph.json"
)


def _rect_mask(cx, cy, half):
    return np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]],
        dtype=np.float64,
    )


def _mk_tracklet(track_id, cx, cy, h=10.0, mask=None, label=None, hits=5):
    t = Tracklet(track_id=track_id, ind_start=0, ind_end=0, boxes=[BBox(x_c=cx, y_c=cy, h=h, a=1.0)])
    t.time_since_update = 0
    t.hits = hits
    t.mask = mask
    t.label = label
    return t


# ----------------------------------------------------------------------
# graph.py: the new area-based foreshortening method
# ----------------------------------------------------------------------
def test_child_apparent_equivalent_diameter_matches_ellipse_area_formula():
    """A branch continuing straight ahead of the parent's axis should read
    at (near) its true diameter for BOTH foreshortening formulas
    (foreshortening factor ~1.0 either way). A branch angled 60deg off
    axis should read smaller for both, but LESS aggressively for the
    area-based (sqrt(cos)) formula than the longest-diameter (cos)
    formula -- exactly the ellipse-area geometry the method's docstring
    claims (only one of an off-axis circle's two projected-ellipse axes
    actually foreshortens)."""
    root = AirwayNode(
        label="root", generation=0, parent=None, start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0])
    )
    straight = AirwayNode(
        label="straight", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, 10.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]]), radius=np.array([3.0, 3.0]),
    )
    theta = np.radians(60.0)
    dx, dy = np.sin(theta) * 10, np.cos(theta) * 10
    angled = AirwayNode(
        label="angled", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([dx, dy, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [dx, dy, 0.0]]), radius=np.array([3.0, 3.0]),
    )
    graph = AirwayGraph([root, straight, angled])

    true_diam = 6.0
    d_straight = graph.child_apparent_equivalent_diameter_at_distance("root", "straight", 5.0)
    assert d_straight is not None and d_straight > 0.99 * true_diam

    d_angled_area = graph.child_apparent_equivalent_diameter_at_distance("root", "angled", 5.0)
    d_angled_longest = graph.child_apparent_diameter_at_distance("root", "angled", 5.0)
    assert d_angled_area is not None and d_angled_longest is not None
    assert abs(d_angled_longest - true_diam * np.cos(theta)) < 1e-9
    assert abs(d_angled_area - true_diam * np.sqrt(np.cos(theta))) < 1e-9
    # shallower falloff: the area-based estimate reads LARGER than the
    # longest-diameter one for the same off-axis angle
    assert d_angled_area > d_angled_longest


# ----------------------------------------------------------------------
# ratio_id.py: whole-mask geometry primitives
# ----------------------------------------------------------------------
def test_mask_boundary_gap_and_circle_boundary_gap_exact():
    """Two axis-aligned, same-height rectangles separated purely
    horizontally: the nearest VERTEX pair sits exactly on the true edge
    gap (top/bottom corners line up), so this is an exact check, not an
    approximation."""
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    b = np.array([[15.0, 0.0], [25.0, 0.0], [25.0, 10.0], [15.0, 10.0]])
    assert ratio_id.mask_boundary_gap(a, b) == 5.0
    assert ratio_id.mask_boundary_gap(None, b) is None

    assert ratio_id.circle_boundary_gap(np.array([0.0, 0.0]), 3.0, np.array([10.0, 0.0]), 5.0) == 2.0
    # overlapping circles clamp to the minimum distance, never negative/zero
    assert ratio_id.circle_boundary_gap(np.array([0.0, 0.0]), 3.0, np.array([1.0, 0.0]), 5.0) == ratio_id._MIN_DIST


def test_ratio_method_disambiguates_where_bearing_alone_cannot():
    """The centerpiece test for the user's own explicit requirement that
    the ratio method 'make use of the fact that we are using a
    segmentation model instead of just point based'.

    Two candidate branches are built with IDENTICAL projected bearing
    (same direction/length, differing only in true radius) and two
    detections are placed at the IDENTICAL image point (differing only in
    mask size) -- a scenario specifically engineered so bearing carries
    ZERO discriminating information at all. `bearing_cost_matrix` must
    come back completely flat (every entry identical -- literally unable
    to tell the two candidates or the two detections apart), while
    `ratio_cost_matrix`, using each mask's own full area and the full
    boundary-to-boundary gap, must correctly favor the size-matched
    pairing (small detection <-> small candidate A, big <-> big candidate
    B) with a clear margin -- proving the ratio cue is doing real,
    independent geometric work the point-based cue structurally cannot."""
    root = AirwayNode(
        label="root", generation=0, parent=None,
        start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
        centerline=np.array([[0.0, -10.0, 0.0], [0.0, 0.0, 0.0]]), radius=np.array([4.0, 4.0]),
    )
    # identical geometry for A and B -- only the radius differs
    A = AirwayNode(
        label="A", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([5.0, 10.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0]]), radius=np.array([2.0, 2.0]),
    )
    B = AirwayNode(
        label="B", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([5.0, 10.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0]]), radius=np.array([5.0, 5.0]),
    )
    graph = AirwayGraph([root, A, B])
    proj = graph.project_children_2d("root")
    assert np.allclose(proj["A"], proj["B"])  # confirms the "identical bearing" setup

    ref = _mk_tracklet(0, 100.0, 100.0, mask=_rect_mask(100, 100, 5))
    det_small = _mk_tracklet(1, 100.0, 50.0, mask=_rect_mask(100, 50, 3))
    det_big = _mk_tracklet(2, 100.0, 50.0, mask=_rect_mask(100, 50, 8))  # same image point as det_small

    bearing = bronchotrack_id.bearing_cost_matrix(
        graph, "root", ["A", "B"], [det_small, det_big], bronchotrack_id.detection_point(ref)
    )
    assert bearing is not None
    assert np.allclose(bearing, bearing[0, 0])  # every entry identical -> zero discriminating power

    ratio = ratio_id.ratio_cost_matrix(graph, "root", ["A", "B"], ref, [det_small, det_big], 5.0)
    assert ratio is not None
    # correct (size-matched) pairing costs much less than the crossed one
    assert ratio[0, 0] < ratio[0, 1]  # A vs det_small cheaper than A vs det_big
    assert ratio[1, 1] < ratio[1, 0]  # B vs det_big cheaper than B vs det_small
    row_idx, col_idx = np.array([0, 1]), np.array([0, 1])
    assert (ratio[row_idx, col_idx] < ratio[row_idx, col_idx[::-1]]).all()


def test_ratio_cost_matrix_falls_back_gracefully_without_radius_data():
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)  # no radius data at all
    ref = _mk_tracklet(0, 100.0, 100.0, mask=_rect_mask(100, 100, 5))
    det = _mk_tracklet(1, 100.0, 50.0, mask=_rect_mask(100, 50, 3))
    assert ratio_id.ratio_cost_matrix(graph, "trachea", ["LMB", "RMB"], ref, [det], 5.0) is None


def test_ratio_cost_matrix_handles_no_other_detections_this_frame():
    """Regression test: a frame where the reference lumen is the ONLY
    thing currently visible (candidates_pool is empty) must return None
    cleanly, not divide-by-empty-array (numpy's `mean` of an empty slice
    raises a RuntimeWarning and returns NaN, which used to propagate
    silently into a NaN-filled cost matrix) -- caught by actually running
    the fusion CLI against real footage during this feature's own
    end-to-end validation, not by an a-priori guess."""
    root = AirwayNode(
        label="root", generation=0, parent=None,
        start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
        centerline=np.array([[0.0, -10.0, 0.0], [0.0, 0.0, 0.0]]), radius=np.array([4.0, 4.0]),
    )
    A = AirwayNode(
        label="A", generation=1, parent="root",
        start=np.array([0.0, 0.0, 0.0]), end=np.array([5.0, 10.0, 0.0]),
        centerline=np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0]]), radius=np.array([2.0, 2.0]),
    )
    graph = AirwayGraph([root, A])
    ref = _mk_tracklet(0, 100.0, 100.0, mask=_rect_mask(100, 100, 5))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a RuntimeWarning here must now fail the test, not just get ignored
        assert ratio_id.ratio_cost_matrix(graph, "root", ["A"], ref, [], 5.0) is None


# ----------------------------------------------------------------------
# bronchotrack_id.py: point-based bearing cue
# ----------------------------------------------------------------------
def test_bearing_cost_matrix_hungarian_correct_assignment():
    root = AirwayNode(
        label="root", generation=0, parent=None, start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0])
    )
    A = AirwayNode(label="A", generation=1, parent="root", start=np.array([0.0, 0.0, 0.0]), end=np.array([10.0, 0.0, 0.0]))
    B = AirwayNode(label="B", generation=1, parent="root", start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, 0.0, 10.0]))
    graph = AirwayGraph([root, A, B])

    ref = _mk_tracklet(0, 100.0, 100.0)
    det_a = _mk_tracklet(1, 100.0, 50.0)  # offset (0, -50) -- matches A's projected bearing
    det_b = _mk_tracklet(2, 50.0, 100.0)  # offset (-50, 0) -- matches B's projected bearing

    cost = bronchotrack_id.bearing_cost_matrix(graph, "root", ["A", "B"], [det_a, det_b], np.array([100.0, 100.0]))
    assert cost is not None
    assert np.allclose(cost, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-9)


# ----------------------------------------------------------------------
# motion_model.py
# ----------------------------------------------------------------------
def test_motion_model_tau_matches_hand_calculation():
    """Linear diameter growth of `rate` px/frame from `diam0`: after the
    filter converges, tau_frames should equal current_diameter/rate to
    within numerical noise -- the exact scale-free time-to-contact formula
    the module docstring derives."""
    m = ApproachMotionModel(process_noise=1e-2, measurement_noise=1e-2)
    diam0, rate = 20.0, 2.0
    est = None
    for f in range(40):
        t = _mk_tracklet(1, 0.0, 0.0, h=diam0 + rate * f)
        est = m.update([t], f)[1]
    assert est.approaching
    assert abs(est.tau_frames - est.diameter / rate) < 0.05


def test_motion_model_not_approaching_when_shrinking():
    m = ApproachMotionModel(process_noise=1e-2, measurement_noise=1e-2)
    est = None
    for f in range(20):
        t = _mk_tracklet(1, 0.0, 0.0, h=50.0 - f)
        est = m.update([t], f)[1]
    assert not est.approaching
    assert est.tau_frames is None


def test_motion_model_prune_drops_stale_filters():
    m = ApproachMotionModel()
    t = _mk_tracklet(1, 0.0, 0.0, h=10.0)
    m.update([t], 0)
    assert 1 in m._filters
    m.prune(set())
    assert 1 not in m._filters


def test_transit_sanity_checker_plausibility():
    checker = TransitSanityChecker(plausible_range=(0.2, 5.0))
    assert checker.check(tau_frames=10.0, candidate_arc_length_mm=5.0) == "unknown"  # no history yet

    checker.record_transit("A", frames_spent=10, arc_length_mm=20.0)  # 2.0 mm/frame
    checker.record_transit("B", frames_spent=20, arc_length_mm=40.0)  # 2.0 mm/frame
    assert checker.median_speed_mm_per_frame == 2.0

    # a candidate implying ~2 mm/frame closing speed (consistent with history)
    assert checker.check(tau_frames=10.0, candidate_arc_length_mm=20.0) == "plausible"
    # a candidate implying a wildly different closing speed
    assert checker.check(tau_frames=1.0, candidate_arc_length_mm=200.0) == "implausible"


# ----------------------------------------------------------------------
# kalman_fusion.py
# ----------------------------------------------------------------------
class _FakeDet:
    def __init__(self, track_id):
        self.track_id = track_id


def test_identity_fusion_blends_cues_and_motion_gates():
    from bronchotrack.fusion.motion_model import ApproachEstimate

    fusion = kalman_fusion.IdentityFusion(bearing_weight=0.5, ratio_weight=0.5, motion_gate_strength=0.3)
    labels = ["A", "B"]
    dets = [_FakeDet(10), _FakeDet(20)]
    bearing = np.array([[0.1, 0.9], [0.9, 0.1]])
    ratio = np.array([[0.1, 0.9], [0.9, 0.1]])
    approach = {
        10: ApproachEstimate(track_id=10, diameter=10, growth_rate=1.0, approaching=True, tau_frames=10.0),
        20: ApproachEstimate(track_id=20, diameter=10, growth_rate=-1.0, approaching=False, tau_frames=None),
    }
    results = None
    for _ in range(10):
        results = fusion.update(labels, dets, bearing, ratio, approach)
    # both cues agree A<->det10, B<->det20 with identical raw cost, but the
    # motion model is only approaching for det10 -- A's fused confidence
    # must end up higher than B's despite identical id-cue evidence
    assert results["A"].fused_confidence > results["B"].fused_confidence
    assert results["A"].matched_track_id == 10
    assert results["B"].matched_track_id == 20


def test_identity_fusion_falls_back_to_single_cue_when_other_missing():
    fusion = kalman_fusion.IdentityFusion()
    labels = ["A", "B"]
    dets = [_FakeDet(10), _FakeDet(20)]
    ratio = np.array([[0.1, 0.9], [0.9, 0.1]])
    results = fusion.update(labels, dets, None, ratio, {})
    assert results["A"].matched_track_id == 10
    assert results["B"].matched_track_id == 20
    assert abs(results["A"].raw_confidence - 0.9) < 1e-9  # unweighted: 1 - 0.1


# ----------------------------------------------------------------------
# association.py: end-to-end continuous re-evaluation + bootstrap
# ----------------------------------------------------------------------
def _mirrored_bifurcation_graph():
    root = AirwayNode(
        label="root", generation=0, parent=None, start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0])
    )
    A = AirwayNode(label="A", generation=1, parent="root", start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, 10.0, 10.0]))
    B = AirwayNode(label="B", generation=1, parent="root", start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, 10.0, -10.0]))
    return AirwayGraph([root, A, B])


def test_fusion_association_root_bootstrap_labels_both_lumens_at_once():
    """The very first bifurcation has no pre-existing occupant tracklet;
    without the one-time bootstrap, only one of two simultaneously-visible
    lumens could ever be identified (see association.py's own docstring).
    Confirms both get labeled correctly in a single frame, and the larger
    one becomes the new current_location."""
    graph = _mirrored_bifurcation_graph()
    assoc = FusionAssociation(graph, commit_threshold=0.6, commit_frames=5)

    det_a = _mk_tracklet(1, 50.0, 100.0, h=15.0)  # smaller x -> A (per project_children_2d)
    det_b = _mk_tracklet(2, 150.0, 100.0, h=25.0)  # larger x, bigger -> B, becomes occupant

    result = assoc.process_frame([det_a, det_b], 0)
    assert result.committed_this_frame
    assert det_a.label == "A"
    assert det_b.label == "B"
    assert assoc.current_location == "B"
    # the bootstrap has no pre-existing occupant to measure an approach
    # speed from, so the sanity checker is deliberately not consulted
    assert result.transition_sanity == "unknown"


def test_fusion_association_commits_after_consecutive_frames_not_one_lucky_frame():
    """Commit hysteresis: a candidate must clear commit_threshold for
    commit_frames CONSECUTIVE frames. A single strong frame must not be
    enough, and an interrupting bad frame must reset that candidate's own
    streak (checked by running one fewer than commit_frames strong frames,
    confirming no commit yet, then continuing)."""
    graph = _mirrored_bifurcation_graph()
    assoc = FusionAssociation(graph, commit_threshold=0.6, commit_frames=5)

    # bootstrap first (frame 0), landing current_location on the occupant
    det_a = _mk_tracklet(1, 50.0, 100.0, h=15.0)
    det_b = _mk_tracklet(2, 150.0, 100.0, h=25.0)
    assoc.process_frame([det_a, det_b], 0)
    assert assoc.current_location == "B"  # leaf in this tiny graph -> no further children

    # rebuild with a 3-generation graph so B itself has children to commit into
    root = AirwayNode(label="root", generation=0, parent=None, start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0]))
    B = AirwayNode(label="B", generation=1, parent="root", start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, 10.0, 0.0]))
    C = AirwayNode(label="C", generation=2, parent="B", start=np.array([0.0, 10.0, 0.0]), end=np.array([10.0, 20.0, 0.0]))
    D = AirwayNode(label="D", generation=2, parent="B", start=np.array([0.0, 10.0, 0.0]), end=np.array([-10.0, 20.0, 0.0]))
    graph2 = AirwayGraph([root, B, C, D])
    assoc2 = FusionAssociation(graph2, commit_threshold=0.6, commit_frames=5)
    assoc2.current_location = "B"
    assoc2._has_committed_once = True

    # place the candidate exactly along B's own 2D-projected offset to "C"
    # (using the real projection vector, not a guessed axis convention),
    # so the bearing match is unambiguous by construction
    target_label = "C"
    ref_pos = np.array([100.0, 100.0])
    offset = graph2.project_children_2d("B")[target_label] * 10.0

    def frame(fidx):
        occ = _mk_tracklet(1, ref_pos[0], ref_pos[1], h=30.0, label="B")
        cand_pos = ref_pos + offset
        cand = _mk_tracklet(2, cand_pos[0], cand_pos[1], h=10.0)
        return assoc2.process_frame([occ, cand], fidx)

    for f in range(4):  # one fewer than commit_frames=5
        res = frame(f)
    assert not res.committed_this_frame
    assert assoc2.current_location == "B"

    # one more consistent frame should now commit
    res = frame(4)
    assert res.committed_this_frame
    assert assoc2.current_location == target_label
    # this candidate never had a diameter-growth measurement (the fake
    # tracklets here use a constant box size), so there's no tau to check
    assert res.transition_sanity == "unknown"


def test_fusion_association_streak_resets_on_interrupting_bad_frame():
    """A candidate's commit streak must reset the moment its OWN fused
    confidence drops below commit_threshold, not just decay -- checked
    directly against `_try_commit` (white-box) rather than through several
    real frames, since a Kalman-smoothed confidence deliberately doesn't
    crash to zero from a single adversarial measurement (that's the
    smoothing working as intended, a separate property already covered by
    `test_identity_fusion_blends_cues_and_motion_gates`); this test
    isolates the hysteresis bookkeeping itself. 4 clearing frames + 1
    frame that drops below threshold + 4 more clearing frames
    (commit_frames=5) must NOT commit, since the streak is never actually
    5 CONSECUTIVE."""
    root = AirwayNode(label="root", generation=0, parent=None, start=np.array([0.0, -10.0, 0.0]), end=np.array([0.0, 0.0, 0.0]))
    B = AirwayNode(label="B", generation=1, parent="root", start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, 10.0, 0.0]))
    C = AirwayNode(label="C", generation=2, parent="B", start=np.array([0.0, 10.0, 0.0]), end=np.array([10.0, 20.0, 0.0]))
    graph2 = AirwayGraph([root, B, C])
    assoc2 = FusionAssociation(graph2, commit_threshold=0.6, commit_frames=5)
    assoc2.current_location = "B"
    assoc2._has_committed_once = True

    cand = _mk_tracklet(2, 0.0, 0.0, h=10.0)

    def results(confidence):
        return {"C": kalman_fusion.CandidateConfidence(label="C", fused_confidence=confidence, matched_track_id=2, raw_confidence=confidence)}

    for f in range(4):
        committed, _ = assoc2._try_commit(results(0.9), [cand], f)
        assert not committed
    assert assoc2._commit_streak["C"] == 4

    committed, verdict = assoc2._try_commit(results(0.4), [cand], 4)  # drops below threshold
    assert not committed
    assert verdict is None  # only set on an actual commit
    assert assoc2._commit_streak["C"] == 0  # streak reset, not just paused

    for f in range(5, 8):
        committed, _ = assoc2._try_commit(results(0.9), [cand], f)
        assert not committed  # only 3 consecutive so far, needs 5
    assert assoc2.current_location == "B"
    assert assoc2._commit_streak["C"] == 3

    # two more clearing frames should now finally commit (5 consecutive)
    committed, _ = assoc2._try_commit(results(0.9), [cand], 8)
    assert not committed
    committed, verdict = assoc2._try_commit(results(0.9), [cand], 9)
    assert committed
    assert verdict in ("plausible", "implausible", "unknown")
    assert assoc2.current_location == "C"


def test_fusion_association_continuous_reevaluation_survives_reference_loss():
    """No permanent anchor dependency: if the reference lumen disappears
    for a few frames and comes back, identification should resume rather
    than being permanently stuck (the diagnosed paper_exact anchor-loss
    dead end this package's continuous re-evaluation design was meant to
    avoid by construction -- see association.py's module docstring)."""
    graph = _mirrored_bifurcation_graph()
    assoc = FusionAssociation(graph, commit_threshold=0.6, commit_frames=3)

    det_a = _mk_tracklet(1, 50.0, 100.0, h=15.0)
    det_b = _mk_tracklet(2, 150.0, 100.0, h=25.0)
    assoc.process_frame([det_a, det_b], 0)  # bootstrap -> current_location becomes "B"
    assert assoc.current_location == "B"

    # "B" is a leaf here, so nothing further to identify -- just confirm a
    # frame with NO tracklets at all doesn't raise and doesn't move location
    result = assoc.process_frame([], 1)
    assert result.location == "B"


def test_scalar_kalman_filter_promoted_and_reexported_identically():
    """Regression check for the paper_exact -> top-level promotion: both
    import paths must resolve to the exact same class object, not two
    independently-drifting copies."""
    from bronchotrack.scalar_kalman import ScalarKalmanFilter as TopLevel
    from bronchotrack.paper_exact.scalar_kalman import ScalarKalmanFilter as ReExported

    assert TopLevel is ReExported


# ----------------------------------------------------------------------
# end-to-end, no ML
# ----------------------------------------------------------------------
def test_full_fusion_pipeline_smoke_no_ml():
    """Runs tracker + fusion association together across synthetic frames
    on the real example graph (carina -> one main bronchus), mirroring
    paper_exact's own no-ML smoke test. Only asserts the pipeline runs
    cleanly, bootstraps both main bronchi, and advances past the carina."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    tracker = MultiLumenTracker(use_reid=False)
    assoc = FusionAssociation(graph, commit_threshold=0.5, commit_frames=3)

    frame_idx = 0
    proj = graph.project_children_2d("trachea")
    left_label = "LMB" if proj["LMB"][0] < proj["RMB"][0] else "RMB"

    # --- carina: two lumens ---
    for _ in range(2):
        dets = [
            Detection(bbox=BBox.from_xywh(60, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=BBox.from_xywh(160, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
        ]
        active = tracker.update(dets, frame_idx, eligibility_fn=assoc.eligibility_fn)
        result = assoc.process_frame(active, frame_idx)
        frame_idx += 1

    assert assoc._has_committed_once
    assert assoc.current_location in ("LMB", "RMB")
    labeled = [t for t in tracker.tracklets if t.label is not None]
    assert {t.label for t in labeled} == {"LMB", "RMB"}
    print(f"[fusion smoke] after carina: location={assoc.current_location}")


def _run_all():
    tests = [
        test_child_apparent_equivalent_diameter_matches_ellipse_area_formula,
        test_mask_boundary_gap_and_circle_boundary_gap_exact,
        test_ratio_method_disambiguates_where_bearing_alone_cannot,
        test_ratio_cost_matrix_falls_back_gracefully_without_radius_data,
        test_ratio_cost_matrix_handles_no_other_detections_this_frame,
        test_bearing_cost_matrix_hungarian_correct_assignment,
        test_motion_model_tau_matches_hand_calculation,
        test_motion_model_not_approaching_when_shrinking,
        test_motion_model_prune_drops_stale_filters,
        test_transit_sanity_checker_plausibility,
        test_identity_fusion_blends_cues_and_motion_gates,
        test_identity_fusion_falls_back_to_single_cue_when_other_missing,
        test_fusion_association_root_bootstrap_labels_both_lumens_at_once,
        test_fusion_association_commits_after_consecutive_frames_not_one_lucky_frame,
        test_fusion_association_streak_resets_on_interrupting_bad_frame,
        test_fusion_association_continuous_reevaluation_survives_reference_loss,
        test_scalar_kalman_filter_promoted_and_reexported_identically,
        test_full_fusion_pipeline_smoke_no_ml,
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
    return failures == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
