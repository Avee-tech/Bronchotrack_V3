"""Smoke / sanity tests for the non-ML parts of the pipeline (graph, kalman,
matching, tracker, association, localization). These do NOT require torch,
torchvision, or ultralytics -- they exercise everything except the actual
YOLOv11 detector and ResNet50 Re-ID embedder, using synthetic BBox/Detection
objects instead.

Runnable either with pytest (`pytest tests/`) or directly:
    python3 tests/test_pipeline_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bronchotrack.association import AirwayAssociation
from bronchotrack.graph import AirwayGraph
from bronchotrack.kalman import KalmanBoxFilter
from bronchotrack.localization import Localizer
from bronchotrack.matching import assign, motion_cost_matrix
from bronchotrack.motion_model import TreeMotionFilter
from bronchotrack.tracker import MultiLumenTracker
from bronchotrack.types import BBox, Detection
from bronchotrack.utils import iou

EXAMPLE_GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "example_graph.json"
)


def test_bbox_conversions():
    b = BBox.from_xyxy(10, 20, 50, 80)
    assert abs(b.x_c - 30) < 1e-6
    assert abs(b.y_c - 50) < 1e-6
    assert abs(b.h - 60) < 1e-6
    assert abs(b.w - 40) < 1e-6

    x1, y1, x2, y2 = b.xyxy
    assert abs(x1 - 10) < 1e-6 and abs(y1 - 20) < 1e-6
    assert abs(x2 - 50) < 1e-6 and abs(y2 - 80) < 1e-6


def test_iou_basic():
    a = BBox.from_xyxy(0, 0, 10, 10)
    b = BBox.from_xyxy(5, 5, 15, 15)
    val = iou(a, b)
    # intersection = 5x5=25, union = 100+100-25=175
    assert abs(val - 25.0 / 175.0) < 1e-6

    c = BBox.from_xyxy(100, 100, 110, 110)
    assert iou(a, c) == 0.0

    assert abs(iou(a, a) - 1.0) < 1e-9


def test_graph_loading_and_queries():
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)

    assert graph.root() == "trachea"
    assert set(graph.children("trachea")) == {"LMB", "RMB"}
    assert graph.parent("LMB") == "trachea"
    assert graph.generation("trachea") == 0
    assert graph.generation("LMB") == 1
    assert graph.generation("RML") == 3

    assert graph.ancestor("RML", 0) == "RML"
    assert graph.ancestor("RML", 1) == "BI"
    assert graph.ancestor("RML", 2) == "RMB"
    assert graph.ancestor("RML", 3) == "trachea"
    assert graph.ancestor("RML", 10) == "trachea"  # clamps at root

    subtree = graph.subtree_labels("RMB")
    assert subtree == {"RMB", "RUL", "BI", "RML", "RLL"}

    assert set(graph.siblings("LUL")) == {"LLL"}

    proj = graph.project_children_2d("trachea")
    assert set(proj.keys()) == {"LMB", "RMB"}
    for v in proj.values():
        assert v.shape == (2,)


def test_from_path_dispatches_json_unchanged():
    """AirwayGraph.from_path with a .json path should behave exactly like
    from_json -- no mesh_to_graph import triggered, no behavior change for
    every existing caller that already passes a pre-built graph JSON."""
    via_from_json = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    via_from_path = AirwayGraph.from_path(EXAMPLE_GRAPH_PATH)
    assert set(via_from_path.nodes.keys()) == set(via_from_json.nodes.keys())
    assert via_from_path.root() == via_from_json.root()


def test_from_path_json_never_imports_mesh_to_graph():
    """Pointing --graph at an ordinary graph JSON must never trigger
    bronchotrack.mesh_to_graph's vtk/vmtk import -- that's the whole point
    of checking the file extension as a plain string before importing
    anything, so a user who never touches a mesh is never asked to have
    vtk/vmtk installed."""
    import sys as _sys

    had_module = "bronchotrack.mesh_to_graph" in _sys.modules
    saved = _sys.modules.pop("bronchotrack.mesh_to_graph", None)
    try:
        AirwayGraph.from_path(EXAMPLE_GRAPH_PATH)
        assert "bronchotrack.mesh_to_graph" not in _sys.modules
    finally:
        if had_module and saved is not None:
            _sys.modules["bronchotrack.mesh_to_graph"] = saved


def test_mesh_to_graph_default_cache_path():
    from bronchotrack.mesh_to_graph import default_cache_path

    assert default_cache_path("patient_models/NewPatient.vtk") == "patient_models/NewPatient_graph.json"
    assert default_cache_path("Model.vtp") == "Model_graph.json"


def test_build_and_cache_graph_reuses_fresh_cache(tmp_path=None):
    """A mesh whose cached graph JSON is already up to date should be
    served straight from the cache -- without re-parsing the 'mesh' file
    at all (proven here by pointing --graph at a mesh path that isn't even
    a valid mesh; if build_and_cache_graph tried to actually load it as a
    surface, this would raise). This is the mechanism that makes a new
    patient's --graph mesh.vtk 'just work' on every run after the first
    without re-running the slow vmtk network extraction every time."""
    import json
    import os
    import tempfile
    import time

    from bronchotrack.mesh_to_graph import build_and_cache_graph

    with tempfile.TemporaryDirectory() as d:
        mesh_path = os.path.join(d, "not_a_real_mesh.vtk")
        cache_path = os.path.join(d, "not_a_real_mesh_graph.json")
        with open(mesh_path, "w") as f:
            f.write("not actually a vtk file")
        fake_graph = {"nodes": [{"label": "trachea", "generation": 0, "parent": None, "start": [0, 0, 0], "end": [0, 1, 0]}]}
        with open(cache_path, "w") as f:
            json.dump(fake_graph, f)
        # ensure the cache's mtime is unambiguously >= the mesh's
        now = time.time()
        os.utime(mesh_path, (now, now))
        os.utime(cache_path, (now + 5, now + 5))

        result = build_and_cache_graph(mesh_path, cache_path=cache_path)
        assert result == fake_graph

        # AirwayGraph.from_path should load the same cached graph end to end.
        graph = AirwayGraph.from_path(mesh_path, graph_cache_path=cache_path)
        assert graph.root() == "trachea"


def test_run_network_extraction_missing_vmtk_gives_actionable_error():
    """If vtk itself is importable but vmtk specifically is not (this
    sandbox's own situation -- vmtk needs its own venv, see
    mesh_to_graph.py's module docstring), building a graph from a mesh
    should fail with a clear, actionable ImportError rather than an
    unrelated traceback deep inside vmtkscripts."""
    try:
        import vtk  # noqa: F401
    except ImportError:
        return  # nothing to check in an environment without vtk at all
    try:
        import vmtk  # noqa: F401
        return  # vmtk IS installed here -- this specific gap doesn't apply
    except ImportError:
        pass

    from bronchotrack.mesh_to_graph import run_network_extraction

    try:
        run_network_extraction(None)
        assert False, "expected ImportError when vmtk is not installed"
    except ImportError as e:
        assert "vmtk" in str(e)


def test_kalman_converges_to_constant_box():
    true_box = BBox(x_c=100.0, y_c=80.0, h=40.0, a=0.6)
    kf = KalmanBoxFilter(true_box)
    for _ in range(15):
        kf.predict()
        # slightly noisy but stationary measurement
        noisy = BBox(
            x_c=true_box.x_c + np.random.randn() * 0.5,
            y_c=true_box.y_c + np.random.randn() * 0.5,
            h=true_box.h + np.random.randn() * 0.5,
            a=true_box.a,
        )
        kf.update(noisy)

    est = kf.as_bbox()
    assert abs(est.x_c - true_box.x_c) < 3.0
    assert abs(est.y_c - true_box.y_c) < 3.0
    assert abs(est.h - true_box.h) < 3.0


def test_matching_assign_simple():
    predicted = [BBox.from_xyxy(0, 0, 10, 10), BBox.from_xyxy(100, 100, 110, 110)]
    dets = [
        Detection(bbox=BBox.from_xyxy(1, 1, 11, 11), confidence=0.9, frame_idx=0),
        Detection(bbox=BBox.from_xyxy(101, 101, 111, 111), confidence=0.9, frame_idx=0),
    ]
    cost = motion_cost_matrix(predicted, dets)
    matches, unmatched_r, unmatched_c = assign(cost, max_cost=0.5)
    assert set(matches) == {(0, 0), (1, 1)}
    assert unmatched_r == []
    assert unmatched_c == []


def test_tracker_maintains_identity_across_frames():
    tracker = MultiLumenTracker(use_reid=False)
    track_ids_over_time = []

    for frame_idx in range(10):
        # a single lumen drifting slowly to the right
        x_c = 100.0 + frame_idx * 2.0
        det = Detection(
            bbox=BBox(x_c=x_c, y_c=80.0, h=40.0, a=0.6),
            confidence=0.9,
            frame_idx=frame_idx,
        )
        active = tracker.update([det], frame_idx)
        track_ids_over_time.append(active[0].track_id if active else None)

    # same physical lumen should keep the same track id throughout
    assert all(tid == track_ids_over_time[0] for tid in track_ids_over_time)
    assert track_ids_over_time[0] is not None


def test_motion_filter_starts_at_root_and_stays_on_graph_edges():
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    mf = TreeMotionFilter(graph)

    label, conf = mf.current_location()
    assert label == graph.root()
    assert abs(conf - 1.0) < 1e-9

    # predict-only (no observations yet): belief should spread onto the
    # root's children/self but never onto an unrelated, non-adjacent branch
    mf.predict(approach_trend=None)
    far_label = next(l for l in graph.all_labels() if graph.generation(l) >= 3)
    assert mf.belief.get(far_label, 0.0) == 0.0
    assert set(mf.belief.keys()) <= ({graph.root()} | set(graph.children(graph.root())))


def test_motion_filter_converges_on_repeated_observation():
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    mf = TreeMotionFilter(graph)

    root_child = graph.children(graph.root())[0]  # e.g. "LMB"
    for _ in range(10):
        mf.predict(approach_trend=0.1)  # mild positive trend -> favors advancing
        mf.update({root_child})

    label, conf = mf.current_location()
    assert label == root_child
    assert conf > 0.85  # sustained, consistent observation should dominate belief
    runner_up = sorted((p for l, p in mf.belief.items() if l != root_child), reverse=True)[0]
    assert conf > 5 * runner_up


def test_motion_filter_rejects_occasional_distant_mislabel():
    """Regression test for a real failure mode found by running the
    pipeline on an actual 916-frame bronchoscopy video: a handful of
    frames where a noisy detection got hard-labeled to a branch several
    generations away from where the scope actually was pulled belief that
    far in a single frame, because every observed label was treated as
    equally strong evidence regardless of how far it was from the current
    estimate. See `update()`'s docstring in motion_model.py."""
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    mf = TreeMotionFilter(graph)
    root = graph.root()
    lmb = graph.children(root)[0]
    lul, lll = graph.children(lmb)  # several hops from root

    import random

    rng = random.Random(1)
    for _ in range(900):
        observed = {root}
        if rng.random() < 0.08:  # ~8% of frames also see a spurious deep mislabel
            observed.add(rng.choice([lul, lll]))
        mf.predict(approach_trend=0.0)
        mf.update(observed)

    label, conf = mf.current_location()
    assert label == root  # true location never moved -- shouldn't have drifted deep
    assert conf > 0.9


def test_motion_filter_still_tracks_genuine_sustained_advance():
    """Regression test for a SECOND real failure mode, found on a follow-up
    real-video run after the fix above: an earlier version discounted
    observed labels by their graph distance from the filter's *own*
    current belief, which is self-referential -- a long, confident,
    correct dwell on one branch made the filter increasingly resistant to
    the very evidence that should later move it off that branch, so it
    could permanently lock on even as the scope genuinely advanced. This
    mirrors that exact shape (a long dwell, long enough to build high
    confidence, followed by a real and fully-evidenced transition) rather
    than a short/easy one, specifically so a future change can't
    reintroduce that self-referential discount as the default without
    this test catching it. See update()'s docstring ("A second real
    failure mode") in motion_model.py.
    """
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    mf = TreeMotionFilter(graph)
    root = graph.root()
    lmb = graph.children(root)[0]

    for _ in range(150):
        mf.predict(approach_trend=0.0)
        mf.update({root})
    label, conf = mf.current_location()
    assert label == root and conf > 0.9  # confidently, correctly settled on root first

    for _ in range(60):
        mf.predict(approach_trend=0.1)
        mf.update({lmb})

    label, conf = mf.current_location()
    assert label == lmb  # must NOT still be locked on root
    assert conf > 0.8


def test_motion_filter_no_observation_frame_keeps_predicted_belief():
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    mf = TreeMotionFilter(graph)
    mf.predict(approach_trend=None)
    before = dict(mf.belief)
    mf.update(set())  # no detections this frame -- should be a no-op
    assert mf.belief == before


def test_full_pipeline_smoke_no_ml():
    """Runs tracker + association + localizer together across several
    synthetic frames (carina -> down one main bronchus -> its two
    children), using hand-placed bounding boxes instead of a real detector.
    Only asserts the pipeline runs cleanly and produces internally
    consistent output -- exact label identity is not checked since the
    synthetic boxes are not derived from the graph's real 3D geometry.
    """
    graph = AirwayGraph.from_json(EXAMPLE_GRAPH_PATH)
    tracker = MultiLumenTracker(use_reid=False)
    association = AirwayAssociation(graph)
    localizer = Localizer(graph)
    motion_filter = TreeMotionFilter(graph)

    def step_motion_filter(labels_this_frame: dict) -> None:
        # mirrors exactly how pipeline.py drives the filter each frame
        motion_filter.predict(association.approach_trend)
        motion_filter.update(set(labels_this_frame.values()))

    frame_idx = 0

    # --- carina: two lumens, left and right ---
    for _ in range(3):
        dets = [
            Detection(bbox=BBox.from_xywh(60, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
            Detection(bbox=BBox.from_xywh(160, 90, 40, 60), confidence=0.9, frame_idx=frame_idx),
        ]
        active = tracker.update(dets, frame_idx, eligibility_fn=association.eligibility_fn)
        labels = association.process_frame(active, frame_idx)
        step_motion_filter(labels)
        loc = localizer.localize(active)
        frame_idx += 1

    assert association.initialized
    labeled = [t for t in tracker.tracklets if t.label is not None]
    assert len(labeled) == 2
    assert {t.label for t in labeled} == {"LMB", "RMB"}
    assert loc == "trachea"  # two primary-level siblings visible -> vote for parent

    # find whichever tracklet advanced further, e.g. "RMB", and simulate
    # advancing into it: its box grows, and two new nested lumens appear.
    # Their image-plane positions are derived from the *actual* graph
    # projection (matching assumes zero roll -- see association.py module
    # docstring) so the synthetic scenario is geometrically self-consistent
    # -- otherwise arbitrary hand-picked offsets can legitimately fail the
    # Hungarian match (the global-optimum assignment need not equal the
    # best single pairing).
    rmb_track = next(t for t in tracker.tracklets if t.label == "RMB")
    other_track = next(t for t in tracker.tracklets if t.label == "LMB")
    initial_box = rmb_track.last_box
    final_box = BBox.from_xyxy(140, 60, 240, 180)

    # grow the RMB box gradually over several frames (as it would in a real
    # video as the scope advances) so IoU-based motion matching keeps the
    # same track identity instead of a discontinuous jump breaking it
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
        labels = association.process_frame(active, frame_idx)
        step_motion_filter(labels)
        loc = localizer.localize(active)
        frame_idx += 1

    # the RMB box has been growing for several frames -- the motion filter
    # should have picked up on that (via approach_trend) and shifted belief
    # off the trachea root, favoring RMB/its subtree over staying put
    mf_label, mf_conf = motion_filter.current_location()
    assert mf_label != graph.root()

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
        labels = association.process_frame(active, frame_idx)
        step_motion_filter(labels)
        loc = localizer.localize(active)
        frame_idx += 1

    # by now RUL/BI have been repeatedly observed -- the motion filter's
    # belief should have settled on one of them (both are valid graph
    # labels reachable from where it started; exact identity isn't checked,
    # matching this feature's "doesn't have to be the exact position"
    # framing -- it should just be topologically consistent)
    mf_label, mf_conf = motion_filter.current_location()
    assert mf_label in graph.all_labels()
    assert mf_conf > 0.0

    # the two nested lumens should have picked up RMB's children as labels
    rmb_children_labels = set(graph.children("RMB"))
    nested_labels = {
        t.label
        for t in tracker.tracklets
        if t.label is not None and t.label in rmb_children_labels
    }
    assert nested_labels == rmb_children_labels, (
        f"expected both of RMB's children to get labeled, got {nested_labels}"
    )
    assert loc is not None and loc in graph.all_labels()

    print("[smoke] final location:", loc, "| all labels seen:", sorted(
        t.label for t in tracker.tracklets if t.label
    ))


def _run_all():
    tests = [
        test_bbox_conversions,
        test_iou_basic,
        test_graph_loading_and_queries,
        test_from_path_dispatches_json_unchanged,
        test_from_path_json_never_imports_mesh_to_graph,
        test_mesh_to_graph_default_cache_path,
        test_build_and_cache_graph_reuses_fresh_cache,
        test_run_network_extraction_missing_vmtk_gives_actionable_error,
        test_kalman_converges_to_constant_box,
        test_matching_assign_simple,
        test_tracker_maintains_identity_across_frames,
        test_motion_filter_starts_at_root_and_stays_on_graph_edges,
        test_motion_filter_converges_on_repeated_observation,
        test_motion_filter_rejects_occasional_distant_mislabel,
        test_motion_filter_still_tracks_genuine_sustained_advance,
        test_motion_filter_no_observation_frame_keeps_predicted_belief,
        test_full_pipeline_smoke_no_ml,
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
