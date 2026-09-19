"""Command-line entry point -- STRICT paper reference version.

    python3 -m bronchotrack.paper_exact.cli \\
        --video path/to/bronchoscopy.mp4 \\
        --graph path/to/airway_graph.json \\
        --weights path/to/lumen_yolo.pt \\
        --reid-weights path/to/reid_resnet50.pt \\
        --out-video out/overlay.mp4 \\
        --out-json out/localization_log.json

Only file-mode video processing is supported here (no --live) -- the live-
stream reader in the main package is a convenience wrapper with no
algorithmic content, so it wasn't worth forking; if you need it, run the
same graph/weights through ``bronchotrack.cli --live`` instead, they're
otherwise unrelated to what makes this build paper-exact.

Most flags that controlled a post-paper addition in the main CLI are gone
entirely: no --no-motion-model or --motion-p-advance (that feature doesn't
exist in this build at all), and no --smoothing-window/--smoothing-min-
frames (localization is always the raw, unsmoothed Eq. 8 vote here). One
non-paper addition IS available here, opt-in: --distance-diameter-weight
(a lumen diameter:distance-from-reference matching cue, default 0.4 --
pass 0 to fall back to strict bearing-only matching). See
bronchotrack.paper_exact's __init__.py and association.py's module
docstring for the full list of deltas against the main package and the
exact formula.

A NEW PATIENT: --graph ALSO ACCEPTS A RAW SURFACE MESH (.vtk/.vtp/.stl)
-------------------------------------------------------------------------
Point --graph directly at a patient's exported 3D model instead of a
pre-built graph JSON:

    python3 -m bronchotrack.paper_exact.cli \\
        --video new_patient.mp4 \\
        --graph NewPatientModel.vtk \\
        ...

The mesh->graph conversion (see bronchotrack.mesh_to_graph's module
docstring) runs automatically the first time, and the result is cached
next to the mesh (<mesh_stem>_graph.json) and reused on every later run
against that same file -- so this only costs the extra time once per
patient, not once per run. Needs vtk/vmtk importable in this environment;
see bronchotrack.mesh_to_graph's docstring if that's not yet installed
here (--precomputed-detections / --weights and everything else about this
CLI has no vtk/vmtk dependency at all). The --mesh-* flags below tune that
one-time conversion; irrelevant, and ignored, when --graph already points
at a graph JSON.
"""
from __future__ import annotations

import argparse
import sys

from ..detection import LumenDetector, PrecomputedDetectionSource
from ..graph import AirwayGraph
from ..reid import ReIDEmbedder
from ..utils import describe_device
from .pipeline import BronchoTrackPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BronchoTrack pipeline CLI (strict paper reference build)")
    p.add_argument("--video", required=True, help="Path to a bronchoscopy video file")
    p.add_argument(
        "--graph",
        required=True,
        help="Path to airway graph JSON (see graph.py), OR a raw surface "
        "mesh (.vtk/.vtp/.stl) to build one from automatically -- see "
        "'A NEW PATIENT' above.",
    )
    p.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="If --graph is a mesh, re-run the mesh->graph conversion even "
        "if a cached graph JSON already exists next to it (default: reuse "
        "the cache when it's up to date with the mesh file). No effect "
        "when --graph is already a graph JSON.",
    )
    p.add_argument(
        "--graph-cache",
        default=None,
        help="If --graph is a mesh, where to cache the built graph JSON "
        "(default: <mesh_stem>_graph.json next to the mesh). No effect "
        "when --graph is already a graph JSON.",
    )
    mesh_group = p.add_argument_group(
        "mesh conversion (only used when --graph is a .vtk/.vtp/.stl mesh)"
    )
    mesh_group.add_argument("--mesh-weld-tol-mm", type=float, default=2.0, help="Endpoint welding tolerance in mm (default: 2.0)")
    mesh_group.add_argument("--mesh-min-branch-length-mm", type=float, default=1.0, help="Drop branches shorter than this, in mm (default: 1.0)")
    mesh_group.add_argument(
        "--mesh-min-radius-mm",
        type=float,
        default=0.4,
        help="Drop branches with median inscribed-sphere radius below this, "
        "in mm (default: 0.4)",
    )
    mesh_group.add_argument("--mesh-smooth-iterations", type=int, default=20, help="Surface smoothing iterations before network extraction (default: 20, 0 to disable)")
    mesh_group.add_argument("--mesh-advancement-ratio", type=float, default=1.05, help="vmtkNetworkExtraction AdvancementRatio (default: 1.05, vmtk's own default)")
    mesh_group.add_argument(
        "--mesh-cap-open-radius-mm",
        type=float,
        default=None,
        help="Radius in mm of the opening cut into a fully closed input "
        "mesh so vmtkNetworkExtraction has a seed opening (default: auto, "
        "~2%% of the mesh's bounding-box diagonal, clamped to 3-10mm).",
    )

    det_group = p.add_mutually_exclusive_group(required=True)
    det_group.add_argument("--weights", help="Path to trained YOLO .pt weights")
    det_group.add_argument(
        "--precomputed-detections", help="Path to precomputed per-frame detections JSON"
    )

    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.1,
        help="Detector confidence threshold (paper: 0.1 -- deliberately low, "
        "since the BYTE-style tracker is designed to use low-confidence "
        "detections rather than discard them). Left at the paper's own "
        "default here after directly testing higher values against it: a "
        "sweep on the merged-dataset yolo11s checkpoint found confirmed-dot "
        "rate is CLEARLY best at 0.10 (66.0%%) and worse at every higher "
        "value tried -- 41.3%% at 0.15, 48.1%% at 0.20, 25.4%% at 0.25 -- and "
        "NOT a clean monotonic slope (0.15 scored worse than 0.20), so this "
        "isn't a smooth tradeoff curve you can interpolate on. If you raise "
        "this, re-validate against a real video/checkpoint rather than "
        "assuming higher is safer; the diameter:distance cue (see viz.py -- "
        "only Tracklet.diameter_distance_match is True ever gets drawn) is "
        "meant to be the noise filter, not this threshold.",
    )
    p.add_argument(
        "--mask-conf-threshold",
        type=float,
        default=0.55,
        help="Segmentation checkpoints only: minimum confidence for a "
        "detection's mask to actually be used (default: 0.55) -- below it, "
        "the detection itself still exists (box, tracking, etc. proceed "
        "normally), but Detection.mask is dropped and every mask-dependent "
        "consumer (the diameter:distance cue, _is_nested's containment "
        "check, the overlay's translucent fill) falls back to its existing "
        "box-based path, same as for a plain non-seg checkpoint. A "
        "SEPARATE, stricter gate on top of --conf-threshold -- see "
        "detection.py module docstring's 'Mask acceptance threshold' "
        "section for why one shared threshold isn't enough (a box can "
        "stay confident on a foreshortened/partly-occluded lumen whose "
        "mask reads noisier). Must be >= --conf-threshold to have any "
        "effect.",
    )
    p.add_argument("--img-size", type=int, default=None, help="Detector inference resolution (default: auto-detect)")
    p.add_argument("--device", default=None, help="'cuda', 'cuda:0', or 'cpu' (default: auto-detect)")
    p.add_argument(
        "--class-id",
        default=None,
        help="Comma-separated class id(s) to treat as detectable lumens (default: all classes)",
    )
    p.add_argument(
        "--reid-weights",
        default=None,
        help="Optional fine-tuned ResNet50 Re-ID checkpoint. If omitted, Re-ID "
        "appearance matching is disabled (motion-only tracking, i.e. "
        "combined cost C falls back to C_m).",
    )

    p.add_argument("--out-video", default=None, help="Optional path to write an annotated overlay video")
    p.add_argument("--out-json", default=None, help="Optional path to write a per-frame localization JSON log")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames (debugging)")

    p.add_argument(
        "--max-match-cost",
        type=float,
        default=0.6,
        help="Airway association: reject a candidate match whose angular cost "
        "exceeds this (default: 0.6). Paper doesn't give an exact number; "
        "kept as a practical gate. Lower = stricter matching.",
    )
    p.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=75.0,
        help="Airway association: candidate children whose take-off angle "
        "against the parent exceeds this are filtered out as implausible "
        "to observe (paper: 'filter candidates based on intersection "
        "angle', no exact number given; default 75deg).",
    )
    p.add_argument(
        "--distance-diameter-weight",
        type=float,
        default=0.4,
        help="Airway association: how much the lumen diameter:distance-from-"
        "reference ratio (angular-width) cue contributes to matching cost, "
        "vs. angular bearing (0=bearing-only/strict-paper, 1=diameter:"
        "distance-only, default: 0.4). NOT part of the paper -- see "
        "association.py module docstring. No effect on graphs without "
        "radius data (falls back to bearing-only automatically).",
    )
    p.add_argument(
        "--virtual-advance-mm",
        type=float,
        default=20.0,
        help="Airway association: how far (mm) past each candidate child's "
        "own bifurcation the diameter:distance cue's 'virtual viewpoint' "
        "sits when it samples that branch's expected diameter and 2D "
        "position from the 3D model (default: 20mm = 2cm -- see "
        "association.py module docstring's 'virtual viewpoint' section). "
        "Clamped to a branch's own length if it's shorter than this.",
    )
    p.add_argument(
        "--virtual-match-threshold",
        type=float,
        default=0.75,
        help="Airway association: how closely a detection's real-image "
        "diameter:distance ratio must agree with the 3D model's virtual-"
        "viewpoint ratio (min/max of the two, both peer-normalized) to mark "
        "that pairing as verified on Tracklet.diameter_distance_match "
        "(default: 0.75, i.e. within 25%%). See association.py module "
        "docstring's 'Virtual verification' section.",
    )
    p.add_argument(
        "--reacquire-max-gap-frames",
        type=int,
        default=90,
        help="Airway association: if every labeled anchor tracklet is lost "
        "at once, how many frames a frozen snapshot of the last real "
        "anchor may still be used as a re-acquisition stand-in before "
        "it's considered too stale to trust (default: 90 = ~3s at 30fps). "
        "0 disables re-acquisition entirely (original one-shot-anchor "
        "behavior -- see association.py module docstring's "
        "'Re-acquisition after total anchor loss' section).",
    )
    p.add_argument(
        "--no-continuous-verification",
        dest="continuous_verification",
        action="store_false",
        default=True,
        help="Airway association: by default, diameter:distance virtual-"
        "model verification (Tracklet.diameter_distance_match, gating the "
        "overlay's dot) re-runs every frame for any group of currently-"
        "visible same-parent labeled tracklets, instead of being decided "
        "once at initial labeling and frozen forever -- this flag disables "
        "that and recovers the original one-shot-only behavior. NOT part "
        "of the paper -- see association.py module docstring's "
        "'Continuous (per-frame) re-verification' section.",
    )

    p.add_argument(
        "--show-graph",
        action="store_true",
        help="Also open a second preview window showing the 3D airway graph "
        "with the current camera location highlighted (needs --display).",
    )
    p.add_argument("--display", action="store_true", help="Show a live on-screen preview window")
    p.add_argument(
        "--out-graph-video",
        default=None,
        help="Optional path to save the airway-graph view as its own video, "
        "synced frame-for-frame with --out-video.",
    )

    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        graph = AirwayGraph.from_path(
            args.graph,
            rebuild_graph=args.rebuild_graph,
            graph_cache_path=args.graph_cache,
            weld_tol_mm=args.mesh_weld_tol_mm,
            min_branch_length_mm=args.mesh_min_branch_length_mm,
            min_radius_mm=args.mesh_min_radius_mm,
            smooth_iterations=args.mesh_smooth_iterations,
            advancement_ratio=args.mesh_advancement_ratio,
            cap_open_radius_mm=args.mesh_cap_open_radius_mm,
        )
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.weights:
        class_id = None
        if args.class_id:
            class_id = [int(c) for c in args.class_id.split(",")]
        detector = LumenDetector(
            args.weights,
            conf_threshold=args.conf_threshold,
            mask_conf_threshold=args.mask_conf_threshold,
            img_size=args.img_size,
            device=args.device,
            class_id=class_id,
        )
        detector.warm_up()
        print(f"Detector device: {describe_device(detector.device)}", file=sys.stderr)
    else:
        detector = PrecomputedDetectionSource.from_json(args.precomputed_detections)

    reid_embedder = None
    if args.reid_weights:
        reid_embedder = ReIDEmbedder(weights_path=args.reid_weights, device=args.device)
        reid_embedder.warm_up()
        print(f"Re-ID device: {describe_device(reid_embedder.device)}", file=sys.stderr)

    pipeline = BronchoTrackPipeline(
        graph=graph,
        detector=detector,
        reid_embedder=reid_embedder,
        association_kwargs={
            "max_match_cost": args.max_match_cost,
            "angle_threshold_deg": args.angle_threshold_deg,
            "distance_diameter_weight": args.distance_diameter_weight,
            "virtual_advance_mm": args.virtual_advance_mm,
            "virtual_match_threshold": args.virtual_match_threshold,
            "reacquire_max_gap_frames": args.reacquire_max_gap_frames,
            "continuous_verification": args.continuous_verification,
        },
    )

    import time

    run_start = time.time()
    results = pipeline.run_on_video(
        video_path=args.video,
        output_json_path=args.out_json,
        output_video_path=args.out_video,
        max_frames=args.max_frames,
        output_graph_video_path=args.out_graph_video,
        display=args.display,
        show_graph=args.show_graph,
    )
    elapsed = time.time() - run_start
    fps = len(results) / elapsed if elapsed > 0 and results else 0.0

    located = sum(1 for r in results if r.location is not None)
    print(f"Processed {len(results)} frames. Location determined on {located} of them.")
    print(f"Processing rate: {fps:.2f} fps ({elapsed:.1f}s total).")
    if results:
        last = results[-1]
        print(f"Final location: {last.location} (generation {last.generation})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
