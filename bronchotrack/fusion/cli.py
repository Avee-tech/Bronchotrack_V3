"""Command-line entry point -- three-model fusion build.

    python3 -m bronchotrack.fusion.cli \\
        --video path/to/bronchoscopy.mp4 \\
        --graph path/to/airway_graph.json \\
        --weights path/to/lumen_yolo.pt \\
        --out-video out/fusion_overlay.mp4 \\
        --out-json out/fusion_log.json

See `bronchotrack.fusion`'s own `__init__.py` for the full design (three
models -- diameter-growth time-to-contact, point-based "bronchotrack
version" bearing matching, whole-mask "ratio method" matching -- combined
via a per-candidate-label Kalman filter). `--graph` accepts a raw surface
mesh (.vtk/.vtp/.stl) exactly like the other two CLIs -- see
`paper_exact.cli`'s own docstring for that flow, unchanged here.
"""
from __future__ import annotations

import argparse
import sys

from ..detection import LumenDetector, PrecomputedDetectionSource
from ..graph import AirwayGraph
from ..reid import ReIDEmbedder
from ..utils import describe_device
from .pipeline import FusionPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BronchoTrack pipeline CLI (three-model fusion build)")
    p.add_argument("--video", required=True, help="Path to a bronchoscopy video file")
    p.add_argument(
        "--graph",
        required=True,
        help="Path to airway graph JSON, OR a raw surface mesh (.vtk/.vtp/.stl) "
        "to build one from automatically (cached next to the mesh on first use).",
    )
    p.add_argument("--rebuild-graph", action="store_true", help="Force re-running mesh->graph conversion")
    p.add_argument("--graph-cache", default=None, help="Where to cache a mesh-built graph JSON")
    mesh_group = p.add_argument_group("mesh conversion (only used when --graph is a .vtk/.vtp/.stl mesh)")
    mesh_group.add_argument("--mesh-weld-tol-mm", type=float, default=2.0)
    mesh_group.add_argument("--mesh-min-branch-length-mm", type=float, default=1.0)
    mesh_group.add_argument("--mesh-min-radius-mm", type=float, default=0.4)
    mesh_group.add_argument("--mesh-smooth-iterations", type=int, default=20)
    mesh_group.add_argument("--mesh-advancement-ratio", type=float, default=1.05)
    mesh_group.add_argument("--mesh-cap-open-radius-mm", type=float, default=None)

    det_group = p.add_mutually_exclusive_group(required=True)
    det_group.add_argument("--weights", help="Path to trained YOLO .pt weights (segmentation checkpoint "
                            "strongly recommended -- the ratio method needs masks to do anything useful; "
                            "without masks it falls back to box-based measurements and effectively "
                            "degrades toward the bronchotrack-version cue)")
    det_group.add_argument("--precomputed-detections", help="Path to precomputed per-frame detections JSON")

    p.add_argument("--conf-threshold", type=float, default=0.25, help="Detector confidence threshold")
    p.add_argument(
        "--mask-conf-threshold",
        type=float,
        default=0.55,
        help="Segmentation checkpoints only: minimum confidence for a detection's "
        "mask to actually be used; below it, falls back to box-based handling. "
        "See detection.py module docstring. Must be >= --conf-threshold to have "
        "any effect.",
    )
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--class-id", default=None, help="Comma-separated class id(s) to treat as lumens")
    p.add_argument("--reid-weights", default=None)

    p.add_argument("--out-video", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-graph-video", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--display", action="store_true")
    p.add_argument("--show-graph", action="store_true")

    fusion_group = p.add_argument_group("fusion weights/thresholds")
    fusion_group.add_argument(
        "--bearing-weight", type=float, default=0.5,
        help="Weight of the point-based bronchotrack-version bearing cue in the blended cost (default 0.5)",
    )
    fusion_group.add_argument(
        "--ratio-weight", type=float, default=0.5,
        help="Weight of the whole-mask ratio-method cue in the blended cost (default 0.5)",
    )
    fusion_group.add_argument(
        "--motion-gate-strength", type=float, default=0.3,
        help="How strongly the diameter-growth motion model's approach signal nudges a "
        "candidate's raw confidence before Kalman smoothing (0=motion model ignored, default 0.3)",
    )
    fusion_group.add_argument("--max-match-cost", type=float, default=0.6)
    fusion_group.add_argument(
        "--commit-threshold", type=float, default=0.65,
        help="Fused confidence a candidate must clear to be eligible to commit as the new location",
    )
    fusion_group.add_argument(
        "--commit-frames", type=int, default=5,
        help="Consecutive frames a candidate must clear --commit-threshold before the location actually advances",
    )
    fusion_group.add_argument("--virtual-advance-mm", type=float, default=20.0)
    fusion_group.add_argument(
        "--display-threshold", type=float, default=0.3,
        help="Minimum fused confidence for a candidate to appear in the HUD/log at all",
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
        class_id = [int(c) for c in args.class_id.split(",")] if args.class_id else None
        detector = LumenDetector(
            args.weights, conf_threshold=args.conf_threshold,
            mask_conf_threshold=args.mask_conf_threshold, img_size=args.img_size,
            device=args.device, class_id=class_id,
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

    pipeline = FusionPipeline(
        graph=graph,
        detector=detector,
        reid_embedder=reid_embedder,
        association_kwargs={
            "bearing_weight": args.bearing_weight,
            "ratio_weight": args.ratio_weight,
            "motion_gate_strength": args.motion_gate_strength,
            "max_match_cost": args.max_match_cost,
            "commit_threshold": args.commit_threshold,
            "commit_frames": args.commit_frames,
            "virtual_advance_mm": args.virtual_advance_mm,
            "display_threshold": args.display_threshold,
        },
    )

    import time

    run_start = time.time()
    results = pipeline.run_on_video(
        video_path=args.video,
        output_json_path=args.out_json,
        output_video_path=args.out_video,
        output_graph_video_path=args.out_graph_video,
        max_frames=args.max_frames,
        display=args.display,
        show_graph=args.show_graph,
    )
    elapsed = time.time() - run_start
    fps = len(results) / elapsed if elapsed > 0 and results else 0.0

    committed = sum(1 for r in results if r.committed_this_frame)
    print(f"Processed {len(results)} frames. {committed} committed branch transitions.")
    print(f"Processing rate: {fps:.2f} fps ({elapsed:.1f}s total).")

    verdicts = [r.transition_sanity for r in results if r.committed_this_frame]
    plausible = verdicts.count("plausible")
    implausible = verdicts.count("implausible")
    unknown = verdicts.count("unknown")
    print(
        f"Motion-model sanity check (diameter-growth tau vs. this run's own empirical "
        f"closing-speed history -- NOT used to calibrate tau, only to flag it): "
        f"{plausible} plausible, {implausible} implausible, {unknown} unknown/insufficient-history."
    )

    if results:
        last = results[-1]
        print(f"Final location: {last.location} (generation {last.generation})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
