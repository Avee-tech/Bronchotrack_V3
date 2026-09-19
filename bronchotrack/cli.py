"""Command-line entry point.

Examples
--------
Run live YOLOv11 inference + full tracking/association pipeline over a
bronchoscopy video, using your 3D-Slicer-derived airway graph, and write
both an annotated overlay video and a per-frame JSON localization log:

    python -m bronchotrack.cli \\
        --video path/to/bronchoscopy.mp4 \\
        --graph path/to/airway_graph.json \\
        --weights path/to/lumen_yolov11.pt \\
        --reid-weights path/to/reid_resnet50.pt \\
        --out-video out/overlay.mp4 \\
        --out-json out/localization_log.json

If you've already run detection separately and saved per-frame boxes to
JSON (see PrecomputedDetectionSource docstring in detection.py for the
expected format), use --precomputed-detections instead of --weights.

For a live RTSP/RTMP/HTTP stream (or a webcam device index) with an
on-screen tracked-overlay preview window, add --live and pass the stream
URL as --video:

    python -m bronchotrack.cli \\
        --video rtsp://192.168.1.50:8554/bronch \\
        --live \\
        --graph path/to/airway_graph.json \\
        --weights path/to/lumen_yolov11.pt \\
        --out-video out/session_recording.mp4 \\
        --out-json out/session_log.json

Press 'q' in the preview window to stop. Add --no-display for headless
live runs (e.g. on a server with no monitor attached).

DroidCam as a source: its network stream is an HTTP MJPEG URL like
http://<phone-ip>:4747/video (verify the exact path in the DroidCam app;
some versions use /mjpegfeed?640x480) -- pass that as --video. If you
instead installed the DroidCam desktop client, it shows up as a regular
webcam device index (e.g. --video 1) instead of a URL. See README.md
"Using DroidCam as the source" for more detail.

A NEW PATIENT: --graph ALSO ACCEPTS A RAW SURFACE MESH (.vtk/.vtp/.stl)
-------------------------------------------------------------------------
Point --graph directly at a patient's exported 3D model instead of a
pre-built graph JSON:

    python -m bronchotrack.cli \\
        --video new_patient.mp4 \\
        --graph NewPatientModel.vtk \\
        ...

The mesh->graph conversion (see bronchotrack.mesh_to_graph's module
docstring) runs automatically the first time, and the result is cached
next to the mesh (<mesh_stem>_graph.json) and reused on every later run
against that same file -- so this only costs the extra time once per
patient, not once per run. Needs vtk/vmtk importable in this environment;
see bronchotrack.mesh_to_graph's docstring if that's not yet installed
here. The --mesh-* flags below tune that one-time conversion; irrelevant,
and ignored, when --graph already points at a graph JSON.
"""
from __future__ import annotations

import argparse
import os
import sys

from .detection import LumenDetector, PrecomputedDetectionSource
from .graph import AirwayGraph
from .pipeline import BronchoTrackPipeline
from .reid import ReIDEmbedder
from .utils import describe_device


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BronchoTrack pipeline CLI")
    p.add_argument(
        "--video",
        required=True,
        help="Path to a bronchoscopy video file, OR (with --live) an RTSP/RTMP/HTTP "
        "stream URL / webcam device index",
    )
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

    p.add_argument(
        "--live",
        action="store_true",
        help="Treat --video as a genuinely live source (network stream or device "
        "index), and process frames in real time via a background reader thread "
        "that always works on the most recent frame -- when processing can't keep "
        "up, older frames are intentionally DROPPED rather than letting the "
        "pipeline drift further and further behind real time. Do NOT use this for "
        "a local video file you want processed exhaustively -- that reader races "
        "ahead through the file as fast as it can decode it, so CPU-bound "
        "inference will fall behind and silently skip most of the file. For a "
        "saved file, leave --live off (the default file mode processes every "
        "single frame, in order, no matter how long it takes).",
    )
    p.add_argument(
        "--display",
        dest="display",
        action="store_true",
        default=None,
        help="Show a live on-screen preview window with the tracked overlay, "
        "for --live or a regular video file. Default: on for --live, off "
        "otherwise (a batch run over a saved file shouldn't pop up a window "
        "unless you ask for it). Press 'q' in the window to stop early -- "
        "for regular (non-live) file playback that means your output "
        "video/JSON will only cover the frames processed so far, not the "
        "whole file; every frame IS still processed in order up to that "
        "point (see run_on_video docstring) -- nothing is skipped along "
        "the way, stopping early just means you didn't run to the end.",
    )
    p.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Run headless, no preview window (e.g. on a server).",
    )
    p.add_argument(
        "--live-fps",
        type=float,
        default=15.0,
        help="(--live only) assumed frame rate for --out-video, since a live stream "
        "doesn't report one reliably up front. Match your source's actual FPS for a "
        "correctly-timed saved recording.",
    )
    p.add_argument(
        "--no-reconnect",
        dest="reconnect",
        action="store_false",
        default=True,
        help="(--live only) don't attempt to reconnect if the live source drops.",
    )

    det_group = p.add_mutually_exclusive_group(required=True)
    det_group.add_argument("--weights", help="Path to trained YOLOv11 .pt weights")
    det_group.add_argument(
        "--precomputed-detections", help="Path to precomputed per-frame detections JSON"
    )

    p.add_argument("--conf-threshold", type=float, default=0.1, help="Detector confidence threshold")
    p.add_argument(
        "--mask-conf-threshold",
        type=float,
        default=0.55,
        help="Segmentation checkpoints only: minimum confidence for a detection's "
        "mask to actually be used (default: 0.55); below it the detection still "
        "exists but falls back to box-based handling, same as a plain non-seg "
        "checkpoint. See detection.py module docstring. Must be >= --conf-threshold "
        "to have any effect.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Detector inference resolution. Default: auto-detected from the "
        "checkpoint's own training config -- only set this if you specifically "
        "want a different resolution than the model was trained at.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda', 'cuda:0', or 'cpu'. Default: auto-detect -- uses your GPU "
        "automatically if a CUDA-capable one is available (e.g. a laptop RTX "
        "4050), falling back to CPU otherwise. The CLI prints which device it "
        "resolved to at startup so you can confirm the GPU is actually being "
        "used; pass this explicitly to force one or the other.",
    )
    p.add_argument(
        "--class-id",
        default=None,
        help="Comma-separated class id(s) to treat as detectable lumens (e.g. '0' or "
        "'0,1'). Default: keep every class the model outputs. Run "
        "`python3 -c \"from bronchotrack.detection import LumenDetector; "
        "print(LumenDetector('WEIGHTS').class_names)\"` if you're not sure what "
        "classes your model has.",
    )
    p.add_argument(
        "--reid-weights",
        default=None,
        help="Optional fine-tuned ResNet50 Re-ID checkpoint. If omitted, "
        "Re-ID appearance matching is disabled (motion-only tracking).",
    )

    p.add_argument("--out-video", default=None, help="Optional path to write an annotated overlay video")
    p.add_argument("--out-json", default=None, help="Optional path to write a per-frame localization JSON log")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames (debugging)")

    p.add_argument(
        "--diameter-weight",
        type=float,
        default=0.5,
        help="Airway association: how much the scale-invariant lumen-diameter-"
        "ratio cue contributes to matching cost, vs. angular bearing (0=angle-"
        "only, 1=size-ratio-only, default: 0.5). See association.py module "
        "docstring for the exact formula. No effect on graphs without radius "
        "data (falls back to angle-only automatically).",
    )
    p.add_argument(
        "--max-match-cost",
        type=float,
        default=0.6,
        help="Airway association: reject a candidate match whose combined "
        "cost exceeds this (default: 0.6). Lower = stricter matching.",
    )
    p.add_argument(
        "--diameter-lookahead",
        type=float,
        default=0.15,
        help="Airway association: fraction of a candidate branch's own arc "
        "length to average over (from its start) when computing its "
        "expected diameter for the size-ratio cue, instead of a single "
        "point sample at the very start (default: 0.15). Set to 0 to use "
        "the exact radius_at_start value. Only affects graphs with a full "
        "centerline.",
    )

    p.add_argument(
        "--show-graph",
        action="store_true",
        help="Also open a second preview window (needs --display, on by "
        "default for --live) showing a schematic view of the 3D airway "
        "graph with the current camera location highlighted and branch "
        "numbers labeled.",
    )
    p.add_argument(
        "--out-graph-video",
        default=None,
        help="Optional path to save the airway-graph view (see --show-graph) "
        "as its own video file, synced frame-for-frame with --out-video.",
    )

    p.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Temporal smoothing: how many recent per-frame location votes to consider "
        "before switching the reported location (default: 5). Set to 1 to report the "
        "raw per-frame vote directly, unsmoothed.",
    )
    p.add_argument(
        "--smoothing-min-frames",
        type=int,
        default=3,
        help="Temporal smoothing: a challenger branch must win the raw per-frame vote "
        "this many times within --smoothing-window frames before the reported location "
        "switches to it (default: 3). For a continuous-insertion recording, this keeps "
        "the reported location from flickering between branches on a single noisy frame.",
    )

    p.add_argument(
        "--no-motion-model",
        dest="use_motion_model",
        action="store_false",
        default=True,
        help="Disable the graph-constrained motion-model filter (see motion_model.py) "
        "that runs alongside the default location/generation output, reported as "
        "'motion_location'/'motion_confidence'/'motion_generation' in --out-json and "
        "as a second 'motion: ...' line in the overlay. On by default.",
    )
    p.add_argument(
        "--motion-p-advance",
        type=float,
        default=0.06,
        help="Motion model: baseline per-frame probability of advancing to a child "
        "branch, before the approach-trend diagnostic nudges it up/down (default: "
        "0.06 -- deliberately low, since real bifurcation crossings are rare on a "
        "per-frame basis; see README 'Motion-model filter'). Higher = the filter "
        "moves down the tree more readily on ambiguous frames.",
    )

    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.live and os.path.isfile(args.video):
        print(
            f"WARNING: --live was given a local file ('{args.video}') as --video. "
            "--live's reader always works on the most recent frame and DROPS "
            "older ones it can't keep up with -- appropriate for a real-time "
            "stream, but it means most of a saved file gets silently skipped "
            "when (CPU-bound) inference can't decode+process as fast as the "
            "file reader races through it. Falling back to regular file mode "
            "instead, which processes every single frame in order.",
            file=sys.stderr,
        )
        args.live = False

    display = args.display if args.display is not None else args.live

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
        detector.warm_up()  # resolve device now so we can report it below
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
        localizer_smoothing_window=args.smoothing_window,
        localizer_min_frames_to_switch=args.smoothing_min_frames,
        association_kwargs={
            "diameter_weight": args.diameter_weight,
            "max_match_cost": args.max_match_cost,
            "diameter_lookahead_fraction": args.diameter_lookahead,
        },
        use_motion_model=args.use_motion_model,
        motion_model_kwargs={"p_advance": args.motion_p_advance},
    )

    import time

    run_start = time.time()

    if args.live:
        results = pipeline.run_live(
            source=args.video,
            display=display,
            output_json_path=args.out_json,
            output_video_path=args.out_video,
            assumed_fps=args.live_fps,
            max_frames=args.max_frames,
            reconnect=args.reconnect,
            show_graph=args.show_graph,
            output_graph_video_path=args.out_graph_video,
        )
    else:
        results = pipeline.run_on_video(
            video_path=args.video,
            output_json_path=args.out_json,
            output_video_path=args.out_video,
            max_frames=args.max_frames,
            output_graph_video_path=args.out_graph_video,
            display=display,
            show_graph=args.show_graph,
        )

    elapsed = time.time() - run_start
    fps = len(results) / elapsed if elapsed > 0 and results else 0.0

    located = sum(1 for r in results if r.location is not None)
    print(f"Processed {len(results)} frames. Location determined on {located} of them.")
    print(
        f"Processing rate: {fps:.2f} fps ({elapsed:.1f}s total"
        f"{', includes waiting on --live source' if args.live else ''})."
    )
    if results:
        last = results[-1]
        print(f"Final location: {last.location} (generation {last.generation})")
        if last.motion_location is not None:
            conf = f"{last.motion_confidence:.2f}" if last.motion_confidence is not None else "?"
            print(
                f"Final motion-model location: {last.motion_location} "
                f"(generation {last.motion_generation}, confidence {conf})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
