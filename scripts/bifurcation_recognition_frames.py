"""Run the pipeline over the whole video once, and for every distinct
bifurcation (`FrameResult.location`) the localizer ever settles on, keep
the single best frame seen there -- "best" meaning the most currently-
tracked children simultaneously confirmed (green, diameter_distance_match
is True), preferring frames where the location estimate itself isn't
stale. That gives one representative, clearly-a-bifurcation frame per
fork actually visited in this video, with the full yellow/red/green
recognition-funnel overlay burned in, so recognition performance can be
inspected fork-by-fork rather than only as one aggregate number over the
whole run.

Also prints a per-bifurcation table: how many children that branch has in
the graph vs. how many were raw-detected (yellow) / tracked (red) /
diameter:distance-confirmed (green) in its representative frame, plus
which specific labels are missing.

Usage:
    python3 scripts/bifurcation_recognition_frames.py \\
        --video inputs/ModelV3_2.mp4 \\
        --graph patient_airway_graph/airway_graph_ModelV3.json \\
        --weights weights/best.pt \\
        --out-dir out/bifurcation_frames
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from bronchotrack.detection import LumenDetector
from bronchotrack.graph import AirwayGraph
from bronchotrack.utils import describe_device
from bronchotrack.paper_exact.boxmot_adapter import BoxMotByteTrackAdapter
from bronchotrack.paper_exact.pipeline import BronchoTrackPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack_red_green_overlay import draw_red_green  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    # Same settings this session has settled on so far.
    p.add_argument("--high-conf-thresh", type=float, default=0.1)
    p.add_argument("--virtual-match-threshold", type=float, default=0.3)
    p.add_argument("--track-buffer", type=int, default=36)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--max-match-cost", type=float, default=0.1)
    p.add_argument("--dynamic-virtual-advance", action="store_true", default=True)
    p.add_argument("--no-dynamic-virtual-advance", dest="dynamic_virtual_advance", action="store_false")
    p.add_argument("--virtual-advance-base-fraction", type=float, default=0.5)
    p.add_argument("--virtual-advance-growth-per-generation", type=float, default=0.10)
    args = p.parse_args(argv)

    graph = AirwayGraph.from_path(args.graph)
    detector = LumenDetector(args.weights, conf_threshold=0.1, mask_conf_threshold=0.55)
    detector.warm_up()
    print(f"Detector device: {describe_device(detector.device)}", file=sys.stderr)

    tracker = BoxMotByteTrackAdapter(
        high_conf_thresh=args.high_conf_thresh,
        track_buffer=args.track_buffer,
        min_hits=args.min_hits,
    )
    pipeline = BronchoTrackPipeline(
        graph=graph,
        detector=detector,
        tracker=tracker,
        association_kwargs={
            "virtual_match_threshold": args.virtual_match_threshold,
            "max_match_cost": args.max_match_cost,
            "dynamic_virtual_advance": args.dynamic_virtual_advance,
            "virtual_advance_base_fraction": args.virtual_advance_base_fraction,
            "virtual_advance_growth_per_generation": args.virtual_advance_growth_per_generation,
            "continuous_verification": True,
        },
    )

    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {args.video}")

    # location -> {score, frame_idx, overlay(np.ndarray), green_labels, red_labels, yellow_count}
    best: dict = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        result = pipeline.process_frame(frame, frame_idx)
        frame_idx += 1

        loc = result.location
        if loc is None:
            continue
        is_live = getattr(result, "location_is_live", None) is not False

        current = [t for t in result.tracklets if t.time_since_update == 0]
        green = [t for t in current if t.diameter_distance_match is True]
        red_labels = sorted({t.label for t in current if t.label})
        green_labels = sorted({t.label for t in green if t.label})

        # score: prefer live localization, then most confirmed children,
        # then most tracked children as a tiebreak.
        score = (1 if is_live else 0, len(green_labels), len(red_labels))

        prev = best.get(loc)
        if prev is None or score > prev["score"]:
            overlay = draw_red_green(frame, result)
            best[loc] = {
                "score": score,
                "frame_idx": frame_idx - 1,
                "overlay": overlay,
                "green_labels": green_labels,
                "red_labels": red_labels,
                "yellow_count": len(result.detections),
                "is_live": is_live,
            }

    cap.release()

    # ---- write per-bifurcation frames + build the summary table ----
    rows = []
    for loc, info in sorted(best.items(), key=lambda kv: (graph.generation(kv[0]) if kv[0] in graph.nodes else 99, kv[0])):
        children = graph.children(loc) if loc in graph.nodes else []
        out_path = os.path.join(args.out_dir, f"{loc}_frame{info['frame_idx']:04d}.png")
        cv2.imwrite(out_path, info["overlay"])
        missing = [c for c in children if c not in info["green_labels"]]
        rows.append({
            "location": loc,
            "generation": graph.generation(loc) if loc in graph.nodes else None,
            "frame_idx": info["frame_idx"],
            "is_live": info["is_live"],
            "graph_children": children,
            "n_graph_children": len(children),
            "yellow_raw_detections": info["yellow_count"],
            "red_tracked_labels": info["red_labels"],
            "green_confirmed_labels": info["green_labels"],
            "n_confirmed": len(info["green_labels"]),
            "missing_children": missing,
            "image": out_path,
        })

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'location':10s} {'gen':>3s} {'frame':>6s} {'live':>4s} {'children':>8s} {'yellow':>6s} {'confirmed':>9s}  labels")
    for r in rows:
        print(
            f"{r['location']:10s} {str(r['generation']):>3s} {r['frame_idx']:6d} "
            f"{'Y' if r['is_live'] else 'n':>4s} {r['n_graph_children']:8d} "
            f"{r['yellow_raw_detections']:6d} {r['n_confirmed']:9d}  "
            f"confirmed={r['green_confirmed_labels']} missing={r['missing_children']}"
        )
    print(f"\nWrote {len(rows)} representative frames to {args.out_dir}, summary at {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
