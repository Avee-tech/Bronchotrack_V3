"""One-off diagnostic overlay: EVERY box ByteTrack currently outputs, in
red, with the subset that actually gets displayed in the normal overlay
(diameter_distance_match is True -- see paper_exact/viz.py's "DISPLAY IS
GATED..." docstring section) drawn again on top in green.

Not part of the pipeline itself -- a debugging/inspection tool requested
to show exactly how much the diameter:distance verification gate is
throwing away vs. what BoxMOT's ByteTrack raw output looks like every
frame, side by side in one video.

Usage:
    python3 scripts/bytetrack_red_green_overlay.py \\
        --video inputs/ModelV3_2.mp4 \\
        --graph patient_airway_graph/airway_graph_ModelV3.json \\
        --weights weights/best.pt \\
        --out out/bytetrack_red_green_overlay.mp4
"""
from __future__ import annotations

import argparse
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

RED = (0, 0, 255)      # BGR
GREEN = (46, 204, 113)  # BGR -- same green as viz.py's palette entry 0
WHITE = (255, 255, 255)


def draw_red_green(frame_bgr: np.ndarray, result) -> np.ndarray:
    out = frame_bgr.copy()

    current = [t for t in result.tracklets if t.time_since_update == 0]
    displayed = [t for t in current if t.diameter_distance_match is True]
    displayed_ids = {t.track_id for t in displayed}

    # Pass 1: every raw ByteTrack output this frame, in red -- drawn first
    # so the green (displayed) boxes always draw on top and stay legible
    # even where a red box and a green box coincide.
    for t in current:
        x1, y1, x2, y2 = (int(round(v)) for v in t.last_box.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, 1, cv2.LINE_AA)
        tag = f"id{t.track_id}"
        if t.label:
            tag += f" {t.label}"
        cv2.putText(out, tag, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)

    # Pass 2: the subset that the normal pipeline would actually display
    # (diameter_distance_match is True), in green, on top.
    for t in displayed:
        x1, y1, x2, y2 = (int(round(v)) for v in t.last_box.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2, cv2.LINE_AA)
        tag = t.label or f"id{t.track_id}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx, ty = x1, max(th + 4, y1 - 6)
        cv2.rectangle(out, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), GREEN, -1)
        cv2.putText(out, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)

    is_stale = getattr(result, "location_is_live", None) is False
    header = (
        f"loc: {result.location or '-'}{' (stale)' if is_stale else ''}  "
        f"gen: {result.generation if result.generation is not None else '-'}  |  "
        f"red: all ByteTrack outputs ({len(current)})  "
        f"green: displayed ({len(displayed)})"
    )
    header_color = (0, 191, 255) if is_stale else WHITE
    cv2.putText(out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--high-conf-thresh",
        type=float,
        default=0.5,
        help="BoxMotByteTrackAdapter's own gate: a NEW track_id is only "
        "ever surfaced as a Tracklet at all the first time its own "
        "detection confidence is at least this (default 0.5, matching "
        "the class default). Lower = more raw detections get a chance to "
        "become a red box in the first place; doesn't change the green "
        "(diameter:distance-verified) gate at all.",
    )
    args = p.parse_args(argv)

    graph = AirwayGraph.from_path(args.graph)
    detector = LumenDetector(args.weights, conf_threshold=0.1, mask_conf_threshold=0.55)
    detector.warm_up()
    print(f"Detector device: {describe_device(detector.device)}", file=sys.stderr)

    tracker = BoxMotByteTrackAdapter(high_conf_thresh=args.high_conf_thresh)
    pipeline = BronchoTrackPipeline(graph=graph, detector=detector, tracker=tracker)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            result = pipeline.process_frame(frame, frame_idx)
            overlay = draw_red_green(frame, result)
            writer.write(overlay)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {frame_idx} frames to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
