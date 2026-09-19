"""Top-level orchestration for the three-model fusion pipeline.

Wires detection -> tracking -> `FusionAssociation` together, frame by
frame -- structurally similar to `paper_exact.pipeline` (same overall
shape: a `FrameResult`-producing `process_frame` plus a `run_on_video`
loop with the same background-writer/JSON-log conveniences) but calling
entirely new association logic (`fusion.association`), not that module's.

Detection (`detection.LumenDetector` / `PrecomputedDetectionSource`),
tracking (`tracker.MultiLumenTracker`), and the airway graph itself are
reused unmodified from the shared top-level package -- these were already
tested, general-purpose primitives with no paper_exact-specific logic in
them at all, exactly the kind of thing the "clean restart" was scoped to
keep rather than reinvent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from ..graph import AirwayGraph
from ..graph_view import AirwayGraphView
from ..tracker import MultiLumenTracker
from ..types import Detection, Tracklet
from .association import AssociationResult, FusionAssociation
from .kalman_fusion import CandidateConfidence


@dataclass
class FrameResult:
    frame_idx: int
    detections: List[Detection]
    tracklets: List[Tracklet]
    location: str
    generation: Optional[int]
    reference_track_id: Optional[int]
    candidates: Dict[str, CandidateConfidence]
    committed_this_frame: bool
    tau_seconds: Dict[int, Optional[float]]  # track_id -> time-to-contact, this frame's approaching tracklets only
    transition_sanity: Optional[str]  # "plausible"/"implausible"/"unknown", only set when committed_this_frame


class FusionPipeline:
    def __init__(
        self,
        graph: AirwayGraph,
        detector,  # LumenDetector or PrecomputedDetectionSource (duck-typed: .infer(frame, idx))
        reid_embedder=None,
        tracker: Optional[MultiLumenTracker] = None,
        association_kwargs: Optional[dict] = None,
        fps: float = 30.0,
    ):
        self.graph = graph
        self.detector = detector
        self.reid_embedder = reid_embedder
        self.fps = fps

        self.tracker = tracker or MultiLumenTracker(use_reid=reid_embedder is not None)
        self.association = FusionAssociation(graph, **(association_kwargs or {}))
        self.graph_view = AirwayGraphView(graph)

    def process_frame(self, frame_bgr: Optional[np.ndarray], frame_idx: int) -> FrameResult:
        detections = self.detector.infer(frame_bgr, frame_idx)

        if self.reid_embedder is not None:
            for det in detections:
                if det.crop is not None and det.crop.size > 0:
                    det.embedding = self.reid_embedder.embed(det.crop)

        active_tracklets = self.tracker.update(
            detections, frame_idx, eligibility_fn=self.association.eligibility_fn
        )

        result: AssociationResult = self.association.process_frame(active_tracklets, frame_idx)

        # this video's own known fps converts the motion model's scale-free
        # tau_frames into a display-time ETA in seconds -- a conversion
        # applied here only, never inside the motion model itself (see
        # `ApproachMotionModel.tau_frames_for`'s own docstring).
        tau_seconds: Dict[int, Optional[float]] = {}
        for cc in result.candidates.values():
            if cc.matched_track_id is None:
                continue
            tau_frames = self.association.motion_model.tau_frames_for(cc.matched_track_id)
            tau_seconds[cc.matched_track_id] = (tau_frames / self.fps) if tau_frames is not None else None

        return FrameResult(
            frame_idx=frame_idx,
            detections=detections,
            tracklets=active_tracklets,
            location=result.location,
            generation=result.generation,
            reference_track_id=result.reference_track_id,
            candidates=result.candidates,
            committed_this_frame=result.committed_this_frame,
            tau_seconds=tau_seconds,
            transition_sanity=result.transition_sanity,
        )

    def run_on_video(
        self,
        video_path: str,
        output_json_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        output_graph_video_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        frame_callback: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
        display: bool = False,
        window_name: str = "BronchoTrack (fusion)",
        graph_window_name: str = "Airway Graph",
        show_graph: bool = False,
    ) -> List[FrameResult]:
        import os

        import cv2

        for out_path in (output_video_path, output_graph_video_path, output_json_path):
            if out_path:
                parent = os.path.dirname(out_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        graph_writer = None
        if output_graph_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            gh, gw = self.graph_view.canvas_h, self.graph_view.canvas_w
            graph_writer = cv2.VideoWriter(output_graph_video_path, fourcc, fps, (gw, gh))

        show_graph_window = display and show_graph

        results: List[FrameResult] = []
        log_entries: List[dict] = []
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break

                result = self.process_frame(frame, frame_idx)
                results.append(result)
                if output_json_path:
                    log_entries.append(self._frame_log_entry(result))

                if frame_callback is not None:
                    frame_callback(result, frame)

                overlay = None
                if writer is not None or display:
                    from .viz import draw_overlay

                    overlay = draw_overlay(frame, result, self.graph)
                    if writer is not None:
                        writer.write(overlay)

                graph_frame = None
                if graph_writer is not None or show_graph_window:
                    graph_frame = self.graph_view.render(
                        result.location, result.generation, visited=self.association.visited
                    )
                    if graph_writer is not None:
                        graph_writer.write(graph_frame)

                if display:
                    cv2.imshow(window_name, overlay)
                    if show_graph_window:
                        cv2.imshow(graph_window_name, graph_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if graph_writer is not None:
                graph_writer.release()
            if display:
                cv2.destroyWindow(window_name)
                if show_graph_window:
                    cv2.destroyWindow(graph_window_name)

        if output_json_path:
            self._write_json_log(log_entries, output_json_path)

        return results

    @staticmethod
    def _frame_log_entry(r: FrameResult) -> dict:
        return {
            "frame_idx": r.frame_idx,
            "location": r.location,
            "generation": r.generation,
            "committed_this_frame": r.committed_this_frame,
            "transition_sanity": r.transition_sanity,
            "tracklets": [
                {
                    "track_id": t.track_id,
                    "label": t.label,
                    "bbox_xyxy": list(t.last_box.xyxy),
                    "confidence": t.confidences[-1] if t.confidences else None,
                }
                for t in r.tracklets
                if t.time_since_update == 0
            ],
            "candidates": [
                {
                    "label": label,
                    "fused_confidence": cc.fused_confidence,
                    "matched_track_id": cc.matched_track_id,
                    "raw_confidence": cc.raw_confidence,
                    "tau_seconds": r.tau_seconds.get(cc.matched_track_id) if cc.matched_track_id else None,
                }
                for label, cc in r.candidates.items()
            ],
        }

    @staticmethod
    def _write_json_log(log_entries: List[dict], path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump(log_entries, f, indent=2)
