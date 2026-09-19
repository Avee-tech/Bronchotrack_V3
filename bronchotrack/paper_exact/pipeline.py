"""Top-level orchestration -- STRICT paper reference version.

Fork of ``bronchotrack.pipeline`` wiring detection -> tracking ->
association -> localization together, frame by frame, using ONLY this
package's paper-exact ``association.py``/``localization.py`` -- no
diameter-ratio cue, no motion-model Bayes filter, no localization
smoothing. See ``bronchotrack.paper_exact`` (this package's
``__init__.py``) for the full list of deltas against the main package.

Detection, tracking (Kalman+Re-ID+BYTE-style two-stage matching), and the
airway graph itself are unmodified and imported directly from the parent
package, since those modules were already a direct, unembellished port of
the paper's method (see their own docstrings).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..graph import AirwayGraph
from ..graph_view import AirwayGraphView
from ..tracker import MultiLumenTracker
from ..types import Detection, Tracklet
from .association import AirwayAssociation
from .localization import Localizer


@dataclass
class FrameResult:
    frame_idx: int
    detections: List[Detection]
    tracklets: List[Tracklet]
    labels: dict  # {track_id: label}
    location: Optional[str]
    generation: Optional[int]
    # False = carried forward, no fresh vote this frame; None = not supplied by
    # the caller (default, so existing FrameResult(...) call sites -- tests,
    # the pseudocode in report/build_report.py -- don't break) -- see
    # localization.py's "Live vs. carried-forward votes" docstring section.
    location_is_live: Optional[bool] = None


class BronchoTrackPipeline:
    def __init__(
        self,
        graph: AirwayGraph,
        detector,  # LumenDetector or PrecomputedDetectionSource (duck-typed: .infer(frame, idx))
        reid_embedder=None,  # ReIDEmbedder or None to disable appearance matching
        tracker: Optional[MultiLumenTracker] = None,
        localizer_ambiguous_k: int = 1,
        association_kwargs: Optional[dict] = None,
    ):
        self.graph = graph
        self.detector = detector
        self.reid_embedder = reid_embedder

        self.tracker = tracker or MultiLumenTracker(use_reid=reid_embedder is not None)
        self.association = AirwayAssociation(graph, **(association_kwargs or {}))
        self.localizer = Localizer(graph, ambiguous_k=localizer_ambiguous_k)
        self.graph_view = AirwayGraphView(graph)

    def process_frame(self, frame_bgr: Optional[np.ndarray], frame_idx: int) -> FrameResult:
        # 1. Lumen detection
        detections = self.detector.infer(frame_bgr, frame_idx)

        # 1b. (optional) appearance embeddings for Re-ID
        if self.reid_embedder is not None:
            for det in detections:
                if det.crop is not None and det.crop.size > 0:
                    det.embedding = self.reid_embedder.embed(det.crop)

        # 2. Multi-lumen tracking (motion + appearance, two-stage BYTE-style)
        active_tracklets = self.tracker.update(
            detections,
            frame_idx,
            eligibility_fn=self.association.eligibility_fn,
            frame_bgr=frame_bgr,
        )

        # 3. Airway association: label propagation from graph (roll-corrected)
        labels = self.association.process_frame(active_tracklets, frame_idx)

        # 4. Voting-based branch-level localization (Eq. 8, unsmoothed)
        location = self.localizer.localize(active_tracklets)
        generation = self.graph.generation(location) if location in self.graph else None
        location_is_live = self.localizer.last_vote_was_live()

        return FrameResult(
            frame_idx=frame_idx,
            detections=detections,
            tracklets=active_tracklets,
            labels=labels,
            location=location,
            generation=generation,
            location_is_live=location_is_live,
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
        window_name: str = "BronchoTrack (paper-exact)",
        graph_window_name: str = "Airway Graph",
        show_graph: bool = False,
    ) -> List[FrameResult]:
        """Process every frame of a saved video file, in order, exhaustively
        (see ``bronchotrack.pipeline.BronchoTrackPipeline.run_on_video`` for
        the full rationale -- identical here, just without the motion-model
        fields in the log/overlay)."""
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
                    from .viz import draw_overlay  # paper-style persistent per-ID color (see viz.py)

                    overlay = draw_overlay(frame, result)
                    if writer is not None:
                        writer.write(overlay)

                graph_frame = None
                if graph_writer is not None or show_graph_window:
                    graph_frame = self.graph_view.render(
                        result.location,
                        result.generation,
                        visited=set(self.association.gallery.keys()),
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
            # False = carried forward from the last real vote with nothing
            # currently visible to re-confirm it, not a fresh Eq. 8 result
            # this frame -- see localization.py's "Live vs. carried-forward
            # votes" docstring section. Without this, "location" alone
            # can't distinguish "still here, just reconfirmed" from "no
            # current evidence, repeating our last guess".
            "location_is_live": r.location_is_live,
            "tracklets": [
                {
                    "track_id": t.track_id,
                    "label": t.label,
                    "bbox_xyxy": list(t.last_box.xyxy),
                    "confidence": t.confidences[-1] if t.confidences else None,
                    # diameter:distance cue's Kalman-filtered virtual-model
                    # verdict (see association.py's "Virtual verification"
                    # docstring section) -- True/False/None. This is the
                    # SAME field viz.py gates the overlay video's dot on
                    # (only True gets drawn), so it was a real gap that the
                    # JSON log never surfaced it: without this, nothing in
                    # the log distinguished a confirmed detection from one
                    # that was tracked/labeled but suppressed from the
                    # video for disagreeing with the 3D model.
                    "diameter_distance_match": t.diameter_distance_match,
                }
                for t in r.tracklets
                if t.time_since_update == 0
            ],
        }

    @staticmethod
    def _write_json_log(log_entries: List[dict], path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump(log_entries, f, indent=2)
