"""Top-level orchestration: wires detection -> tracking -> association ->
localization together, frame by frame, exactly following the paper's
module order (see package docstring in __init__.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .association import AirwayAssociation
from .graph import AirwayGraph
from .graph_view import AirwayGraphView
from .localization import Localizer
from .motion_model import TreeMotionFilter
from .tracker import MultiLumenTracker
from .types import Detection, Tracklet


@dataclass
class FrameResult:
    frame_idx: int
    detections: List[Detection]
    tracklets: List[Tracklet]
    labels: dict  # {track_id: label}
    location: Optional[str]
    location_raw: Optional[str]  # unsmoothed per-frame vote, see localization.py
    generation: Optional[int]
    approach_trend: Optional[float] = None  # diagnostic only -- see association.py docstring
    motion_location: Optional[str] = None  # see motion_model.py -- graph-constrained Bayes filter
    motion_confidence: Optional[float] = None
    motion_generation: Optional[int] = None


class BronchoTrackPipeline:
    def __init__(
        self,
        graph: AirwayGraph,
        detector,  # LumenDetector or PrecomputedDetectionSource (duck-typed: .infer(frame, idx))
        reid_embedder=None,  # ReIDEmbedder or None to disable appearance matching
        tracker: Optional[MultiLumenTracker] = None,
        localizer_ambiguous_k: int = 1,
        localizer_smoothing_window: int = 5,
        localizer_min_frames_to_switch: int = 3,
        association_kwargs: Optional[dict] = None,
        use_motion_model: bool = True,
        motion_model_kwargs: Optional[dict] = None,
    ):
        self.graph = graph
        self.detector = detector
        self.reid_embedder = reid_embedder

        self.tracker = tracker or MultiLumenTracker(use_reid=reid_embedder is not None)
        self.association = AirwayAssociation(graph, **(association_kwargs or {}))
        self.localizer = Localizer(
            graph,
            ambiguous_k=localizer_ambiguous_k,
            smoothing_window=localizer_smoothing_window,
            min_frames_to_switch=localizer_min_frames_to_switch,
        )
        self.motion_filter: Optional[TreeMotionFilter] = (
            TreeMotionFilter(graph, **(motion_model_kwargs or {})) if use_motion_model else None
        )
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
            detections, frame_idx, eligibility_fn=self.association.eligibility_fn
        )

        # 3. Airway association: label propagation from graph
        labels = self.association.process_frame(active_tracklets, frame_idx)

        # 4. Voting-based branch-level localization (smoothed; see localization.py)
        location = self.localizer.localize(active_tracklets)
        location_raw = self.localizer.last_raw
        generation = self.graph.generation(location) if location in self.graph else None

        # 4b. (optional) graph-constrained motion-model filter -- an
        # alternative, probabilistic "which lumen is the camera in" estimate
        # that runs alongside the vote-based localizer above rather than
        # replacing it (see motion_model.py docstring)
        motion_location = motion_confidence = motion_generation = None
        if self.motion_filter is not None:
            self.motion_filter.predict(self.association.approach_trend)
            self.motion_filter.update(set(labels.values()))
            motion_location, motion_confidence = self.motion_filter.current_location()
            if motion_location is not None:
                motion_generation = self.graph.generation(motion_location)

        return FrameResult(
            frame_idx=frame_idx,
            detections=detections,
            tracklets=active_tracklets,
            labels=labels,
            location=location,
            location_raw=location_raw,
            generation=generation,
            approach_trend=self.association.approach_trend,
            motion_location=motion_location,
            motion_confidence=motion_confidence,
            motion_generation=motion_generation,
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
        window_name: str = "BronchoTrack",
        graph_window_name: str = "Airway Graph",
        show_graph: bool = False,
    ) -> List[FrameResult]:
        """Run the full pipeline over a video file, sequentially and
        exhaustively: every single frame the file contains is decoded and
        processed in order via plain `cv2.VideoCapture.read()` calls, with
        no frame ever skipped regardless of how slow inference is. This is
        different from `run_live` on purpose -- that path intentionally
        *drops* frames it can't keep up with, because it's built for
        genuinely live sources where staying current matters more than
        seeing every frame. For a saved file, there's no "falling behind"
        to avoid, so this path just takes as long as it takes and gets
        every frame. If you've been passing a local video file to
        `--live`/`run_live`, switch to this method (or drop `--live` on
        the CLI) to stop losing frames.

        `frame_callback(result, frame_bgr)` is called after each frame is
        processed, before the overlay is written -- use it for custom
        overlays/logging/streaming without subclassing.

        `display=True` opens an on-screen preview window (same overlay as
        `run_live`) while processing -- press 'q' to stop early. Add
        `show_graph=True` for a second window with the airway-graph view
        (see graph_view.AirwayGraphView).

        `output_graph_video_path`, if given, writes a second video --
        synced frame-for-frame with `output_video_path` -- showing the
        airway graph with the current localized branch highlighted and
        every branch labeled.

        Caveat on the returned `List[FrameResult]`: each `FrameResult.tracklets`
        holds references to the *same* mutable `Tracklet` objects used
        throughout the run, so inspecting `results[i].tracklets[j].label` (or
        `.time_since_update`, `.last_box`, etc.) *after* the whole run has
        finished reflects that tracklet's state as of the LAST frame it was
        updated, not its state at frame `i`. For historically-accurate
        per-frame data, read the JSON log written to `output_json_path`
        (built as a plain-dict snapshot at the correct moment -- see
        `_frame_log_entry`), or use `frame_callback` to capture what you need
        while it's still current.
        """
        import cv2

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
                    # snapshot to a plain dict *now*, while tracklet state is
                    # still current for this frame -- see _frame_log_entry
                    log_entries.append(self._frame_log_entry(result))

                if frame_callback is not None:
                    frame_callback(result, frame)

                overlay = None
                if writer is not None or display:
                    from .viz import draw_overlay

                    overlay = draw_overlay(frame, result)
                    if writer is not None:
                        writer.write(overlay)

                graph_frame = None
                if graph_writer is not None or show_graph_window:
                    graph_frame = self.graph_view.render(result.location, result.generation)
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

    def run_live(
        self,
        source: str,
        display: bool = True,
        window_name: str = "BronchoTrack",
        graph_window_name: str = "Airway Graph",
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        output_graph_video_path: Optional[str] = None,
        show_graph: bool = False,
        assumed_fps: float = 15.0,
        max_frames: Optional[int] = None,
        frame_callback: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
        reconnect: bool = True,
    ) -> List[FrameResult]:
        """Process a live video source (RTSP/RTMP/HTTP stream, or a webcam
        device index passed as a string/int) in real time.

        `show_graph=True` (with `display=True`) opens a second preview
        window showing a live schematic view of the airway graph -- every
        branch labeled with its id/number, color-coded by generation, with
        the currently-localized branch highlighted -- so you can watch the
        scope's progress down the tree alongside the camera feed. Set
        `output_graph_video_path` to also save that view as its own video
        file, synced frame-for-frame with `output_video_path`. See
        graph_view.AirwayGraphView.

        Unlike `run_on_video`, this always works on the *most recently
        arrived* frame (via `live.LiveVideoStream`, a background reader
        thread) rather than draining a file sequentially -- so if
        processing briefly falls behind, frames are dropped rather than
        the pipeline drifting further and further behind live.

        `display=True` opens a `cv2.imshow` preview window with the same
        overlay drawn by `viz.draw_overlay` (boxes, track IDs, branch
        labels, current location/generation); press 'q' in that
        window to stop. Set `display=False` for a headless run (e.g. on a
        server with no attached monitor) -- results are still returned,
        and still optionally written to `output_video_path` /
        `output_json_path`.

        Note: `output_video_path` here is written at `assumed_fps` since a
        live stream doesn't report a reliable frame rate up front the way
        a file does; adjust to your actual source's frame rate for a
        correctly-timed saved copy.
        """
        import cv2

        from .live import LiveVideoStream
        from .viz import draw_overlay

        # cv2.VideoCapture accepts an int device index; allow "0" etc. from CLI
        cap_source = int(source) if isinstance(source, str) and source.isdigit() else source

        stream = LiveVideoStream(cap_source, reconnect=reconnect)
        stream.start()

        writer = None
        graph_writer = None
        results: List[FrameResult] = []
        log_entries: List[dict] = []
        frame_idx = 0
        last_seen_idx = -1
        show_graph_window = display and show_graph

        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                frame, last_seen_idx = stream.read_new(last_seen_idx)
                if frame is None:
                    if not stream.is_connected and not stream._running:
                        # background thread gave up (reconnect exhausted or fatal error)
                        break
                    continue

                result = self.process_frame(frame, frame_idx)
                results.append(result)
                if output_json_path:
                    log_entries.append(self._frame_log_entry(result))

                if frame_callback is not None:
                    frame_callback(result, frame)

                overlay = None
                if display or output_video_path is not None:
                    overlay = draw_overlay(frame, result)

                graph_frame = None
                if show_graph_window or output_graph_video_path is not None:
                    graph_frame = self.graph_view.render(result.location, result.generation)

                if display:
                    cv2.imshow(window_name, overlay)
                    if show_graph_window:
                        cv2.imshow(graph_window_name, graph_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if output_video_path is not None:
                    if writer is None:
                        h, w = overlay.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_video_path, fourcc, assumed_fps, (w, h))
                    writer.write(overlay)

                if output_graph_video_path is not None:
                    if graph_writer is None:
                        gh, gw = graph_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        graph_writer = cv2.VideoWriter(
                            output_graph_video_path, fourcc, assumed_fps, (gw, gh)
                        )
                    graph_writer.write(graph_frame)

                frame_idx += 1
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
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
        """Snapshot a FrameResult into a plain-dict JSON log entry.

        MUST be called immediately after `process_frame` returns for that
        same frame, not deferred to the end of a run: `Tracklet` objects are
        mutable and shared by reference across frames (the same object
        instance is updated in place on every later `tracker.update()` call
        for as long as that track stays alive), so reading `t.label` /
        `t.time_since_update` / `t.last_box` from an *old* FrameResult after
        later frames have already run would silently report each
        tracklet's most recent state instead of its state at that historical
        frame -- e.g. a track that was actively matched at frame 103 but
        later dropped by frame 200 would incorrectly show as absent when
        logged retroactively. Snapshotting per-frame avoids that entirely.
        """
        return {
            "frame_idx": r.frame_idx,
            "location": r.location,
            "location_raw": r.location_raw,
            "generation": r.generation,
            "approach_trend": r.approach_trend,
            "motion_location": r.motion_location,
            "motion_confidence": r.motion_confidence,
            "motion_generation": r.motion_generation,
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
        }

    @staticmethod
    def _write_json_log(log_entries: List[dict], path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump(log_entries, f, indent=2)
