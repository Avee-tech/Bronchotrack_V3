"""Experimental drop-in tracker: BoxMOT's ByteTrack instead of this
project's own hand-written `bronchotrack.tracker.MultiLumenTracker`.

NOT a paper mechanism, NOT wired in by default -- opt-in via
`--tracker bytetrack` on the paper_exact CLI (see cli.py). Exists to answer
a concrete question: does swapping the custom two-stage BYTE-style
tracker for a maintained, widely-used implementation of the same family of
ideas (ByteTrack, Zhang et al. 2022 -- the paper's own tracker is already
"BYTE-style" by design, see tracker.py's docstring) change anything
downstream, on this video, with everything else (detector, airway
association, localization) held fixed?

What's actually different from `MultiLumenTracker`, honestly:
  * BoxMOT's ByteTrack is motion-only (IoU-based Kalman association) --
    it has no appearance/Re-ID stage, so `reid.py` embeddings computed
    upstream are simply unused here even if `--reid-weights` is passed.
    The custom tracker's stage-1 match blends motion + appearance
    (lambda=0.5 default); this tracker is motion cost only, full stop.
  * No `eligibility_fn` support. The custom tracker filters WHICH existing
    tracklets a detection is even allowed to match against, using the
    paper's "airway labels beyond 3 generations from current location are
    ineligible" rule (see association.py's `eligibility_fn`). BoxMOT's
    ByteTrack has no such hook -- every detection can match every existing
    track regardless of airway-graph distance. This is a real behavioral
    gap, not a paper-faithful simplification; flagged, not hidden.
  * `min_hits` (BoxMOT default 3): a new track only appears in this
    tracker's *output* after 3 consecutive frames of being matched, so a
    lumen that flickers in and out for 1-2 frames never gets a chance to
    be labeled at all here, unlike the custom tracker (which spawns a
    tracklet from a single unmatched high-confidence detection
    immediately).

  Real, log-confirmed consequence of the gap above -- kept at BoxMOT's own
  default (3) after directly testing `min_hits=1` and finding it breaks
  this exact video: with `min_hits=1`, a single spurious low-confidence
  duplicate detection at frame 0 (the detector briefly double-firing on
  one still-wide-open lumen before the carina is even visible -- see the
  frame-0 overlay saved during this experiment) immediately became a
  second reportable track, which happened to satisfy the carina
  bootstrap's naive "exactly 2 detections" trigger (association.py's
  `_try_initialize`) 175 frames too early, seeding a wrong-but-internally-
  self-consistent L/R split for the entire rest of the run. The custom
  tracker never hits this failure mode on this video specifically because
  it implements BYTE's OWN defining rule -- an unmatched low-confidence
  detection is discarded, never allowed to spawn a new tracklet (see
  tracker.py's docstring, step 5) -- which this adapter did not initially
  reproduce. `min_hits=3` doesn't fully restore that same guarantee (a
  low-conf detection that happens to recur 3 frames running can still
  seed a track), but it did prevent this specific one-off false start in
  testing, and is BoxMOT's own tuned default rather than an arbitrary
  override.

  Real, log-confirmed consequence of THAT gap, in turn: `min_hits=3` alone
  was NOT enough -- the same spurious frame-0 box recurred more than 3
  frames running and still triggered the false init. What actually fixed
  it is `high_conf_thresh` below, which reproduces `tracker.py`'s real
  guard directly (discard an unmatched low-confidence detection outright,
  never let it spawn a track), rather than relying on BoxMOT's `min_hits`
  as a proxy for the same idea.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..types import BBox, Detection, Tracklet

EligibilityFn = Callable[[Tracklet], bool]


class BoxMotByteTrackAdapter:
    """Same `.update(detections, frame_idx, eligibility_fn=None,
    frame_bgr=None) -> List[Tracklet]` contract as
    `bronchotrack.tracker.MultiLumenTracker`, so it's a drop-in
    replacement for `BronchoTrackPipeline(tracker=...)` -- nothing else in
    the pipeline needs to know which one is running.
    """

    def __init__(
        self,
        track_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        min_hits: int = 3,
        high_conf_thresh: float = 0.5,
    ):
        """
        Parameters
        ----------
        track_thresh : BoxMOT's ByteTrack internal high/low confidence
            split (its own analogue of `high_conf_thresh` above). Left at
            a low value consistent with the paper's own low detector
            threshold (0.1) so ByteTrack's own two-stage logic still has
            low-confidence detections to use, same reasoning as
            `detection.py`'s own docstring.
        match_thresh : IoU gate for its association step.
        track_buffer : frames a lost track is kept before being dropped
            (its analogue of `max_time_since_update`).
        min_hits : consecutive matched frames before a new track is
            reported at all. Left at BoxMOT's own default (3) -- see the
            module docstring for why `min_hits=1` was tried and rejected
            (a real false carina-initialization at frame 0 on this video),
            and why `min_hits=3` alone turned out not to be enough either.
        high_conf_thresh : a track is only ever newly surfaced to the rest
            of the pipeline (i.e. this adapter creates a `Tracklet` for it
            at all) the first time ITS OWN detection confidence is at
            least this. Reproduces `tracker.py`'s actual guard --
            "unmatched low-confidence detections spawn nothing, ever" --
            which BoxMOT's own `min_hits` does not provide on its own (a
            low-confidence box that happens to recur `min_hits` frames in
            a row still gets reported). See module docstring.
        """
        from boxmot import ByteTrack

        self._bt = ByteTrack(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            min_hits=min_hits,
        )
        self.high_conf_thresh = high_conf_thresh
        self.tracklets: Dict[int, Tracklet] = {}
        self.max_time_since_update = track_buffer

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        eligibility_fn: Optional[EligibilityFn] = None,  # accepted, unused -- see module docstring
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[Tracklet]:
        if frame_bgr is None:
            # BoxMOT's ByteTrack is motion-only and doesn't actually read
            # pixel content, but its API wants an image-shaped array for
            # internal bookkeeping (e.g. clipping boxes to frame bounds).
            # Fall back to a plausible blank frame only if the caller
            # truly has no image (shouldn't happen via the CLI's video
            # path, but keeps this usable against precomputed-detection
            # sources with no image at all).
            frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)

        if detections:
            dets_arr = np.array(
                [
                    [*d.bbox.xyxy, d.confidence, float(d.class_id)]
                    for d in detections
                ],
                dtype=np.float64,
            )
        else:
            dets_arr = np.zeros((0, 6), dtype=np.float64)

        out = self._bt.update(dets_arr, frame_bgr)

        seen_ids = set()
        for row in out:
            x1, y1, x2, y2, track_id, conf, cls_id, det_ind = row
            track_id = int(track_id)
            det = detections[int(det_ind)] if 0 <= int(det_ind) < len(detections) else None

            t = self.tracklets.get(track_id)
            bbox = BBox.from_xyxy(x1, y1, x2, y2)
            if t is None:
                if conf < self.high_conf_thresh:
                    # This track_id exists inside BoxMOT's own internal
                    # state now, but we deliberately never surface it here
                    # -- see `high_conf_thresh` above / module docstring.
                    # `seen_ids` intentionally does NOT get this id either,
                    # so nothing downstream (including a later frame's
                    # staleness check) ever learns it exists unless/until
                    # it reappears at high confidence, at which point this
                    # same branch creates it for the first time then.
                    continue
                seen_ids.add(track_id)
                t = Tracklet(
                    track_id=track_id,
                    ind_start=frame_idx,
                    ind_end=frame_idx,
                    boxes=[bbox],
                    confidences=[float(conf)],
                    hits=1,
                    time_since_update=0,
                )
                self.tracklets[track_id] = t
            else:
                seen_ids.add(track_id)
                t.boxes.append(bbox)
                t.confidences.append(float(conf))
                t.ind_end = frame_idx
                t.hits += 1
                t.time_since_update = 0

            if det is not None and det.mask is not None:
                t.mask = det.mask
            if det is not None and det.embedding is not None:
                t.embedding = det.embedding  # unused by this tracker's own matching; kept for parity

        # age out anything ByteTrack itself no longer reports this frame
        stale = [tid for tid in self.tracklets if tid not in seen_ids]
        for tid in stale:
            t = self.tracklets[tid]
            t.time_since_update += 1
            if t.time_since_update > self.max_time_since_update:
                del self.tracklets[tid]

        return list(self.tracklets.values())
