"""Multi-lumen tracking module (paper section 3).

Implements the paper's two-stage, BYTE-style association per frame:

  1. Split detections into high-confidence / low-confidence sets.
  2. Match high-confidence detections against existing tracklets using the
     *combined* cost C = lambda*C_a + (1-lambda)*C_m (motion + appearance),
     via the Hungarian algorithm, gated by `match_thresh_high`.
  3. Re-match still-unmatched tracklets against the low-confidence
     detections using *motion cost only*, gated by the looser
     `match_thresh_low` (low-confidence boxes are noisier, so appearance
     similarity is not trusted for them).
  4. Any remaining unmatched high-confidence detections spawn new
     tracklets; unmatched low-confidence detections are discarded.
  5. Tracklets that go unmatched for too many consecutive frames are
     dropped.

An optional `eligibility_fn(tracklet) -> bool` hook lets the caller (see
pipeline.py / association.py) implement the paper's rule "filter tracklets
if airway labels exceed 3 generations from current location" without this
module needing to know about the airway graph itself.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .kalman import KalmanBoxFilter
from .matching import appearance_cost_matrix, assign, combined_cost_matrix, motion_cost_matrix
from .types import BBox, Detection, Tracklet

EligibilityFn = Callable[[Tracklet], bool]


class MultiLumenTracker:
    def __init__(
        self,
        high_conf_thresh: float = 0.5,
        match_thresh_high: float = 0.4,
        match_thresh_low: float = 0.8,
        max_time_since_update: int = 30,
        lam: float = 0.5,
        use_reid: bool = True,
    ):
        """
        Parameters
        ----------
        high_conf_thresh : detections with confidence >= this are "high
            confidence" (matched with combined motion+appearance cost);
            below it, "low confidence" (matched with motion cost only,
            second stage). The detector itself should already be run at a
            low overall threshold (paper: 0.1) so low-confidence-but-real
            detections are available for this second stage.
        match_thresh_high : max allowed combined cost for a stage-1 match
            (paper: 0.4).
        match_thresh_low : max allowed motion cost for a stage-2 match
            (paper: 0.7-0.9 range; default splits the difference at 0.8).
        max_time_since_update : frames a tracklet may go unmatched before
            being dropped.
        lam : weight on appearance cost in the combined cost (paper: 0.5).
        use_reid : if False, stage 1 uses motion cost only too (useful if
            you haven't wired up reid.py / don't have detection crops).
        """
        self.high_conf_thresh = high_conf_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_time_since_update = max_time_since_update
        self.lam = lam
        self.use_reid = use_reid

        self.tracklets: List[Tracklet] = []
        self._filters: dict[int, KalmanBoxFilter] = {}
        self._next_id = 1

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        eligibility_fn: Optional[EligibilityFn] = None,
    ) -> List[Tracklet]:
        # 1. predict every existing tracklet forward one frame
        predicted_boxes: List[BBox] = []
        for t in self.tracklets:
            kf = self._filters[t.track_id]
            predicted_boxes.append(kf.predict())

        eligible_idx = [
            i
            for i, t in enumerate(self.tracklets)
            if eligibility_fn is None or eligibility_fn(t)
        ]
        ineligible_idx = [i for i in range(len(self.tracklets)) if i not in eligible_idx]

        high_dets = [d for d in detections if d.confidence >= self.high_conf_thresh]
        low_dets = [d for d in detections if d.confidence < self.high_conf_thresh]

        matched_track_idx = set()
        matched_det_idx = set()

        # 2. stage 1: high-confidence detections vs eligible tracklets, combined cost
        if eligible_idx and high_dets:
            stage1_boxes = [predicted_boxes[i] for i in eligible_idx]
            stage1_tracklets = [self.tracklets[i] for i in eligible_idx]

            motion_c = motion_cost_matrix(stage1_boxes, high_dets)
            if self.use_reid:
                embeddings = [t.embedding for t in stage1_tracklets]
                appearance_c = appearance_cost_matrix(embeddings, high_dets)
                cost = combined_cost_matrix(motion_c, appearance_c, self.lam)
            else:
                cost = motion_c

            matches, _, _ = assign(cost, self.match_thresh_high)
            for local_r, local_c in matches:
                track_i = eligible_idx[local_r]
                self._apply_match(track_i, high_dets[local_c], frame_idx)
                matched_track_idx.add(track_i)
                matched_det_idx.add(local_c)

        # 3. stage 2: remaining tracklets (eligible + ineligible) vs low-confidence
        #    detections, motion cost only
        remaining_idx = [i for i in range(len(self.tracklets)) if i not in matched_track_idx]
        if remaining_idx and low_dets:
            stage2_boxes = [predicted_boxes[i] for i in remaining_idx]
            motion_c = motion_cost_matrix(stage2_boxes, low_dets)
            matches, _, _ = assign(motion_c, self.match_thresh_low)
            for local_r, local_c in matches:
                track_i = remaining_idx[local_r]
                self._apply_match(track_i, low_dets[local_c], frame_idx)
                matched_track_idx.add(track_i)

        # 4. unmatched tracklets: age them, drop if stale
        keep_tracklets: List[Tracklet] = []
        for i, t in enumerate(self.tracklets):
            if i in matched_track_idx:
                keep_tracklets.append(t)
                continue
            t.time_since_update += 1
            if t.time_since_update <= self.max_time_since_update:
                keep_tracklets.append(t)
            else:
                del self._filters[t.track_id]
        self.tracklets = keep_tracklets

        # 5. unmatched high-confidence detections spawn new tracklets;
        #    unmatched low-confidence detections are discarded (paper)
        for j, det in enumerate(high_dets):
            if j not in matched_det_idx:
                self._spawn_tracklet(det, frame_idx)

        return self.tracklets

    # ------------------------------------------------------------------
    def _apply_match(self, track_idx: int, det: Detection, frame_idx: int) -> None:
        t = self.tracklets[track_idx]
        kf = self._filters[t.track_id]
        kf.update(det.bbox)

        t.boxes.append(kf.as_bbox())
        t.confidences.append(det.confidence)
        t.ind_end = frame_idx
        t.time_since_update = 0
        t.hits += 1
        t.kalman_state = kf.state
        t.kalman_covariance = kf.covariance

        if det.embedding is not None:
            from .reid import update_tracklet_embedding

            update_tracklet_embedding(t, det.embedding)

        if det.mask is not None:
            t.mask = det.mask  # latest observed polygon; see types.py -- not smoothed

    def _spawn_tracklet(self, det: Detection, frame_idx: int) -> None:
        track_id = self._next_id
        self._next_id += 1

        kf = KalmanBoxFilter(det.bbox)
        self._filters[track_id] = kf

        t = Tracklet(
            track_id=track_id,
            ind_start=frame_idx,
            ind_end=frame_idx,
            boxes=[det.bbox],
            confidences=[det.confidence],
            kalman_state=kf.state,
            kalman_covariance=kf.covariance,
            hits=1,
            time_since_update=0,
        )
        if det.embedding is not None:
            t.embedding = det.embedding / (np.linalg.norm(det.embedding) + 1e-9)

        if det.mask is not None:
            t.mask = det.mask

        self.tracklets.append(t)

    def active_tracklets(self, min_hits: int = 1) -> List[Tracklet]:
        return [t for t in self.tracklets if t.hits >= min_hits]
