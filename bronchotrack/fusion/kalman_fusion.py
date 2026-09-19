"""Combine `bronchotrack_id`'s bearing cue, `ratio_id`'s whole-mask cue,
and `motion_model`'s approach signal into one persistent, Kalman-smoothed
confidence per candidate branch label -- "finally use a kalman filter to
combine all three models" (the user's own words).

Design
-------
The two identification cues each produce a (candidates x detections) COST
matrix for the same frame (0 = perfect agreement). They're blended into
one matrix (weighted average, same pattern as
`paper_exact.association._match_candidates`'s own cost blend) and a
single Hungarian assignment is run on the blend -- one assignment, not two
independently reconciled ones, so there's never a frame where the two
cues silently "vote" for different detections without that disagreement
being visible in the blended cost itself.

The motion model doesn't get a matrix entry of its own: on its own, a
diameter-growth rate can't identify WHICH branch is ahead, only whether
whatever's currently in view is being approached. Instead, once a
candidate label is Hungarian-matched to a detection this frame, that
detection's own `ApproachEstimate` (if any) nudges the raw confidence
before it's smoothed -- approaching detections get pulled toward higher
confidence, non-approaching ones get pulled down -- because a branch the
scope is NOT actually heading toward has no physical reason to be growing
in the frame, however well its bearing/ratio happens to line up this
instant. This is the concrete way all three models combine into one
number, per label, per frame:

    raw = 1 - blended_cost[label, matched_detection]
    if matched detection is approaching:    raw = raw*(1-gate) + 1.0*gate
    elif matched detection is NOT approaching: raw = raw*(1-gate)
    (gate = motion_gate_strength, default 0.3 -- a nudge, not a veto)

That per-frame `raw` measurement is then fed into a per-LABEL (not
per-track_id -- track IDs come and go across frames as the tracker loses
and re-spawns tracklets over occlusions; the candidate branch LABEL is
what's actually persistent across a bifurcation's whole dwell time)
`ScalarKalmanFilter`, exactly the same predict/update recursion
`paper_exact.association` uses for its own scalar smoothing. A label with
no detection matched to it this frame gets a predict-only step (the
filter's belief decays toward its own last velocity estimate rather than
being force-reset), so a candidate that drops out of view for a few
frames doesn't instantly lose all accumulated confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..scalar_kalman import ScalarKalmanFilter
from ..types import Tracklet
from ..utils import hungarian
from .motion_model import ApproachEstimate

DEFAULT_BEARING_WEIGHT = 0.5
DEFAULT_RATIO_WEIGHT = 0.5
DEFAULT_MOTION_GATE_STRENGTH = 0.3
DEFAULT_MAX_MATCH_COST = 0.6
DEFAULT_PROCESS_NOISE = 1e-2
DEFAULT_MEASUREMENT_NOISE = 5e-2


@dataclass
class CandidateConfidence:
    label: str
    fused_confidence: float  # Kalman-filtered, [0, 1]-ish (not hard-clamped -- see class docstring)
    matched_track_id: Optional[int]
    raw_confidence: Optional[float]  # this frame's pre-smoothing measurement, None if unmatched


class IdentityFusion:
    """One instance per (parent branch) bifurcation being evaluated --
    `fusion.association` creates a fresh one each time `current_location`
    changes, since a new bifurcation means a new, unrelated set of
    candidate labels (carrying over confidence for a label that happens
    to share a name across two different bifurcations, which can't
    actually happen in a well-formed tree, would be a bug anyway)."""

    def __init__(
        self,
        bearing_weight: float = DEFAULT_BEARING_WEIGHT,
        ratio_weight: float = DEFAULT_RATIO_WEIGHT,
        motion_gate_strength: float = DEFAULT_MOTION_GATE_STRENGTH,
        max_match_cost: float = DEFAULT_MAX_MATCH_COST,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
    ):
        total = bearing_weight + ratio_weight
        self.bearing_weight = bearing_weight / total if total > 0 else 0.5
        self.ratio_weight = ratio_weight / total if total > 0 else 0.5
        self.motion_gate_strength = motion_gate_strength
        self.max_match_cost = max_match_cost
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self._filters: Dict[str, ScalarKalmanFilter] = {}

    def update(
        self,
        candidate_labels: List[str],
        detections: List[Tracklet],
        bearing_cost: Optional[np.ndarray],
        ratio_cost: Optional[np.ndarray],
        approach_estimates: Dict[int, ApproachEstimate],
    ) -> Dict[str, CandidateConfidence]:
        blended = self._blend(candidate_labels, detections, bearing_cost, ratio_cost)

        raw_by_label: Dict[str, float] = {}
        matched_track_by_label: Dict[str, int] = {}
        if blended is not None and detections:
            row_idx, col_idx = hungarian(blended)
            for r, c in zip(row_idx, col_idx):
                if blended[r, c] > self.max_match_cost:
                    continue
                label = candidate_labels[r]
                det = detections[c]
                raw = 1.0 - float(blended[r, c])
                raw = self._apply_motion_gate(raw, approach_estimates.get(det.track_id))
                raw_by_label[label] = raw
                matched_track_by_label[label] = det.track_id

        results: Dict[str, CandidateConfidence] = {}
        for label in candidate_labels:
            fused = self._smoothed(label, raw_by_label.get(label))
            results[label] = CandidateConfidence(
                label=label,
                fused_confidence=fused,
                matched_track_id=matched_track_by_label.get(label),
                raw_confidence=raw_by_label.get(label),
            )
        return results

    def _blend(
        self,
        candidate_labels: List[str],
        detections: List[Tracklet],
        bearing_cost: Optional[np.ndarray],
        ratio_cost: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if bearing_cost is None and ratio_cost is None:
            return None
        shape = (len(candidate_labels), len(detections))
        if bearing_cost is None:
            return ratio_cost
        if ratio_cost is None:
            return bearing_cost
        assert bearing_cost.shape == shape and ratio_cost.shape == shape
        return self.bearing_weight * bearing_cost + self.ratio_weight * ratio_cost

    def _apply_motion_gate(self, raw: float, estimate: Optional[ApproachEstimate]) -> float:
        if estimate is None:
            return raw
        gate = self.motion_gate_strength
        if estimate.approaching:
            return raw * (1.0 - gate) + 1.0 * gate
        return raw * (1.0 - gate)

    def _smoothed(self, label: str, raw: Optional[float]) -> float:
        kf = self._filters.get(label)
        if kf is None:
            init = raw if raw is not None else 0.0
            kf = ScalarKalmanFilter(init, process_noise=self.process_noise, measurement_noise=self.measurement_noise)
            self._filters[label] = kf
            return kf.value
        kf.predict()
        if raw is not None:
            kf.update(raw)
        return kf.value

    def confidence(self, label: str) -> Optional[float]:
        kf = self._filters.get(label)
        return kf.value if kf is not None else None
