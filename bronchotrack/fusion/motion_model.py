"""Diameter-growth motion model -- scale-free time-to-contact.

Not a paper mechanism (the paper has no motion-model-driven ETA at all --
`bronchotrack.motion_model.TreeMotionFilter`, the main package's OTHER
motion model, is a graph-constrained Bayes filter over *which branch* the
scope is in, not a time estimate; do not confuse the two). This module
answers a different question: given how fast the lumen ahead of the scope
is visually growing, how many frames until the scope reaches it?

The core idea (a "looming"/time-to-contact estimator, the same principle
behind e.g. Lee's 1976 tau function for visual approach): for an object of
true size S approaching a camera at a roughly constant closing speed v,
its image size s(t) is proportional to S / depth(t), and depth(t) shrinks
roughly linearly as the object is approached. Differentiating,

    tau(t) = s(t) / s'(t)

is, to that same approximation, the remaining TIME to contact (in
whatever units t is measured in -- here, frames) -- and critically, it
needs no absolute size or camera calibration at all: S and any unknown
px-per-mm scale factor both cancel out of the ratio. This is exactly the
"scale-free time-to-contact" the diameter-growth motion model was asked
for: track a tracklet's own apparent diameter over time, and its current
value divided by its own current rate of change already IS the frames-
until-contact estimate, with no absolute-distance calibration step
anywhere in the computation.

"Apparent diameter" here is `utils.mask_equivalent_diameter` (area-based,
uses the whole detected mask) when a segmentation mask is available,
falling back to the box's (w+h)/2 otherwise -- same fallback convention
`paper_exact.association._detection_diameter` uses, just the area-
equivalent measure rather than the longest-diameter one (there's no
"widest extent at a glance" reasoning needed here the way there was for
that cue; a smooth, whole-mask size estimate is preferable for a
derivative-based rate calculation, since it's less sensitive to a single
noisy boundary point than a longest-diameter Feret measurement would be).

Per-tracklet Kalman smoothing
------------------------------
A single frame-to-frame diameter delta is noisy (detector jitter, partial
occlusion, mask boundary noise), and dividing by a noisy `s'(t)` is
exactly the kind of computation small-denominator noise wrecks. Each
tracklet gets its own `ScalarKalmanFilter` (constant-velocity state
[diameter, d(diameter)/dt] -- the same 2-state recursion
`paper_exact.association` already uses for its own scalar smoothing) so
`tau` is computed from the FILTERED value and rate, not the raw
frame-to-frame diameter difference.

Sanity-checking against the airway graph (NOT calibration)
-------------------------------------------------------------
The user explicitly asked for the known branch length to be used only to
sanity-check the motion model's predictions, not to calibrate them --
i.e. `tau` itself must stay purely image-derived (scale-free), and the
graph is only consulted after the fact to ask "does this look physically
reasonable". `TransitSanityChecker` (below) does this by building its own
empirical mm-per-frame closing-speed distribution from this run's own
completed branch transits (arc_length_mm / frames_spent_in_branch, logged
each time the association module records a committed transition) and
flagging a live `tau` prediction implausible only if the closing speed it
implies for the candidate branch's own known length falls far outside
that empirically observed range. No branch length is ever added to,
subtracted from, or used to rescale `tau` itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..scalar_kalman import ScalarKalmanFilter
from ..types import Tracklet
from ..utils import mask_equivalent_diameter

DEFAULT_PROCESS_NOISE = 5e-3
DEFAULT_MEASUREMENT_NOISE = 4.0  # apparent diameter is in pixels; boxes/masks are noisier than a ratio
MIN_GROWTH_RATE = 1e-3  # px/frame; below this, "approaching" is not distinguishable from noise


@dataclass
class ApproachEstimate:
    """One tracklet's current diameter-growth state."""

    track_id: int
    diameter: float  # Kalman-filtered apparent diameter, px
    growth_rate: float  # Kalman-filtered d(diameter)/dt, px/frame
    approaching: bool  # growth_rate > MIN_GROWTH_RATE
    tau_frames: Optional[float]  # None if not (measurably) approaching


def _detection_diameter(t: Tracklet) -> Optional[float]:
    if t.mask is not None:
        d = mask_equivalent_diameter(t.mask)
        if d is not None:
            return d
    box = t.last_box
    return (box.w + box.h) / 2.0


class ApproachMotionModel:
    """Owns one `ScalarKalmanFilter` per currently-tracked tracklet, fed
    that tracklet's own apparent diameter once per frame it's actually
    updated (mirroring `paper_exact.association._smoothed_ratio`'s
    once-per-frame-per-track_id discipline)."""

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._filters: Dict[int, ScalarKalmanFilter] = {}
        self._updated_frame: Dict[int, int] = {}

    def update(self, tracklets: List[Tracklet], frame_idx: int) -> Dict[int, ApproachEstimate]:
        """Advance every live tracklet's filter by one frame and return
        the resulting `ApproachEstimate` for each currently-visible
        (`time_since_update == 0`) tracklet. Tracklets not seen this frame
        keep their filter (predict-only, via `prune`'s survivors) but are
        not included in the returned dict -- there's no fresh diameter
        measurement to report an estimate from."""
        estimates: Dict[int, ApproachEstimate] = {}
        for t in tracklets:
            if t.time_since_update != 0:
                continue
            diameter = _detection_diameter(t)
            if diameter is None:
                continue
            estimates[t.track_id] = self._observe(t.track_id, diameter, frame_idx)
        return estimates

    def _observe(self, track_id: int, diameter: float, frame_idx: int) -> ApproachEstimate:
        kf = self._filters.get(track_id)
        if kf is None:
            kf = ScalarKalmanFilter(
                diameter,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
            self._filters[track_id] = kf
            self._updated_frame[track_id] = frame_idx
        elif self._updated_frame.get(track_id) != frame_idx:
            kf.predict()
            kf.update(diameter)
            self._updated_frame[track_id] = frame_idx
        # else: already observed this exact frame_idx (shouldn't normally
        # happen -- each tracklet appears once per frame -- but keep the
        # method idempotent within a frame rather than double-updating).

        value, rate = kf.value, kf.rate
        approaching = rate > MIN_GROWTH_RATE
        tau = (value / rate) if approaching and value > 0 else None
        return ApproachEstimate(
            track_id=track_id, diameter=value, growth_rate=rate, approaching=approaching, tau_frames=tau
        )

    def tau_frames_for(self, track_id: int) -> Optional[float]:
        """The current scale-free time-to-contact estimate (frames) for
        `track_id`'s own filter, or None if there's no filter for it yet
        or it isn't measurably approaching. Public accessor so callers
        (`fusion.association`'s sanity-check wiring, `fusion.pipeline`'s
        `tau_seconds` conversion) don't need to reach into `_filters`
        directly."""
        kf = self._filters.get(track_id)
        if kf is None:
            return None
        value, rate = kf.value, kf.rate
        return (value / rate) if rate > MIN_GROWTH_RATE and value > 0 else None

    def prune(self, live_track_ids: "set") -> None:
        """Drop filter state for tracklets the tracker has fully dropped
        (see `paper_exact.association._prune_ratio_filters` -- same
        reasoning: don't grow unboundedly over a long video)."""
        for track_id in list(self._filters.keys()):
            if track_id not in live_track_ids:
                del self._filters[track_id]
                self._updated_frame.pop(track_id, None)


@dataclass
class TransitRecord:
    label: str
    frames_spent: int
    arc_length_mm: float

    @property
    def speed_mm_per_frame(self) -> float:
        return self.arc_length_mm / max(self.frames_spent, 1)


class TransitSanityChecker:
    """Empirical, self-referential sanity check for `tau_frames`, built
    entirely from this run's own history -- see module docstring's
    "Sanity-checking against the airway graph" section for why this
    doesn't calibrate `tau` itself.

    Call `record_transit` every time the association module commits a
    branch-to-branch transition (with how many frames were spent in the
    branch just left, and that branch's own `arc_length_mm()`); call
    `check` with a live `tau_frames` prediction and the candidate branch's
    own `arc_length_mm()` to get a plausibility verdict against the
    closing-speed distribution observed so far this run.
    """

    def __init__(self, plausible_range: tuple = (0.2, 5.0)):
        self.plausible_range = plausible_range
        self.transits: List[TransitRecord] = []

    def record_transit(self, label: str, frames_spent: int, arc_length_mm: float) -> None:
        if frames_spent <= 0 or arc_length_mm <= 0:
            return
        self.transits.append(TransitRecord(label, frames_spent, arc_length_mm))

    @property
    def median_speed_mm_per_frame(self) -> Optional[float]:
        if not self.transits:
            return None
        return float(np.median([r.speed_mm_per_frame for r in self.transits]))

    def check(self, tau_frames: Optional[float], candidate_arc_length_mm: Optional[float]) -> str:
        """Returns "plausible", "implausible", or "unknown" (not enough
        information yet -- either no completed transits to compare
        against, no tau prediction, or no graph length for the
        candidate)."""
        if tau_frames is None or tau_frames <= 0 or candidate_arc_length_mm is None or candidate_arc_length_mm <= 0:
            return "unknown"
        median_speed = self.median_speed_mm_per_frame
        if median_speed is None or median_speed <= 0:
            return "unknown"
        implied_speed = candidate_arc_length_mm / tau_frames
        lo, hi = self.plausible_range
        ratio = implied_speed / median_speed
        return "plausible" if lo <= ratio <= hi else "implausible"
