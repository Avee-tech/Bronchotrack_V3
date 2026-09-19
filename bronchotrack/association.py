"""Airway association module (paper section 4).

This is the module that turns "a bunch of tracked lumen bounding boxes" into
"which anatomical branch is each one, and therefore where is the scope" --
the paper's training-free, patient-generic part of the system (everything
upstream, detection + tracking, has no notion of *this specific patient's*
anatomy; this module is where the pre-op airway graph M gets used).

Faithfully implemented from the paper:
  * gallery G = {label: tracklets seen there}
  * carina initialization (exactly the root's children visible -> label
    them left/right main bronchus)
  * label propagation outward from already-labeled "anchor" tracklets to
    their still-unlabeled neighbors in the current frame, using the airway
    graph structure (children / siblings) plus an intersection-angle filter
    for unlikely-to-be-visible children
  * Hungarian matching between graph-predicted candidate positions and
    detected lumen positions

One deliberate simplification from the literal paper description, flagged
here rather than hidden: the paper is not fully explicit about the metric
used to match 2D-projected graph points to image detections. Because the
pipeline has no absolute depth/scale (paper: depth is inferred only through
*which* branches are visible, not metrically), this implementation matches
on **angular position** (bearing from the anchor) rather than absolute 2D
distance -- i.e. "is this candidate child/sibling roughly in the same
direction from the anchor as this graph node predicts", which is scale-free
and matches the paper's own framing of depth as topological rather than
metric. See README "Fidelity notes".

No roll correction: the paper's roll-angle estimation (rotating the
graph's projected 2D positions to match the bronchoscope's current
rotation about its own axis before this bearing comparison) is
deliberately NOT implemented here -- candidate matching assumes the
camera's roll relative to the graph's own coordinate frame is fixed at 0,
i.e. graph-space bearings are compared to image-space bearings directly,
unrotated. This is a simplification, not a paper-faithful reading: bearing
matching will degrade if the scope's actual rotation drifts far from the
graph's default orientation. If that turns out to matter for your data,
`flip_v` (a static, one-time mirror -- not per-frame roll tracking) is
still available for a fixed axis-orientation mismatch between the graph
and the camera.

Diameter-ratio matching
------------------------
On top of bearing, candidates are also scored on a **scale-invariant size
cue** -- but, per real-video testing (see README "Fidelity notes"), this is
now a *peer-comparison*, not a distance-from-anchor comparison. The
original anchor-distance formulation (comparing each candidate's
distance-from-anchor-over-diameter ratio between image and graph space)
was measured to give no benefit over angle-only matching at default
weight, and to actively hurt tree traversal when weighted fully -- most
likely because a YOLO bounding box is a noisy proxy for true lumen
diameter, and that noise gets baked directly into a *distance* term that
also has to be estimated from the same noisy boxes. Comparing lumens only
to their own simultaneously-visible peers removes the distance term
entirely and needs just the relative sizes to be roughly right, not their
absolute positions:

    ratio_image(detection) = image_diameter(detection) / mean(image_diameter(peer detections))
    ratio_graph(candidate) = graph_diameter(candidate)  / mean(graph_diameter(peer candidates))

where `image_diameter` is a detection's own bounding-box size (average of
width/height), `graph_diameter` is a candidate branch's real diameter
(see below), and "peers" means the *other* detections/candidates present
in this same match call -- e.g. if two child lumens are both visible at a
bifurcation, each one's size is judged relative to the other, not to an
absolute distance from wherever the anchor happens to be. This only means
something when there is an actual peer to compare against, so the cue is
skipped entirely (falls back to angle-only) unless **at least two**
candidate labels have known graph diameters **and at least two** detections
are visible in the current frame; with only one lumen in view there is
nothing to normalize against.

`graph_diameter` is taken as the **length-weighted mean radius over the
first `diameter_lookahead_fraction` of the branch's own arc length**
(default: first 15%, via `AirwayNode.mean_radius`) rather than a single
point sample at exactly `radius_at_start` -- a small smoothing correction
so the cue isn't overly sensitive to exactly where the centerline
extraction happened to place its first sample, and to the fact that a
candidate lumen is usually seen a short distance past its own birth point,
not at the literal bifurcation plane. Falls back to `radius_at_start` on
graphs with only start/end radii (no full centerline), and is skipped
automatically (falls back to angle-only) for graphs without radius data at
all.

Both ratios are log-transformed and compared as
|log(ratio_image) - log(ratio_graph)| for every (candidate label,
detection) pair, capped at `_LOG_RATIO_CAP`, then combined with the
angular/bearing cost above into a single matching cost (weighted by
`diameter_weight`).

Approach-trend diagnostic (NOT a depth/position estimate)
-----------------------------------------------------------
`self.approach_trend` reports the normalized slope of the current anchor's
own (Kalman-smoothed) image-space diameter over its last few frames -- a
cheap, purely diagnostic signal for "is the tracked lumen growing (camera
advancing toward it) or shrinking/stable right now". This is deliberately
**not** used anywhere in the matching cost above, and is not a substitute
for the "virtual model predicts where a lumen will appear, and how big it
will be" idea it's tempting to reach for here: turning monocular image size
into real distance-into-branch requires knowing the camera-to-tissue depth,
and image size is dominated by that unknown depth, not by the airway's own
(usually mild) taper along a branch -- so this trend cannot be reliably
inverted into "how far along the branch is the scope" the way a true
render-and-compare or learned-depth approach (e.g. a CycleGAN-based depth
estimator, as used in some bronchoscopy-navigation literature) can. It's
exposed alongside the location output purely as a soft, human-readable
diagnostic ("does it look like we're approaching a bifurcation"), not fed
back into any matching decision -- see README "Not implemented" for what a
real depth/pose-aware version of this would require.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from .graph import AirwayGraph
from .types import BBox, GalleryEntry, Tracklet
from .utils import contains, hungarian

DEFAULT_ANGLE_THRESHOLD_DEG = 75.0
DEFAULT_MAX_MATCH_COST = 0.6
DEFAULT_DIAMETER_WEIGHT = 0.5  # 0 = angle-only (previous behavior), 1 = size-ratio-only
DEFAULT_DIAMETER_LOOKAHEAD_FRACTION = 0.15  # smooth over the first 15% of a candidate branch
_LOG_RATIO_CAP = 2.0  # ~7.4x size-ratio mismatch -> fully penalized
_MIN_RATIO = 1e-3
_APPROACH_TREND_WINDOW = 8  # frames of anchor box history used for the diagnostic trend


class AirwayAssociation:
    def __init__(
        self,
        graph: AirwayGraph,
        max_generation_gap: int = 3,
        angle_threshold_deg: float = DEFAULT_ANGLE_THRESHOLD_DEG,
        max_match_cost: float = DEFAULT_MAX_MATCH_COST,
        diameter_weight: float = DEFAULT_DIAMETER_WEIGHT,
        diameter_lookahead_fraction: float = DEFAULT_DIAMETER_LOOKAHEAD_FRACTION,
        containment_tol: float = 2.0,
        flip_v: bool = False,
        max_angular_cost: Optional[float] = None,
    ):
        """
        diameter_weight : how much the scale-invariant lumen-diameter-ratio
            cue (see module docstring) contributes to the matching cost,
            relative to angular bearing. 0.5 weighs them equally; 0 disables
            it (angle-only, the previous default); 1 uses size-ratio only.
            Automatically has no effect on graphs without radius data (all
            candidates then fall back to angle-only, regardless of this
            setting).
        diameter_lookahead_fraction : fraction of a candidate branch's own
            arc length (from its start) to average over when computing its
            expected diameter for the ratio cue, instead of a single point
            sample at the very start (default 0.15 = first 15%). 0 recovers
            the old exact-`radius_at_start` behavior. Only has an effect on
            graphs with a full `centerline` (falls back to `radius_at_start`
            otherwise).
        max_angular_cost : deprecated alias for `max_match_cost` (kept so
            existing configs/scripts using the old name don't silently
            break); if both are given, `max_match_cost` wins.
        """
        self.graph = graph
        self.max_generation_gap = max_generation_gap
        self.angle_threshold = np.radians(angle_threshold_deg)
        self.max_match_cost = max_angular_cost if max_angular_cost is not None else max_match_cost
        self.diameter_weight = diameter_weight
        self.diameter_lookahead_fraction = diameter_lookahead_fraction
        self.containment_tol = containment_tol
        self.flip_v = flip_v

        self.gallery: Dict[str, GalleryEntry] = {}
        self.current_location: str = graph.root()
        self.approach_trend: Optional[float] = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # public entry point, called once per frame by pipeline.py
    # ------------------------------------------------------------------
    def process_frame(self, tracklets: List[Tracklet], frame_idx: int) -> Dict[int, str]:
        current = [t for t in tracklets if t.time_since_update == 0]

        if not self._initialized:
            self._try_initialize(current, frame_idx)

        if not self._initialized:
            return {}

        anchors = [t for t in current if t.label is not None]
        anchors.sort(key=lambda t: -t.age_frames)  # oldest / most established first

        used_labels: Set[str] = {t.label for t in anchors}
        unlabeled = [t for t in current if t.label is None]

        for anchor in anchors:
            unlabeled = self._propagate_from_anchor(
                anchor, unlabeled, used_labels, frame_idx
            )

        self._update_gallery(current, frame_idx)

        return {t.track_id: t.label for t in current if t.label is not None}

    def eligibility_fn(self, tracklet: Tracklet) -> bool:
        """Hook passed to MultiLumenTracker.update(): 'filter tracklets if
        airway labels exceed 3 generations from current location' (paper)."""
        if tracklet.label is None or tracklet.label not in self.graph:
            return True
        gen_t = self.graph.generation(tracklet.label)
        gen_cur = self.graph.generation(self.current_location)
        return abs(gen_t - gen_cur) <= self.max_generation_gap

    # ------------------------------------------------------------------
    # carina initialization
    # ------------------------------------------------------------------
    def _try_initialize(self, current: List[Tracklet], frame_idx: int) -> None:
        root = self.graph.root()
        root_children = self.graph.children(root)
        if len(root_children) == 0:
            # degenerate single-branch graph; just start labeled at root
            self._initialized = True
            self.current_location = root
            return

        if len(current) != len(root_children):
            return  # wait for exactly the expected number of lumens to appear

        proj = self.graph.project_children_2d(root)
        dets_sorted = sorted(current, key=lambda t: t.last_box.x_c)
        labels_sorted = sorted(root_children, key=lambda l: proj[l][0])

        for t, label in zip(dets_sorted, labels_sorted):
            t.label = label
            t.label_history.append((frame_idx, label))

        self.current_location = root
        self._initialized = True

    # ------------------------------------------------------------------
    # label propagation
    # ------------------------------------------------------------------
    def _propagate_from_anchor(
        self,
        anchor: Tracklet,
        unlabeled: List[Tracklet],
        used_labels: Set[str],
        frame_idx: int,
    ) -> List[Tracklet]:
        if not unlabeled:
            return unlabeled

        remaining = list(unlabeled)

        # --- children: detections nested inside the anchor's box ---
        child_idx = [
            i
            for i, t in enumerate(remaining)
            if contains(anchor.last_box, t.last_box, tol=self.containment_tol)
        ]
        if child_idx:
            matched_ids = self._match_candidates(
                anchor=anchor,
                candidate_labels=self._filtered_children(anchor.label),
                detections=[remaining[i] for i in child_idx],
                used_labels=used_labels,
                frame_idx=frame_idx,
            )
            remaining = [
                t
                for t in remaining
                if t.track_id not in matched_ids
            ]

        # --- siblings: remaining nearby detections at the same level ---
        parent_label = self.graph.parent(anchor.label)
        if parent_label is not None:
            sibling_idx = [
                i
                for i, t in enumerate(remaining)
                if not contains(anchor.last_box, t.last_box, tol=self.containment_tol)
                and not contains(t.last_box, anchor.last_box, tol=self.containment_tol)
            ]
            if sibling_idx:
                sibling_labels = [
                    l for l in self.graph.children(parent_label) if l != anchor.label
                ]
                matched_ids = self._match_candidates(
                    anchor=anchor,
                    candidate_labels=sibling_labels,
                    detections=[remaining[i] for i in sibling_idx],
                    used_labels=used_labels,
                    frame_idx=frame_idx,
                    reference_label=parent_label,  # project relative to parent, not anchor
                )
                remaining = [t for t in remaining if t.track_id not in matched_ids]

        return remaining

    def _filtered_children(self, parent_label: str) -> List[str]:
        """Candidate children of `parent_label`, dropping branches whose
        take-off angle is too steep to plausibly be visible (paper:
        'filter candidates based on intersection angle')."""
        children = self.graph.children(parent_label)
        return [
            c
            for c in children
            if self.graph.intersection_angle(parent_label, c) <= self.angle_threshold
        ]

    def _match_candidates(
        self,
        anchor: Tracklet,
        candidate_labels: List[str],
        detections: List[Tracklet],
        used_labels: Set[str],
        frame_idx: int,
        reference_label: Optional[str] = None,
    ) -> Set[int]:
        """Hungarian-match `candidate_labels` (graph nodes) against
        `detections` (unlabeled tracklets), combining two scale-invariant
        cues: angular position around a reference point (graph-space
        bearing compared directly to image-space bearing, assuming zero
        roll -- see module docstring), and the lumen-diameter distance
        ratio (see module docstring). Returns the set of matched
        track_ids."""
        candidate_labels = [l for l in candidate_labels if l not in used_labels]
        if not candidate_labels or not detections:
            return set()

        ref_label = reference_label or anchor.label
        proj = self.graph.project_children_2d(ref_label)
        proj = {l: proj[l] for l in candidate_labels if l in proj}
        if not proj:
            return set()

        labels = list(proj.keys())
        graph_pts = np.stack([proj[l] for l in labels], axis=0)
        if self.flip_v:
            graph_pts = graph_pts * np.array([1.0, -1.0])

        ref_box = anchor.last_box
        ref_center = np.array([ref_box.x_c, ref_box.y_c])
        image_offsets = np.stack(
            [
                np.array([d.last_box.x_c, d.last_box.y_c]) - ref_center
                for d in detections
            ],
            axis=0,
        )

        cost = self._angular_cost_matrix(graph_pts, image_offsets)

        if self.diameter_weight > 0:
            diam_cost = self._diameter_ratio_cost_matrix(labels, detections)
            if diam_cost is not None:
                cost = (1.0 - self.diameter_weight) * cost + self.diameter_weight * diam_cost

        row_idx, col_idx = hungarian(cost)

        matched_ids: Set[int] = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] > self.max_match_cost:
                continue
            label = labels[r]
            det = detections[c]
            det.label = label
            det.label_history.append((frame_idx, label))
            used_labels.add(label)
            matched_ids.add(det.track_id)
        return matched_ids

    @staticmethod
    def _angular_cost_matrix(graph_pts: np.ndarray, image_offsets: np.ndarray) -> np.ndarray:
        def unit(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n = np.where(n < 1e-9, 1.0, n)
            return v / n

        g = unit(graph_pts)
        d = unit(image_offsets)
        return 1.0 - g @ d.T  # (n_labels, n_detections), 0 = perfectly aligned bearing

    def _diameter_ratio_cost_matrix(
        self,
        labels: List[str],
        detections: List[Tracklet],
    ) -> Optional[np.ndarray]:
        """Scale-invariant size-ratio cost (see module docstring) -- **no
        anchor involved**. Each candidate label's diameter is compared to
        the mean diameter of its own peer group of candidate labels (the
        other branches being considered for this same match, in graph
        space); each detection's diameter is independently compared to the
        mean diameter of its own peer group of currently-visible detections
        (in image space). The two normalized (log) ratios are then compared
        the same way as before, per (label, detection) pair.

        This only makes sense as a *relative* size check, so it requires an
        actual peer to compare against: at least two candidate labels with
        known graph diameters, AND at least two currently-visible
        detections. With only one lumen visible there is nothing to compare
        its size to, so this returns None (falls back to angle-only) rather
        than inventing a single-item "ratio".
        """
        graph_diameters = np.array(
            [2.0 * d if d is not None else np.nan for d in (self._candidate_diameter(l) for l in labels)]
        )
        valid = ~np.isnan(graph_diameters)
        if valid.sum() < 2 or len(detections) < 2:
            return None

        mean_graph_diameter = np.nanmean(graph_diameters[valid])
        ratio_graph = graph_diameters / max(mean_graph_diameter, _MIN_RATIO)  # (n_labels,)

        image_diameters = np.array(
            [(d.last_box.w + d.last_box.h) / 2.0 for d in detections]
        )
        mean_image_diameter = max(np.mean(image_diameters), _MIN_RATIO)
        ratio_image = image_diameters / mean_image_diameter  # (n_dets,)

        log_ratio_graph = np.log(np.maximum(ratio_graph, _MIN_RATIO))
        log_ratio_image = np.log(np.maximum(ratio_image, _MIN_RATIO))

        # (n_labels, n_dets)
        diff = np.abs(log_ratio_graph[:, None] - log_ratio_image[None, :])
        cost = np.minimum(diff, _LOG_RATIO_CAP) / _LOG_RATIO_CAP  # normalize to [0, 1]

        # candidates with no radius data get a neutral cost (angle carries the match)
        nan_rows = ~valid
        cost[nan_rows, :] = 0.5
        return cost

    def _candidate_diameter(self, label: str) -> Optional[float]:
        """Expected diameter for `label`'s branch, smoothed over the first
        `diameter_lookahead_fraction` of its own arc length (see module
        docstring). Falls back to the plain `radius_at_start` point sample
        when `diameter_lookahead_fraction` is 0 or the graph has no full
        centerline to smooth over."""
        node = self.graph.get(label)
        if self.diameter_lookahead_fraction <= 0:
            return node.radius_at_start
        smoothed = node.mean_radius(0.0, self.diameter_lookahead_fraction)
        return smoothed if smoothed is not None else node.radius_at_start

    @staticmethod
    def _diameter_trend(tracklet: Tracklet, window: int = _APPROACH_TREND_WINDOW) -> Optional[float]:
        """Diagnostic-only: normalized slope of `tracklet`'s own image-space
        diameter over its last `window` (Kalman-smoothed) boxes -- positive
        means the tracked lumen has been growing (camera advancing toward
        it), negative means shrinking. See module docstring's "Approach-
        trend diagnostic" section for why this is NOT a distance/position
        estimate. Returns None with fewer than 3 boxes of history."""
        boxes = tracklet.boxes[-window:]
        if len(boxes) < 3:
            return None
        diam = np.array([(b.w + b.h) / 2.0 for b in boxes])
        mean_diam = diam.mean()
        if mean_diam < 1e-6:
            return None
        idx = np.arange(len(diam))
        slope = np.polyfit(idx, diam, 1)[0]
        return float(slope / mean_diam)

    # ------------------------------------------------------------------
    # gallery bookkeeping
    # ------------------------------------------------------------------
    def _update_gallery(self, current: List[Tracklet], frame_idx: int) -> None:
        best_label, best_gen, best_tracklet = None, -1, None
        for t in current:
            if t.label is None:
                continue
            entry = self.gallery.setdefault(
                t.label, GalleryEntry(label=t.label, first_seen_frame=frame_idx)
            )
            if t.track_id not in entry.track_ids:
                entry.track_ids.append(t.track_id)
            entry.last_seen_frame = frame_idx

            gen = self.graph.generation(t.label) if t.label in self.graph else -1
            if gen > best_gen:
                best_gen = gen
                best_label = t.label
                best_tracklet = t

        if best_label is not None:
            self.current_location = best_label
            self.approach_trend = self._diameter_trend(best_tracklet)
