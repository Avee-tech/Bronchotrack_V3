"""Airway association module -- STRICT paper reference version (paper section 4).

This is a fork of ``bronchotrack.association`` with every post-paper
addition removed, and one paper-described mechanism that the main package
never implemented -- roll-angle correction -- added in. See
``bronchotrack.paper_exact`` (this package's ``__init__.py``) for the full
list of deltas against the main package.

Faithfully implemented from the paper:
  * gallery G = {label: tracklets seen there}
  * carina initialization (exactly the root's children visible -> label
    them left/right main bronchus)
  * label propagation outward from already-labeled "anchor" tracklets to
    their still-unlabeled neighbors in the current frame, using the airway
    graph structure (children / siblings) plus an intersection-angle filter
    for unlikely-to-be-visible children
  * roll-angle estimation (Eq. 6/7) rotating the graph's projected 2D
    candidate positions to compensate for the bronchoscope's rotation about
    its own optical axis, before comparing them to image-space detections
  * Hungarian matching between graph-predicted candidate positions and
    detected lumen positions

Deliberate simplifications, flagged rather than hidden (the paper is not
fully explicit about these):

1. Matching metric. The paper does not spell out the exact metric used to
   compare 2D-projected graph points to image detections. Because the
   pipeline has no absolute depth/scale (paper: depth is inferred only
   through *which* branches are visible, not metrically), this matches on
   **angular position** (bearing from the anchor) rather than absolute 2D
   distance -- scale-free, and consistent with the paper's own framing of
   depth as topological rather than metric.

2. Roll-angle sign. The paper's Eq. 7 as written uses arccos of a cosine
   similarity, which is inherently unsigned (range [0, pi]) and so cannot,
   by itself, distinguish a clockwise from a counter-clockwise roll. To
   actually rotate the graph the correct direction, this implementation
   uses the standard signed-angle form (atan2 of the 2D cross and dot
   product) between the reference-pair vector at the last roll-estimate
   frame and its current position -- numerically equal to the paper's
   arccos magnitude, but with the sign a real rotation requires.

   That signed angle is a per-frame MEASUREMENT of roll, not the value
   used directly -- box centers are noisy, so a raw running sum of these
   measurements (an earlier version of this module) jitters frame to
   frame and drags every rotated graph_pts computation along with it. This
   version instead runs the raw measurement through a small constant-
   velocity Kalman filter (`scalar_kalman.ScalarKalmanFilter`, same
   predict/update recursion as the paper's own box-tracking Kalman filter
   in `kalman.py`, just over a 1D [roll, roll-velocity] state) and uses
   the filtered estimate as `self.roll`. Not a paper mechanism -- the
   paper doesn't say how (or whether) to smooth its own roll estimate --
   but a natural extension of a filtering approach it already uses
   elsewhere.

3. Reference pair lifetime. "roll_t = roll_m + angle(...)" compares
   positions "at frame m" (historical) against "at frame t" (current) for
   what the paper calls "two oldest tracklet positions". Read literally,
   this means: pick the two oldest currently-labeled tracklets as the
   reference pair; while that *same* pair stays visible, m is fixed at the
   frame the pair was first adopted (not incremented every frame, which
   would just accumulate per-frame noise). If the set of "two oldest"
   tracklets changes (one leaves the frame, a new anchor becomes older),
   a new reference pair is adopted, its "m" reset to the current frame,
   and the running roll estimate carried forward as that new pair's roll_m
   baseline.

Diameter:distance matching cue (NOT in the paper -- opt-in, on by default)
----------------------------------------------------------------------------
On top of bearing, candidates are also scored on the ratio of each lumen's
diameter to its distance from a reference center:

    ratio_image(detection) = image_diameter(detection) / |image_offset|
    ratio_graph(candidate) = graph_diameter(candidate)  / |graph_point|

`diameter / distance` is, to a small-angle approximation, the lumen's
*angular width* as seen from the reference point -- the geometry (how the
diameter and the two centers it's measured between are found) is described
below; this is comparing the SAME quantity computed two ways: once from the
real detected lumens in the current video frame, once from the pre-op 3D
model, "standing" a bit past the bifurcation the way the scope actually
will be by the time it's looking straight at these children (the "virtual
viewpoint" below) -- an independent geometric sanity check that the label
about to be assigned actually looks like the branch it's being matched to,
not just that its *bearing* lines up.

Diameter: the longest diameter, not the area-equivalent one
-------------------------------------------------------------
`image_diameter` is the detected mask's own **longest diameter** -- the
maximum distance between any two of its polygon's vertices (see
`utils.polygon_longest_diameter`) -- rather than the area-equivalent
circle diameter this cue used previously. A lumen opening viewed at an
angle is routinely non-circular (foreshortened, partly occluded, or simply
oval), and "how wide does this look at a glance" tracks its widest visible
extent more closely than a same-area circle would; falls back to
max(box.w, box.h) when there's no mask (see `_detection_diameter`) -- the
larger side, not the average, for the same "widest extent" reasoning.
`graph_diameter`
is the branch's own known cross-sectional diameter (2x its centerline
radius) at the virtual viewpoint described below -- the model doesn't need
an estimate here, since the pre-op scan already measured it directly.

Center: the average of opposing boundary points, not the plain centroid
---------------------------------------------------------------------------
Both `image_offset` (the vector the distance above is the length of) and
the video's on-screen marker (see `paper_exact.viz`) use a lumen's
`utils.polygon_opposing_center`, not its plain vertex/area centroid or its
box center: for every mask boundary point, pair it with whichever other
boundary point sits most nearly diametrically opposite it (by angle around
a preliminary centroid), and average the midpoints of every such pair. A
lopsided mask (partly occluded, foreshortened, mid-transition between two
branches) drags a plain centroid toward whichever side has more boundary
detail; averaging "opposite wall to opposite wall" instead estimates the
middle of the *open space*, independent of how much of the boundary is on
each side. Falls back to the box center (x_c, y_c) when there's no mask;
see `_detection_center`.

The reference-relative geometry (`image_offset` = one lumen's center minus
another's, `graph_point` = the projected virtual-viewpoint position below)
is otherwise structurally the same computation `_match_candidates` already
does for the angular-bearing cost, just re-derived here from these two new
per-lumen quantities rather than reused wholesale, since the bearing cost
intentionally keeps using plain box centers (it's paper-faithful geometry
this addition shouldn't perturb).

The virtual viewpoint: a couple of centimetres past the bifurcation
-----------------------------------------------------------------------
`graph_point` and `graph_diameter` are NOT evaluated at the candidate
branch's very first point (right at the bifurcation, where a child lumen
is typically foreshortened/oblique and unrepresentative of what it'll
actually look like once you're looking at it head-on) or at its plain
midpoint. Instead, using the current localization estimate (which
bifurcation the scope is at) together with the airway graph's own
centerline, this projects each candidate child's position and samples its
diameter `virtual_advance_mm` (default 20mm = 2cm, matching how far a
scope typically has to advance past a fork before a child's opening reads
clearly) along that child's *own* centerline from its start -- see
`AirwayGraph.project_children_at_distance` /
`child_apparent_diameter_at_distance`. This is, in effect, "running the
same measurement on the 3D model that just got run on the video frame":
since the pre-op model's true geometry at that point is already known
exactly, there's no need to synthesize a picture and re-detect anything in
it -- the model already knows what's there directly. (Clamped to the
branch's own length if `virtual_advance_mm` would run past its own next
bifurcation.)

Critically, `graph_diameter` is NOT that branch's true, straight-on
cross-sectional diameter -- it's foreshortening-corrected. A real detected
mask is never the true cross-section either: the near airway wall
partially occludes any child opening that isn't dead-ahead, even a couple
of centimetres past the fork, so a real, in-frame measurement of an
off-axis child is *expected* to read smaller than its actual anatomical
diameter. Comparing that against the model's idealized, unoccluded true
diameter would show a mismatch (and could fail virtual verification, see
below) even when the label being assigned is exactly correct. Instead,
`child_apparent_diameter_at_distance` projects the branch's true diameter
by the cosine of the angle between the parent's own forward axis at the
bifurcation (`end_tangent` -- the scope hasn't committed to any specific
child yet, so this is the one fixed direction it's actually looking along,
the same axis `project_children_2d`/`project_children_at_distance` already
use) and the child's own local tangent at the virtual point -- 1.0 (no
correction) when the child continues nearly straight ahead of the parent,
and shrinking toward 0 as its opening turns away from that axis and would,
in reality, become increasingly occluded by the near wall. (This is
deliberately NOT a ray recomputed per child from the bifurcation to that
child's own virtual point -- for a roughly straight branch such a ray is
itself nearly parallel to the branch's own tangent by construction, which
would make every branch read as face-on no matter how sharply it actually
forks off the parent, defeating the correction entirely.) The two sides of
this comparison are estimating the same thing (what the opening would
actually look like from here), not "what the video shows" against "what
the branch truly measures".

That angular width is directly comparable *within* a domain, but NOT
directly comparable *across* domains as an absolute number: image space is
in pixels and graph space is in millimetres, related by an unknown,
depth-dependent camera scale factor this pipeline has no way to calibrate.
(A first version of this cue skipped this step and compared raw ratios
directly -- a synthetic unit test with a deliberately large pixel/mm scale
mismatch caught it immediately: the size difference between candidates was
swamped by the constant scale offset between domains, so the "wrong"
candidate won every time regardless of which one actually matched. This is
a distinct bug from what sank the main package's own *original*,
now-abandoned anchor-distance cue -- its docstring attributes that failure
to YOLO box noise feeding a distance term, not a cross-domain scale
mismatch -- but the fix below happens to guard against both.) The fix,
applied here: normalize each domain's ratios by that domain's own mean
first --

    norm_ratio_graph(l) = ratio_graph(l) / mean(ratio_graph over all
                            candidate labels with known radius data)
    norm_ratio_image(d) = ratio_image(d) / mean(ratio_image over all
                            currently-visible detections in this match)

-- which cancels the unknown scale factor exactly (it's the same multiplier
on every candidate/detection within a call, so it divides out of the
mean-normalized ratio), leaving only each item's angular width *relative to
its peers* -- the same trick the main package's peer-comparison cue uses,
but applied to diameter:distance rather than diameter alone, so it also
captures "this candidate is unusually close to the bifurcation for its
size" rather than just "this candidate is unusually big". The two
normalized ratios are then log-transformed and compared as
|log(norm_ratio_image) - log(norm_ratio_graph)|, capped and normalized to
[0, 1], then blended with the angular cost by `distance_diameter_weight`
(default 0.4; 0 disables this cue entirely and recovers pure bearing-only
matching). The cue is automatically skipped (falls back to angle-only)
when no candidate in a given match call has any radius data.

Virtual verification (`Tracklet.diameter_distance_match`)
-------------------------------------------------------------
Beyond feeding the matching cost, each accepted (candidate, detection)
pairing is separately checked as a pass/fail: if
`min(norm_ratio_image, norm_ratio_graph) / max(...) >= virtual_match_threshold`
(default 0.75, i.e. the two normalized angular widths agree to within 25%)
the pairing is considered independently verified by the 3D model, recorded
on the winning detection's `Tracklet.diameter_distance_match` (True/False;
stays None if there was no radius data to check against at all, e.g. an
unlabeled graph or a candidate with no centerline). `virtual_match_threshold`
is a constructor argument (and `--virtual-match-threshold` on the CLI) --
this compares the same peer-normalized ratios already computed for the
matching cost above, not raw ratios, for the same cross-domain-scale reason.

This verdict is now refreshed every frame a tracklet has a currently-visible
sibling to compare against, not decided once and frozen -- see "Continuous
(per-frame) re-verification" below, which subsumes the one-shot version this
section originally described.

Continuous (per-frame) re-verification (NOT in the paper -- opt-in, on by default)
-------------------------------------------------------------------------------------
An earlier version of this module set `diameter_distance_match` exactly
once, at the single frame a tracklet was first labeled (inside
`_match_candidates`, or `_verify_bootstrap_matches` for the carina split),
and never again -- because `_match_candidates` only ever runs on tracklets
that are still *unlabeled*, and a tracklet is removed from that pool for
good the moment it gets a label. Diagnosed as a real, log-confirmed
consequence rather than a hypothetical: on a 1084-frame real-patient run
where re-acquisition (above) raised labeled-frame coverage from 15.3% to
39.9%, the confirmed-dot rate stayed essentially flat regardless. Tracing
every track_id's `diameter_distance_match` value through that run showed it
never changes across a tracklet's lifetime (0 of 11 labeled tracks), and
over half (6 of 11) sat at `None` forever -- never verified even once.
Two distinct causes, both structural:

  * Tier-1 self-reacquisition (`_reacquire`) relabels a tracklet directly
    by IoU against the frozen snapshot, deliberately skipping
    `_match_candidates` entirely ("no Hungarian matching needed, since
    this isn't 'which child is this'") -- and with it, the only place a
    verdict ever got assigned. Since re-acquisition is exactly the
    mechanism that drives most of the coverage gain, most of the newly
    labeled frames were, by construction, unable to ever earn a dot.
  * Even ordinary Hungarian-matched labeling only verifies once, at
    assignment time; a candidate briefly lacking a visible peer to compare
    against at that exact frame stays `None` forever even if a peer
    reappears later, and a `False` from one noisy frame's geometry is
    never revisited even as later frames' (already Kalman-smoothed)
    ratios would clearly pass.

The fix generalizes `_verify_bootstrap_matches`'s mutual-peer technique
(each newly-labeled tracklet verified using the *other* just-labeled
tracklets as its own reference, no single designated anchor needed) from a
one-shot bootstrap-only step into a per-frame pass (`_continuous_verify`,
via `_peer_verify`) that regroups ALL currently-visible labeled tracklets
by shared parent branch every frame and, for every group of two or more,
re-runs the same mutual verification -- overwriting
`diameter_distance_match` with this frame's result rather than leaving a
stale one in place. This applies uniformly regardless of how a tracklet got
its label (bootstrap, ordinary matching, or tier-1/tier-2 re-acquisition),
so a reacquired tracklet that later shares a frame with a visible sibling
now gets verified same as any other.

Groups smaller than two currently-visible tracklets are left untouched, not
reset to `None` -- there is no well-posed way to verify a single, solo
lumen this way: the diameter:distance ratio is inherently a *relative*,
peer-normalized quantity (see "Diameter:distance matching cue" above), and
there is no live parent tracklet in frame to measure a real image-space
distance against once the scope has advanced past it. A verdict earned
while a sibling was briefly visible is therefore left standing once that
sibling leaves frame again, rather than being wiped.

Opt-in via `continuous_verification` (constructor argument / CLI's
`--no-continuous-verification` to disable) -- set False to recover the
original one-shot-only behavior exactly. No effect when
`distance_diameter_weight <= 0` (cue disabled entirely).

Re-acquisition after total anchor loss (NOT in the paper -- opt-in, on by default)
-----------------------------------------------------------------------------------
Label propagation above only ever runs from a currently-visible LABELED
tracklet ("anchor"). Read literally, that means once every anchor is lost
at once (a long occlusion, a burst of missed detections, the tracker
dropping every tracklet outright), nothing in `process_frame` can ever
assign a new label again for the rest of the video: `anchors` stays
permanently empty, `_propagate_from_anchor` never runs, and
`current_location` just freezes at whatever it last was -- a real,
diagnosed dead end in earlier runs of this pipeline (confirmed by log
analysis: one real video showed zero displayed locations for 82% of its
length after its last anchor was lost around frame 200 of 1084).

The fix keeps a frozen snapshot of the most-established real anchor
(`_last_real_anchor`, plus the frame it was taken) updated every frame a
real one exists. The moment a frame has NO real anchors, `_reacquire`
tries two things, in order, both bounded by `reacquire_max_gap_frames`
(default 90 = 3s at 30fps -- past that, the snapshot is treated as too
stale to trust and re-acquisition gives up for this frame, though it
keeps checking every later frame in case a real anchor reappears on its
own):

1. SELF re-acquisition (the dominant real-world case -- the very lumen
   the scope is already inside drops out for a few frames, e.g. motion
   blur or a burst of missed detections, then reappears as itself, not as
   a newly-visible child). Among this frame's still-unlabeled tracklets,
   whichever one has the highest IoU against the snapshot's own last box
   is checked against `reacquire_iou_threshold` (default 0.3); a
   sufficiently overlapping box is relabeled DIRECTLY with the snapshot's
   own label -- no Hungarian matching needed, since this isn't "which
   child is this", it's "is this probably the same lumen coming back".
   That tracklet becomes a real anchor again immediately, in this same
   frame, and (falling through to the normal code path below) can go on
   to have its OWN children/siblings matched immediately too.

2. Virtual-anchor propagation (fallback, if step 1 finds no sufficiently
   overlapping box -- e.g. the scope kept advancing while the anchor was
   lost, so nothing in the current frame still resembles its old
   position): the frozen snapshot itself -- a plain `Tracklet` carrying
   the last known label/box/mask -- is handed to `_propagate_from_anchor`
   exactly as a live tracklet would be (that method only ever reads
   `.label`, `.last_box`, `.mask`, none of which care whether the object
   is still being tracked), so genuinely NEW child/sibling lumens can
   still be matched against where the anchor last was.

Either way, the moment anything gets relabeled it becomes a real anchor
again from the very next frame onward -- this is a per-frame fallback,
not a second one-shot initialization, so it keeps working across
repeated gaps for the rest of the run. Set `reacquire_max_gap_frames=0`
to disable re-acquisition entirely and reproduce the original dead-end
behavior.

The root split (the carina's first L/R -- or however many-way -- labeling,
`_try_initialize`) is the one place labels get assigned WITHOUT going
through this matching step: there's no pre-existing anchor to match
against yet, so labels are handed out directly by sorted position instead
of Hungarian assignment. `_verify_bootstrap_matches` runs the same
verification separately right after, using each newly-labeled tracklet as
the others' reference point in turn -- otherwise these tracklets'
`diameter_distance_match` would sit at its default `None` forever (they
never become "unlabeled candidates" being matched again, which is the
only other place the verdict gets set), and since the overlay video only
draws a dot once it's `True`, the very first, most load-bearing labels of
the whole run would never be displayed at all.

`image_diameter` (a detection's own size, longest-diameter as above) is the
mask-derived value when the detector is a segmentation checkpoint and a
mask came back for this detection (see `detection.py`'s docstring and
`Detection.mask`), falling back to the box's (w+h)/2 otherwise. This also
means `_is_nested` (the children-vs-siblings split feeding into which
candidate set a detection is even matched against) uses point-in-polygon
against the anchor's own mask when available, instead of box-in-box
containment, for the same reason (unaffected by the changes above -- still
uses the detection's plain box center as the point being tested, only the
*container* switched to mask-based).

What segmentation does NOT change here: multi-lumen *tracking* (`kalman.py`
+ `matching.py` + `tracker.py`) stays entirely box-based, on purpose. The
Kalman filter only predicts where a box will be next frame (constant-
velocity over x_c/y_c/h/a) -- there's no equivalent "predicted mask shape"
without a materially bigger redesign (tracking an ellipse or spline
parameterization instead of a box), so comparing a *predicted* box against
a *detected* mask for motion-cost IoU would be comparing two different
kinds of thing. Mask IoU is a reasonable idea for tracking two *observed*
masks against each other, but doesn't fit this architecture's predict/
detect split cleanly enough to be worth the redesign here.

`ratio_image` per detection is itself Kalman-smoothed over time before it
enters the cost above: each tracklet gets its own `ScalarKalmanFilter`
(see `scalar_kalman.py`), predicted and updated at most once per frame
(a detection can be considered against both a "children" and a "siblings"
candidate set in the same frame -- see `_propagate_from_anchor` -- so the
filter is only fed one real measurement per frame, not once per call).
This is on top of, not instead of, the box's own Kalman smoothing in
`kalman.py` (the diameter and distance that go into ratio_image are
already built from smoothed box center/height/aspect state) -- ratio is a
*nonlinear* combination of two already-smoothed quantities, so smoothing
the inputs doesn't automatically smooth their ratio, and this filter
catches what's left. Filters are created lazily per track_id and dropped
once a tracklet is no longer active (see `process_frame`'s pruning step).

This is a genuinely non-paper addition -- the paper's own "observable
branch likelihood" (Eq. 6) is described only qualitatively, as a negative
correlation with intersection angle (see `_filtered_children`), with no
size or distance term at all. It's included here, opt-in, because you
asked for it; set `distance_diameter_weight=0` (or `--distance-diameter-
weight 0` on the CLI) to fall back to the strict paper-only bearing match.

Everything below this docstring is otherwise a direct, unmodified port of
the geometry/matching logic in ``bronchotrack.association``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..graph import AirwayGraph
from ..types import Tracklet
from ..utils import (
    contains,
    hungarian,
    iou,
    polygon_contains_point,
    polygon_longest_diameter,
    polygon_opposing_center,
    rotate_2d,
)
from .scalar_kalman import ScalarKalmanFilter

DEFAULT_ANGLE_THRESHOLD_DEG = 75.0
DEFAULT_MAX_MATCH_COST = 0.6  # paper does not give an exact number; kept as a practical gate
DEFAULT_DISTANCE_DIAMETER_WEIGHT = 0.4  # 0 = bearing-only (strict paper), see docstring
DEFAULT_DIAMETER_LOOKAHEAD_FRACTION = 0.15
DEFAULT_VIRTUAL_ADVANCE_MM = 20.0  # "a couple of centimetres past the bifurcation", see docstring
DEFAULT_VIRTUAL_MATCH_THRESHOLD = 0.75  # min(ratio)/max(ratio) needed to call it a match
DEFAULT_REACQUIRE_MAX_GAP_FRAMES = 90  # ~3s at 30fps; 0 disables re-acquisition entirely
DEFAULT_REACQUIRE_IOU_THRESHOLD = 0.3  # min IoU against the frozen snapshot to call it "the same lumen"
DEFAULT_CONTINUOUS_VERIFICATION = True  # re-verify every frame vs. once at initial labeling; see docstring
_LOG_RATIO_CAP = 2.0  # ~7.4x diameter:distance mismatch -> fully penalized
_MIN_RATIO = 1e-3
_MIN_DIST = 1e-6  # guard against divide-by-near-zero distance


class AirwayAssociation:
    def __init__(
        self,
        graph: AirwayGraph,
        max_generation_gap: int = 3,
        angle_threshold_deg: float = DEFAULT_ANGLE_THRESHOLD_DEG,
        max_match_cost: float = DEFAULT_MAX_MATCH_COST,
        containment_tol: float = 2.0,
        distance_diameter_weight: float = DEFAULT_DISTANCE_DIAMETER_WEIGHT,
        virtual_advance_mm: float = DEFAULT_VIRTUAL_ADVANCE_MM,
        virtual_match_threshold: float = DEFAULT_VIRTUAL_MATCH_THRESHOLD,
        reacquire_max_gap_frames: int = DEFAULT_REACQUIRE_MAX_GAP_FRAMES,
        reacquire_iou_threshold: float = DEFAULT_REACQUIRE_IOU_THRESHOLD,
        continuous_verification: bool = DEFAULT_CONTINUOUS_VERIFICATION,
    ):
        self.graph = graph
        self.max_generation_gap = max_generation_gap
        self.angle_threshold = np.radians(angle_threshold_deg)
        self.max_match_cost = max_match_cost
        self.containment_tol = containment_tol
        self.distance_diameter_weight = distance_diameter_weight
        self.virtual_advance_mm = virtual_advance_mm
        self.virtual_match_threshold = virtual_match_threshold
        self.reacquire_max_gap_frames = reacquire_max_gap_frames
        self.reacquire_iou_threshold = reacquire_iou_threshold
        self.continuous_verification = continuous_verification

        self.gallery: Dict[str, "GalleryEntry"] = {}
        self.current_location: str = graph.root()
        self._initialized = False

        # -- re-acquisition state -- see module docstring's own section
        self._last_real_anchor: Optional[Tracklet] = None
        self._last_real_anchor_frame: Optional[int] = None

        # -- roll-angle state (Eq. 6/7), Kalman-smoothed -- see module docstring
        self._roll_kf = ScalarKalmanFilter(0.0, process_noise=1e-3, measurement_noise=5e-2)
        self._roll_ref_pair: Optional[Tuple[int, int]] = None  # (track_id, track_id)
        self._roll_ref_positions: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (c1^m, c2^m)
        self._roll_ref_roll: float = 0.0  # filtered roll_m for the current reference pair

        # -- per-tracklet diameter:distance ratio smoothing -- see module docstring
        self._ratio_filters: Dict[int, ScalarKalmanFilter] = {}
        self._ratio_filters_updated_frame: Dict[int, int] = {}

    @property
    def roll(self) -> float:
        """The current Kalman-filtered roll estimate, radians."""
        return self._roll_kf.value

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # public entry point, called once per frame by pipeline.py
    # ------------------------------------------------------------------
    def process_frame(self, tracklets: List[Tracklet], frame_idx: int) -> Dict[int, str]:
        current = [t for t in tracklets if t.time_since_update == 0]

        self._prune_ratio_filters(tracklets)

        if not self._initialized:
            self._try_initialize(current, frame_idx)

        if not self._initialized:
            return {}

        self._update_roll(current)

        anchors = [t for t in current if t.label is not None]
        anchors.sort(key=lambda t: -t.age_frames)  # oldest / most established first

        if anchors:
            self._last_real_anchor = Tracklet(
                track_id=-1, ind_start=frame_idx, ind_end=frame_idx,
                boxes=[anchors[0].last_box], label=anchors[0].label, mask=anchors[0].mask,
            )
            self._last_real_anchor_frame = frame_idx
        else:
            anchors = self._reacquire(current, frame_idx)

        used_labels: Set[str] = {t.label for t in anchors}
        unlabeled = [t for t in current if t.label is None]

        for anchor in anchors:
            unlabeled = self._propagate_from_anchor(anchor, unlabeled, used_labels, frame_idx)

        self._continuous_verify(current, frame_idx)

        self._update_gallery(current, frame_idx)

        return {t.track_id: t.label for t in current if t.label is not None}

    def _reacquire(self, current: List[Tracklet], frame_idx: int) -> List[Tracklet]:
        """Called only when NO real anchor is currently visible -- see
        module docstring's "Re-acquisition after total anchor loss"
        section for the two-tier strategy this implements. Returns a list
        of zero or one `Tracklet` for `process_frame` to treat as this
        frame's anchor(s): the tracklet SELF-reacquired via IoU (now
        directly labeled, a real anchor from here on), the frozen snapshot
        itself as a virtual anchor to propagate children/siblings from, or
        `[]` if neither applies (same as the original one-shot-anchor
        behavior: no relabeling happens this frame)."""
        if self.reacquire_max_gap_frames <= 0:
            return []
        if self._last_real_anchor is None or self._last_real_anchor_frame is None:
            return []
        if not current:
            return []
        if frame_idx - self._last_real_anchor_frame > self.reacquire_max_gap_frames:
            return []

        snapshot = self._last_real_anchor
        unlabeled = [t for t in current if t.label is None]
        if unlabeled:
            best = max(unlabeled, key=lambda t: iou(snapshot.last_box, t.last_box))
            if iou(snapshot.last_box, best.last_box) >= self.reacquire_iou_threshold:
                best.label = snapshot.label
                best.label_history.append((frame_idx, best.label))
                return [best]

        return [snapshot]

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

        # Unlike every OTHER label assignment (which goes through
        # _match_candidates and gets a diameter_distance_match verdict as
        # a side effect), this bootstrap hands out labels directly with no
        # Hungarian matching step -- there being no established anchor yet
        # to match against is exactly why this exists. Left alone, these
        # tracklets would keep diameter_distance_match at its default None
        # forever: they never become "unlabeled candidates" again, so
        # _propagate_from_anchor's matching (the only place the verdict
        # gets set) never runs for them again either. Since viz.py only
        # draws a dot once diameter_distance_match is True, that meant the
        # very first, most load-bearing labels in the whole run (the root
        # split every later label ultimately hangs off) could never be
        # displayed at all. Verify them explicitly here instead.
        self._verify_bootstrap_matches(root, dets_sorted, frame_idx)

        self.current_location = root
        self._initialized = True

    def _verify_bootstrap_matches(
        self, ref_label: str, tracklets: List[Tracklet], frame_idx: int
    ) -> None:
        """Runs the same diameter:distance virtual-verification
        `_match_candidates` applies after Hungarian assignment, but for a
        set of tracklets that were just labeled directly (see
        `_try_initialize`) rather than matched against a pre-existing
        anchor -- otherwise the very first, most load-bearing labels of the
        whole run would sit at `diameter_distance_match = None` until
        `_continuous_verify` next gets a chance to run on them (which, for
        the carina split, is immediately -- see `process_frame` -- but this
        keeps the explicit call for clarity at the one call site that has
        no pre-existing anchor to lean on). Thin wrapper around
        `_peer_verify`, the general (now per-frame, not one-shot) version
        of this same mutual-reference technique -- see module docstring's
        "Continuous (per-frame) re-verification" section."""
        self._peer_verify(ref_label, tracklets, frame_idx)

    def _continuous_verify(self, current: List[Tracklet], frame_idx: int) -> None:
        """Re-run diameter:distance peer verification every frame for every
        group of currently-visible labeled tracklets that share a parent
        branch, updating `Tracklet.diameter_distance_match` in place rather
        than leaving it at whatever it was decided the single frame each
        tracklet was first labeled. See module docstring's "Continuous
        (per-frame) re-verification" section for why this exists (in
        short: tier-1 self-reacquisition never verifies at all on its own,
        and a one-shot verdict can't reflect a peer that only becomes
        visible later). No-op if disabled (`continuous_verification=False`)
        or the diameter:distance cue itself is off."""
        if not self.continuous_verification or self.distance_diameter_weight <= 0:
            return

        groups: Dict[str, List[Tracklet]] = defaultdict(list)
        for t in current:
            if t.time_since_update != 0 or t.label is None or t.label not in self.graph:
                continue
            parent = self.graph.parent(t.label)
            if parent is None:
                continue
            groups[parent].append(t)

        for parent_label, group in groups.items():
            if len(group) < 2:
                continue  # no peer to verify against -- see docstring; verdict left standing, not reset
            self._peer_verify(parent_label, group, frame_idx)

    def _peer_verify(
        self, ref_label: str, tracklets: List[Tracklet], frame_idx: int
    ) -> None:
        """Mutually verify a group of currently-visible, same-parent
        labeled tracklets against the 3D model, each using the *other*
        members as its own diameter:distance reference point (mirroring
        the "siblings" case in `_propagate_from_anchor`, just without a
        single designated anchor -- there isn't one when every member of
        the group is already labeled). For the common two-child (L/R)
        case, this simplifies to each verifying the other. Called both
        from `_verify_bootstrap_matches` (the carina split's one-shot
        case) and `_continuous_verify` (every frame thereafter, for any
        parent group -- see module docstring's "Continuous (per-frame)
        re-verification" section). No-op if the diameter:distance cue is
        disabled or fewer than two tracklets were passed."""
        if self.distance_diameter_weight <= 0 or len(tracklets) < 2:
            return
        labels = [t.label for t in tracklets]
        for i, anchor in enumerate(tracklets):
            others = [t for j, t in enumerate(tracklets) if j != i]
            other_labels = [l for j, l in enumerate(labels) if j != i]
            diam_result = self._diameter_distance_cost_matrix(
                other_labels, ref_label, anchor, others, frame_idx
            )
            if diam_result is None:
                continue
            _, norm_ratio_graph, norm_ratio_image, valid = diam_result
            for k, det in enumerate(others):
                if not valid[k]:
                    continue
                a, b = norm_ratio_graph[k], norm_ratio_image[k]
                lo, hi = min(a, b), max(a, b, _MIN_RATIO)
                det.diameter_distance_match = bool((lo / hi) >= self.virtual_match_threshold)

    # ------------------------------------------------------------------
    # roll-angle estimation (Eq. 6/7) -- see docstring points 2 and 3
    # ------------------------------------------------------------------
    def _update_roll(self, current: List[Tracklet]) -> None:
        labeled = [t for t in current if t.label is not None]
        if len(labeled) < 2:
            self._roll_kf.predict()  # no measurement this frame; propagate the prior
            return

        oldest_two = sorted(labeled, key=lambda t: -t.age_frames)[:2]
        pair_ids = tuple(sorted(t.track_id for t in oldest_two))
        by_id = {t.track_id: t for t in oldest_two}
        c1 = np.array([by_id[pair_ids[0]].last_box.x_c, by_id[pair_ids[0]].last_box.y_c])
        c2 = np.array([by_id[pair_ids[1]].last_box.x_c, by_id[pair_ids[1]].last_box.y_c])

        if pair_ids != self._roll_ref_pair:
            # adopt a new reference pair: m := now, roll_m := current filtered estimate
            self._roll_ref_pair = pair_ids
            self._roll_ref_positions = (c1, c2)
            self._roll_ref_roll = self.roll
            self._roll_kf.predict()  # roll_t == roll_m this frame; still no new measurement
            return

        c1_m, c2_m = self._roll_ref_positions
        v_m = c1_m - c2_m
        v_t = c1 - c2
        self._roll_kf.predict()
        if np.linalg.norm(v_m) < 1e-6 or np.linalg.norm(v_t) < 1e-6:
            return  # degenerate (coincident boxes); no usable measurement this frame

        # signed angle from v_m to v_t (see docstring point 2)
        cross = v_m[0] * v_t[1] - v_m[1] * v_t[0]
        dot = float(np.dot(v_m, v_t))
        signed_delta = float(np.arctan2(cross, dot))

        measurement = self._roll_ref_roll + signed_delta
        self._roll_kf.update(measurement)

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

        # --- children: detections nested inside the anchor's opening ---
        child_idx = [i for i, t in enumerate(remaining) if self._is_nested(anchor, t)]
        if child_idx:
            matched_ids = self._match_candidates(
                anchor=anchor,
                candidate_labels=self._filtered_children(anchor.label),
                detections=[remaining[i] for i in child_idx],
                used_labels=used_labels,
                frame_idx=frame_idx,
            )
            remaining = [t for t in remaining if t.track_id not in matched_ids]

        # --- siblings: remaining nearby detections at the same level ---
        parent_label = self.graph.parent(anchor.label)
        if parent_label is not None:
            sibling_idx = [
                i
                for i, t in enumerate(remaining)
                if not self._is_nested(anchor, t) and not self._is_nested(t, anchor)
            ]
            if sibling_idx:
                sibling_labels = [l for l in self.graph.children(parent_label) if l != anchor.label]
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

    def _is_nested(self, outer: Tracklet, inner: Tracklet) -> bool:
        """True if `inner` is a "child" of `outer` for the purposes of
        `_propagate_from_anchor`'s children-vs-siblings split. Uses
        `outer`'s segmentation mask (point-in-polygon on `inner`'s box
        center) when available -- a real lumen opening is rarely
        axis-aligned or convex, so its box can nest an unrelated sibling's
        box (or fail to nest a real child's) in cases its actual mask
        wouldn't. Falls back to box-in-box containment when either
        tracklet has no mask (e.g. a plain, non-segmentation detector)."""
        if outer.mask is not None:
            center = (inner.last_box.x_c, inner.last_box.y_c)
            return polygon_contains_point(outer.mask, center)
        return contains(outer.last_box, inner.last_box, tol=self.containment_tol)

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
        `detections` (unlabeled tracklets) on angular position around a
        reference point: graph-space bearing, rotated by the current roll
        estimate (see `_update_roll`), compared to image-space bearing."""
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
        graph_pts = rotate_2d(graph_pts, self.roll)  # Eq. 6/7 roll correction

        ref_box = anchor.last_box
        ref_center = np.array([ref_box.x_c, ref_box.y_c])
        image_offsets = np.stack(
            [np.array([d.last_box.x_c, d.last_box.y_c]) - ref_center for d in detections],
            axis=0,
        )

        cost = self._angular_cost_matrix(graph_pts, image_offsets)

        diam_result = None
        if self.distance_diameter_weight > 0:
            diam_result = self._diameter_distance_cost_matrix(labels, ref_label, anchor, detections, frame_idx)
            if diam_result is not None:
                diam_cost, _, _, _ = diam_result
                cost = (
                    (1.0 - self.distance_diameter_weight) * cost
                    + self.distance_diameter_weight * diam_cost
                )

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

            # virtual verification -- see module docstring's "Virtual
            # verification" section. Only meaningful when the diameter:
            # distance cue actually ran and this specific candidate had
            # radius data to compare against.
            if diam_result is not None:
                _, norm_ratio_graph, norm_ratio_image, valid = diam_result
                if valid[r]:
                    a, b = norm_ratio_graph[r], norm_ratio_image[c]
                    lo, hi = min(a, b), max(a, b, _MIN_RATIO)
                    match_ok = (lo / hi) >= self.virtual_match_threshold
                    det.diameter_distance_match = bool(match_ok)
                else:
                    det.diameter_distance_match = None
            else:
                det.diameter_distance_match = None
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

    def _diameter_distance_cost_matrix(
        self,
        labels: List[str],
        ref_label: str,
        anchor: Tracklet,
        detections: List[Tracklet],
        frame_idx: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Diameter:distance (angular-width) cost -- see module docstring's
        "Diameter:distance matching cue" section for the geometry (longest
        diameter, opposing-point center, virtual viewpoint a couple of
        centimetres past the bifurcation). Returns None (falls back to
        angle-only) if none of `labels` has graph radius data at all;
        otherwise `(cost, norm_ratio_graph, norm_ratio_image, valid)` --
        the cost matrix plus the peer-normalized ratios and validity mask
        the caller needs afterward to compute `Tracklet.
        diameter_distance_match` for whichever pairing Hungarian actually
        picks (see `_match_candidates`). `frame_idx`, if given, is used to
        Kalman-smooth each detection's own ratio_image over time (at most
        one predict/update per tracklet per frame -- see module
        docstring); omit it (or pass None) to use the raw per-call ratio
        with no temporal smoothing, e.g. for one-off/offline evaluation."""
        graph_diam = np.array([self._candidate_diameter(ref_label, l) for l in labels], dtype=np.float64)
        valid = ~np.isnan(graph_diam)
        if not valid.any():
            return None

        virtual_pts = self.graph.project_children_at_distance(ref_label, self.virtual_advance_mm)
        graph_pts = np.stack(
            [virtual_pts.get(l, np.zeros(2)) for l in labels], axis=0
        )  # (n_labels, 2) -- the "virtual viewpoint" 2D positions, see docstring
        graph_dist = np.maximum(np.linalg.norm(graph_pts, axis=1), _MIN_DIST)
        ratio_graph = graph_diam / graph_dist  # (n_labels,), NaN where invalid
        mean_graph_ratio = np.nanmean(ratio_graph[valid])
        norm_ratio_graph = ratio_graph / max(mean_graph_ratio, _MIN_RATIO)

        anchor_center = self._detection_center(anchor)
        image_diam = np.array([self._detection_diameter(d) for d in detections])
        image_offsets = np.stack(
            [self._detection_center(d) - anchor_center for d in detections], axis=0
        )
        image_dist = np.maximum(np.linalg.norm(image_offsets, axis=1), _MIN_DIST)
        raw_ratio_image = image_diam / image_dist  # (n_dets,)
        ratio_image = np.array(
            [
                self._smoothed_ratio(d.track_id, r, frame_idx)
                for d, r in zip(detections, raw_ratio_image)
            ]
        )
        mean_image_ratio = max(np.mean(ratio_image), _MIN_RATIO)
        norm_ratio_image = ratio_image / mean_image_ratio

        log_ratio_graph = np.log(np.maximum(norm_ratio_graph, _MIN_RATIO))
        log_ratio_image = np.log(np.maximum(norm_ratio_image, _MIN_RATIO))

        diff = np.abs(log_ratio_graph[:, None] - log_ratio_image[None, :])  # (n_labels, n_dets)
        cost = np.minimum(diff, _LOG_RATIO_CAP) / _LOG_RATIO_CAP

        cost[~valid, :] = 0.5  # no radius data for this candidate -> neutral, angle carries it
        return cost, norm_ratio_graph, norm_ratio_image, valid

    @staticmethod
    def _detection_diameter(detection: Tracklet) -> float:
        """A detection's lumen diameter: its mask's own longest diameter
        (max distance between any two of its boundary points -- see
        `utils.polygon_longest_diameter`) when a segmentation mask is
        available, falling back to max(box.w, box.h) otherwise (a plain,
        non-segmentation detector, or a frame where the mask came back
        degenerate). The box fallback deliberately takes the LARGER side,
        not the average -- consistent with the mask path above always
        reporting the longest extent rather than something averaged down,
        and because a foreshortened/partly-occluded lumen's box is often
        narrower on one axis than the true opening, so the larger side is
        the closer stand-in for its real apparent width. See module
        docstring's "Diameter: the longest diameter" section."""
        if detection.mask is not None:
            d = polygon_longest_diameter(detection.mask)
            if d is not None:
                return d
        return max(detection.last_box.w, detection.last_box.h)

    @staticmethod
    def _detection_center(detection: Tracklet) -> np.ndarray:
        """A detection's center: the average of opposing mask-boundary
        points (see `utils.polygon_opposing_center` and module docstring's
        "Center" section) when a segmentation mask is available, falling
        back to the box center (x_c, y_c) otherwise."""
        if detection.mask is not None:
            c = polygon_opposing_center(detection.mask)
            if c is not None:
                return c
        return np.array([detection.last_box.x_c, detection.last_box.y_c])

    def _smoothed_ratio(self, track_id: int, raw_ratio: float, frame_idx: Optional[int]) -> float:
        """Kalman-smoothed ratio_image for one tracklet (see module
        docstring). Creates a filter on first sight of this track_id;
        predicts+updates at most once per frame_idx even if this
        tracklet is considered in more than one candidate set this frame
        (children, then siblings -- see `_propagate_from_anchor`)."""
        kf = self._ratio_filters.get(track_id)
        if kf is None:
            kf = ScalarKalmanFilter(raw_ratio, process_noise=5e-3, measurement_noise=2e-2)
            self._ratio_filters[track_id] = kf
            self._ratio_filters_updated_frame[track_id] = frame_idx
            return kf.value

        if frame_idx is not None and self._ratio_filters_updated_frame.get(track_id) == frame_idx:
            return kf.value  # already updated this frame from a different candidate set

        kf.predict()
        kf.update(raw_ratio)
        self._ratio_filters_updated_frame[track_id] = frame_idx
        return kf.value

    def _prune_ratio_filters(self, tracklets: List[Tracklet]) -> None:
        """Drop smoothing state for tracklets the tracker has fully
        dropped, so `_ratio_filters` doesn't grow unboundedly over a long
        video."""
        live_ids = {t.track_id for t in tracklets}
        for track_id in list(self._ratio_filters.keys()):
            if track_id not in live_ids:
                del self._ratio_filters[track_id]
                self._ratio_filters_updated_frame.pop(track_id, None)

    def _candidate_diameter(self, ref_label: str, label: str) -> float:
        """Expected *apparent* diameter for `label`'s branch as seen from
        `ref_label`'s own bifurcation, at the virtual viewpoint
        `virtual_advance_mm` down `label`'s own centerline -- see
        `AirwayGraph.child_apparent_diameter_at_distance` and module
        docstring's "The virtual viewpoint" section. This is
        foreshortening-corrected, not the branch's true straight-on
        diameter (see that same section for why: a real detected mask is
        never the true cross-section either, since the near airway wall
        partially occludes any child that isn't dead-ahead -- comparing a
        real, foreshortened measurement against an idealized true diameter
        would read as a mismatch even when the tracking is correct). NaN
        if the graph has no radius data at all for this branch."""
        d = self.graph.child_apparent_diameter_at_distance(ref_label, label, self.virtual_advance_mm)
        return d if d is not None else float("nan")

    # ------------------------------------------------------------------
    # gallery bookkeeping
    # ------------------------------------------------------------------
    def _update_gallery(self, current: List[Tracklet], frame_idx: int) -> None:
        from ..types import GalleryEntry

        best_label, best_gen = None, -1
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

        if best_label is not None:
            self.current_location = best_label
