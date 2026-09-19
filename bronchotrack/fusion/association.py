"""Per-frame orchestration of the three fused models into a branch
location -- fresh logic for this package (does NOT import
`paper_exact.association`; see `fusion/__init__.py` for why).

Continuous re-evaluation, not one-shot anchor propagation
-------------------------------------------------------------
`paper_exact.association.AirwayAssociation._try_initialize` runs exactly
once, guarded by `self._initialized`; once every currently-labeled
"anchor" tracklet is lost, nothing in that module can ever acquire a new
label again for the rest of the video (a real, previously-diagnosed dead
end in this project -- see the pipeline's own history). This module has
no such one-shot gate: every single frame, `process_frame` re-derives
candidate scores from whatever is CURRENTLY visible against
`self.current_location`'s own children, with no dependency on any
specific tracklet surviving from an earlier frame. Losing every tracklet
for a while just means a few frames with no candidate measurements (the
Kalman filters in `IdentityFusion` predict-only through the gap); as soon
as a lumen reappears, scoring resumes from there -- there's no permanent
failure mode by construction.

Reference point / reference opening
--------------------------------------
Both identification cues need a reference position to measure candidates
relative to (image-side) and a reference branch to project from
(graph-side, `ref_label = self.current_location`). The natural reference
detection is whichever currently-visible tracklet already carries
`label == self.current_location` (the lumen we're recognized as sitting
inside). Right after a fresh commit this exists immediately (the
just-matched detection is labeled at commit time -- see `_try_commit`).
Before the very first commit (nothing has ever been identified yet,
typically at the tree root looking at the first bifurcation), there is no
such tracklet BY DEFINITION -- the root branch itself is where the camera
already is, never itself a detected lumen. `_pick_reference` falls back,
in that case only, to the single largest currently-visible tracklet as a
stand-in vantage point.

The root bootstrap, and why it's scoped to fire at most once
------------------------------------------------------------------
Falling back to a single stand-in reference has a real consequence at the
very first bifurcation specifically: `candidates_pool` excludes whichever
tracklet was picked as the reference, so with exactly two visible lumens
(a typical trachea -> {LMB, RMB} split) only ONE of them is ever actually
scored as a candidate -- the other is permanently "spent" playing the
reference role and can never itself be identified this way. `_try_bootstrap`
fixes this the same way the paper's own carina initialization
(`paper_exact.association._try_initialize`) does: the moment the number
of currently-visible tracklets exactly matches `self.current_location`'s
number of children AND no branch has EVER been committed yet in this
run's history, label all of them at once by sorted bearing position (no
reference point needed for this one comparison, since it's relative
labels against each other, not against a vantage point), then adopt
whichever one is currently largest as the new `current_location`.

Deliberately scoped to `not self._has_committed_once`, checked and then
permanently latched True the moment ANY commit happens (bootstrap or
normal) -- this must NOT re-trigger at some other 2-child bifurcation
deeper in the tree (a bronchial tree is full of 2-child splits), only at
the genuine start where no reference has ever existed. Once it has fired
once, or once a normal `_try_commit` has happened even without it firing,
every later bifurcation always has a real occupant tracklet already
labeled from its own parent's commit, so the normal continuous
Hungarian-matching path (below) always has a proper reference and never
needs this special case again -- this is a one-time initial condition,
not a recurring dependency the way `paper_exact`'s one-shot
`_try_initialize` guard ends up being for its entire run (see this
module's own top docstring's "Continuous re-evaluation" section).

Commit hysteresis
--------------------
A candidate's fused confidence must clear `commit_threshold` for
`commit_frames` CONSECUTIVE frames before `self.current_location` is
actually advanced to it -- a single lucky frame of high agreement
shouldn't relabel the scope's position; a sustained one should. Each
non-qualifying frame resets that candidate's own streak counter (not
every candidate's -- one candidate having a bad frame shouldn't cost
another candidate's independently-accumulating streak).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..graph import AirwayGraph
from ..types import GalleryEntry, Tracklet
from ..utils import mask_equivalent_diameter
from . import bronchotrack_id, ratio_id
from .kalman_fusion import CandidateConfidence, IdentityFusion
from .motion_model import ApproachMotionModel, TransitSanityChecker

DEFAULT_COMMIT_THRESHOLD = 0.65
DEFAULT_COMMIT_FRAMES = 5
DEFAULT_VIRTUAL_ADVANCE_MM = 20.0
DEFAULT_DISPLAY_THRESHOLD = 0.3  # below this, a candidate isn't worth showing at all


@dataclass
class AssociationResult:
    location: str
    generation: Optional[int]
    reference_track_id: Optional[int]
    candidates: Dict[str, CandidateConfidence] = field(default_factory=dict)
    committed_this_frame: bool = False
    transition_sanity: Optional[str] = None  # "plausible"/"implausible"/"unknown" -- only set when
    # committed_this_frame is True; see TransitSanityChecker and _try_commit/_try_bootstrap below.


def _detection_size(t: Tracklet) -> float:
    if t.mask is not None:
        d = mask_equivalent_diameter(t.mask)
        if d is not None:
            return d
    return (t.last_box.w + t.last_box.h) / 2.0


class FusionAssociation:
    def __init__(
        self,
        graph: AirwayGraph,
        bearing_weight: float = 0.5,
        ratio_weight: float = 0.5,
        motion_gate_strength: float = 0.3,
        max_match_cost: float = 0.6,
        commit_threshold: float = DEFAULT_COMMIT_THRESHOLD,
        commit_frames: int = DEFAULT_COMMIT_FRAMES,
        virtual_advance_mm: float = DEFAULT_VIRTUAL_ADVANCE_MM,
        display_threshold: float = DEFAULT_DISPLAY_THRESHOLD,
    ):
        self.graph = graph
        self.bearing_weight = bearing_weight
        self.ratio_weight = ratio_weight
        self.motion_gate_strength = motion_gate_strength
        self.max_match_cost = max_match_cost
        self.commit_threshold = commit_threshold
        self.commit_frames = commit_frames
        self.virtual_advance_mm = virtual_advance_mm
        self.display_threshold = display_threshold

        self.current_location: str = graph.root()
        self.motion_model = ApproachMotionModel()
        self.sanity_checker = TransitSanityChecker()
        self.gallery: Dict[str, GalleryEntry] = {}

        self._fusion: Optional[IdentityFusion] = None
        self._fusion_location: Optional[str] = None
        self._commit_streak: Dict[str, int] = {}
        self._location_entered_frame: int = 0
        self._has_committed_once: bool = False

        self._update_gallery(self.current_location, 0)

    @property
    def visited(self) -> Set[str]:
        return set(self.gallery.keys())

    def eligibility_fn(self, tracklet: Tracklet) -> bool:
        """No generation-gap dropout in this package (see `fusion/__init__
        .py` -- a deliberate scope reduction, not an oversight): every
        tracklet the tracker maintains is eligible for matching. The
        continuous re-evaluation design above makes a stale, far-away
        label far less costly to carry than in `paper_exact` (a bad label
        just never wins Hungarian assignment against real candidates
        again), so the paper's own dropout heuristic isn't load-bearing
        here the way it is there."""
        return True

    def process_frame(self, tracklets: List[Tracklet], frame_idx: int) -> AssociationResult:
        current = [t for t in tracklets if t.time_since_update == 0]

        approach_estimates = self.motion_model.update(tracklets, frame_idx)
        self.motion_model.prune({t.track_id for t in tracklets})

        children = self.graph.children(self.current_location)
        if not children:
            return AssociationResult(
                location=self.current_location,
                generation=self._generation(self.current_location),
                reference_track_id=None,
            )

        if not self._has_committed_once and len(current) == len(children):
            if self._try_bootstrap(children, current, frame_idx):
                return AssociationResult(
                    location=self.current_location,
                    generation=self._generation(self.current_location),
                    reference_track_id=None,
                    candidates={},
                    committed_this_frame=True,
                    # the bootstrap has no pre-existing occupant to have
                    # measured an approach speed from -- nothing to check
                    transition_sanity="unknown",
                )

        reference = self._pick_reference(current)
        if reference is None:
            return AssociationResult(
                location=self.current_location,
                generation=self._generation(self.current_location),
                reference_track_id=None,
            )

        candidates_pool = [t for t in current if t.track_id != reference.track_id]

        bearing_cost = bronchotrack_id.bearing_cost_matrix(
            self.graph,
            self.current_location,
            children,
            candidates_pool,
            reference_point=bronchotrack_id.detection_point(reference),
        )
        ratio_cost = ratio_id.ratio_cost_matrix(
            self.graph,
            self.current_location,
            children,
            reference,
            candidates_pool,
            self.virtual_advance_mm,
        )

        fusion = self._get_fusion(self.current_location)
        results = fusion.update(children, candidates_pool, bearing_cost, ratio_cost, approach_estimates)

        committed, sanity_verdict = self._try_commit(results, current, frame_idx)

        visible_candidates = {
            label: cc for label, cc in results.items() if cc.fused_confidence >= self.display_threshold
        }

        return AssociationResult(
            location=self.current_location,
            generation=self._generation(self.current_location),
            reference_track_id=reference.track_id,
            candidates=visible_candidates,
            committed_this_frame=committed,
            transition_sanity=sanity_verdict,
        )

    # ------------------------------------------------------------------
    def _pick_reference(self, current: List[Tracklet]) -> Optional[Tracklet]:
        labeled_here = [t for t in current if t.label == self.current_location]
        if labeled_here:
            return max(labeled_here, key=lambda t: t.hits)
        if not current:
            return None
        return max(current, key=_detection_size)

    def _try_bootstrap(self, children: List[str], current: List[Tracklet], frame_idx: int) -> bool:
        """Paper-style one-time carina initialization -- see class
        docstring's "The root bootstrap" section. Labels every currently-
        visible tracklet directly by sorted bearing position (no Hungarian
        matching -- there's no single reference point to measure FROM yet,
        this is a simultaneous relative ordering instead), then adopts the
        largest of them as the new `current_location`. No-op (returns
        False) if `self.current_location`'s children aren't all
        projectable yet (missing graph geometry)."""
        proj = self.graph.project_children_2d(self.current_location)
        proj = {l: proj[l] for l in children if l in proj}
        if len(proj) != len(children):
            return False

        dets_sorted = sorted(current, key=lambda t: bronchotrack_id.detection_point(t)[0])
        labels_sorted = sorted(children, key=lambda l: proj[l][0])
        for t, label in zip(dets_sorted, labels_sorted):
            t.label = label
            t.label_history.append((frame_idx, label))

        occupant = max(dets_sorted, key=_detection_size)

        frames_spent = frame_idx - self._location_entered_frame
        arc_length_mm = self.graph.get(self.current_location).arc_length_mm()
        self.sanity_checker.record_transit(self.current_location, frames_spent, arc_length_mm)

        self.current_location = occupant.label
        self._location_entered_frame = frame_idx
        self._fusion = None
        self._commit_streak = {}
        self._has_committed_once = True
        self._update_gallery(occupant.label, frame_idx)
        return True

    def _get_fusion(self, location: str) -> IdentityFusion:
        if self._fusion is None or self._fusion_location != location:
            self._fusion = IdentityFusion(
                bearing_weight=self.bearing_weight,
                ratio_weight=self.ratio_weight,
                motion_gate_strength=self.motion_gate_strength,
                max_match_cost=self.max_match_cost,
            )
            self._fusion_location = location
            self._commit_streak = {}
        return self._fusion

    def _try_commit(
        self, results: Dict[str, CandidateConfidence], current: List[Tracklet], frame_idx: int
    ) -> Tuple[bool, Optional[str]]:
        for label, cc in results.items():
            if cc.fused_confidence >= self.commit_threshold:
                self._commit_streak[label] = self._commit_streak.get(label, 0) + 1
            else:
                self._commit_streak[label] = 0

        commit_label = next(
            (l for l, streak in self._commit_streak.items() if streak >= self.commit_frames), None
        )
        if commit_label is None:
            return False, None

        matched_tid = results[commit_label].matched_track_id
        if matched_tid is not None:
            for t in current:
                if t.track_id == matched_tid:
                    t.label = commit_label
                    t.label_history.append((frame_idx, commit_label))

        # Sanity-check verdict FIRST (against history accumulated before
        # this transit), using the just-committed candidate's own last
        # motion-model tau prediction and its known branch length -- see
        # TransitSanityChecker's docstring for why this never calibrates
        # tau itself, only flags it. Then record this transit into the
        # checker's own history for future checks.
        candidate_tau_frames = self.motion_model.tau_frames_for(matched_tid) if matched_tid is not None else None
        candidate_arc_length_mm = self.graph.get(commit_label).arc_length_mm()
        sanity_verdict = self.sanity_checker.check(candidate_tau_frames, candidate_arc_length_mm)

        frames_spent = frame_idx - self._location_entered_frame
        left_arc_length_mm = self.graph.get(self.current_location).arc_length_mm()
        self.sanity_checker.record_transit(self.current_location, frames_spent, left_arc_length_mm)

        self.current_location = commit_label
        self._location_entered_frame = frame_idx
        self._fusion = None
        self._commit_streak = {}
        self._has_committed_once = True
        self._update_gallery(commit_label, frame_idx)
        return True, sanity_verdict

    def _update_gallery(self, label: str, frame_idx: int) -> None:
        entry = self.gallery.setdefault(label, GalleryEntry(label=label, first_seen_frame=frame_idx))
        entry.last_seen_frame = frame_idx

    def _generation(self, label: str) -> Optional[int]:
        return self.graph.generation(label) if label in self.graph else None
