"""bronchotrack.fusion -- a three-model fused localization pipeline, kept
side by side with both `bronchotrack.pipeline` (the enhanced main package)
and `bronchotrack.paper_exact` (the strict paper reference) rather than
replacing either.

Built per an explicit request for: "create a motion model where you use
the lumen diameter increase speed to get the approach speed[, and] using
that predict when you will reach the next bifurcation of the given
branch... next use the ratio method and airway graph as in bronchotrack
paper to determine what lumen we are looking at[. F]or the bronchotrack
version use the center of the segmentation[. B]ut for the ratio method
use the entire identified segmentation to get ratio to distances[.
F]inally use a kalman filter to combine all three models and display the
child branch[es] on the live video... a clean restart of the code" --
plus a follow-up clarification that the "ratio method" must genuinely
leverage full mask geometry, not reduce to points the way a first draft
of this package did.

The three models
-------------------
1. `motion_model.ApproachMotionModel` -- diameter-growth motion model.
   Tracks each tracklet's own apparent diameter (Kalman-smoothed) and
   derives a SCALE-FREE time-to-contact, tau = diameter / d(diameter)/dt
   -- no camera calibration anywhere in the computation. Cross-checked
   (never calibrated) against the airway graph's own known branch lengths
   via `motion_model.TransitSanityChecker`, an empirical, self-referential
   plausibility check built from this run's own completed transits.

2. `bronchotrack_id` -- "the bronchotrack version": point-based bearing
   matching using each lumen's segmentation CENTER (`utils.
   polygon_opposing_center`), the paper's own core angular-position idea
   (paper section 4), re-derived fresh here rather than importing
   `paper_exact.association`'s implementation (see "Why a clean restart,
   concretely" below).

3. `ratio_id` -- "the ratio method": genuinely whole-mask, not
   point-based. Size is the mask's own AREA-equivalent diameter (a
   shoelace integral over every boundary vertex, not two extreme points);
   distance is the nearest boundary-to-boundary GAP between two full mask
   polygons (an all-pairs search over both boundaries), mirrored on the
   graph side by an analytic circle-boundary-gap using each opening's own
   true radius. See `ratio_id.py`'s own module docstring for the full
   geometric derivation and why it's a materially different measurement
   from `bronchotrack_id`, not a renamed copy of it.

`kalman_fusion.IdentityFusion` blends (1) and (2)'s cost matrices into one
Hungarian assignment per frame, per candidate branch label; the winning
assignment's raw confidence is nudged by whether model (3)'s matched
tracklet is currently "approaching" (motion-model gating), and the result
is fed into a per-label `ScalarKalmanFilter` -- literally "use a kalman
filter to combine all three models" into one smoothed, persistent
per-candidate confidence. `association.FusionAssociation` re-runs this
EVERY frame against whatever's currently visible (see its own docstring's
"Continuous re-evaluation" section) and commits a branch transition only
once a candidate clears `commit_threshold` for `commit_frames` consecutive
frames.

`viz.draw_overlay` shows every currently-scored candidate child (not just
the current position) as a HUD panel of name / fused-confidence bar / ETA,
plus dots on the actually-tracked lumens in frame -- see that module's own
docstring for why the candidates are a HUD list rather than dots
overlaid at fabricated screen positions (no camera calibration exists
anywhere in this pipeline to place them accurately).

Why a clean restart, concretely
-----------------------------------
This package does NOT import `paper_exact.association`, `.localization`,
or `.pipeline`. It DOES reuse, unmodified, the primitives that were
already general-purpose and paper_exact-agnostic: `graph.py` (plus one
new method added there, `child_apparent_equivalent_diameter_at_distance`
-- the ratio method's area-based foreshortening formula, alongside the
existing longest-diameter one used elsewhere), `types.py`, `utils.py`,
`detection.py`, `tracker.py`, and the Kalman filter machinery
(`scalar_kalman.ScalarKalmanFilter`, promoted from inside `paper_exact/`
to this package-shared top-level location once both packages needed it;
`paper_exact.scalar_kalman` still re-exports it unchanged for backward
compatibility -- see that module).

Deliberate scope reductions for this first version (flagged, not hidden)
------------------------------------------------------------------------
* No roll-angle correction (`paper_exact.association`'s Eq. 6/7
  mechanism) -- `bronchotrack_id`'s bearing matching is plain, unrotated
  cosine similarity. Reimplementing roll estimation fresh here (rather
  than importing paper_exact's) was judged not worth the scope for a
  first version; the continuous per-frame re-evaluation design already
  tolerates more bearing noise gracefully than a one-shot anchor
  propagation scheme would (a bad frame just doesn't win Hungarian
  assignment; it doesn't corrupt a persistent anchor chain).
* No multi-generation "ambiguous cohort" localization smoothing
  (`paper_exact.localization`'s g^{k-1}/g^k voting) -- `current_location`
  always advances one bifurcation at a time to a committed child, which
  fits "predict when you will reach the NEXT bifurcation" more directly
  than that mechanism would.
* No tracker eligibility dropout (`FusionAssociation.eligibility_fn`
  always returns True) -- see that method's own docstring for why the
  continuous re-evaluation design makes this less necessary than in
  `paper_exact` (where a stale label could otherwise never be displaced).
"""
from .association import AssociationResult, FusionAssociation
from .kalman_fusion import CandidateConfidence, IdentityFusion
from .motion_model import ApproachEstimate, ApproachMotionModel, TransitSanityChecker
from .pipeline import FrameResult, FusionPipeline

__all__ = [
    "FusionAssociation",
    "AssociationResult",
    "IdentityFusion",
    "CandidateConfidence",
    "ApproachMotionModel",
    "ApproachEstimate",
    "TransitSanityChecker",
    "FusionPipeline",
    "FrameResult",
]
