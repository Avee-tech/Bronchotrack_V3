"""bronchotrack.paper_exact -- a strict reference implementation of the
BronchoTrack paper (arXiv:2402.12763), kept side by side with the main
``bronchotrack`` package rather than replacing it.

Why this package exists
------------------------
The main package started as a paper-faithful port and grew, over several
iterations, a handful of deliberate improvements that go beyond what the
paper describes (a diameter-ratio matching cue, a graph-constrained
motion-model filter, localization output smoothing). Those additions are
real improvements for actual bronchoscopy videos, but they mean running
``bronchotrack.pipeline`` no longer reproduces *only* what the paper
specifies. This package is the answer to "what does the paper's own method,
and nothing else, actually output" -- useful as a baseline to compare the
enhanced pipeline against, and as the thing to point to if you need to show
your thesis committee an implementation that matches the paper exactly.

Run it with ``python3 -m bronchotrack.paper_exact.cli`` (see that module's
docstring for the exact flags), on the same video/graph/weights you'd pass
to the main CLI.

What's reused, unmodified, from the main package
--------------------------------------------------
These were already direct, unembellished ports of the paper's method (each
one's own docstring cites the exact paper section/equation), so there was
nothing to fork:

* ``kalman.py``       -- the paper's constant-velocity Kalman state vector
                          x_k = [x_c, y_c, h, a, xdot_c, ydot_c, hdot].
* ``reid.py``          -- ResNet50 appearance embedding + the EMA update
                          e_i^t = alpha*e_i^{t-1} + (1-alpha)*f_i^t, alpha=0.9.
* ``matching.py``      -- C_m = 1-IoU, C_a = 1-cosine, C = lambda*C_a +
                          (1-lambda)*C_m (lambda=0.5), Hungarian assignment.
* ``tracker.py``       -- the two-stage BYTE-style association (high-conf
                          combined cost, low-conf motion-only cost) and the
                          "drop tracklets whose label is >3 generations from
                          the current location" eligibility rule.
* ``graph.py``         -- airway graph representation, the paper's
                          coordinate standardization (y = trachea axis, x =
                          plane of the L/R main-bronchus origins, z =
                          orthogonal), intersection-angle and 2D-projection
                          helpers.
* ``detection.py``     -- YOLO-based lumen detector wrapper. See "The one
                          unavoidable substitution" below.
* ``graph_view.py`` -- pure rendering, no algorithmic content (its
  ``render()`` gained one new optional ``visited`` kwarg for this package's
  own use, see "What's added" below; default behavior for existing callers
  is unchanged).

What's removed here (present in the main package, absent from the paper)
---------------------------------------------------------------------------
* The main package's *peer-comparison* diameter-ratio matching cue
  (``association._diameter_ratio_cost_matrix`` / ``--diameter-weight``,
  which compares each detection/candidate's size to the mean size of its
  own currently-visible peers) -- not present here. This package has its
  own, differently-formulated diameter cue instead; see "What's added"
  below.
* The graph-constrained motion-model Bayes filter (main package's
  ``motion_model.TreeMotionFilter`` / ``--no-motion-model`` /
  ``motion_location``/``motion_confidence``/``motion_generation``) -- not
  imported, not run, not in ``FrameResult`` at all here.
* Localization temporal smoothing (main package's
  ``localization.Localizer``'s ``smoothing_window``/``min_frames_to_switch``
  "sticky" hysteresis) -- ``paper_exact.localization.Localizer`` reports the
  raw per-frame Eq. 8 vote every single frame, unsmoothed.
* The "approach trend" diagnostic (main package's
  ``association.approach_trend`` / image-size slope) -- a purely
  informational addition with no paper basis; dropped.
* ``flip_v`` (main package's static graph/camera axis-mirror workaround) --
  a fix for a specific coordinate-mismatch issue, not a paper concept.

What's added here (paper-described, missing from the main package)
-----------------------------------------------------------------------
* Roll-angle correction (paper Eq. 6/7): the main package's
  ``association.py`` explicitly disclosed that it skips roll estimation
  entirely and compares graph-space and image-space bearings unrotated.
  ``paper_exact.association.AirwayAssociation`` implements the paper's roll
  estimate and rotates the graph's projected candidate points by it before
  the angular-bearing match. Two necessary interpretive choices are made
  explicit in that module's own docstring: the paper's arccos-based formula
  is direction-less by construction, so the signed form (atan2 of the 2D
  cross/dot product) is used instead; and the "frame m" reference is read as
  the frame a given oldest-tracklet-pair was first adopted, not
  incremented every single frame.

* A diameter:distance ("angular width") matching cue -- opt-in, on by
  default at weight 0.4 (``--distance-diameter-weight``, 0 disables it).
  For each candidate branch / detection, this compares
  ``diameter / distance-between-two-lumen-centers`` between graph space
  and image space, log-differenced and blended into the matching cost.
  The diameter is each mask's own **longest diameter** (max distance
  between any two of its boundary points), and the center is the
  **average of opposing boundary points** rather than a plain centroid --
  see ``utils.polygon_longest_diameter`` / ``polygon_opposing_center``.
  On the graph side, instead of the candidate branch's very first point or
  plain midpoint, this evaluates a **virtual viewpoint a couple of
  centimetres past the bifurcation** (``--virtual-advance-mm``, default
  20mm) along the current localization estimate's candidate child --
  effectively re-running the same diameter/distance measurement directly
  against the known 3D geometry at the point the scope will actually be
  looking at, rather than synthesizing and re-detecting a picture. Each
  accepted pairing is additionally checked pass/fail against
  ``--virtual-match-threshold`` (default 0.75) and recorded on
  ``Tracklet.diameter_distance_match``. This is NOT a paper mechanism --
  the paper's own Eq. 6 "observable branch likelihood" is described only
  qualitatively, as a negative correlation with intersection angle, with
  no size or distance term. It's a deliberately different formulation from
  the main package's peer-comparison cue: diameter:distance is meaningful
  on its own (to a small-angle approximation it's the lumen's angular
  width as seen from the reference point) rather than needing a peer group
  to normalize against, so it applies even with only one candidate/
  detection visible. See ``association.py``'s module docstring for the
  exact formula and the geometric reasoning behind it.

* Kalman smoothing for two of the above (``scalar_kalman.py``, a minimal
  constant-velocity Kalman filter reused for both -- same predict/update
  recursion as the paper's own box-tracking filter in ``kalman.py``, just
  over a 1D state): the roll-angle estimate (a shared filter, since it's
  one global rotation), and each tracklet's own diameter:distance ratio
  in the cue above (one filter per tracklet, since each one has its own
  independent noise). Neither is a paper mechanism -- both are smoothing
  added on top of this package's own non-paper additions -- but both
  reuse the exact filtering approach the paper already specifies
  elsewhere rather than inventing a different smoothing technique.

* Center-dot overlay video (``paper_exact/viz.py``, forked from the main
  package's ``viz.py``): rather than a box or mask outline, each tracked
  lumen is drawn as a filled circle at its ``polygon_opposing_center`` (the
  same point the diameter:distance cue and its virtual-model verification
  use), with its assigned branch name above it. Display is gated on
  ``Tracklet.diameter_distance_match is True`` -- i.e. only lumens the
  diameter:distance cue's Kalman-filtered virtual-model comparison has
  actually confirmed against the 3D graph get drawn at all, not merely
  "currently visible" or "labeled" ones. This is deliberate: it lets
  ``--conf-threshold`` be set permissively (admitting far more raw
  candidate detections than a stricter cutoff would) while keeping the
  video itself clean, by leaning on that physically-grounded consistency
  check to filter out the resulting false positives instead of a blunt
  per-frame confidence cutoff. Each track_id keeps its own persistent
  color from a fixed palette for as long as it's tracked, the same
  "follow one lumen's identity across frames by color alone" idea the
  paper's own qualitative-results figures (Figs. 3 and 6) use, just
  carried onto a marker instead of a box.

* A "driving path" trail in the graph view (``graph_view.py``'s new
  ``visited`` kwarg, wired up in ``paper_exact.pipeline``): the paper's
  Fig. 5(a) shows the scope's full route through the tree, not just its
  current position. `pipeline.py` passes the association module's gallery
  (every branch label ever visited this run) so `graph_view.render()` now
  highlights the whole traversed path, with the single current-location
  marker still drawn on top for the "you are here" point.

* Re-acquisition after total anchor loss (``--reacquire-max-gap-frames``,
  default 90 = ~3s at 30fps; 0 disables it): label propagation only ever
  runs from a currently-visible labeled tracklet, so a real, diagnosed
  failure mode existed here -- once every anchor was lost at once (a long
  occlusion, a burst of missed detections), NOTHING could ever be labeled
  again for the rest of the video (confirmed on a real run: 82% of a
  1084-frame video showed zero locations after its last anchor was lost
  around frame 200). ``association.py`` now keeps a frozen snapshot of
  the most-established real anchor and, on any frame with no real
  anchors, hands that snapshot back as a stand-in "virtual anchor" to
  propagate from -- bounded by ``reacquire_max_gap_frames`` so a very old
  snapshot isn't trusted indefinitely. See that module's own docstring
  section for the full mechanism.

* Continuous (per-frame) diameter:distance re-verification
  (``--no-continuous-verification`` to disable; on by default): another
  real, log-confirmed failure mode, this one diagnosed *from* the
  re-acquisition fix above rather than independently of it --
  re-acquisition alone raised labeled-frame coverage 15.3% -> 39.9% on
  that same 1084-frame run, but the confirmed-dot rate (the overlay's
  display gate, ``Tracklet.diameter_distance_match is True``) stayed flat.
  Root cause: that field was decided exactly once, at the single frame a
  tracklet was first labeled, then frozen for its whole remaining life --
  and tier-1 self-reacquisition relabels a tracklet directly by IoU with
  no Hungarian matching step at all, i.e. no verification either, so
  reacquired tracklets could structurally never earn a dot. ``association.py``
  now regroups all currently-visible labeled tracklets by shared parent
  branch every frame and re-verifies any group of two or more, overwriting
  the previous verdict -- applying uniformly regardless of how a tracklet
  got its label. See that module's own docstring section ("Continuous
  (per-frame) re-verification") for the full mechanism and why groups
  smaller than two are left standing rather than reset.

* Live vs. carried-forward location (``paper_exact.localization.Localizer
  .last_vote_was_live()`` / ``FrameResult.location_is_live`` / the JSON
  log's ``"location_is_live"``): the paper's own Eq. 8 carries the last
  known location forward when nothing labeled is currently visible ("the
  scope having left the airway" isn't implied just because there's
  nothing to vote with this frame) -- correct paper behavior, but it means
  ``location`` alone can't distinguish "still here, freshly reconfirmed"
  from "no current evidence, repeating our last guess", which matters
  since the scope may have silently advanced past a bifurcation while
  every anchor was lost. This is purely additive bookkeeping alongside the
  paper's own carry-forward rule, not a change to it -- see
  ``localization.py``'s "Live vs. carried-forward votes" docstring
  section.

The one unavoidable substitution
-----------------------------------
The paper trains a YOLOv7 detector (256x256 input, 0.1 confidence
threshold, E-ELAN backbone, trainable bag-of-freebies) on 7,644 hand-labeled
frames from ten patients, plus a from-scratch ResNet50 Re-ID head on 3,630
lumen crops -- both from a proprietary hospital dataset this project has no
access to. Retraining a literal YOLOv7 from that description, without the
paper's own data, would not actually be more paper-faithful than reusing
your own trained detector -- it would just be a differently-wrong model.
``detection.py`` (reused unmodified from the main package) wraps whatever
Ultralytics YOLO checkpoint you give it via ``--weights``; the modeling
choices around it (0.1 confidence default, low-confidence detections kept
for the tracker's second stage, class filtering) all match the paper's
description even though the exact network architecture is YOLOv11 rather
than YOLOv7. This is the same substitution the main package already makes,
carried over here since there's no paper-truer alternative available in
this environment.

What's explicitly NOT specified by the paper (kept as configurable, with
the numeric default flagged as ours, not the paper's)
---------------------------------------------------------------------------
* The exact metric for matching 2D-projected graph points to image
  detections (angular bearing is used here -- scale-free, consistent with
  the paper's own framing of depth as topological rather than metric; see
  ``association.py`` docstring point 1).
* ``max_match_cost`` (rejects an over-cost Hungarian match), and
  ``angle_threshold_deg`` (the intersection-angle child-visibility filter)
  -- the paper names both mechanisms but not their numeric thresholds.
* The exact high/low confidence split threshold and the two match-cost
  gates in the tracker (paper: "high and low confidence candidates" and a
  cost threshold, no numbers given) -- these were already parameters in the
  main package's ``tracker.py``, reused unmodified here.
"""
from .association import AirwayAssociation
from .localization import Localizer
from .pipeline import BronchoTrackPipeline, FrameResult

__all__ = ["AirwayAssociation", "Localizer", "BronchoTrackPipeline", "FrameResult"]
