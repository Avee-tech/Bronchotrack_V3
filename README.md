# BronchoTrack Pipeline

A modular, from-scratch re-implementation of the tracking + branch-level
localization pipeline from:

> Tian, S., Liao, X. et al. **"BronchoTrack: Airway Lumen Tracking for
> Branch-Level Bronchoscopic Localization."** arXiv:2402.12763.

It is built to plug directly into the two artifacts you already have:

- **your trained YOLOv11 lumen-detection weights** (`.pt`) -> `bronchotrack/detection.py`
- **your 3D airway graph from 3D Slicer** (nodes/edges/coordinates/labels, as JSON) -> `bronchotrack/graph.py`

and turn them into per-frame **branch-level bronchoscope localization**.

## What's implemented

Per your requested scope, this covers the paper's base pipeline plus Re-ID
(loop closure / BronchoTrack-LC is **not** implemented -- see "Not
implemented" below; roll-angle estimation was implemented and later
**removed** -- candidate matching now assumes zero roll, see "Fidelity
notes").

| Paper module | File(s) | Status |
|---|---|---|
| 1. Lumen Detection | `detection.py` | YOLOv11 wrapper (Ultralytics API) + precomputed-detections loader |
| 2a. Motion tracking (Kalman) | `kalman.py` | 7-D constant-velocity filter, paper's exact state vector |
| 2b. Appearance Re-ID | `reid.py` | ResNet50 embedding + EMA update (needs your own fine-tuned weights for best results) |
| 2c. Cost + assignment | `matching.py` | `C_m`, `C_a`, `C = lambda*C_a + (1-lambda)*C_m`, Hungarian |
| 2d. Multi-lumen tracker | `tracker.py` | Two-stage BYTE-style association, track lifecycle |
| 3a. Airway association | `association.py` | Gallery, label propagation, angle-filtered candidate children, 2D bearing + diameter-ratio matching (assumes zero roll) |
| 3b. Localization | `localization.py` | Majority-vote branch-level location, temporally smoothed |
| Orchestration | `pipeline.py`, `cli.py` | Wires it all together, video I/O, JSON/overlay output, device (GPU/CPU) reporting |
| Live streaming | `live.py` | Background-threaded RTSP/RTMP/HTTP/webcam reader with auto-reconnect, feeding `pipeline.run_live()` |
| Airway graph view | `graph_view.py` | Live schematic view of the graph with branch labels + current location highlighted |
| Motion-model filter (extra, beyond the paper) | `motion_model.py` | Graph-constrained Bayes filter, seeded at the trachea, predicting/updating which branch the camera is in; runs alongside 3b, not in place of it |

## Quick start

```bash
pip install -r requirements.txt   # or: pip install -e ".[detection,reid]"

python3 -m bronchotrack.cli \
    --video path/to/bronchoscopy.mp4 \
    --graph path/to/your_airway_graph.json \
    --weights path/to/your_lumen_yolov11.pt \
    --out-video out/overlay.mp4 \
    --out-json out/localization_log.json
```

Or in Python:

```python
from bronchotrack import AirwayGraph, BronchoTrackPipeline
from bronchotrack.detection import LumenDetector
from bronchotrack.reid import ReIDEmbedder

graph = AirwayGraph.from_json("your_airway_graph.json")
detector = LumenDetector("your_lumen_yolov11.pt", conf_threshold=0.1, img_size=256)
reid = ReIDEmbedder(weights_path=None)  # or your fine-tuned checkpoint

pipeline = BronchoTrackPipeline(graph=graph, detector=detector, reid_embedder=reid)
results = pipeline.run_on_video("bronchoscopy.mp4", output_json_path="log.json")

for r in results[:5]:
    print(r.frame_idx, r.location, r.generation)
```

Run `python3 tests/test_pipeline_smoke.py` to sanity-check the graph /
tracking / association / localization logic on synthetic data (no torch or
ultralytics required for this).

## Live video (RTSP stream, on-screen preview)

For a live source instead of a file -- an RTSP/RTMP/HTTP stream, or a
webcam/capture-card device index -- add `--live` and pass the stream URL as
`--video`:

```bash
python3 -m bronchotrack.cli \
    --video rtsp://192.168.1.50:8554/bronch \
    --live \
    --graph path/to/your_airway_graph.json \
    --weights path/to/your_lumen_yolov11.pt \
    --out-video out/session_recording.mp4 \
    --out-json out/session_log.json
```

This opens a live `cv2.imshow` preview window (boxes, track IDs, branch
labels, current location/generation overlaid) that updates in real
time; press `q` in that window to stop. Use `--no-display` for a headless
run (no monitor attached, e.g. on a server) -- it still processes frames,
still optionally records `--out-video`/`--out-json`, just without a window.

Under the hood (`bronchotrack/live.py`), a background thread continuously
reads from the stream into a single-slot "latest frame" buffer and
auto-reconnects if the stream drops; the main loop always grabs the most
recent frame rather than draining a backlog, so if processing briefly falls
behind, frames are dropped instead of the display drifting further and
further behind real time. Isolated dropped/corrupt frames (common on WiFi
sources) are retried in place rather than triggering a full reconnect --
only `fast_retry_limit` (default 15) *consecutive* failures are treated as
a real disconnect. `--live-fps` sets the frame rate used when writing
`--out-video`, since a live stream doesn't reliably report one the way a
file does -- match it to your source's actual FPS for a correctly timed
recording.

### Using DroidCam as the source

DroidCam gives you two different kinds of `--video` source depending on how
it's connected:

- **Network stream (works cross-platform, no extra drivers):** with the
  DroidCam app running on your phone and both devices on the same WiFi
  network, it serves an MJPEG stream directly -- point `--video` at
  `http://<phone-ip>:4747/video` (check the exact port/path in the
  DroidCam app itself; some versions use `/mjpegfeed?640x480` instead).
  Worth opening that URL in a regular browser first to confirm it streams
  before wiring it into the pipeline. This is an HTTP source, not RTSP, but
  `cv2.VideoCapture` (and therefore `run_live`/`--live`) handles both the
  same way -- no code changes needed, just the right URL.
- **Virtual webcam (via the DroidCam desktop client):** if you've installed
  the DroidCam client on the same machine running this pipeline, it
  registers as a regular camera device. Skip the URL and pass the device
  index instead, e.g. `--video 1 --live` (try `0`, `1`, `2`... until you
  hit the right one -- `ffmpeg -f v4l2 -list_devices true -i dummy` on
  Linux, or the camera list in your OS's settings, will tell you which
  index it landed on).

Either way, USB-tethered DroidCam tends to be far more stable than WiFi for
a real-time tracking pipeline -- WiFi introduces the frame drops/latency
`live.py`'s retry logic is designed to absorb, but for a bronchoscopy-style
application you'll generally want the most consistent feed you can get.

From Python directly:

```python
pipeline.run_live(
    source="rtsp://192.168.1.50:8554/bronch",
    display=True,
    output_video_path="out/session_recording.mp4",
    output_json_path="out/session_log.json",
)
```

`tests/test_live.py` exercises the background reader's connect/read/stop
lifecycle against a synthetic local video file (no real RTSP source
needed) -- run it with `python3 tests/test_live.py`.

## Airway graph live view

Alongside the tracked-overlay camera window, you can open a second window
showing a schematic top-down-ish view of the airway graph itself -- every
branch drawn and labeled with its id/number, color-coded by generation
(orange = trachea, shading towards blue for deeper generations), with the
currently-localized branch highlighted (brighter, thicker, with a marker at
its distal end) so you can watch the scope's progress down the tree
alongside the live camera feed:

```
--live --show-graph
```

To also save it as its own video file (synced frame-for-frame with
`--out-video`), add `--out-graph-video out/graph_view.mp4` -- this works for
both `--live` and regular file playback (no `--live` needed).

This view (`graph_view.py`, `AirwayGraphView`) is a debugging/demo
convenience, not part of the paper's algorithm. Since your graph likely
doesn't define `coordinate_system` (see below), it fits its own 2D
projection plane via PCA over every branch's centerline/endpoint points --
the plane of greatest positional spread in the tree -- which recovers a
reasonable front-on layout without needing any external anatomical axis
info, and works regardless of how the graph happens to be oriented in its
own coordinate system. The projection is computed once when the pipeline
starts (the tree's geometry doesn't change at runtime); only the
highlighting/labels are redrawn per frame.

## Motion-model filter (graph-constrained localization)

Beyond what the paper itself specifies, `motion_model.py`'s
`TreeMotionFilter` adds a discrete Bayes (predict/update) filter over the
airway graph's branch labels, seeded fully certain at the trachea and
constrained to only ever move along real graph edges -- to a branch's
children, its parent, or staying put; never to an unrelated branch
elsewhere in the tree. It's on by default (`--no-motion-model` to disable)
and reported on every frame as `motion_location` / `motion_confidence` /
`motion_generation`, alongside (not replacing) the existing `location` /
`generation` from the paper's own majority-vote `Localizer`.

Each frame it:

1. **Predicts** -- splits the previous frame's belief across "stayed"
   (favored by default -- see below), "advanced to a child", and "retreated
   to the parent", nudged by `association.py`'s `approach_trend` diagnostic
   (a growing tracked lumen shifts probability toward advancing, a
   shrinking one toward retreating). `--motion-p-advance` tunes the
   baseline forward-motion rate.
2. **Updates** -- treats the branch labels actually hard-matched to a
   visible tracklet this frame (from `association.py`'s existing
   diameter-ratio + bearing matching) as a noisy observation, boosting
   their probability and mildly decaying everything else (not zeroing it --
   a missed/occluded detection for one frame isn't strong evidence the
   scope left that branch), **discounted by how far that observed label is
   (in tree hops) from the filter's current best estimate** -- see "A real
   failure mode, found on real video" below for why that discount exists.

### A real failure mode, found on real video

The first version of this filter defaulted to a much higher baseline
`p_advance` (0.55) and gave every observed label equal weight regardless
of where it was in the tree. Tested on a real 916-frame recording
(`scripts/analyze_motion_model.py` against a `--out-json` log), it agreed
with the paper's vote-based `Localizer` on only 22% of frames, and finished
3 generations deeper (`generation 5` vs. the localizer's `generation 2`) --
i.e. it was racing ahead of what the tracked evidence actually supported,
not just disagreeing on noise. Two things were compounding: (1) a flat 55%
per-frame chance of advancing is wildly too high against how rarely a real
bifurcation crossing actually happens per frame (11 branch switches over
916 frames in that same run, ~1.2%), so the predict step alone provided a
persistent forward drift; and (2) `association.py` can legitimately
hard-label several tracklets in the same frame (an anchor plus its
children/siblings), and real detections are noisy enough (~0.45 mean
confidence in this project's own CPU baseline, `baseline_box_detection.md`)
that an occasional mislabel several generations away from the true
location is expected, not exceptional -- and the update step was treating
that stray label exactly as strongly as a well-established nearby one.

The first fix: `p_advance` dropped to 0.06 (predict alone no longer
provides meaningful forward drift without real evidence), *and* `update()`
was changed to discount each observed label's evidence strength by its
tree-hop distance from the current best estimate
(`distance_discount_per_hop`, floored at `min_discounted_confidence` so
*sustained* evidence at a distant branch could still pull belief there --
just not a single noisy frame). That combination fixed the 22%-agreement
run above -- but introduced a second, worse problem, caught the same way:
by re-running on real video rather than trusting the unit tests alone.

### A second real failure mode: self-referential discounting causes lock-in

Re-running the fixed version on the same real recording made agreement
*worse* (7.3%, down from 22%) and flipped the direction of the error: the
filter now finished stuck at `generation 0` (the trachea) for the back half
of a 916-frame run the vote-based localizer tracked all the way to
`generation 2`. The distance discount above measures distance from the
filter's *own current belief* -- which is fine while that belief is
correct, but self-defeating once it's stale: the longer and more
confidently the filter sits on one branch, the more it discounts *any*
new evidence far from that branch, including the genuine evidence that the
scope has since moved on. It can lock onto a branch and then actively
resist the observations that would correct it, getting more resistant the
longer it stays locked -- exactly what an idealized clean two-state
simulation (a long, unambiguous dwell on one branch, then a long,
unambiguous dwell on a different one) reproduced once tried, even though
neither the earlier synthetic tests nor a clean simulation without that
long-dwell-then-switch shape had caught it.

Isolating the two changes in simulation showed `p_advance = 0.06` **alone**
(no distance discount at all) already resisted the original
occasional-stray-mislabel scenario just as well as the combination did --
the discount was solving a problem `p_advance` had already solved, while
adding a new, worse failure mode on top. So `distance_discount_per_hop`
now defaults to `0.0` (fully off); `p_advance` is the sole default defense
against per-frame noise. The mechanism is still there and exposed via
`TreeMotionFilter(..., distance_discount_per_hop=...)` if you want to
experiment with it, but re-enabling it should be paired with re-running
`scripts/analyze_motion_model.py` on a real recording afterward and
specifically watching for this lock-in signature: the motion-model
location sitting unmoving on one branch across a long stretch where the
vote-based localizer has already moved on.

Both failure modes are now covered by regression tests in
`tests/test_pipeline_smoke.py`
(`test_motion_filter_rejects_occasional_distant_mislabel` for the first,
`test_motion_filter_still_tracks_genuine_sustained_advance` -- built
around a long confident dwell followed by a real transition, specifically
to reproduce the second -- for the lock-in), built directly from these two
real-video runs rather than only the original synthetic scenario. Still
worth re-running `scripts/analyze_motion_model.py` on your own recordings
after any further retuning, rather than trusting the unit tests alone --
that's exactly how both of these were actually found.

Reported location is the highest-probability branch, with its probability
as a confidence value -- deliberately branch-level, not a continuous
in-branch position (see the "Position-aware matching vs. real depth/pose
estimation" note under "Not implemented" for why a true continuous-position
version is a much larger, separate undertaking). Worth noting explicitly:
under repeated confirming observations of the same branch, confidence
settles at a stable equilibrium well under 1.0 rather than climbing to
near-certainty -- that's intentional, not a bug. The predict step keeps
diffusing some belief forward onto that branch's own children every frame
(because the model assumes the scope may still be moving), so the filter
never claims more certainty than a forward-motion assumption actually
supports.

Because it runs alongside the paper's original vote-based `Localizer`
rather than replacing it, the two outputs are directly comparable on the
same run -- useful as an ablation (does the graph-constrained filter track
branch transitions more smoothly / recover faster from a missed detection
than simple majority-voting?) rather than only having one localization
number to report.

## Plugging in what you already have

### Your airway graph (from 3D Slicer)

`AirwayGraph.from_json(path)` expects (see `examples/example_graph.json`):

```json
{
  "coordinate_system": {              // optional
    "origin": [x, y, z],
    "y_axis": [x, y, z],               // trachea direction
    "x_axis": [x, y, z],               // plane of L/R main-bronchus origins
    "z_axis": [x, y, z]                // optional; computed if omitted
  },
  "nodes": [
    { "label": "trachea", "generation": 0, "parent": null, "start": [x,y,z], "end": [x,y,z] },
    { "label": "LMB", "generation": 1, "parent": "trachea", "start": [x,y,z], "end": [x,y,z] },
    ...
  ]
}
```

The loader is tolerant of common key-name variants (`id`/`name` for
`label`, `parent_id`, `p0`/`p1`/`start_node`/`end_node` for start/end,
`level`/`depth` for `generation`, a top-level `"branches"` list instead of
`"nodes"`, a top-level `"root_branch"` id instead of a `parent: null`
sentinel, or an `"edges": [[parent, child], ...]` list instead of a
per-node `"parent"` field) -- see `_LABEL_KEYS` / `_PARENT_KEYS` / etc. at
the top of `graph.py`. Generations are auto-inferred by BFS from the root
if not provided.

If each node also carries a full sampled centerline (`"centerline"` /
`"centerline_mm"`, a list of `[x,y,z]` points from proximal to distal end)
and/or per-sample radius (`"radius"` / `"radius_mm"`), those are picked up
too and used for a more accurate **local tangent at the bifurcation**
(`AirwayNode.start_tangent` / `.end_tangent`) instead of the branch's
overall start->end secant -- this matters for `intersection_angle()` and
`project_children_2d()` on branches that curve before reaching their
bifurcation (common for the trachea and main bronchi in a real segmentation).
Without a centerline, both fall back to the plain secant direction.

**If your Slicer export doesn't match either shape** (e.g. it's a raw VTK
centerline/polydata, or Markups curves), the cleanest integration is a
short adapter script that reads your Slicer export and either (a)
constructs `AirwayNode`/`AirwayGraph` objects directly, or (b) writes out
the canonical JSON above via `AirwayGraph(...).to_json(path)`. That keeps
all Slicer-specific code (vtkMRMLModelNode handling, centerline curve
walking, etc.) out of this package.

### Your YOLOv11 detections

Two ways to feed detections in, both implementing the same `.infer(frame,
frame_idx) -> List[Detection]` interface so `BronchoTrackPipeline` doesn't
care which one you use:

- **`LumenDetector(weights_path, ...)`** -- runs your `.pt` weights live via
  `ultralytics.YOLO(...).predict(...)` per frame.
- **`PrecomputedDetectionSource.from_json(path)`** -- if you already ran
  inference separately, replay saved per-frame boxes:
  `{"0": [{"xyxy": [x1,y1,x2,y2], "confidence": 0.93}, ...], "1": [...]}`.

## Fidelity notes (where this deviates from / interprets the paper)

The paper's text describes the system at a level that leaves some
implementation choices unspecified. Where that happened, the choice made
here is documented in the corresponding module's docstring; the two most
consequential ones:

- **`association.py` candidate matching metric.** The paper doesn't fully
  specify the metric used once graph-predicted 2D positions and image
  detections are compared. Because the system has no absolute depth/scale
  (this matches the paper's own framing -- depth is inferred topologically,
  not metrically), this implementation matches primarily on **angular
  position (bearing) from the anchor**, not absolute distance. If your
  graph's projection axes are mirrored relative to the camera image plane,
  pass `AirwayAssociation(..., flip_v=True)`.
- **`association.py` diameter-ratio matching cue.** Bearing alone can't
  disambiguate two candidate branches that happen to sit in roughly the
  same direction from the anchor. As an additional cue, when **two or more**
  lumens are simultaneously visible, each candidate's **real lumen
  diameter** (`radius_mm` from the graph) is compared to the mean diameter
  of its own peer group of candidate branches, and each detection's
  bounding-box size is independently compared to the mean size of its own
  peer group of currently-visible detections -- the two resulting
  normalized ratios (graph-side and image-side) are then compared directly,
  since normalizing against peers cancels out the unknown camera-to-tissue
  scale factor on both sides without needing to know depth. This is a
  **peer/group comparison, not a distance-from-anchor comparison** -- an
  earlier version compared each candidate's distance-from-anchor-over-
  diameter ratio instead, but real-video ablation (`compare_ratio_ablation.py`
  on `testvideo3.mp4`) showed that version gave no measurable benefit over
  angle-only matching at the default weight, and was actively worse when
  weighted fully (shallower tree traversal, more stalling) -- most likely
  because folding a noisy YOLO-box diameter into a *distance* estimate
  compounds the noise. The peer-comparison version needs only relative
  sizes to be roughly right, and needs an actual peer to compare against,
  so it's skipped (falls back to angle-only) automatically whenever fewer
  than two lumens are visible, not just when radius data is missing.
  Combined with the bearing cost via `diameter_weight` (default 0.5, `0` =
  angle-only, `1` = size-ratio-only). See `association.py`'s module
  docstring for the exact formula.

  The graph-side diameter is not a single point sample at the exact branch
  origin -- it's smoothed (length-weighted mean) over the first
  `diameter_lookahead_fraction` of that candidate branch's own arc length
  (default 0.15, i.e. the first 15%; `--diameter-lookahead` on the CLI, 0 =
  old exact-`radius_at_start` behavior) via the new `AirwayNode.radius_at` /
  `mean_radius` centerline-interpolation helpers in `graph.py`. This is
  intentionally a small, local smoothing correction, not a full "predict
  where the scope is in 3D and render what it should see" system -- see the
  new "Position-aware matching vs. real depth/pose estimation" entry under
  "Not implemented" below for why that bigger version isn't in scope here,
  and what a real version of it would need.

  A related, purely diagnostic field, `FrameResult.approach_trend` /
  `AirwayAssociation.approach_trend` (also written to the JSON log and
  shown on the overlay as "approaching"/"receding"), reports the normalized
  growth trend of the current anchor's own tracked box size over its last
  few frames. It is *not* fed into the matching cost and is not a distance
  or position estimate -- see `association.py`'s module docstring
  ("Approach-trend diagnostic") for exactly why monocular image size alone
  can't be safely inverted into that.
- **`localization.py` voting rule.** The paper's `g^{k-1}` / `g^k` formula
  is reproduced with a concrete, documented reading: a tracklet's "cohort"
  is the currently-visible labeled tracklets sharing its parent; if more
  than one distinct label is visible in a cohort, that cohort votes for
  their shared parent branch (ambiguous -- haven't committed to a specific
  child yet); if exactly one, it votes for itself.
- **No roll correction.** The paper's roll-angle estimation (rotating the
  graph's projected 2D positions to match the bronchoscope's current
  rotation about its own axis, before the bearing comparison above) was
  implemented and then removed -- candidate matching now assumes the
  camera's roll relative to the graph's coordinate frame is fixed at 0.
  This is a real simplification (not a "the paper is ambiguous here" case
  like the others in this list): bearing matching will degrade if the
  scope's actual rotation drifts far from the graph's default orientation.
  See "Not implemented" below.

None of these change the paper's high-level design -- they fill in gaps
left by the summary-level algorithmic description in a way that's internally
consistent and testable (see `tests/test_pipeline_smoke.py`).

## Not implemented

**Loop closure (BronchoTrack-LC)** -- the SLAM-inspired keyframe gallery +
LoFTR dense feature matching for drift recovery -- was explicitly left out
of this build's scope. The gallery structure in `association.py`
(`AirwayAssociation.gallery`) already stores per-branch tracklet history,
so adding keyframe images + a LoFTR matcher there is the natural extension
point if you want it later.

**Roll-angle estimation** was implemented (carina-init roll from the two
main-bronchus lumen centers, plus a per-frame update from the two most
stable tracked lumens, with a signed-angle fix for the paper's
magnitude-only `arccos` formula) and has since been removed by request.
Candidate matching in `association.py` now assumes zero roll throughout
(see "Fidelity notes" above). If you want it back: it was a self-contained
module (no other file depended on more than `initial_roll` /
`pick_stable_pair` / `update_roll` and a `self.roll` float carried through
`AirwayAssociation`), so re-adding it is a matter of restoring that module
and threading a roll estimate back through `_match_candidates`'s graph-point
rotation step, rather than a redesign.

**Position-aware matching vs. real depth/pose estimation.** The
diameter-ratio cue (see "Fidelity notes" above) now smooths its expected
branch diameter over a short lookahead along the candidate's own
centerline, and a diagnostic (`approach_trend`) reports whether the
current anchor's tracked box is growing or shrinking. Both stop short of
what a full "virtual model predicts where a lumen will appear and how big
it will be, matched live against the real image" system would need: a
continuous 6-DoF pose estimate of the scope inside the 3D airway model,
not just a discrete branch label. Monocular image size alone can't be
reliably inverted into that -- it's dominated by the unknown camera-to-
tissue distance, not by a branch's own (usually mild) taper -- which is
why the literature that does full pose/depth-aware bronchoscope tracking
(e.g. CycleGAN-based synthetic depth from real frames, compared against a
CT-derived model) trains a dedicated depth-estimation network rather than
deriving depth from a size heuristic. Building that here would mean: (1) a
renderer for the segmented airway mesh (the BronchoPose dataset's `.obj`
files are a plausible source) that can produce a synthetic view from an
arbitrary camera pose, (2) a continuous pose-tracking module (e.g. an EKF
over 6-DoF state, likely initialized from the existing discrete branch
localization and refined every frame) instead of `AirwayAssociation`'s
current branch-label state, and (3) a way to measure the rendered/expected
lumen diameter at that pose to compare against the real detection -- a
substantially larger subsystem than the two additions above, not a small
extension to `association.py`.

## Tuning knobs worth knowing about

- `MultiLumenTracker(high_conf_thresh, match_thresh_high, match_thresh_low, lam, use_reid)`
- `AirwayAssociation(max_generation_gap, angle_threshold_deg, max_match_cost, diameter_weight, diameter_lookahead_fraction, flip_v)`
  -- `max_match_cost` (default 0.6) replaces the old `max_angular_cost` name
  (still accepted as a deprecated alias); `diameter_weight` (default 0.5)
  controls how much the diameter-ratio cue (see Fidelity notes above)
  contributes vs. bearing; `diameter_lookahead_fraction` (default 0.15)
  controls how much of a candidate branch's own centerline is smoothed over
  for that cue's expected diameter (0 = old exact-`radius_at_start`
  behavior). All three are exposed on the CLI as `--max-match-cost` /
  `--diameter-weight` / `--diameter-lookahead`.
- `Localizer(ambiguous_k, smoothing_window=5, min_frames_to_switch=3)` --
  the latter two control temporal smoothing of the reported location (see
  `localization.py` module docstring); exposed on the CLI as
  `--smoothing-window` / `--smoothing-min-frames`.
- `TreeMotionFilter(graph, p_advance=0.06, p_stay=0.92, p_retreat=0.02, trend_gain=1.5, observation_confidence=0.85, distance_discount_per_hop=0.0, min_discounted_confidence=0.5)`
  -- see "Motion-model filter" above (and its two "real failure mode" notes
  for why `p_advance` defaults so low and why `distance_discount_per_hop`
  defaults to *off* despite being built specifically to fix the first
  failure -- it caused a worse one); `p_advance` is exposed on the CLI as
  `--motion-p-advance`, the others are constructor-only for now (pass via
  `BronchoTrackPipeline(..., motion_model_kwargs={...})`). Disable the
  whole filter with `use_motion_model=False` / `--no-motion-model`.
- `LumenDetector(conf_threshold=0.1, img_size=256)` -- paper's defaults;
  the *tracker* is designed to make use of low-confidence detections
  (second-stage motion-only matching) rather than you raising this
  threshold to clean up detections yourself.
- `AirwayGraphView(graph, canvas_size=(640, 640), margin=50)` -- the live
  airway-graph view (see below); tweak `canvas_size` if 640x640 is too
  small/large, or `margin` if the tree runs close to the window edge.

## Re-ID weights

`reid.py` will run with ImageNet-pretrained ResNet50 as a generic feature
extractor if you don't pass `weights_path`, but for the appearance term to
actually help (vs. just adding noise) you should fine-tune ResNet50 on
crops of your own detected lumens (paper: 3,630 crops, 128x128, softmax
classification loss over patient identity or tracklet identity) and pass
that checkpoint in. If you'd like, this can be built out next as a small
training script once you have exported lumen crops.
