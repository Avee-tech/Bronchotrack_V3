"""Overlay rendering -- full-mask style: each confirmed lumen is drawn as
its actual segmentation polygon (translucent fill + outline) plus its
branch name, rather than just a center marker. An earlier version of this
module drew only a fixed-radius dot at each tracklet's center point; that
was replaced on explicit request to show the entire mask instead. Not part
of the paper's algorithm -- purely a visualization convenience, forked
from ``bronchotrack.viz`` because the *style* is a deliberate choice
specific to this package.

Each tracklet still gets a color from a fixed, visually distinct palette,
keyed by `track_id % len(palette)` -- so a given physical lumen keeps the
same fill/outline color for as long as its track survives, and a new color
only repeats once more than `len(palette)` tracks are alive simultaneously
(rare in practice).

When a tracklet has a segmentation polygon (`Tracklet.mask`, YOLO-seg
checkpoints only -- see `detection.py`), that exact polygon is drawn: a
translucent fill plus a solid outline. When it doesn't (a plain-detector
checkpoint, or a frame where the polygon didn't survive tracking), this
falls back to the previous fixed-radius center dot, drawn at the same
point `association.AirwayAssociation._detection_center` uses -- the
average of opposing mask-boundary points (see `utils.polygon_opposing_center`)
when a mask is available, or the box center otherwise -- so the fallback
marker is still exactly the point the diameter:distance cue and its
virtual-model verification are computed from. The branch name is drawn
above the mask (or the dot, in the fallback case) only once a tracklet has
actually been assigned a label (see `association.py`).

DISPLAY IS GATED ON THE KALMAN-FILTERED VIRTUAL-MODEL MATCH, NOT JUST
"CURRENTLY VISIBLE" -- an explicit request to run the detector permissively
(a low --conf-threshold, e.g. 0.4, to admit far more raw candidate
detections than a stricter cutoff would) while keeping the on-screen
result clean by leaning on the diameter:distance cue's own Kalman-smoothed
virtual verification (`Tracklet.diameter_distance_match` -- see
`association.py`'s "Virtual verification" docstring section: each
tracklet's own diameter:distance ratio is Kalman-filtered over time via
`scalar_kalman.py`, the same filtering approach as the paper's own box
Kalman filter, and compared against the 3D airway graph's own predicted
ratio at the virtual viewpoint) to do the false-positive filtering
instead. A tracklet only gets a dot once `diameter_distance_match is True`
-- `None` (cue disabled, or not enough graph radius data to evaluate it
yet) and `False` (currently visible, matched to a label, but disagreeing
with the 3D model) are both suppressed, not just unlabeled/unmatched
tracklets. This trades "show everything currently tracked" (the previous
behavior) for "show only what's been confirmed against the 3D model" --
deliberately, so a permissive detector threshold doesn't clutter the
video with noise. Note this means the diameter:distance cue must actually
be enabled (`--distance-diameter-weight` > 0, the default) and the graph
must have radius data for anything to ever display at all; see the CLI's
own docstring.
"""
from __future__ import annotations

import numpy as np

from ..utils import polygon_opposing_center

# A fixed, high-contrast qualitative palette (BGR, OpenCV order) -- chosen
# for pairwise distinguishability rather than any particular meaning; loosely
# in the spirit of the tab10 categorical palette used in the paper's own
# plots (Fig. 4/5), adapted to BGR and to work as line/box colors on a dark
# endoscopic background.
_PALETTE_BGR = [
    (46, 204, 113),   # green
    (219, 152, 52),   # blue
    (60, 76, 231),    # red
    (232, 199, 41),   # cyan-ish
    (154, 89, 182),   # purple
    (44, 165, 245),   # orange
    (185, 128, 41),   # steel blue
    (63, 191, 26),    # lime
    (180, 119, 230),  # pink/magenta
    (98, 98, 246),    # coral
]


def _track_color(track_id: int) -> tuple:
    return _PALETTE_BGR[track_id % len(_PALETTE_BGR)]


_DOT_RADIUS = 8  # pixels -- fallback marker when no mask polygon is available
_MASK_FILL_ALPHA = 0.35  # opacity of the translucent mask fill
_MASK_OUTLINE_THICKNESS = 2


def _center(t) -> tuple:
    """Same center a tracklet's diameter:distance geometry uses -- see
    `association.AirwayAssociation._detection_center` (duplicated here
    rather than imported, since that's a method of a stateful class and
    this only needs the pure geometry)."""
    if t.mask is not None and len(t.mask) >= 3:
        c = polygon_opposing_center(t.mask)
        if c is not None:
            return float(c[0]), float(c[1])
    return float(t.last_box.x_c), float(t.last_box.y_c)


def _mask_points(t) -> "np.ndarray":
    """t.mask as an (N, 1, 2) int32 array, the shape cv2's polygon/fill
    functions expect."""
    return np.round(t.mask).astype(np.int32).reshape((-1, 1, 2))


def draw_overlay(frame_bgr: np.ndarray, result) -> np.ndarray:
    import cv2

    out = frame_bgr.copy()

    to_draw = [
        t
        for t in result.tracklets
        if t.time_since_update == 0 and t.diameter_distance_match is True
        # not (yet) confirmed against the Kalman-filtered virtual-model
        # comparison -- see module docstring's "DISPLAY IS GATED..."
    ]

    # Pass 1: translucent mask fills, composited in one blend so overlapping
    # masks don't double-darken and untouched regions of the frame are
    # never blended against themselves.
    has_any_mask = any(t.mask is not None and len(t.mask) >= 3 for t in to_draw)
    if has_any_mask:
        fill_layer = out.copy()
        for t in to_draw:
            if t.mask is not None and len(t.mask) >= 3:
                cv2.fillPoly(fill_layer, [_mask_points(t)], _track_color(t.track_id))
        out = cv2.addWeighted(fill_layer, _MASK_FILL_ALPHA, out, 1 - _MASK_FILL_ALPHA, 0)

    # Pass 2: outlines (or the fallback center dot), plus labels, drawn
    # crisp on top of the blended fills.
    for t in to_draw:
        color = _track_color(t.track_id)
        has_mask = t.mask is not None and len(t.mask) >= 3

        if has_mask:
            pts = _mask_points(t)
            cv2.polylines(out, [pts], True, color, _MASK_OUTLINE_THICKNESS, cv2.LINE_AA)
            cv2.polylines(out, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
            cx, _ = _center(t)
            label_anchor = (int(round(cx)), int(t.mask[:, 1].min()))
        else:
            # no polygon for this tracklet (plain-detector checkpoint, or
            # the mask didn't survive tracking this frame) -- fall back to
            # the original center-dot marker rather than drawing nothing.
            cx, cy = _center(t)
            cx, cy = int(round(cx)), int(round(cy))
            cv2.circle(out, (cx, cy), _DOT_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), _DOT_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)
            label_anchor = (cx, cy - _DOT_RADIUS)

        if t.label:
            text = t.label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ax, ay = label_anchor
            tx, ty = ax - tw // 2, ay - 8
            cv2.rectangle(out, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), color, -1)
            cv2.putText(
                out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

    # "(stale)" + amber text when location_is_live is False -- this frame's
    # `location` was carried forward from the last real vote, not freshly
    # reconfirmed (see localization.py's "Live vs. carried-forward votes"
    # docstring section). location_is_live is None for callers that never
    # set it (e.g. the pseudocode in report/build_report.py) -- treated the
    # same as live/white rather than flagged, since we have no information
    # either way.
    is_stale = getattr(result, "location_is_live", None) is False
    header = f"loc: {result.location or '-'}{' (stale)' if is_stale else ''}  gen: {result.generation if result.generation is not None else '-'}"
    header_color = (0, 191, 255) if is_stale else (255, 255, 255)  # amber (BGR) vs white
    cv2.putText(
        out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_color, 2, cv2.LINE_AA
    )
    return out
