"""Overlay rendering for the fusion pipeline -- shows both the currently
tracked lumens AND the predicted/candidate child branches, per the user's
explicit request ("display the child branhces on the live video"), not
just the single currently-occupied one the way `paper_exact.viz` does.

Two things are drawn, deliberately kept visually distinct:

1. Tracked lumens -- one dot per currently-visible tracklet at
   `bronchotrack_id.detection_point` (the same opposing-boundary center
   the point-based identification cue itself uses), colored by a fixed
   per-track_id palette (same convention as `paper_exact.viz`, duplicated
   rather than imported since it's a few lines of pure styling with no
   shared state). A tracklet that has actually been committed to a label
   (`Tracklet.label` is set -- see `association.FusionAssociation
   ._try_commit`) gets that name drawn above its dot.

2. A candidate HUD panel -- one row per currently-scored candidate child
   of `result.location` (from `FrameResult.candidates`), each showing its
   name, a Kalman-fused confidence bar, and its ETA (`tau_seconds`, from
   the matched tracklet's own diameter-growth motion model) if it's
   currently being approached. This is NOT drawn as dots placed at the
   graph-projected candidate positions overlaid on the real camera image:
   `AirwayGraph.project_children_2d`'s 2D coordinates live in an
   arbitrary local tangent-plane basis with no established correspondence
   to real pixel coordinates (no camera intrinsics/extrinsics are
   available anywhere in this pipeline -- see `paper_exact.association`'s
   own docstring on why depth/position is treated as topological, not
   metric). Drawing a fabricated dot at a plausible-looking but
   physically ungrounded screen position would misrepresent the pipeline
   as having a calibrated camera model it doesn't have; a HUD list makes
   exactly the same information available (which candidates, how
   confident, how soon) without that false precision.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..graph import AirwayGraph
from .bronchotrack_id import detection_point

if TYPE_CHECKING:
    from .pipeline import FrameResult

_PALETTE_BGR = [
    (46, 204, 113),
    (219, 152, 52),
    (60, 76, 231),
    (232, 199, 41),
    (154, 89, 182),
    (44, 165, 245),
    (185, 128, 41),
    (63, 191, 26),
    (180, 119, 230),
    (98, 98, 246),
]
_DOT_RADIUS = 8
_HUD_BAR_W = 160
_HUD_BAR_H = 12
_HUD_ROW_H = 30
_HUD_ORIGIN = (10, 56)  # below the location header


def _track_color(track_id: int) -> tuple:
    return _PALETTE_BGR[track_id % len(_PALETTE_BGR)]


def draw_overlay(frame_bgr: np.ndarray, result: "FrameResult", graph: AirwayGraph) -> np.ndarray:
    import cv2

    out = frame_bgr.copy()

    for t in result.tracklets:
        if t.time_since_update != 0:
            continue
        color = _track_color(t.track_id)
        cx, cy = detection_point(t)
        cx, cy = int(round(float(cx))), int(round(float(cy)))

        cv2.circle(out, (cx, cy), _DOT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), _DOT_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)

        if t.label:
            text = t.label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tx, ty = cx - tw // 2, cy - _DOT_RADIUS - 8
            cv2.rectangle(out, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), color, -1)
            cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    header = f"loc: {result.location or '-'}  gen: {result.generation if result.generation is not None else '-'}"
    cv2.putText(out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    _draw_candidate_hud(out, result)
    return out


def _draw_candidate_hud(out: np.ndarray, result: "FrameResult") -> None:
    import cv2

    x0, y0 = _HUD_ORIGIN
    ranked = sorted(result.candidates.items(), key=lambda kv: -kv[1].fused_confidence)
    for i, (label, cc) in enumerate(ranked):
        y = y0 + i * _HUD_ROW_H
        conf = max(0.0, min(1.0, cc.fused_confidence))

        color = _track_color(cc.matched_track_id) if cc.matched_track_id is not None else (200, 200, 200)
        cv2.putText(out, label, (x0, y + _HUD_BAR_H), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        bar_x = x0 + 60
        cv2.rectangle(out, (bar_x, y), (bar_x + _HUD_BAR_W, y + _HUD_BAR_H), (90, 90, 90), 1, cv2.LINE_AA)
        fill_w = int(_HUD_BAR_W * conf)
        if fill_w > 0:
            cv2.rectangle(out, (bar_x, y), (bar_x + fill_w, y + _HUD_BAR_H), color, -1, cv2.LINE_AA)

        eta_text = f"{conf * 100:.0f}%"
        tau = result.tau_seconds.get(cc.matched_track_id) if cc.matched_track_id is not None else None
        if tau is not None:
            eta_text += f"  ETA {tau:.1f}s"
        cv2.putText(
            out,
            eta_text,
            (bar_x + _HUD_BAR_W + 8, y + _HUD_BAR_H),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
