"""Optional overlay rendering for debugging/demo videos. Not part of the
paper's algorithm -- purely a convenience for `pipeline.run_on_video(...,
output_video_path=...)` and the CLI."""
from __future__ import annotations

import numpy as np


def draw_overlay(frame_bgr: np.ndarray, result) -> np.ndarray:
    import cv2

    out = frame_bgr.copy()
    for t in result.tracklets:
        if t.time_since_update != 0:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.last_box.xyxy]
        color = (0, 200, 0) if t.label is not None else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"#{t.track_id} {t.label or '?'}"
        cv2.putText(
            out, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    header = f"loc: {result.location or '-'}  gen: {result.generation if result.generation is not None else '-'}"
    raw = getattr(result, "location_raw", None)
    if raw is not None and raw != result.location:
        header += f"  (raw: {raw})"
    trend = getattr(result, "approach_trend", None)
    if trend is not None and abs(trend) > 0.01:
        header += "  approaching" if trend > 0 else "  receding"
    cv2.putText(
        out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )

    motion_loc = getattr(result, "motion_location", None)
    if motion_loc is not None:
        conf = getattr(result, "motion_confidence", None)
        motion_line = f"motion: {motion_loc}" + (f" ({conf:.2f})" if conf is not None else "")
        cv2.putText(
            out, motion_line, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
        )
    return out
