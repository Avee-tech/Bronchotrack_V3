"""Live visualization of the airway graph M and the bronchoscope's current
location within it. This is not part of the paper's algorithm -- it's a
debugging/demo convenience analogous to viz.py's tracked-overlay video,
except it draws the *graph* rather than the *camera frame*: a schematic
view of the bronchial tree with the currently-localized branch highlighted
and every branch labeled with its number/id, so you can watch the scope's
progress down the tree alongside the live video.

Projection approach
--------------------
The graph is defined in 3D (from CT-derived centerlines), but we only have
a 2D screen to show it on. Rather than requiring `coordinate_system` to be
set on the graph (useful for the paper's roll/bearing math, but your real
`airway_graph.json` export doesn't provide one), this fits a 2D projection
plane by PCA over every branch's centerline/endpoint points: the plane
spanned by the two directions of greatest positional spread in the tree.
For a bronchial tree this recovers something close to a natural "unrolled"
front-on view without needing any external anatomical axis info, and works
for *any* graph regardless of how it happens to be oriented in its own
coordinate system.

The projection (and the resulting fixed pixel layout of every branch) is
computed once at construction, since the tree's geometry doesn't change at
runtime -- `render()` is cheap and just redraws colors/highlighting/labels
on top of that cached layout, once per frame.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

from .graph import AirwayGraph

DEFAULT_CANVAS_SIZE = (640, 640)
DEFAULT_MARGIN = 50


class AirwayGraphView:
    def __init__(
        self,
        graph: AirwayGraph,
        canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
        margin: int = DEFAULT_MARGIN,
    ):
        self.graph = graph
        self.canvas_w, self.canvas_h = canvas_size
        self.margin = margin

        self._pca_mean = np.zeros(3)
        self._basis = self._fit_projection_plane()
        polylines_3d = self._collect_polylines()
        polylines_2d_raw = {label: self._project(pts) for label, pts in polylines_3d.items()}
        self._scale, self._offset = self._fit_to_canvas(polylines_2d_raw)
        self._polylines_px: Dict[str, np.ndarray] = {
            label: self._to_pixels(pts) for label, pts in polylines_2d_raw.items()
        }
        self._label_anchor_px: Dict[str, np.ndarray] = {
            label: pts[len(pts) // 2] for label, pts in self._polylines_px.items()
        }
        self._max_gen = max((n.generation for n in graph.nodes.values()), default=1)
        self._max_gen = max(self._max_gen, 1)

    # ------------------------------------------------------------------
    # one-time setup: 3D graph -> fitted 2D pixel coordinates
    # ------------------------------------------------------------------
    def _fit_projection_plane(self) -> np.ndarray:
        """PCA over all branch geometry -> (2, 3) basis of the two
        highest-variance directions (see module docstring)."""
        pts = []
        for node in self.graph.nodes.values():
            pts.append(np.atleast_2d(node.start))
            pts.append(np.atleast_2d(node.end))
            if node.centerline is not None and len(node.centerline):
                pts.append(np.atleast_2d(node.centerline))
        all_pts = np.concatenate(pts, axis=0)
        self._pca_mean = all_pts.mean(axis=0)
        centered = all_pts - self._pca_mean
        # SVD on centered points: right singular vectors are the principal axes
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[:2]  # (2, 3), top-2 principal directions, most spread first

    def _project(self, pts_3d: np.ndarray) -> np.ndarray:
        centered = pts_3d - self._pca_mean
        return centered @ self._basis.T  # (N, 2)

    def _collect_polylines(self) -> Dict[str, np.ndarray]:
        out = {}
        for label, node in self.graph.nodes.items():
            if node.centerline is not None and len(node.centerline) >= 2:
                out[label] = node.centerline
            else:
                out[label] = np.stack([node.start, node.end], axis=0)
        return out

    def _fit_to_canvas(self, polylines_2d: Dict[str, np.ndarray]) -> Tuple[float, np.ndarray]:
        all_2d = np.concatenate(list(polylines_2d.values()), axis=0)
        lo = all_2d.min(axis=0)
        hi = all_2d.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        avail_w = max(self.canvas_w - 2 * self.margin, 1)
        avail_h = max(self.canvas_h - 2 * self.margin, 1)
        scale = min(avail_w / span[0], avail_h / span[1])
        center_data = (lo + hi) / 2.0
        center_canvas = np.array([self.canvas_w / 2.0, self.canvas_h / 2.0])
        offset = center_canvas - center_data * scale
        return scale, offset

    def _to_pixels(self, pts_2d: np.ndarray) -> np.ndarray:
        px = pts_2d * self._scale + self._offset
        px = px.copy()
        px[:, 1] = self.canvas_h - px[:, 1]  # flip: image y grows downward
        return px.astype(np.int32)

    # ------------------------------------------------------------------
    # per-frame rendering
    # ------------------------------------------------------------------
    def render(
        self,
        current_location: Optional[str] = None,
        current_generation: Optional[int] = None,
        visited: Optional[Set[str]] = None,
    ) -> np.ndarray:
        """Render the graph view for the current frame: every branch drawn
        as its (projected) centerline, color-coded by generation, labeled
        with its branch id/number, with `current_location` highlighted
        (thicker line, bright color, marker at its distal end) if given.

        `visited`, if given, is a set of branch labels visited at any point
        so far (e.g. the association module's gallery keys) -- drawn as a
        highlighted trail (medium thickness, brighter than an unvisited
        branch but dimmer than `current_location`) so the rendered path
        traces the scope's route through the tree so far, not just its
        current position -- closer to the paper's own Fig. 5(a) "driving
        path" airway diagrams than a single highlighted node alone."""
        import cv2

        visited = visited or set()
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 30, dtype=np.uint8)

        for label, node in self.graph.nodes.items():
            px = self._polylines_px[label]
            is_current = label == current_location
            is_visited = label in visited and not is_current
            color = self._generation_color(node.generation, dim=not (is_current or is_visited))
            if is_current:
                thickness = 4
            elif is_visited:
                thickness = 3
            else:
                thickness = 2
            pts = px.reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)

        for label in self.graph.nodes:
            anchor = self._label_anchor_px[label]
            is_current = label == current_location
            is_visited = label in visited and not is_current
            if is_current:
                text_color, font_scale = (255, 255, 255), 0.45
            elif is_visited:
                text_color, font_scale = (215, 215, 215), 0.38
            else:
                text_color, font_scale = (170, 170, 170), 0.35
            cv2.putText(
                canvas,
                str(label),
                (int(anchor[0]) + 4, int(anchor[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

        if current_location is not None and current_location in self._polylines_px:
            px = self._polylines_px[current_location]
            marker = (int(px[-1][0]), int(px[-1][1]))
            cv2.circle(canvas, marker, 7, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, marker, 9, (20, 20, 20), 1, cv2.LINE_AA)

        header = f"location: {current_location or '-'}"
        if current_generation is not None:
            header += f"  gen: {current_generation}"
        if visited:
            header += f"  visited: {len(visited)}"
        cv2.putText(
            canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        return canvas

    def _generation_color(self, gen: int, dim: bool) -> Tuple[int, int, int]:
        import cv2

        t = 0.0 if self._max_gen <= 0 else min(max(gen / self._max_gen, 0.0), 1.0)
        hue = int(20 + t * 140)  # orange (trachea, gen 0) -> green -> blue (deep generations)
        sat = 90 if dim else 200
        val = 110 if dim else 230
        hsv = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
