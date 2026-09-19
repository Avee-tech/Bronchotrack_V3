"""Airway graph M: the pre-operative, patient-specific bronchial tree built
from a CT scan in 3D Slicer (segmentation -> skeletonization -> branch
labeling), loaded here as a JSON node/edge graph.

This is the ``M`` referenced throughout the paper: nodes are branch segments
between bifurcations, edges are parent/child anatomical relationships, and
each node carries a unique anatomical label (e.g. "RB1", "LB6") plus its
generation (0 = trachea, 1 = main bronchi, ...).

--------------------------------------------------------------------------
Expected JSON schema (canonical form written out by ``AirwayGraph.to_json``)
--------------------------------------------------------------------------
{
  "coordinate_system": {          # optional; omit to skip standardization
      "origin": [x, y, z],
      "y_axis": [x, y, z],         # trachea direction
      "x_axis": [x, y, z],         # plane of L/R main bronchus origins
      "z_axis": [x, y, z]          # orthogonal to both
  },
  "nodes": [
      {
        "label": "trachea",
        "generation": 0,
        "parent": null,
        "start": [x, y, z],
        "end":   [x, y, z]
      },
      {
        "label": "LMB",
        "generation": 1,
        "parent": "trachea",
        "start": [x, y, z],
        "end":   [x, y, z]
      },
      ...
  ]
}

``AirwayGraph.from_json`` is deliberately tolerant of common variations you
are likely to have coming out of a 3D Slicer export script -- see the alias
lists in ``_NODE_KEY_ALIASES`` below. If your export uses an "edges" list
of ``[parent_label, child_label]`` pairs instead of a per-node "parent"
field, that is also supported (``"edges": [["trachea", "LMB"], ...]``).

If your graph does not match either shape, the cleanest fix is a ~20 line
adapter script that reads your Slicer export and calls
``AirwayGraph(nodes=[...])`` directly with :class:`AirwayNode` objects, or
dumps to the canonical JSON above via ``AirwayGraph.to_json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .utils import angle_between, project_to_tangent_plane


_LABEL_KEYS = ("label", "id", "name", "branch_label", "branch_id")
_PARENT_KEYS = ("parent", "parent_label", "parent_id")
_START_KEYS = ("start", "start_point", "p0", "proximal", "proximal_point", "start_node")
_END_KEYS = ("end", "end_point", "p1", "distal", "distal_point", "end_node")
_GEN_KEYS = ("generation", "level", "depth", "gen")
_CENTERLINE_KEYS = ("centerline", "centerline_mm", "polyline", "points")
_RADIUS_KEYS = ("radius", "radius_mm", "radii")


def _first_present(d: dict, keys: Tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


@dataclass
class AirwayNode:
    label: str
    generation: int
    parent: Optional[str]
    start: np.ndarray  # (3,) proximal point of the branch segment
    end: np.ndarray  # (3,) distal point (near the next bifurcation)
    children: List[str] = field(default_factory=list)
    centerline: Optional[np.ndarray] = None  # (N, 3) full sampled centerline, if available
    radius: Optional[np.ndarray] = None  # (N,) radius at each centerline sample, if available

    @property
    def direction(self) -> np.ndarray:
        """Unit vector pointing from proximal (start) to distal (end) --
        the branch's net/overall direction (used for coordinate
        standardization and as a fallback when no centerline is available)."""
        d = self.end - self.start
        n = np.linalg.norm(d)
        if n < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return d / n

    def _tangent(self, at_end: bool, window: int = 3) -> np.ndarray:
        """Local tangent direction at the proximal (at_end=False) or distal
        (at_end=True) end of the branch, estimated from the last/first
        `window` centerline samples. This is more anatomically accurate
        than the overall secant `direction` for branches that curve before
        reaching their bifurcation -- which matters for intersection-angle
        filtering and 2D child projection (see AirwayGraph.intersection_angle
        / project_children_2d), both of which care about the branch's
        trajectory *right at* the bifurcation, not its average heading.
        Falls back to `direction` if no (or too short a) centerline exists.
        """
        if self.centerline is None or len(self.centerline) < 2:
            return self.direction
        pts = self.centerline
        w = max(2, min(window, len(pts)))
        if at_end:
            d = pts[-1] - pts[-w]
        else:
            d = pts[w - 1] - pts[0]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return self.direction
        return d / n

    @property
    def start_tangent(self) -> np.ndarray:
        """Direction the branch heads off in, just after its proximal end
        (i.e. right after leaving its parent's bifurcation)."""
        return self._tangent(at_end=False)

    @property
    def end_tangent(self) -> np.ndarray:
        """Direction the branch is heading in, just before its distal end
        (i.e. right at the bifurcation into its own children)."""
        return self._tangent(at_end=True)

    @property
    def radius_at_start(self) -> Optional[float]:
        return float(self.radius[0]) if self.radius is not None and len(self.radius) else None

    @property
    def radius_at_end(self) -> Optional[float]:
        return float(self.radius[-1]) if self.radius is not None and len(self.radius) else None

    def _arc_length_fractions(self) -> Optional[np.ndarray]:
        """Cumulative distance along `centerline`, normalized to [0, 1]
        (0 = proximal/start, 1 = distal/end). None if there's no usable
        centerline to integrate over (missing, too short, or zero length)."""
        if self.centerline is None or len(self.centerline) < 2:
            return None
        seg_lengths = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cum[-1]
        if total < 1e-9:
            return None
        return cum / total

    def radius_at(self, t: float) -> Optional[float]:
        """Radius interpolated at fractional arc-length position `t` along
        this branch (0 = start, 1 = end; clamped to that range). This is the
        'continuous along the centerline' counterpart to the single-point
        `radius_at_start` / `radius_at_end` -- it lets a caller ask "how
        wide is this branch a bit past where it's born" instead of only at
        the exact first sample.

        Falls back to a straight lerp between the first and last radius
        samples if there's no matching centerline to compute true arc-length
        fractions from (e.g. only start/end radii, no full polyline), and to
        the single value if there's only one radius sample at all. Returns
        None if there's no radius data whatsoever.
        """
        if self.radius is None or len(self.radius) == 0:
            return None
        t = min(max(t, 0.0), 1.0)
        if len(self.radius) == 1:
            return float(self.radius[0])
        fracs = self._arc_length_fractions()
        if fracs is None or len(fracs) != len(self.radius):
            return float(self.radius[0] + t * (self.radius[-1] - self.radius[0]))
        return float(np.interp(t, fracs, self.radius))

    def tangent_at(self, t: float, window: int = 3) -> np.ndarray:
        """Local tangent direction at fractional arc-length position `t`
        (0=start, 1=end; clamped) -- the direction counterpart to
        `point_at`/`radius_at`, generalizing `start_tangent`/`end_tangent`
        to an arbitrary point along the branch rather than just its two
        ends. Used to estimate how face-on vs. edge-on a child's opening
        is to a given viewing ray (see `AirwayGraph.
        child_apparent_diameter_at_distance`). Falls back to `direction`
        if there's no (or too short a) centerline to estimate a local
        window from."""
        t = min(max(t, 0.0), 1.0)
        if self.centerline is None or len(self.centerline) < 2:
            return self.direction
        fracs = self._arc_length_fractions()
        if fracs is None:
            return self.direction
        idx = int(np.clip(np.searchsorted(fracs, t), 0, len(fracs) - 1))
        w = max(1, min(window, len(fracs) - 1))
        lo = max(0, idx - w)
        hi = min(len(fracs) - 1, idx + w)
        if lo == hi:
            return self.direction
        d = self.centerline[hi] - self.centerline[lo]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return self.direction
        return d / n

    def point_at(self, t: float) -> np.ndarray:
        """3D point interpolated at fractional arc-length position `t`
        along this branch (0 = start, 1 = end; clamped to that range) --
        the position counterpart to `radius_at`. Used by
        `AirwayGraph.project_child_at_distance` to find "a couple of
        centimetres past the bifurcation" rather than only the branch's
        very first sample or its plain start/end.

        Falls back to a straight lerp between `start` and `end` if there's
        no centerline to walk along.
        """
        t = min(max(t, 0.0), 1.0)
        if self.centerline is None or len(self.centerline) < 2:
            return self.start + t * (self.end - self.start)
        fracs = self._arc_length_fractions()
        if fracs is None:
            return self.start + t * (self.end - self.start)
        return np.array([np.interp(t, fracs, self.centerline[:, k]) for k in range(3)])

    def arc_length_mm(self) -> float:
        """Total centerline length, falling back to the straight-line
        start->end distance if there's no centerline."""
        if self.centerline is not None and len(self.centerline) >= 2:
            fracs = self._arc_length_fractions()
            if fracs is not None:
                return float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum())
        return float(np.linalg.norm(self.end - self.start))

    def mean_radius(self, t0: float = 0.0, t1: float = 1.0, n: int = 8) -> Optional[float]:
        """Length-weighted-ish mean radius over the fractional arc-length
        window [t0, t1] (each clamped to [0, 1]), sampled at `n` evenly
        spaced points via `radius_at`. A smoothed alternative to a single
        point sample -- e.g. for `association.py`'s diameter-ratio cue,
        which cares about "roughly how wide is this branch right near its
        own bifurcation" and shouldn't be overly sensitive to exactly where
        the centerline extraction happened to place its first sample point.
        Returns None if no radius data is available at all.
        """
        t0, t1 = min(max(t0, 0.0), 1.0), min(max(t1, 0.0), 1.0)
        if t1 < t0:
            t0, t1 = t1, t0
        samples = [self.radius_at(t) for t in np.linspace(t0, t1, max(2, n))]
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        return float(np.mean(samples))

    @property
    def midpoint(self) -> np.ndarray:
        return (self.start + self.end) / 2.0


class AirwayGraph:
    """Tree of :class:`AirwayNode`, indexed by anatomical label.

    This is a thin, dependency-free structure on purpose: it should be easy
    to construct directly from whatever your 3D Slicer pipeline already
    produces (skeleton + branch labeling), without needing this repo to
    understand Slicer's own data model (vtkMRMLModelNode, centerline
    curves, etc.) -- do that conversion once, upstream, and hand this class
    a flat list of nodes.
    """

    def __init__(self, nodes: List[AirwayNode], root_label: Optional[str] = None):
        self.nodes: Dict[str, AirwayNode] = {n.label: n for n in nodes}
        self._derive_children()
        self.root_label = root_label or self._infer_root()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _derive_children(self) -> None:
        for node in self.nodes.values():
            node.children = []
        for node in self.nodes.values():
            if node.parent is not None and node.parent in self.nodes:
                self.nodes[node.parent].children.append(node.label)

    def _infer_root(self) -> str:
        for node in self.nodes.values():
            if node.parent is None:
                return node.label
        # fall back to the lowest-generation node
        return min(self.nodes.values(), key=lambda n: n.generation).label

    @classmethod
    def from_json(cls, path: str) -> "AirwayGraph":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_path(
        cls,
        path: str,
        rebuild_graph: bool = False,
        graph_cache_path: Optional[str] = None,
        **mesh_build_kwargs,
    ) -> "AirwayGraph":
        """Like `from_json`, but also accepts a raw surface mesh (.vtk/
        .vtp/.stl) directly -- the entry point behind both CLIs' `--graph`
        flag, so a new patient is "upload the model instead of the graph
        and work as normal": pass the mesh path where you'd otherwise pass
        a pre-built graph JSON, and the mesh->graph conversion (weld/BFS/
        merge/filter/label -- see `mesh_to_graph`'s module docstring) runs
        automatically.

        A mesh is only ever converted once: the built graph is cached to
        `graph_cache_path` (default: `mesh_to_graph.default_cache_path`,
        i.e. `<mesh_stem>_graph.json` next to the mesh itself) and reused
        on every later run against the same mesh, unless `rebuild_graph`
        is set or the mesh file has been touched more recently than the
        cache. `**mesh_build_kwargs` (weld_tol_mm, min_branch_length_mm,
        min_radius_mm, smooth_iterations, advancement_ratio,
        cap_open_radius_mm) are forwarded to `mesh_to_graph.build_graph`
        and ignored entirely when `path` is already a graph JSON.

        Building from a mesh needs `vtk`/`vmtk` importable in this Python
        environment -- NOT a hard dependency of the rest of this package,
        so `mesh_to_graph` is only imported here, lazily, the moment a
        mesh path is actually given; see that module's own docstring for
        install instructions if this raises ImportError.
        """
        # Checked as a plain string tuple (NOT imported from mesh_to_graph)
        # so that pointing --graph at an ordinary graph JSON never triggers
        # mesh_to_graph's vtk/vmtk import at all -- only a mesh path does.
        if path.lower().endswith((".vtk", ".vtp", ".stl")):
            from .mesh_to_graph import build_and_cache_graph

            data = build_and_cache_graph(
                path,
                cache_path=graph_cache_path,
                force_rebuild=rebuild_graph,
                **mesh_build_kwargs,
            )
            return cls.from_dict(data)
        return cls.from_json(path)

    @classmethod
    def from_dict(cls, data: dict) -> "AirwayGraph":
        raw_nodes = data.get("nodes", data.get("branches"))
        if raw_nodes is None:
            raise ValueError(
                "Airway graph JSON must have a top-level 'nodes' (or 'branches') list"
            )
        edges = data.get("edges")  # optional [[parent, child], ...] form

        parent_of: Dict[str, Optional[str]] = {}
        if edges:
            for parent_label, child_label in edges:
                parent_of[child_label] = parent_label

        nodes: List[AirwayNode] = []
        for raw in raw_nodes:
            label = _first_present(raw, _LABEL_KEYS)
            if label is None:
                raise ValueError(f"Airway graph node missing a label field: {raw}")
            label = str(label)

            parent = _first_present(raw, _PARENT_KEYS)
            if parent is None:
                parent = parent_of.get(label)
            parent = str(parent) if parent is not None else None

            start = _first_present(raw, _START_KEYS)
            end = _first_present(raw, _END_KEYS)
            if start is None or end is None:
                raise ValueError(
                    f"Airway graph node '{label}' is missing start/end 3D "
                    f"coordinates (looked for keys {_START_KEYS} / {_END_KEYS})"
                )

            generation = _first_present(raw, _GEN_KEYS)

            centerline_raw = _first_present(raw, _CENTERLINE_KEYS)
            centerline = (
                np.asarray(centerline_raw, dtype=np.float64) if centerline_raw else None
            )
            radius_raw = _first_present(raw, _RADIUS_KEYS)
            radius = np.asarray(radius_raw, dtype=np.float64) if radius_raw else None

            nodes.append(
                AirwayNode(
                    label=label,
                    generation=int(generation) if generation is not None else -1,
                    parent=parent,
                    start=np.asarray(start, dtype=np.float64),
                    end=np.asarray(end, dtype=np.float64),
                    centerline=centerline,
                    radius=radius,
                )
            )

        root_branch = data.get("root_branch")
        root_label = str(root_branch) if root_branch is not None else None
        graph = cls(nodes, root_label=root_label)

        if any(n.generation < 0 for n in graph.nodes.values()):
            graph._infer_generations()

        coord_sys = data.get("coordinate_system")
        if coord_sys:
            graph.standardize_coordinates(
                origin=np.asarray(coord_sys["origin"], dtype=np.float64),
                y_axis=np.asarray(coord_sys["y_axis"], dtype=np.float64),
                x_axis=np.asarray(coord_sys["x_axis"], dtype=np.float64),
                z_axis=np.asarray(coord_sys.get("z_axis"))
                if coord_sys.get("z_axis") is not None
                else None,
            )
        return graph

    def _infer_generations(self) -> None:
        """BFS generation numbering from the root when the source data didn't
        provide explicit generation numbers."""
        root = self._infer_root()
        self.nodes[root].generation = 0
        queue = [root]
        while queue:
            label = queue.pop(0)
            gen = self.nodes[label].generation
            for child in self.nodes[label].children:
                self.nodes[child].generation = gen + 1
                queue.append(child)

    def standardize_coordinates(
        self,
        origin: np.ndarray,
        y_axis: np.ndarray,
        x_axis: np.ndarray,
        z_axis: Optional[np.ndarray] = None,
    ) -> None:
        """Re-express every node's start/end points in the paper's standard
        frame: y aligned with the trachea direction, x in the plane formed by
        the left/right main-bronchus origins, z orthogonal to both."""
        y = y_axis / (np.linalg.norm(y_axis) + 1e-12)
        x = x_axis - np.dot(x_axis, y) * y  # orthogonalize against y
        x = x / (np.linalg.norm(x) + 1e-12)
        z = np.cross(x, y) if z_axis is None else z_axis / (np.linalg.norm(z_axis) + 1e-12)
        rot = np.stack([x, y, z], axis=0)  # rows are the new basis vectors

        for node in self.nodes.values():
            node.start = rot @ (node.start - origin)
            node.end = rot @ (node.end - origin)
            if node.centerline is not None:
                node.centerline = (node.centerline - origin) @ rot.T

    def to_json(self, path: str) -> None:
        data = {
            "nodes": [
                {
                    "label": n.label,
                    "generation": n.generation,
                    "parent": n.parent,
                    "start": n.start.tolist(),
                    "end": n.end.tolist(),
                    **({"centerline": n.centerline.tolist()} if n.centerline is not None else {}),
                    **({"radius": n.radius.tolist()} if n.radius is not None else {}),
                }
                for n in self.nodes.values()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # tree queries used by tracker.py / association.py / localization.py
    # ------------------------------------------------------------------
    def __contains__(self, label: str) -> bool:
        return label in self.nodes

    def get(self, label: str) -> AirwayNode:
        return self.nodes[label]

    def parent(self, label: str) -> Optional[str]:
        return self.nodes[label].parent

    def children(self, label: str) -> List[str]:
        return list(self.nodes[label].children)

    def siblings(self, label: str) -> List[str]:
        p = self.parent(label)
        if p is None:
            return []
        return [c for c in self.children(p) if c != label]

    def generation(self, label: str) -> int:
        return self.nodes[label].generation

    def ancestor(self, label: str, k: int) -> Optional[str]:
        """g^k(l): the k-th ancestor of branch l (k=0 -> l itself,
        k=1 -> parent, k=2 -> grandparent, ...). Returns None if the tree
        does not go up that far (i.e. clamps at the root)."""
        cur = label
        for _ in range(k):
            p = self.parent(cur)
            if p is None:
                return cur
            cur = p
        return cur

    def subtree_labels(self, label: str) -> Set[str]:
        """M_{l_i}: the set of all labels in the subgraph rooted at `label`
        (label itself plus every descendant)."""
        out: Set[str] = set()
        stack = [label]
        while stack:
            cur = stack.pop()
            out.add(cur)
            stack.extend(self.children(cur))
        return out

    def intersection_angle(self, parent_label: str, child_label: str) -> float:
        """Angle (radians) between the parent branch's trajectory *at the
        bifurcation* (end_tangent) and the child branch's initial takeoff
        direction (start_tangent) -- used to filter unlikely-to-be-visible
        candidate children (steep angles are less likely to be visible from
        the parent's lumen view). Uses local tangents from the centerline
        when available (more accurate for curved branches), falling back to
        each branch's overall start->end secant otherwise."""
        return angle_between(
            self.nodes[parent_label].end_tangent, self.nodes[child_label].start_tangent
        )

    def project_children_2d(self, parent_label: str) -> Dict[str, np.ndarray]:
        """Project the children of `parent_label` onto the 2D plane
        orthogonal to the parent branch's own axis, centered at the parent's
        distal (end) point -- i.e. the tangent plane an observer looking
        down the parent lumen would see the children's openings arranged in.
        Returns {child_label: (u, v)}.
        """
        parent = self.nodes[parent_label]
        origin = parent.end
        axis = parent.end_tangent
        children = self.children(parent_label)
        if not children:
            return {}
        pts = np.stack([self.nodes[c].midpoint for c in children], axis=0)
        coords_2d = project_to_tangent_plane(pts, origin, axis)
        return {c: coords_2d[i] for i, c in enumerate(children)}

    def project_children_at_distance(
        self, parent_label: str, distance_mm: float
    ) -> Dict[str, np.ndarray]:
        """Like `project_children_2d`, but instead of projecting each
        child's overall midpoint, projects the 3D point `distance_mm` along
        each child's *own* centerline from its start (clamped to that
        child's actual length if it's shorter than `distance_mm`).

        This is the "virtual viewpoint a couple of centimetres past the
        bifurcation" used by `paper_exact.association`'s diameter:distance
        cue: right at a bifurcation (distance_mm=0, closer to what
        `project_children_2d`'s plain midpoint approximates for a short
        branch) a child lumen is often foreshortened or oblique, in exactly
        the way a real bronchoscopy view is once the scope has actually
        advanced a bit past the fork -- sampling a bit further down the
        branch gives a geometry closer to what the camera will actually be
        looking at by the time it matters, and lets the same distance be
        used consistently for both the projected bearing/distance here and
        the expected diameter (`AirwayNode.radius_at` at the same t).

        Returns {child_label: (u, v)} in the same parent-tangent-plane
        coordinates as `project_children_2d`.
        """
        parent = self.nodes[parent_label]
        origin = parent.end
        axis = parent.end_tangent
        children = self.children(parent_label)
        if not children:
            return {}
        pts = []
        for c in children:
            node = self.nodes[c]
            length = node.arc_length_mm()
            t = 0.0 if length <= 1e-9 else min(distance_mm, length) / length
            pts.append(node.point_at(t))
        pts = np.stack(pts, axis=0)
        coords_2d = project_to_tangent_plane(pts, origin, axis)
        return {c: coords_2d[i] for i, c in enumerate(children)}

    def child_radius_at_distance(self, child_label: str, distance_mm: float) -> Optional[float]:
        """True (straight-on, unoccluded) radius of `child_label`'s branch
        at `distance_mm` along its own centerline from its start (clamped
        to its actual length) -- the radius counterpart to
        `project_children_at_distance`, sampled at the exact same virtual
        point rather than an independent lookahead-fraction window. See
        `child_apparent_diameter_at_distance` for the foreshortening-
        corrected version actually used by the diameter:distance cue."""
        node = self.nodes[child_label]
        length = node.arc_length_mm()
        t = 0.0 if length <= 1e-9 else min(distance_mm, length) / length
        return node.radius_at(t)

    def child_apparent_diameter_at_distance(
        self, parent_label: str, child_label: str, distance_mm: float
    ) -> Optional[float]:
        """The diameter `child_label`'s opening would actually *appear* to
        have from `parent_label`'s own vantage point, at the virtual
        viewpoint `distance_mm` down the child's centerline -- NOT its
        true, straight-on cross-sectional diameter.

        A lumen opening's face is roughly perpendicular to its own local
        centerline direction (like the mouth of a tube). The camera itself
        doesn't move per child: it's a single scope sitting at
        `parent_label`'s own bifurcation, looking forward along the
        parent's own trajectory there (`parent.end_tangent`) -- the same
        fixed axis `project_children_2d`/`project_children_at_distance`/
        `intersection_angle` already use for exactly this "the scope hasn't
        committed to a child yet, it's looking at all of them from one
        vantage point" reason. A child whose opening, at the sampled
        point, faces back toward that axis (its local tangent nearly
        parallel to the parent's forward direction -- i.e. it continues
        nearly straight ahead) is seen close to face-on, at close to its
        full diameter; a child whose opening has turned off to the side
        presents it increasingly edge-on -- the near airway wall
        increasingly occludes it -- and it reads smaller, exactly the
        effect a real bronchoscopy frame shows once you're a couple of
        centimetres out from a fork but not squarely facing every child.
        (Deliberately NOT a ray recomputed per child from the bifurcation
        to that child's own virtual point: for a roughly straight branch
        that ray is itself nearly parallel to the branch's own tangent by
        construction, which would make every branch read as face-on
        regardless of how sharply it actually forks off the parent --
        exactly the case this correction needs to catch.)
        This approximates that as a circular aperture's projection: true
        diameter times the cosine of the angle between the parent's
        forward axis and the child's own local tangent at the sampled
        point (from `AirwayNode.tangent_at`) -- 1.0 (no foreshortening)
        when the child continues straight ahead of the parent, shrinking
        toward 0 as its opening turns away from the camera.

        Returns None if there's no radius data for `child_label` at all.
        """
        parent = self.nodes[parent_label]
        child = self.nodes[child_label]
        length = child.arc_length_mm()
        t = 0.0 if length <= 1e-9 else min(distance_mm, length) / length

        radius = child.radius_at(t)
        if radius is None:
            return None

        view_dir = parent.end_tangent
        view_norm = np.linalg.norm(view_dir)
        if view_norm < 1e-9:
            return 2.0 * radius  # degenerate (no well-defined parent axis)
        view_dir = view_dir / view_norm

        local_tangent = child.tangent_at(t)
        foreshortening = abs(float(np.dot(view_dir, local_tangent)))  # == |cos(angle_between(...))|
        return 2.0 * radius * foreshortening

    def child_apparent_equivalent_diameter_at_distance(
        self, parent_label: str, child_label: str, distance_mm: float
    ) -> Optional[float]:
        """Area-based counterpart to `child_apparent_diameter_at_distance`,
        used by ``fusion.ratio_id``'s whole-mask ratio method instead of
        that method's longest-diameter counterpart (used by
        ``paper_exact.association``'s point-based cue).

        Same viewing geometry (a circular aperture of true radius `radius`,
        viewed off-axis from `parent_label`'s own fixed forward direction
        at the angle between that axis and the child's local tangent at
        the sampled point) but a different projection: viewed off-axis, a
        true circle of radius r projects to an ELLIPSE with one axis
        unforeshortened (semi-axis r, lying in the plane of the turn) and
        the other foreshortened by cos(theta) (semi-axis r*cos(theta)).
        That ellipse's area is pi*r*(r*cos(theta)) = pi*r^2*cos(theta), so
        its area-equivalent circle (`utils.mask_equivalent_diameter`'s
        same "circle with the same area" idea, applied to the *predicted*
        opening rather than a detected mask) has diameter
        2*r*sqrt(cos(theta)) -- a shallower falloff with angle than the
        longest-diameter version's plain 2*r*cos(theta), because only one
        of the ellipse's two axes actually shrinks.

        This is deliberately a different formula from
        `child_apparent_diameter_at_distance`, not just a rename: the two
        feed genuinely different image-side measurements
        (`utils.polygon_longest_diameter`'s Feret diameter vs.
        `utils.mask_equivalent_diameter`'s area-equivalent diameter) and
        should be compared against a graph-side prediction using the same
        projection assumption, or the cross-domain comparison is
        apples-to-oranges.

        Returns None if there's no radius data for `child_label` at all.
        """
        parent = self.nodes[parent_label]
        child = self.nodes[child_label]
        length = child.arc_length_mm()
        t = 0.0 if length <= 1e-9 else min(distance_mm, length) / length

        radius = child.radius_at(t)
        if radius is None:
            return None

        view_dir = parent.end_tangent
        view_norm = np.linalg.norm(view_dir)
        if view_norm < 1e-9:
            return 2.0 * radius  # degenerate (no well-defined parent axis)
        view_dir = view_dir / view_norm

        local_tangent = child.tangent_at(t)
        cos_theta = abs(float(np.dot(view_dir, local_tangent)))
        return 2.0 * radius * float(np.sqrt(cos_theta))

    def root(self) -> str:
        return self.root_label

    def all_labels(self) -> List[str]:
        return list(self.nodes.keys())
