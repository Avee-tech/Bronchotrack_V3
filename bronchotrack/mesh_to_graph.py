"""Build a BronchoTrack airway graph directly from a CT-derived airway
surface mesh (a .vtk/.vtp/.stl model exported from 3D Slicer's Segment
Editor, or any other segmentation tool) -- the shared implementation behind
both ``scripts/build_airway_graph.py`` (the standalone CLI) and the
``--graph`` flag on ``bronchotrack.cli`` / ``bronchotrack.paper_exact.cli``
accepting a mesh path directly instead of a pre-built graph JSON (see
``AirwayGraph.from_path``).

WHY NOT SLICER'S OWN CENTERLINE EXTRACTION -- Slicer's Extract Centerline
module repeatedly failed on this dataset (that's why this exists at all).
This instead uses vmtk's vmtkNetworkExtraction directly, which is a
different (simpler, more robust for this kind of noisy, non-capped airway
tree) algorithm -- it estimates a topological skeleton + per-point radius
straight from the closed surface without needing capped/sealed openings or
manually-picked source/target points.

PIPELINE
--------
1. Load the surface mesh, keep only its largest connected component (drops
   small segmentation-noise islands), lightly smooth it.
2. Run vmtk's vmtkNetworkExtraction -> a raw "network" of many short
   polyline segments ("cells"), each with a per-point inscribed-sphere
   Radius and a Topology [start_id, end_id] junction pair (-1 = free/leaf
   end). This raw network is typically noisy: dozens to hundreds of tiny
   spurious cells, and junction ids that don't always coincide exactly in
   space where they anatomically should.
3. WELD: cluster every raw endpoint (both real junction ids and -1 leaf
   ends) by spatial proximity (default tolerance 2mm), rather than trusting
   the raw Topology ids directly -- this reconstructs a clean connectivity
   graph purely from geometry, robust to whatever ids vmtk happened to
   assign.
4. Pick a root branch = the cell with the largest median radius (the
   trachea is reliably the widest structure in the tree). Orient it using
   whichever end is a true dead-end (degree 1) as the proximal/start side.
5. BFS the welded cell-adjacency graph from the root to assign
   generation + parent to every reachable cell. Anything not reached
   (disconnected segmentation noise elsewhere in the mesh) is dropped.
6. FILTER: drop spurious near-zero-length or too-thin branches (default
   < 1.0mm long or < 0.4mm median radius) and splice their children onto
   the nearest surviving ancestor, instead of just deleting the subtree.
7. LABEL: root = "trachea". At the root, its two children are labeled "R"
   / "L" by median radius (the right main bronchus is anatomically wider
   and shorter than the left -- a much more reliable heuristic than
   trusting the mesh file's own +/-X coordinate convention, which isn't
   guaranteed consistent across exports). At every deeper level, children
   are labeled by appending "1", "2", "3", ... to the parent's label, in
   descending order of median radius (thicker branch = lower digit).
8. Return the graph in the exact schema ``AirwayGraph.from_dict()`` expects:
   a top-level {"nodes": [...]} with one entry per branch (label,
   generation, parent, start, end, centerline, radius, length_mm,
   median_radius_mm, _source_cell_id).

This was reverse-engineered/rebuilt from an original interactive session's
intermediate artifacts and validated by re-running it on the original model
and confirming: same root ("trachea"), same top-level R/L assignment (by
radius), matching branch count in the same ballpark, and matching radius/
length ranges as the hand-built graph already in this repo. It will NOT
reproduce a hand-built graph's deeper digit-suffix numbering (e.g. "L11"
vs "L12") byte-for-byte, since that ordering wasn't itself based on a
single consistent rule in the original ad-hoc build -- but the tree
topology, generation numbers, and R/L split are robust and consistent.

REQUIREMENTS -- DELIBERATELY NOT A HARD DEPENDENCY OF THE PACKAGE
-------------------------------------------------------------------
Needs `vmtk` (which bundles its own VTK build) -- NOT the plain `vtk`
package alone. Nothing else in ``bronchotrack`` imports vtk/vmtk at all
(the running pipeline only ever needs the already-built graph JSON), so
this module is imported *lazily* -- only when a caller actually hands a
mesh path to ``AirwayGraph.from_path`` / ``build_and_cache_graph`` -- and
importing it with vmtk missing raises a clear, actionable ImportError
rather than crashing the whole CLI for users who never touch a mesh file.
Install into its own virtualenv if you'd rather not put vmtk in your main
one:

    python3 -m venv vmtk_env
    vmtk_env/bin/pip install vmtk

Then run the standalone script with THAT interpreter, e.g.:

    vmtk_env/bin/python3 scripts/build_airway_graph.py \\
        --mesh NewPatientModel.vtk \\
        --out patient_airway_graph/airway_graph_newpatient.json

Everything below only needs numpy + vtk (both vmtk dependencies already
provide these) -- no scipy required.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError as e:  # pragma: no cover -- exercised via the CLI's own error path
    raise ImportError(
        "Building a graph from a mesh needs `vtk` (and, for the network-"
        "extraction step, `vmtk`) importable in this Python environment. "
        "Install with `pip install vmtk` (it bundles a compatible vtk) "
        "here, or build the graph once in a separate vmtk-enabled venv "
        "with `scripts/build_airway_graph.py` and pass the resulting "
        ".json to --graph instead."
    ) from e


# Mesh file extensions this module knows how to load (used by
# AirwayGraph.from_path to decide whether --graph is a mesh or a graph
# JSON).
MESH_EXTENSIONS = (".vtk", ".vtp", ".stl")


# ---------------------------------------------------------------------
# 1. mesh loading / cleanup
# ---------------------------------------------------------------------
def load_surface(path: str) -> "vtk.vtkPolyData":
    if path.lower().endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif path.lower().endswith(".stl"):
        reader = vtk.vtkSTLReader()
    else:
        reader = vtk.vtkPolyDataReader()  # legacy .vtk
    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    if surface.GetNumberOfPoints() == 0:
        raise ValueError(f"Loaded 0 points from {path} -- wrong reader for this file format?")
    return surface


def largest_connected_component(surface: "vtk.vtkPolyData") -> "vtk.vtkPolyData":
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(surface)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    return conn.GetOutput()


def boundary_edge_count(surface: "vtk.vtkPolyData") -> int:
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(surface)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()


def open_surface_if_closed(surface: "vtk.vtkPolyData", cap_radius_mm: Optional[float] = None) -> "vtk.vtkPolyData":
    """vmtkNetworkExtraction requires the surface to have at least one
    opening -- a fully watertight mesh (as segmentation tools typically
    export, auto-capping every terminal bronchiole and the proximal
    tracheal cut) has to have a small hole cut into it first, or
    extraction silently returns an empty network.

    Cuts a small spherical cap out of the surface at whichever point is
    most extremal along the mesh's longest bounding-box axis (in this
    airway anatomy, that reliably lands at or very near the proximal
    tracheal cut -- the widest, most natural place for an opening). Any
    single hole works for vmtkNetworkExtraction to seed from; it does not
    need to be anatomically exact.
    """
    if boundary_edge_count(surface) > 0:
        return surface  # already has an opening, nothing to do

    from vtk.util.numpy_support import vtk_to_numpy as _v2n

    pts = _v2n(surface.GetPoints().GetData())
    bounds = surface.GetBounds()
    ranges = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
    axis = int(np.argmax(ranges))
    apex = pts[np.argmax(pts[:, axis])]

    if cap_radius_mm is None:
        diag = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
        cap_radius_mm = float(np.clip(0.02 * diag, 3.0, 10.0))

    sphere = vtk.vtkSphere()
    sphere.SetCenter(apex.tolist())
    sphere.SetRadius(cap_radius_mm)
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(surface)
    clip.SetClipFunction(sphere)
    clip.InsideOutOff()  # keep the OUTSIDE of the sphere -> removes a cap, opens a hole
    clip.Update()
    opened = clip.GetOutput()

    print(
        f"  input surface was fully closed (0 boundary edges) -- cut a "
        f"{cap_radius_mm:.1f}mm-radius opening at {apex.tolist()} "
        f"(longest-axis extremum) so vmtkNetworkExtraction has a seed opening",
        file=sys.stderr,
    )
    return opened


def smooth_surface(surface: "vtk.vtkPolyData", iterations: int = 20) -> "vtk.vtkPolyData":
    if iterations <= 0:
        return surface
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(0.1)
    smoother.SetBoundarySmoothing(False)
    smoother.SetFeatureEdgeSmoothing(False)
    smoother.SetNonManifoldSmoothing(True)
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


# ---------------------------------------------------------------------
# 2. vmtk network extraction
# ---------------------------------------------------------------------
def run_network_extraction(surface: "vtk.vtkPolyData", advancement_ratio: float = 1.05) -> "vtk.vtkPolyData":
    try:
        from vmtk import vmtkscripts
    except ImportError as e:
        raise ImportError(
            "`vtk` is importable here but `vmtk` itself is not. Install "
            "with `pip install vmtk` in this environment, or build the "
            "graph once in a separate vmtk-enabled venv with "
            "scripts/build_airway_graph.py and pass the resulting .json "
            "to --graph instead."
        ) from e

    ne = vmtkscripts.vmtkNetworkExtraction()
    ne.Surface = surface
    ne.AdvancementRatio = advancement_ratio
    ne.Execute()
    network = ne.Network
    if network.GetNumberOfCells() == 0:
        raise RuntimeError(
            "vmtkNetworkExtraction produced an empty network -- check that the "
            "input mesh is a single watertight-ish airway surface with at "
            "least one opening."
        )
    return network


# ---------------------------------------------------------------------
# 3. raw branch extraction
# ---------------------------------------------------------------------
class RawBranch:
    __slots__ = ("cell_id", "points", "radius", "topo_start", "topo_end")

    def __init__(self, cell_id: int, points: np.ndarray, radius: np.ndarray, topo_start: int, topo_end: int):
        self.cell_id = cell_id
        self.points = points  # (N,3), ordered as stored by vmtk (not yet oriented to root)
        self.radius = radius  # (N,)
        self.topo_start = topo_start
        self.topo_end = topo_end

    @property
    def length_mm(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    @property
    def median_radius(self) -> float:
        return float(np.median(self.radius)) if len(self.radius) else 0.0


def _sanitize_radius(radius: np.ndarray) -> np.ndarray:
    """vmtk's inscribed-sphere radius estimate occasionally comes back NaN
    at an isolated point (self-intersection / degenerate mesh region).
    Replace any such points by linear interpolation from their nearest
    valid neighbors so nothing downstream (JSON output, the median, the
    diameter:distance cue) ever sees a NaN."""
    radius = radius.astype(np.float64).copy()
    bad = ~np.isfinite(radius)
    if not bad.any():
        return radius
    good = ~bad
    if not good.any():
        return np.zeros_like(radius)  # entire branch is garbage -- caller's length/radius filter will drop it
    idx = np.arange(len(radius))
    radius[bad] = np.interp(idx[bad], idx[good], radius[good])
    return radius


def extract_raw_branches(network: "vtk.vtkPolyData") -> List[RawBranch]:
    points_all = vtk_to_numpy(network.GetPoints().GetData())
    radius_all = vtk_to_numpy(network.GetPointData().GetArray("Radius"))
    topo = vtk_to_numpy(network.GetCellData().GetArray("Topology"))

    branches = []
    for cid in range(network.GetNumberOfCells()):
        cell = network.GetCell(cid)
        ids = [cell.GetPointIds().GetId(i) for i in range(cell.GetPointIds().GetNumberOfIds())]
        if len(ids) < 2:
            continue
        branches.append(
            RawBranch(
                cell_id=cid,
                points=points_all[ids].copy(),
                radius=_sanitize_radius(radius_all[ids]),
                topo_start=int(topo[cid][0]),
                topo_end=int(topo[cid][1]),
            )
        )
    return branches


# ---------------------------------------------------------------------
# 4. spatial welding of endpoints (union-find, tolerance-based)
# ---------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def weld_endpoints(branches: List[RawBranch], tol_mm: float) -> Dict[Tuple[int, str], int]:
    """Returns {(branch_index, 'start'|'end'): welded_cluster_id}."""
    endpoint_keys: List[Tuple[int, str]] = []
    endpoint_coords: List[np.ndarray] = []
    for i, b in enumerate(branches):
        endpoint_keys.append((i, "start"))
        endpoint_coords.append(b.points[0])
        endpoint_keys.append((i, "end"))
        endpoint_coords.append(b.points[-1])

    coords = np.array(endpoint_coords)
    n = len(coords)
    uf = UnionFind(n)

    # O(n^2) pairwise weld -- fine at this scale (a few hundred endpoints).
    for i in range(n):
        d = np.linalg.norm(coords[i + 1 :] - coords[i], axis=1)
        close = np.where(d <= tol_mm)[0]
        for j in close:
            uf.union(i, i + 1 + j)

    # relabel roots to compact 0..k-1 ids
    roots = {}
    cluster_of = {}
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        cluster_of[endpoint_keys[i]] = roots[r]
    return cluster_of


# ---------------------------------------------------------------------
# 5. root selection + BFS tree construction over the welded cell graph
# ---------------------------------------------------------------------
def build_adjacency(branches: List[RawBranch], cluster_of: Dict[Tuple[int, str], int]) -> Dict[int, List[int]]:
    """cluster_id -> list of branch indices touching it."""
    adj: Dict[int, List[int]] = {}
    for i in range(len(branches)):
        for end in ("start", "end"):
            c = cluster_of[(i, end)]
            adj.setdefault(c, []).append(i)
    return adj


def pick_root(branches: List[RawBranch], cluster_of: Dict[Tuple[int, str], int], adj: Dict[int, List[int]]) -> Tuple[int, str]:
    """Returns (root_branch_index, proximal_end) where proximal_end in {'start','end'}."""
    root_idx = max(range(len(branches)), key=lambda i: branches[i].median_radius)
    start_cluster = cluster_of[(root_idx, "start")]
    end_cluster = cluster_of[(root_idx, "end")]
    start_degree = len(adj[start_cluster])
    end_degree = len(adj[end_cluster])

    if start_degree == end_degree:
        # tie-break: proximal end is the one further from the whole
        # network's centroid (the free tip of the trachea tends to be an
        # extremal point of the point cloud).
        all_pts = np.concatenate([b.points for b in branches], axis=0)
        centroid = all_pts.mean(axis=0)
        d_start = np.linalg.norm(branches[root_idx].points[0] - centroid)
        d_end = np.linalg.norm(branches[root_idx].points[-1] - centroid)
        proximal = "start" if d_start >= d_end else "end"
    else:
        # the end with FEWER other branches attached is the dead-end/
        # proximal side; the busier end is where the tree branches off.
        proximal = "start" if start_degree < end_degree else "end"
    return root_idx, proximal


def bfs_tree(
    branches: List[RawBranch],
    cluster_of: Dict[Tuple[int, str], int],
    adj: Dict[int, List[int]],
    root_idx: int,
    root_proximal_end: str,
) -> Tuple[Dict[int, Optional[int]], Dict[int, int], Dict[int, str]]:
    """BFS from the root branch outward.

    Returns:
        parent_of: branch_index -> parent branch_index (None for root)
        gen_of: branch_index -> generation (root = 0)
        entry_end[i]: which end of branch i ('start'|'end') faces its
            parent / the root -- i.e. the end that should be oriented as
            this branch's "start" in the final output.
    """
    parent_of: Dict[int, Optional[int]] = {root_idx: None}
    gen_of: Dict[int, int] = {root_idx: 0}
    entry_end: Dict[int, str] = {root_idx: root_proximal_end}

    distal_end = "end" if root_proximal_end == "start" else "start"
    frontier_cluster = cluster_of[(root_idx, distal_end)]
    queue: List[Tuple[int, int]] = [(root_idx, frontier_cluster)]  # (branch_idx, cluster to expand from)
    visited_clusters = set()

    while queue:
        cur_idx, cluster = queue.pop(0)
        if cluster in visited_clusters:
            continue
        visited_clusters.add(cluster)
        for nb_idx in adj[cluster]:
            if nb_idx in parent_of:
                continue
            # which end of nb_idx sits at `cluster`?
            nb_entry = "start" if cluster_of[(nb_idx, "start")] == cluster else "end"
            parent_of[nb_idx] = cur_idx
            gen_of[nb_idx] = gen_of[cur_idx] + 1
            entry_end[nb_idx] = nb_entry
            nb_distal = "end" if nb_entry == "start" else "start"
            queue.append((nb_idx, cluster_of[(nb_idx, nb_distal)]))

    return parent_of, gen_of, entry_end


# ---------------------------------------------------------------------
# 5b. merge straight-through (non-bifurcating) raw cells into one branch
# ---------------------------------------------------------------------
class MergedBranch:
    """One or more raw vmtkNetworkExtraction cells, chained end-to-end,
    that together form a single continuous stretch of airway with no
    bifurcation along the way -- vmtkNetworkExtraction routinely splits one
    physical branch into several raw cells wherever the centerline bends,
    which is NOT the same thing as an anatomical fork. This merges those
    back into one logical branch, so the final graph's nodes correspond to
    real bifurcation-to-bifurcation segments rather than raw skeleton
    fragments."""

    __slots__ = ("raw_cell_ids", "points", "radius")

    def __init__(self, raw_cell_ids: List[int], points: np.ndarray, radius: np.ndarray):
        self.raw_cell_ids = raw_cell_ids
        self.points = points
        self.radius = radius

    @property
    def cell_id(self) -> int:
        return self.raw_cell_ids[0]

    @property
    def length_mm(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    @property
    def median_radius(self) -> float:
        return float(np.median(self.radius)) if len(self.radius) else 0.0


def merge_straight_through(
    raw_branches: List[RawBranch],
    parent_of: Dict[int, Optional[int]],
    gen_of: Dict[int, int],
    entry_end: Dict[int, str],
    root_idx: int,
) -> Tuple[List[MergedBranch], Dict[int, Optional[int]], int]:
    """Collapses chains of raw cells that have exactly one child in a row
    into single MergedBranch entries.

    Returns (merged_branches, parent_of_by_merged_index, root_merged_index).
    A branch boundary is placed at the root, at every true bifurcation
    (>=2 children), and at every leaf (0 children) -- i.e. exactly the
    points that matter anatomically.
    """
    children: Dict[int, List[int]] = {}
    for i, p in parent_of.items():
        if p is not None:
            children.setdefault(p, []).append(i)

    def oriented(cell_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        b = raw_branches[cell_idx]
        if entry_end[cell_idx] == "start":
            return b.points, b.radius
        return b.points[::-1], b.radius[::-1]

    merged_branches: List[MergedBranch] = []
    merged_parent: Dict[int, Optional[int]] = {}
    # raw "chain-start" cell id -> its assigned merged branch index
    start_cell_to_merged_idx: Dict[int, int] = {}

    # BFS over chain-start cells: the root, and every cell that is one of
    # >=2 children at its own entry junction (a true fork).
    queue: List[Tuple[int, Optional[int]]] = [(root_idx, None)]  # (raw start cell, parent raw start cell or None)
    while queue:
        start_cell, parent_start_cell = queue.pop(0)

        raw_chain = [start_cell]
        p0, r0 = oriented(start_cell)
        all_pts, all_rad = [p0], [r0]
        cur = start_cell
        while len(children.get(cur, [])) == 1:
            cur = children[cur][0]
            p_, r_ = oriented(cur)
            all_pts.append(p_[1:])  # drop the duplicate shared junction point
            all_rad.append(r_[1:])
            raw_chain.append(cur)
        merged_points = np.concatenate(all_pts, axis=0)
        merged_radius = np.concatenate(all_rad, axis=0)
        last_cell = raw_chain[-1]

        idx = len(merged_branches)
        merged_branches.append(MergedBranch(raw_chain, merged_points, merged_radius))
        start_cell_to_merged_idx[start_cell] = idx
        merged_parent[idx] = start_cell_to_merged_idx[parent_start_cell] if parent_start_cell is not None else None

        for fork_child in children.get(last_cell, []):
            queue.append((fork_child, start_cell))

    root_merged_idx = start_cell_to_merged_idx[root_idx]
    return merged_branches, merged_parent, root_merged_idx


# ---------------------------------------------------------------------
# 6. filter spurious short branches, splicing children onto survivors
# ---------------------------------------------------------------------
def filter_short_branches(
    branches: List[MergedBranch],
    parent_of: Dict[int, Optional[int]],
    root_idx: int,
    min_length_mm: float,
    min_radius_mm: float = 0.0,
) -> Tuple[List[int], Dict[int, Optional[int]], Dict[int, int]]:
    """Drops branches that are too short OR too thin to be a trustworthy
    lumen (vmtk's inscribed-sphere radius estimate gets noisy on very thin,
    marching-cubes-mesh-derived structures -- below roughly 0.3-0.4mm it's
    picking up surface digitization noise as often as a real small airway),
    splicing each removed branch's children onto its nearest surviving
    ancestor rather than dropping their whole subtree."""
    keep = {
        i for i in parent_of
        if i == root_idx
        or (branches[i].length_mm >= min_length_mm and branches[i].median_radius >= min_radius_mm)
    }

    def effective_parent(i: int) -> Optional[int]:
        p = parent_of[i]
        while p is not None and p not in keep:
            p = parent_of[p]
        return p

    new_parent = {i: effective_parent(i) for i in keep}

    # recompute generation cleanly via BFS over the reduced parent map
    children: Dict[Optional[int], List[int]] = {}
    for i, p in new_parent.items():
        children.setdefault(p, []).append(i)

    new_gen: Dict[int, int] = {root_idx: 0}
    queue = [root_idx]
    while queue:
        cur = queue.pop(0)
        for c in children.get(cur, []):
            new_gen[c] = new_gen[cur] + 1
            queue.append(c)

    kept_ordered = sorted(keep, key=lambda i: new_gen[i])
    return kept_ordered, new_parent, new_gen


# ---------------------------------------------------------------------
# 7. assign labels, emit the graph dict
#    (points are already oriented root-to-tip by merge_straight_through)
# ---------------------------------------------------------------------
def assign_labels(
    kept: List[int],
    parent: Dict[int, Optional[int]],
    root_idx: int,
    branches: List[MergedBranch],
) -> Dict[int, str]:
    children: Dict[int, List[int]] = {}
    for i in kept:
        p = parent[i]
        if p is not None:
            children.setdefault(p, []).append(i)

    labels: Dict[int, str] = {root_idx: "trachea"}

    def order_by_radius_desc(idxs: List[int]) -> List[int]:
        return sorted(idxs, key=lambda i: -branches[i].median_radius)

    root_children = order_by_radius_desc(children.get(root_idx, []))
    if len(root_children) == 2:
        labels[root_children[0]] = "R"  # wider -> right main bronchus (anatomically wider+shorter)
        labels[root_children[1]] = "L"
    else:
        # unusual topology (trifurcation etc. at the carina) -- fall back
        # to a generic, still-deterministic scheme and flag it.
        if root_children:
            print(
                f"WARNING: root has {len(root_children)} children (expected 2 "
                f"for a normal trachea->L/R split); labeling them trachea1, "
                f"trachea2, ... by descending radius instead of R/L.",
                file=sys.stderr,
            )
        for n, c in enumerate(root_children, start=1):
            labels[c] = f"trachea{n}"

    # BFS the rest, appending digits in descending-radius order
    queue = list(root_children)
    while queue:
        cur = queue.pop(0)
        kids = order_by_radius_desc(children.get(cur, []))
        for n, c in enumerate(kids, start=1):
            labels[c] = labels[cur] + str(n)
            queue.append(c)

    return labels


def build_graph(
    mesh_path: str,
    weld_tol_mm: float = 2.0,
    min_branch_length_mm: float = 1.0,
    smooth_iterations: int = 20,
    advancement_ratio: float = 1.05,
    cap_open_radius_mm: Optional[float] = None,
    min_radius_mm: float = 0.4,
) -> dict:
    """Runs the full mesh -> graph pipeline and returns a dict in the same
    {"nodes": [...]} schema ``AirwayGraph.from_dict`` reads -- the single
    entry point both ``scripts/build_airway_graph.py`` and
    ``AirwayGraph.from_path`` call."""
    print(f"Loading {mesh_path} ...", file=sys.stderr)
    surface = load_surface(mesh_path)
    print(f"  {surface.GetNumberOfPoints()} points, {surface.GetNumberOfCells()} cells", file=sys.stderr)

    surface = largest_connected_component(surface)
    surface = open_surface_if_closed(surface, cap_open_radius_mm)
    surface = smooth_surface(surface, smooth_iterations)

    print("Running vmtkNetworkExtraction ...", file=sys.stderr)
    network = run_network_extraction(surface, advancement_ratio)
    branches = extract_raw_branches(network)
    print(f"  raw network: {len(branches)} branches", file=sys.stderr)

    cluster_of = weld_endpoints(branches, weld_tol_mm)
    adj = build_adjacency(branches, cluster_of)

    root_idx, root_proximal = pick_root(branches, cluster_of, adj)
    print(
        f"  root branch = raw cell {branches[root_idx].cell_id} "
        f"(median radius {branches[root_idx].median_radius:.2f}mm, proximal end = {root_proximal})",
        file=sys.stderr,
    )

    parent_of, gen_of, entry_end = bfs_tree(branches, cluster_of, adj, root_idx, root_proximal)
    unreached = len(branches) - len(parent_of)
    if unreached:
        print(f"  {unreached} raw branches were not reachable from the root (disconnected noise) -- dropped", file=sys.stderr)

    merged, merged_parent_raw, merged_root_idx = merge_straight_through(branches, parent_of, gen_of, entry_end, root_idx)
    print(f"  merged non-bifurcating chains: {len(branches)} raw cells -> {len(merged)} anatomical branches", file=sys.stderr)

    kept, final_parent, final_gen = filter_short_branches(
        merged, merged_parent_raw, merged_root_idx, min_branch_length_mm, min_radius_mm
    )
    print(
        f"  after filtering branches shorter than {min_branch_length_mm}mm or "
        f"thinner than {min_radius_mm}mm median radius: {len(kept)} branches kept",
        file=sys.stderr,
    )

    labels = assign_labels(kept, final_parent, merged_root_idx, merged)

    nodes = []
    for i in kept:
        b = merged[i]
        pts, rad = b.points, b.radius  # already oriented root-to-tip by merge_straight_through
        parent_i = final_parent[i]
        nodes.append(
            {
                "label": labels[i],
                "generation": final_gen[i],
                "parent": labels[parent_i] if parent_i is not None else None,
                "start": pts[0].tolist(),
                "end": pts[-1].tolist(),
                "centerline": pts.tolist(),
                "radius": rad.tolist(),
                "length_mm": round(b.length_mm, 2),
                "median_radius_mm": round(b.median_radius, 3),
                "_source_cell_id": b.cell_id,
            }
        )

    nodes.sort(key=lambda n: (n["generation"], n["label"]))
    return {"nodes": nodes}


def default_cache_path(mesh_path: str) -> str:
    """Where a mesh's built graph gets cached if the caller doesn't name
    one explicitly: alongside the mesh, `<stem>_graph.json`."""
    stem, _ = os.path.splitext(mesh_path)
    return stem + "_graph.json"


def build_and_cache_graph(
    mesh_path: str,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    **build_kwargs,
) -> dict:
    """Like ``build_graph``, but reuses a previously-built graph JSON next
    to the mesh instead of re-running the (slow) vmtk network extraction
    every single time the CLI is invoked. This is what lets `--graph
    NewPatient.vtk` "just work" on every run after the first, the same as
    if you'd pointed --graph at an already-built .json all along.

    `cache_path` defaults to `default_cache_path(mesh_path)`. The cache is
    reused whenever it exists and is newer than the mesh file, unless
    `force_rebuild=True`; otherwise (or if the cache doesn't exist yet) the
    mesh is (re)converted and the result written to `cache_path`.
    """
    import json

    if cache_path is None:
        cache_path = default_cache_path(mesh_path)

    if (
        not force_rebuild
        and os.path.exists(cache_path)
        and os.path.getmtime(cache_path) >= os.path.getmtime(mesh_path)
    ):
        print(f"Reusing cached graph {cache_path} (up to date with {mesh_path})", file=sys.stderr)
        with open(cache_path) as f:
            return json.load(f)

    graph = build_graph(mesh_path, **build_kwargs)

    out_dir = os.path.dirname(cache_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(graph, f)
    print(f"Wrote {len(graph['nodes'])} branches to {cache_path}", file=sys.stderr)

    return graph
