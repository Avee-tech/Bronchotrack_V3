"""Render the semantic airway graph *inside* a translucent 3D airway
surface (built from the same graph's own centerline + radius data --
there's no separate raw CT mesh checked into this repo, so the "3D model"
here is the graph's own tube reconstruction, exactly the thing
`association.py`'s diameter:distance cue reasons about, just made
solid/visible instead of an abstract set of radius numbers), with the
"virtual viewpoint" points used by `AirwayGraph.project_children_at_distance`
/ `child_apparent_diameter_at_distance` marked explicitly at whatever
`--virtual-advance-mm` is passed.

Three visual layers, one interactive HTML:
  1. TRANSLUCENT TUBE MESH -- one lofted tube per branch, radius taken
     directly from the graph's own per-sample `radius` array, opacity low
     enough that the centerline graph and virtual points are clearly
     visible through the "airway wall".
  2. SEMANTIC GRAPH -- each branch's centerline drawn as a line (the
     branch itself), plus a marker+label at each branch's start (i.e.
     every bifurcation/graph node), colored by generation.
  3. VIRTUAL DISTANCE POINTS -- for every parent->child pair, the exact
     3D point `distance_mm` down the child's own centerline from its
     start (clamped to the child's length) -- i.e. literally the point
     `project_children_at_distance` / `child_apparent_diameter_at_distance`
     sample from, marked as a distinct diamond marker with hover text
     showing the true radius and the foreshortening-corrected apparent
     diameter at that point.

Usage:
    python3 scripts/semantic_graph_in_translucent_model.py \\
        --graph patient_airway_graph/airway_graph_ModelV3.json \\
        --virtual-advance-mm 30 \\
        --out out/semantic_graph_in_model.html
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go

from bronchotrack.graph import AirwayGraph
from bronchotrack.paper_exact.association import AirwayAssociation

TUBE_SIDES = 14
TUBE_COLOR = "rgb(224, 172, 150)"  # translucent tissue tan
GRAPH_LINE_COLOR = "rgb(30, 40, 60)"
NODE_MARKER_COLOR_BY_GEN = [
    "#1f2937", "#2563eb", "#0d9488", "#7c3aed", "#c2410c", "#be123c", "#4d7c0f", "#0369a1",
]
VIRTUAL_POINT_COLOR = "rgb(255, 196, 0)"


def rotation_minimizing_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (normal, binormal) perpendicular to the local tangent,
    propagated along the polyline so consecutive rings don't twist
    (double-reflection RMF, Wang et al. 2008 -- simplified single-pass
    version, adequate for a visualization tube, not a certified geometry
    kernel)."""
    n = len(points)
    tangents = np.zeros((n, 3))
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tangents = tangents / norms

    normals = np.zeros((n, 3))
    binormals = np.zeros((n, 3))

    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, tangents[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    n0 = seed - np.dot(seed, tangents[0]) * tangents[0]
    n0 /= max(np.linalg.norm(n0), 1e-9)
    normals[0] = n0
    binormals[0] = np.cross(tangents[0], n0)

    for i in range(1, n):
        t_prev, t_cur = tangents[i - 1], tangents[i]
        n_prev = normals[i - 1]
        axis = np.cross(t_prev, t_cur)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            n_cur = n_prev
        else:
            axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(t_prev, t_cur), -1.0, 1.0))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            n_cur = (
                n_prev * cos_a
                + np.cross(axis, n_prev) * sin_a
                + axis * np.dot(axis, n_prev) * (1 - cos_a)
            )
        n_cur = n_cur - np.dot(n_cur, t_cur) * t_cur
        n_cur /= max(np.linalg.norm(n_cur), 1e-9)
        normals[i] = n_cur
        binormals[i] = np.cross(t_cur, n_cur)

    return normals, binormals


def tube_mesh_for_branch(centerline: np.ndarray, radius: np.ndarray, sides: int = TUBE_SIDES):
    """Lofted tube surface (vertices, triangle-index-triples) for one
    branch's centerline + per-sample radius."""
    normals, binormals = rotation_minimizing_frames(centerline)
    thetas = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    n_rings = len(centerline)
    verts = np.zeros((n_rings, sides, 3))
    for i in range(n_rings):
        r = max(radius[i], 0.05)
        verts[i] = (
            centerline[i]
            + r * np.outer(cos_t, normals[i])
            + r * np.outer(sin_t, binormals[i])
        )
    verts_flat = verts.reshape(-1, 3)

    tris = []
    for i in range(n_rings - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a = i * sides + j
            b = i * sides + j2
            c = (i + 1) * sides + j
            d = (i + 1) * sides + j2
            tris.append((a, b, c))
            tris.append((b, d, c))
    return verts_flat, np.array(tris)


def build_figure(graph: AirwayGraph, assoc: AirwayAssociation) -> go.Figure:
    fig = go.Figure()

    # ---- Layer 1: translucent tube mesh, all branches combined into one
    # Mesh3d trace for a single legend entry / one translucent surface. ----
    all_verts, all_tris = [], []
    offset = 0
    for label, node in graph.nodes.items():
        if node.centerline is None or len(node.centerline) < 2 or node.radius is None:
            continue
        centerline = np.asarray(node.centerline, dtype=float)
        radius = np.asarray(node.radius, dtype=float)
        if len(radius) != len(centerline):
            radius = np.full(len(centerline), np.nanmean(radius) if len(radius) else 3.0)
        radius = np.nan_to_num(radius, nan=2.0)
        radius[radius <= 0] = 0.05
        verts, tris = tube_mesh_for_branch(centerline, radius)
        all_verts.append(verts)
        all_tris.append(tris + offset)
        offset += len(verts)

    V = np.concatenate(all_verts, axis=0)
    F = np.concatenate(all_tris, axis=0)
    fig.add_trace(
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color=TUBE_COLOR,
            opacity=0.22,
            flatshading=False,
            lighting=dict(ambient=0.55, diffuse=0.6, specular=0.15, roughness=0.9, fresnel=0.1),
            lightposition=dict(x=100, y=200, z=150),
            name="3D airway model (translucent)",
            hoverinfo="skip",
        )
    )

    # ---- Layer 2: semantic graph -- centerlines + bifurcation nodes ----
    for label, node in graph.nodes.items():
        centerline = np.asarray(node.centerline, dtype=float) if node.centerline is not None else np.stack([node.start, node.end])
        color = NODE_MARKER_COLOR_BY_GEN[node.generation % len(NODE_MARKER_COLOR_BY_GEN)]
        fig.add_trace(
            go.Scatter3d(
                x=centerline[:, 0], y=centerline[:, 1], z=centerline[:, 2],
                mode="lines",
                line=dict(color=GRAPH_LINE_COLOR, width=5),
                name=f"graph: {label}",
                legendgroup="semantic_graph",
                showlegend=False,
                hovertext=f"{label} (gen {node.generation}, parent={node.parent})",
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[node.start[0]], y=[node.start[1]], z=[node.start[2]],
                mode="markers+text",
                marker=dict(size=4, color=color, symbol="circle", line=dict(color="white", width=1)),
                text=[label],
                textposition="top center",
                textfont=dict(size=11, color=color),
                name=f"node: {label}",
                legendgroup="semantic_graph",
                showlegend=False,
                hovertext=(
                    f"{label}  gen={node.generation}  parent={node.parent}<br>"
                    f"length={node.length_mm if hasattr(node, 'length_mm') else node.arc_length_mm():.1f}mm  "
                    f"radius@start={node.radius_at_start:.2f}mm"
                ),
                hoverinfo="text",
            )
        )
    # one dummy legend entry for the whole semantic-graph layer
    fig.add_trace(
        go.Scatter3d(
            x=[None], y=[None], z=[None], mode="lines",
            line=dict(color=GRAPH_LINE_COLOR, width=5),
            name="Semantic graph (centerline + bifurcation labels)",
        )
    )

    # ---- Layer 3: virtual-distance points -- one distance per bifurcation
    # when assoc.dynamic_virtual_advance is on (see association.py's
    # "_virtual_advance_mm_for": half the parent's diameter at the
    # bifurcation, compounded by assoc.virtual_advance_growth_per_generation
    # per generation deeper), or the same fixed distance everywhere
    # otherwise -- both go through the exact same pipeline code path. ----
    vx, vy, vz, vtext = [], [], [], []
    for parent_label, parent in graph.nodes.items():
        if not parent.children:
            continue
        distance_mm = assoc._virtual_advance_mm_for(parent_label)
        for child_label in parent.children:
            child = graph.nodes[child_label]
            length = child.arc_length_mm()
            t = 0.0 if length <= 1e-9 else min(distance_mm, length) / length
            pt = child.point_at(t)
            true_r = child.radius_at(t)
            apparent_d = graph.child_apparent_diameter_at_distance(parent_label, child_label, distance_mm)
            vx.append(pt[0]); vy.append(pt[1]); vz.append(pt[2])
            clamped_note = " (clamped to branch end)" if distance_mm > length else ""
            parent_diam = 2 * parent.radius_at_end if parent.radius_at_end is not None else float("nan")
            vtext.append(
                f"virtual viewpoint for {parent_label} → {child_label}<br>"
                f"distance used: {distance_mm:.2f}mm "
                f"({'dynamic: 0.5×parent Ø(' + f'{parent_diam:.2f}mm' + f')×1.1^gen{parent.generation}' if assoc.dynamic_virtual_advance else 'fixed'})<br>"
                f"{min(distance_mm, length):.1f}mm down {child_label}'s centerline{clamped_note}<br>"
                f"true radius here: {true_r:.2f}mm (Ø2 = {2*true_r:.2f}mm true diameter)<br>"
                f"foreshortening-corrected apparent diameter: "
                f"{apparent_d:.2f}mm" if apparent_d is not None else "apparent diameter: n/a"
            )
    marker_name = (
        "Virtual viewpoints (adaptive: 0.5×parent diameter, ×1.1 per generation)"
        if assoc.dynamic_virtual_advance
        else f"Virtual viewpoints ({assoc.virtual_advance_mm:g}mm past each bifurcation, fixed)"
    )
    fig.add_trace(
        go.Scatter3d(
            x=vx, y=vy, z=vz,
            mode="markers",
            marker=dict(size=6, color=VIRTUAL_POINT_COLOR, symbol="diamond", line=dict(color="black", width=1)),
            name=marker_name,
            hovertext=vtext,
            hoverinfo="text",
        )
    )

    title_sub = (
        "adaptive virtual viewpoints: half the parent's diameter at each bifurcation, "
        f"×1.1 compounded per generation deeper (base_fraction={assoc.virtual_advance_base_fraction:g}, "
        f"growth={assoc.virtual_advance_growth_per_generation:g}/gen)"
        if assoc.dynamic_virtual_advance
        else f"virtual viewpoints marked at a fixed {assoc.virtual_advance_mm:g}mm past each bifurcation"
    )
    fig.update_layout(
        title=dict(
            text=(
                f"Semantic airway graph inside its translucent 3D model<br>"
                f"<sub>{title_sub}</sub>"
            ),
        ),
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
            aspectmode="data",
            bgcolor="rgb(250, 249, 246)",
        ),
        paper_bgcolor="white",
        legend=dict(itemsizing="constant", x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=70, b=0),
        height=900,
    )
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--virtual-advance-mm", type=float, default=30.0)
    p.add_argument(
        "--dynamic-virtual-advance",
        action="store_true",
        help="Mark each bifurcation's virtual viewpoint at half that "
        "parent's own diameter, compounded by --virtual-advance-growth-"
        "per-generation per generation deeper, instead of the fixed "
        "--virtual-advance-mm distance everywhere (see association.py's "
        "'Dynamic virtual advance' docstring section).",
    )
    p.add_argument("--virtual-advance-base-fraction", type=float, default=0.5)
    p.add_argument("--virtual-advance-growth-per-generation", type=float, default=0.10)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    graph = AirwayGraph.from_path(args.graph)
    assoc = AirwayAssociation(
        graph,
        virtual_advance_mm=args.virtual_advance_mm,
        dynamic_virtual_advance=args.dynamic_virtual_advance,
        virtual_advance_base_fraction=args.virtual_advance_base_fraction,
        virtual_advance_growth_per_generation=args.virtual_advance_growth_per_generation,
    )
    fig = build_figure(graph, assoc)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.write_html(args.out, include_plotlyjs=True, full_html=True)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
