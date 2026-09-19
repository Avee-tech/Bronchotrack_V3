#!/usr/bin/env python3
"""Standalone CLI wrapper around ``bronchotrack.mesh_to_graph.build_graph``
-- kept as its own entry point (rather than folded entirely into the main
CLIs) for the workflow where you build a patient's graph once, in a
dedicated vmtk-enabled venv, on its own, before ever touching a video:

    python3 -m venv vmtk_env
    vmtk_env/bin/pip install vmtk

    vmtk_env/bin/python3 scripts/build_airway_graph.py \\
        --mesh NewPatientModel.vtk \\
        --out patient_airway_graph/airway_graph_newpatient.json

As of this script's latest version, this manual step is now OPTIONAL: both
``bronchotrack.cli`` and ``bronchotrack.paper_exact.cli`` accept a mesh
path directly on ``--graph`` and will run this same conversion themselves
on demand (see ``bronchotrack.graph.AirwayGraph.from_path`` /
``bronchotrack.mesh_to_graph.build_and_cache_graph``), caching the result
next to the mesh so it isn't rebuilt on every run. Use this script instead
when you specifically want vmtk isolated to its own venv, want to inspect/
version-control the JSON before running anything else, or want to pass
mesh-conversion parameters that the main CLIs don't bother exposing.

See ``bronchotrack.mesh_to_graph``'s own module docstring for the full
pipeline explanation (WHY NOT SLICER, weld/BFS/merge/filter/label steps,
REQUIREMENTS).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bronchotrack.mesh_to_graph import build_graph


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="Path to the airway surface mesh (.vtk/.vtp/.stl)")
    p.add_argument("--out", required=True, help="Path to write airway_graph.json to")
    p.add_argument("--weld-tol", type=float, default=2.0, help="Endpoint welding tolerance in mm (default: 2.0)")
    p.add_argument("--min-branch-length", type=float, default=1.0, help="Drop branches shorter than this, in mm (default: 1.0)")
    p.add_argument(
        "--min-radius",
        type=float,
        default=0.4,
        help="Drop branches with median inscribed-sphere radius below this, "
        "in mm (default: 0.4 -- below this, vmtk's radius estimate on a "
        "marching-cubes mesh is as likely to be surface digitization noise "
        "as a real small airway).",
    )
    p.add_argument("--smooth-iterations", type=int, default=20, help="Surface smoothing iterations before network extraction (default: 20, 0 to disable)")
    p.add_argument("--advancement-ratio", type=float, default=1.05, help="vmtkNetworkExtraction AdvancementRatio (default: 1.05, vmtk's own default)")
    p.add_argument(
        "--cap-open-radius",
        type=float,
        default=None,
        help="If the input mesh is fully closed (no openings -- common for "
        "segmentation exports), radius in mm of the cap cut open at the "
        "mesh's longest-axis extremum so vmtkNetworkExtraction has a seed "
        "opening (default: auto, ~2%% of the mesh's bounding-box diagonal, "
        "clamped to 3-10mm). Ignored if the mesh already has an opening.",
    )
    args = p.parse_args(argv)

    graph = build_graph(
        mesh_path=args.mesh,
        weld_tol_mm=args.weld_tol,
        min_branch_length_mm=args.min_branch_length,
        smooth_iterations=args.smooth_iterations,
        advancement_ratio=args.advancement_ratio,
        cap_open_radius_mm=args.cap_open_radius,
        min_radius_mm=args.min_radius,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(graph, f)

    labels = [n["label"] for n in graph["nodes"]]
    print(f"Wrote {len(labels)} branches to {args.out}")
    print(f"Root: trachea. Top-level split: {sorted(l for l in labels if len(l) == 1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
