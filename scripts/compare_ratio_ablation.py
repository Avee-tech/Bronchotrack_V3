"""Compare N runs of the SAME video/graph/weights against DIFFERENT
`--diameter-weight` settings, to answer: does the diameter-ratio matching
cue (association.py's scale-invariant size cue, on top of bearing) do
better than the paper's own original method (bearing/angle-only matching,
`--diameter-weight 0`)?

This does NOT re-run the pipeline itself -- produce one --out-json log per
weight setting first (detection is deterministic at inference, so running
the same video multiple times with only --diameter-weight changed is a
clean, isolated ablation), then point this script at all of them:

    python3 -m bronchotrack.cli \\
        --video testvideo3.mp4 --graph patient_airway_graph/airway_graph.json \\
        --weights Lumen_detection_results/best.pt --class-id 0 \\
        --diameter-weight 0 --no-motion-model \\
        --out-json out/log_angle_only.json

    python3 -m bronchotrack.cli \\
        --video testvideo3.mp4 --graph patient_airway_graph/airway_graph.json \\
        --weights Lumen_detection_results/best.pt --class-id 0 \\
        --diameter-weight 0.5 --no-motion-model \\
        --out-json out/log_balanced.json

    python3 -m bronchotrack.cli \\
        --video testvideo3.mp4 --graph patient_airway_graph/airway_graph.json \\
        --weights Lumen_detection_results/best.pt --class-id 0 \\
        --diameter-weight 1.0 --no-motion-model \\
        --out-json out/log_ratio_only.json

    python3 scripts/compare_ratio_ablation.py \\
        --run "angle-only (paper)=out/log_angle_only.json" \\
        --run "balanced (default)=out/log_balanced.json" \\
        --run "ratio-only=out/log_ratio_only.json" \\
        --out out/ratio_ablation_report.md

No ground truth is available here, so "better" is proxied by: how much of
the video actually got labeled/localized at all (coverage), how many
distinct branches were reached (does it get further into the tree, not
just stall at the carina), and how stable the reported location is
(fewer implausible flip-flops between unrelated branches). None of these
alone prove correctness -- read them together, and sanity-check the
"distinct branches reached" list against what you'd expect from watching
the actual video.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple


def _switch_count(seq: List[Optional[str]]) -> int:
    count = 0
    prev = None
    for v in seq:
        if v is None:
            continue
        if prev is not None and v != prev:
            count += 1
        prev = v
    return count


def _coverage(seq: List[Optional[str]]) -> float:
    if not seq:
        return 0.0
    return sum(1 for v in seq if v is not None) / len(seq)


def _label_coverage(entries: List[dict]) -> float:
    """Fraction of (frame, active-tracklet) slots that had a non-null
    label -- i.e. how often association actually assigned an anatomical
    label to something it was currently tracking, not just whether a
    *location* was reported (which can be stale, see localization.py's
    smoothing/freeze behavior)."""
    total = 0
    labeled = 0
    for e in entries:
        for t in e.get("tracklets", []):
            total += 1
            if t.get("label") is not None:
                labeled += 1
    return labeled / total if total else 0.0


def _multi_lumen_frequency(entries: List[dict]) -> float:
    """Fraction of frames with 2+ simultaneously-visible tracklets -- the
    condition the anchor-free peer-comparison diameter-ratio cue requires
    to fire at all (association.py's `_diameter_ratio_cost_matrix` returns
    None, falling back to angle-only, whenever fewer than two lumens are
    visible OR fewer than two candidate labels have known graph diameters;
    this counts only the visibility half of that condition, since the
    label-diameter half isn't in the log). A low value here is the
    expected explanation if `--diameter-weight 0.5` ("balanced") looks
    nearly identical to `--diameter-weight 0` ("angle-only"): the cue is
    skipped on every frame where it isn't satisfied, regardless of the
    configured weight, so a low firing rate caps how much any weight
    setting *can* change the result."""
    if not entries:
        return 0.0
    return sum(1 for e in entries if len(e.get("tracklets", [])) >= 2) / len(entries)


def summarize(entries: List[dict]) -> Dict:
    locations = [e.get("location") for e in entries]
    generations = [g for g in (e.get("generation") for e in entries) if g is not None]
    return {
        "n_frames": len(entries),
        "location_coverage": _coverage(locations),
        "label_coverage": _label_coverage(entries),
        "multi_lumen_frequency": _multi_lumen_frequency(entries),
        "distinct_branches": sorted({l for l in locations if l is not None}, key=lambda x: (len(x), x)),
        "max_generation": max(generations) if generations else None,
        "location_switches": _switch_count(locations),
        "final_location": locations[-1] if locations else None,
        "final_generation": entries[-1].get("generation") if entries else None,
    }


def analyze_run(label: str, path: str) -> Dict:
    with open(path, "r") as f:
        entries = json.load(f)
    if not entries:
        raise ValueError(f"{path} has no frames logged")
    half = len(entries) // 2
    return {
        "label": label,
        "path": path,
        "whole_run": summarize(entries),
        "first_half": summarize(entries[:half]),
    }


def format_report(runs: List[Dict]) -> str:
    lines = ["# Diameter-ratio matching ablation", ""]
    lines.append(
        "Comparing `--diameter-weight` settings on the same video/graph/weights. "
        "See each run's own log for exact command."
    )
    lines.append("")
    for r in runs:
        lines.append(f"- **{r['label']}** -- `{r['path']}`")
    lines.append("")

    def table(section: str) -> List[str]:
        out = [f"## {section}", ""]
        out.append("| Metric | " + " | ".join(r["label"] for r in runs) + " |")
        out.append("|---|" + "---|" * len(runs))
        rows: List[Tuple[str, str]] = [
            ("Location coverage", "location_coverage", "pct"),
            ("Label coverage (per active tracklet)", "label_coverage", "pct"),
            ("Multi-lumen frames (ratio cue can fire)", "multi_lumen_frequency", "pct"),
            ("Distinct branches reached", "distinct_branches", "count"),
            ("Max generation reached", "max_generation", "raw"),
            ("Location switches", "location_switches", "raw"),
            ("Final location", "final_location", "raw"),
            ("Final generation", "final_generation", "raw"),
        ]
        for label, key, kind in rows:
            vals = []
            for r in runs:
                v = r[section.lower().replace(" ", "_")][key]
                if kind == "pct":
                    vals.append(f"{v:.1%}")
                elif kind == "count":
                    vals.append(str(len(v)))
                else:
                    vals.append(str(v))
            out.append(f"| {label} | " + " | ".join(vals) + " |")
        out.append("")
        for r in runs:
            branches = r[section.lower().replace(" ", "_")]["distinct_branches"]
            out.append(f"- **{r['label']}** distinct branches: {', '.join(branches) if branches else '(none)'}")
        out.append("")
        return out

    lines += table("Whole run")
    lines += table("First half")
    lines.append(
        "(First half is separated out because a detection/tracking dropout partway "
        "through some real recordings can freeze the second half identically across "
        "every setting -- see README/motion-model investigation. If whole-run and "
        "first-half numbers tell different stories, trust first-half more.)"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help='One run as "label=path/to/log.json". Repeat --run for each --diameter-weight setting.',
    )
    p.add_argument("--out", default=None, help="Optional path to also write the report as markdown")
    args = p.parse_args(argv)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            print(f"--run must be 'label=path.json', got: {spec}", file=sys.stderr)
            return 1
        label, path = spec.split("=", 1)
        runs.append(analyze_run(label.strip(), path.strip()))

    report = format_report(runs)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nWritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
