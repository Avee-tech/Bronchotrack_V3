#!/usr/bin/env python3
"""One-at-a-time parameter sensitivity sweep for BronchoTrack's paper_exact
CLI. Varies each of 7 tunable numeric parameters individually at
-10%/-5%/0%/+5%/+10% of a baseline value, holding everything else at
baseline -- 29 total runs (1 baseline + 7 params x 4 non-zero offsets).

Why one-at-a-time rather than a full grid: a true 5-value grid across all 7
parameters is 5**7 = 78,125 combinations. Even at a fast ~2s/frame-equivalent
this is not a reasonable amount of compute to spend on a single video. This
sweep instead measures each parameter's individual effect on the pipeline's
confirmed-dot rate (and coverage/live-rate/final-location, for context),
holding the other 6 at their defaults -- it will NOT catch interaction
effects between two parameters moving together, but it tells you which
parameters are worth a deeper (smaller, targeted) grid afterward.

Usage
-----
    python3 param_sweep.py \\
      --video /path/to/ModelV3_2.mp4 \\
      --graph /path/to/airway_graph_ModelV3.json \\
      --weights /path/to/merged_lumen_yolo11s.pt \\
      --repo-root /path/to/bronchotrack_pipeline \\
      --device cuda \\
      --out-dir sweep_out \\
      --results sweep_results.jsonl

    # optional: cap frames per run for a faster/coarser first pass
    python3 param_sweep.py ... --max-frames 250

Results append to --results as JSON Lines (one run per line) so you can
Ctrl+C and resume later (already-run run_ids are skipped automatically).
After it finishes, rank results yourself with e.g.:

    python3 -c "
import json
rows = [json.loads(l) for l in open('sweep_results.jsonl')]
rows = [r for r in rows if r.get('metrics')]
rows.sort(key=lambda r: r['metrics']['confirmed_rate'], reverse=True)
for r in rows[:5]:
    print(r['run_id'], r['metrics'])
"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# (name, cli flag, baseline value, is_int) -- baseline values match this
# pipeline's current CLI defaults (bronchotrack/paper_exact/cli.py) as of
# when this script was written; update BASELINE below if those defaults
# ever change.
PARAMS = [
    ("conf_threshold", "--conf-threshold", 0.1, False),
    ("max_match_cost", "--max-match-cost", 0.6, False),
    ("angle_threshold_deg", "--angle-threshold-deg", 75.0, False),
    ("distance_diameter_weight", "--distance-diameter-weight", 0.4, False),
    ("virtual_advance_mm", "--virtual-advance-mm", 20.0, False),
    ("virtual_match_threshold", "--virtual-match-threshold", 0.75, False),
    ("reacquire_max_gap_frames", "--reacquire-max-gap-frames", 90, True),
]
BASELINE = {name: val for name, _flag, val, _int in PARAMS}


def build_runs():
    runs = [("baseline", dict(BASELINE))]
    for name, _flag, base_val, is_int in PARAMS:
        for pct in (-10, -5, 5, 10):
            val = base_val * (1 + pct / 100)
            val = round(val) if is_int else round(val, 6)
            params = dict(BASELINE)
            params[name] = val
            runs.append((f"{name}_{pct:+d}pct", params))
    return runs


def analyze_log(path: Path):
    data = json.loads(path.read_text())
    frames = data if isinstance(data, list) else data.get("frames", data)
    n = len(frames)
    if n == 0:
        return None
    n_loc = sum(1 for f in frames if f.get("location"))
    n_live = sum(1 for f in frames if f.get("location_is_live") is True)
    n_confirmed = sum(1 for f in frames if any(t.get("diameter_distance_match") for t in f.get("tracklets", [])))
    final = [f for f in frames if f.get("location")]
    return {
        "n_frames": n,
        "coverage": n_loc / n,
        "live_rate": n_live / n,
        "confirmed_rate": n_confirmed / n,
        "final_location": final[-1].get("location") if final else None,
        "final_generation": final[-1].get("generation") if final else None,
    }


def already_done(results_path: Path):
    if not results_path.is_file():
        return set()
    done = set()
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("metrics"):
                done.add(row["run_id"])
        except json.JSONDecodeError:
            continue
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path, help="bronchotrack_pipeline checkout, run as cwd (so `python3 -m bronchotrack.paper_exact.cli` resolves)")
    p.add_argument("--device", default="cuda", help="'cuda', 'cuda:0', or 'cpu' (default: cuda)")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames per run for speed; omit to use the full video")
    p.add_argument("--out-dir", type=Path, default=Path("sweep_out"))
    p.add_argument("--results", type=Path, default=Path("sweep_results.jsonl"))
    p.add_argument("--timeout-s", type=int, default=1800, help="Per-run subprocess timeout (default: 30 min)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = build_runs()
    done = already_done(args.results)
    todo = [(rid, params) for rid, params in runs if rid not in done]
    print(f"{len(runs)} total runs, {len(done)} already done, {len(todo)} remaining")

    with open(args.results, "a") as f:
        for i, (run_id, params) in enumerate(todo):
            out_json = args.out_dir / f"{run_id}.json"
            cmd = [
                "python3", "-m", "bronchotrack.paper_exact.cli",
                "--video", str(args.video),
                "--graph", str(args.graph),
                "--weights", str(args.weights),
                "--device", args.device,
                "--img-size", str(args.img_size),
                "--out-json", str(out_json),
            ]
            if args.max_frames:
                cmd += ["--max-frames", str(args.max_frames)]
            for name, flag, _base, is_int in PARAMS:
                v = params[name]
                cmd += [flag, str(int(v) if is_int else v)]

            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=args.repo_root, capture_output=True, text=True, timeout=args.timeout_s)
                elapsed = time.time() - t0
                if proc.returncode != 0:
                    result = {"run_id": run_id, "params": params, "error": proc.stderr[-2000:], "elapsed_s": elapsed}
                else:
                    result = {"run_id": run_id, "params": params, "metrics": analyze_log(out_json), "elapsed_s": elapsed}
            except subprocess.TimeoutExpired:
                result = {"run_id": run_id, "params": params, "error": "timeout", "elapsed_s": time.time() - t0}

            f.write(json.dumps(result) + "\n")
            f.flush()
            m = result.get("metrics")
            tag = f"confirmed={m['confirmed_rate']:.1%} live={m['live_rate']:.1%} cov={m['coverage']:.1%}" if m else f"ERROR: {result.get('error', '?')[:150]}"
            print(f"[{i+1}/{len(todo)}] {run_id}: {tag} ({result['elapsed_s']:.0f}s)", flush=True)

    print(f"\nDone. Results in {args.results.resolve()}")
    print("Rank with the snippet in this script's module docstring.")


if __name__ == "__main__":
    sys.exit(main())
