"""Analyze a BronchoTrack --out-json localization log, comparing the
paper's majority-vote Localizer output (`location`/`generation`) against
the new graph-constrained TreeMotionFilter output (`motion_location`/
`motion_generation`/`motion_confidence`, see bronchotrack/motion_model.py)
on the same run.

This does NOT re-run the pipeline -- it's a post-hoc comparison over
whatever --out-json log you already produced. Run the CLI first (with
whatever --video/--graph/--weights/--device config you'd normally use),
then point this script at the resulting JSON:

    python3 -m bronchotrack.cli \\
        --video path/to/your_video.mp4 \\
        --graph path/to/airway_graph.json \\
        --weights path/to/best.pt \\
        --class-id 0 \\
        --out-json out/localization_log.json

    python3 scripts/analyze_motion_model.py --json out/localization_log.json \\
        --out out/motion_model_report.md

The motion-model filter is on by default (see --no-motion-model on the
CLI) so a fresh run's JSON should already have everything this script
needs -- if it doesn't (an older log from before this feature existed, or
one made with --no-motion-model), this script will say so rather than
silently reporting empty/misleading numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional


def _switch_count(seq: List[Optional[str]]) -> int:
    """Number of times consecutive non-None entries change value (skips
    over None gaps rather than counting a None->value transition)."""
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


def label_observation_stats(entries: List[dict]) -> Dict:
    """Diagnostic: how often was each branch label actually hard-matched to
    a visible tracklet this frame (i.e. what the motion filter's update()
    sees as `observed_labels`), split into the first vs. second half of the
    run. Doesn't need the graph file -- generations are inferred from
    whatever `generation`/`motion_generation` value co-occurred with each
    label across the log, since a label's generation is constant.

    This exists to answer "why" when the summary numbers above look wrong
    (e.g. the motion filter stuck at one branch for half a run) without
    guessing blind -- see README "Motion-model filter" for the real
    investigation this was built for.
    """
    n = len(entries)
    half = n // 2

    label_gen: Dict[str, int] = {}
    for e in entries:
        if e.get("location") is not None and e.get("generation") is not None:
            label_gen.setdefault(e["location"], e["generation"])
        if e.get("motion_location") is not None and e.get("motion_generation") is not None:
            label_gen.setdefault(e["motion_location"], e["motion_generation"])

    def label_counts(subset: List[dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in subset:
            for t in e.get("tracklets", []):
                label = t.get("label")
                if label is None:
                    continue
                counts[label] = counts.get(label, 0) + 1
        return counts

    first_half_counts = label_counts(entries[:half])
    second_half_counts = label_counts(entries[half:])
    overall_counts = label_counts(entries)

    def tracklet_presence(subset: List[dict]) -> Dict[str, float]:
        if not subset:
            return {"frames_with_any_tracklet": 0.0, "frames_with_labeled_tracklet": 0.0, "mean_tracklets_per_frame": 0.0}
        n_sub = len(subset)
        any_tracklet = sum(1 for e in subset if e.get("tracklets"))
        any_labeled = sum(
            1 for e in subset if any(t.get("label") is not None for t in e.get("tracklets", []))
        )
        total_tracklets = sum(len(e.get("tracklets", [])) for e in subset)
        return {
            "frames_with_any_tracklet": any_tracklet / n_sub,
            "frames_with_labeled_tracklet": any_labeled / n_sub,
            "mean_tracklets_per_frame": total_tracklets / n_sub,
        }

    presence_first_half = tracklet_presence(entries[:half])
    presence_second_half = tracklet_presence(entries[half:])

    max_streak: Dict[str, int] = {}
    current_streak: Dict[str, int] = {}
    for e in entries:
        seen_this_frame = {t.get("label") for t in e.get("tracklets", []) if t.get("label") is not None}
        for label in list(current_streak.keys()):
            if label not in seen_this_frame:
                current_streak[label] = 0
        for label in seen_this_frame:
            current_streak[label] = current_streak.get(label, 0) + 1
            max_streak[label] = max(max_streak.get(label, 0), current_streak[label])

    return {
        "n_frames": n,
        "half_index": half,
        "label_generation": label_gen,
        "overall_counts": overall_counts,
        "first_half_counts": first_half_counts,
        "second_half_counts": second_half_counts,
        "max_streak": max_streak,
        "presence_first_half": presence_first_half,
        "presence_second_half": presence_second_half,
    }


def format_label_stats(stats: Dict, top_n: int = 8) -> str:
    lines = []
    lines.append("")
    lines.append("## Observed-label diagnostic (what update() actually saw each frame)")
    lines.append(
        f"Frames 0-{stats['half_index'] - 1} = first half, "
        f"{stats['half_index']}-{stats['n_frames'] - 1} = second half."
    )
    lines.append("")
    lines.append("### Tracklet presence (is anything being tracked at all, regardless of label?)")
    for name, p in [("First half", stats["presence_first_half"]), ("Second half", stats["presence_second_half"])]:
        lines.append(
            f"- {name}: {p['frames_with_any_tracklet']:.1%} of frames had >=1 active tracklet, "
            f"{p['frames_with_labeled_tracklet']:.1%} had >=1 *labeled* tracklet, "
            f"{p['mean_tracklets_per_frame']:.2f} tracklets/frame on average"
        )
    lines.append("")

    def fmt_table(counts: Dict[str, int], title: str) -> List[str]:
        out = [f"### {title}"]
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
        if not top:
            out.append("- (no hard-matched labels)")
            return out
        for label, count in top:
            gen = stats["label_generation"].get(label)
            gen_str = f"gen {gen}" if gen is not None else "gen ?"
            streak = stats["max_streak"].get(label, 0)
            out.append(f"- `{label}` ({gen_str}): seen in {count} frames, longest streak {streak}")
        return out

    lines += fmt_table(stats["overall_counts"], "Most-observed labels, whole run")
    lines.append("")
    lines += fmt_table(stats["first_half_counts"], "Most-observed labels, first half")
    lines.append("")
    lines += fmt_table(stats["second_half_counts"], "Most-observed labels, second half")
    return "\n".join(lines)


def analyze(log_path: str) -> Dict:
    with open(log_path, "r") as f:
        entries = json.load(f)

    n = len(entries)
    if n == 0:
        raise ValueError(f"{log_path} has no frames logged")

    locations = [e.get("location") for e in entries]
    generations = [e.get("generation") for e in entries]
    motion_locations = [e.get("motion_location") for e in entries]
    motion_generations = [e.get("motion_generation") for e in entries]
    motion_confidences_all = [e.get("motion_confidence") for e in entries]
    motion_confidences = [c for c in motion_confidences_all if c is not None]

    if not any(m is not None for m in motion_locations):
        raise ValueError(
            f"{log_path} has no motion_location data in it -- this log looks like it was "
            "produced before the motion-model filter was added, or --no-motion-model was "
            "passed for this run. Re-run the CLI (without --no-motion-model) and try again."
        )

    both_present = [
        (l, m) for l, m in zip(locations, motion_locations) if l is not None and m is not None
    ]
    agree = sum(1 for l, m in both_present if l == m)
    agreement_rate = agree / len(both_present) if both_present else 0.0

    both_gen = [
        (g, mg) for g, mg in zip(generations, motion_generations) if g is not None and mg is not None
    ]
    gen_agree = sum(1 for g, mg in both_gen if g == mg)
    gen_agreement_rate = gen_agree / len(both_gen) if both_gen else 0.0

    # when they disagree on the exact branch, is the motion filter deeper
    # (further along) or shallower than the vote-based localizer, on
    # average, at that same frame?
    disagree_deltas = [mg - g for g, mg in both_gen if g != mg]
    avg_disagree_delta = sum(disagree_deltas) / len(disagree_deltas) if disagree_deltas else None

    final_conf = motion_confidences_all[-1]

    return {
        "n_frames": n,
        "localizer_coverage": _coverage(locations),
        "motion_coverage": _coverage(motion_locations),
        "localizer_switch_count": _switch_count(locations),
        "motion_switch_count": _switch_count(motion_locations),
        "branch_agreement_rate": agreement_rate,
        "generation_agreement_rate": gen_agreement_rate,
        "n_frames_compared": len(both_present),
        "avg_disagree_generation_delta": avg_disagree_delta,
        "n_disagreements": len(disagree_deltas),
        "motion_confidence_mean": (sum(motion_confidences) / len(motion_confidences))
        if motion_confidences
        else None,
        "motion_confidence_min": min(motion_confidences) if motion_confidences else None,
        "motion_confidence_max": max(motion_confidences) if motion_confidences else None,
        "final_localizer_location": locations[-1],
        "final_localizer_generation": generations[-1],
        "final_motion_location": motion_locations[-1],
        "final_motion_generation": motion_generations[-1],
        "final_motion_confidence": final_conf,
    }


def format_report(result: Dict, log_path: str) -> str:
    lines = []
    lines.append(f"# Motion-model vs. vote-based localizer -- `{log_path}`")
    lines.append("")
    lines.append(f"Frames logged: {result['n_frames']}")
    lines.append("")
    lines.append("## Coverage (fraction of frames with a non-null location)")
    lines.append(f"- Vote-based Localizer: {result['localizer_coverage']:.1%}")
    lines.append(f"- Motion-model filter:  {result['motion_coverage']:.1%}")
    lines.append("")
    lines.append("## Stability (branch switches over the whole run -- lower is smoother)")
    lines.append(f"- Vote-based Localizer: {result['localizer_switch_count']} switches")
    lines.append(f"- Motion-model filter:  {result['motion_switch_count']} switches")
    lines.append("")
    lines.append("## Agreement between the two methods")
    lines.append(
        f"- Exact same branch: {result['branch_agreement_rate']:.1%} "
        f"({result['n_frames_compared']} frames where both had a location)"
    )
    lines.append(
        f"- Same generation (depth), even when the exact branch differs: "
        f"{result['generation_agreement_rate']:.1%}"
    )
    if result["avg_disagree_generation_delta"] is not None:
        direction = "deeper than" if result["avg_disagree_generation_delta"] > 0 else "shallower than"
        lines.append(
            f"- Across the {result['n_disagreements']} frames where the exact branch "
            f"differed, the motion filter was on average "
            f"{abs(result['avg_disagree_generation_delta']):.2f} generations {direction} "
            "the vote-based localizer at that same frame"
        )
    lines.append("")
    lines.append("## Motion-model confidence")
    if result["motion_confidence_mean"] is not None:
        lines.append(f"- Mean: {result['motion_confidence_mean']:.3f}")
        lines.append(f"- Range: {result['motion_confidence_min']:.3f} - {result['motion_confidence_max']:.3f}")
    else:
        lines.append("- No confidence values logged")
    lines.append("")
    lines.append("## Final frame")
    lines.append(
        f"- Vote-based Localizer: {result['final_localizer_location']} "
        f"(generation {result['final_localizer_generation']})"
    )
    conf_str = f", confidence {result['final_motion_confidence']:.3f}" if result["final_motion_confidence"] is not None else ""
    lines.append(
        f"- Motion-model filter:  {result['final_motion_location']} "
        f"(generation {result['final_motion_generation']}{conf_str})"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compare the vote-based Localizer against the TreeMotionFilter on a "
        "completed --out-json run.",
    )
    p.add_argument("--json", required=True, help="Path to a --out-json localization log from bronchotrack.cli")
    p.add_argument("--out", default=None, help="Optional path to also write the report as markdown")
    args = p.parse_args(argv)

    with open(args.json, "r") as f:
        entries = json.load(f)

    result = analyze(args.json)
    report = format_report(result, args.json)
    report += "\n" + format_label_stats(label_observation_stats(entries))
    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nWritten to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
