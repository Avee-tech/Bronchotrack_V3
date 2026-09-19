# BronchoTrack — how to run everything

All commands assume you're inside an activated venv with `ultralytics` and
`pyyaml` installed, and your terminal's current directory is wherever you
unzipped `bronchotrack_pipeline_code.zip` (or your existing checkout).

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ultralytics pyyaml
```

Every command below assumes that venv is active (`(venv)` shown in your
prompt). Swap every `/path/to/...` for your actual files.

---

## 1. Merge multiple Roboflow exports into one training set

`scripts/merge_yolo_datasets.py` — combines datasets that annotate the same
object under different class names, drops any class you don't want (e.g.
Trachea), and skips any source with zero real annotations automatically.

```bash
python3 scripts/merge_yolo_datasets.py \
  --source "/path/to/BronchoTrack V2" \
  --source "/path/to/Airwaydetection" \
  --source "/path/to/Aveeshas Workspace" \
  --class-map "Lumen=lumen" \
  --class-map "Airwaydetection=lumen" \
  --class-map "Trachea=DROP" \
  --dest merged_pool
```

Output: `merged_pool/images/`, `merged_pool/labels/`, `merged_pool/data.yaml`.

---

## 2. Split a merged (or any unsplit) export into train/val/test

`scripts/split_yolo_dataset.py` — 85/10/5 by default, seeded/reproducible.

```bash
python3 scripts/split_yolo_dataset.py --source merged_pool --dest split_dataset_merged
```

Output: `split_dataset_merged/{train,val,test}/{images,labels}/` and a
`data.yaml` pointing at them.

---

## 3. Train a checkpoint

`scripts/train_yolov12.py` — works for any `yoloXX(n/s/m/l/x)[-seg].pt` base,
auto-detects detect vs. segment from your labels.

```bash
# nano (fast, smaller model)
python3 scripts/train_yolov12.py --data split_dataset_merged/data.yaml --model yolo11n.pt

# small (matches the original best.pt architecture -- the strongest result so far)
python3 scripts/train_yolov12.py --data split_dataset_merged/data.yaml --model yolo11s.pt

# yolo12 nano
python3 scripts/train_yolov12.py --data split_dataset_merged/data.yaml --model yolo12n.pt
```

Best checkpoint lands at `runs/<name>/weights/best.pt` (default `--name
bronchotrack_yolov12`, override with `--name` if running multiple trainings
so they don't overwrite each other).

---

## 4. Run the full BronchoTrack pipeline on a video

`bronchotrack.paper_exact.cli` — this is the actual pipeline: detection →
tracking → airway association → localization, with overlay + graph-view
video output.

```bash
python3 -m bronchotrack.paper_exact.cli \
  --video /path/to/ModelV3_2.mp4 \
  --graph patient_airway_graph/airway_graph_ModelV3.json \
  --weights /path/to/your_checkpoint.pt \
  --device cuda \
  --img-size 640 \
  --out-video out/overlay.mp4 \
  --out-graph-video out/graph_view.mp4 \
  --out-json out/log.json
```

Notes:
- `--device cuda` uses your GPU; use `cpu` if you don't have one set up.
- Current tuned defaults (all confirmed via real A/B testing on this video,
  see below): `--conf-threshold 0.1`, `--mask-conf-threshold 0.55`. You
  don't need to pass these explicitly unless overriding them.
- Add `--max-frames 250` to process only the first N frames for a quick
  check instead of the full video.
- Full list of every tunable flag: `python3 -m bronchotrack.paper_exact.cli --help`

### Reading the results

```bash
python3 -c "
import json
data = json.load(open('out/log.json'))
frames = data if isinstance(data, list) else data.get('frames', data)
n = len(frames)
n_loc = sum(1 for f in frames if f.get('location'))
n_live = sum(1 for f in frames if f.get('location_is_live') is True)
n_confirmed = sum(1 for f in frames if any(t.get('diameter_distance_match') for t in f.get('tracklets', [])))
final = [f for f in frames if f.get('location')]
print(f'coverage: {n_loc}/{n} ({n_loc/n:.1%})')
print(f'live votes: {n_live}/{n} ({n_live/n:.1%})')
print(f'confirmed-dot rate: {n_confirmed}/{n} ({n_confirmed/n:.1%})')
if final:
    print(f'final location: {final[-1][\"location\"]}  gen {final[-1][\"generation\"]}')
"
```

---

## 5. Parameter sensitivity sweep

`scripts/param_sweep.py` — one-at-a-time sweep across 7 tunable parameters
at -10%/-5%/0%/+5%/+10% of their defaults (29 runs total). See the script's
own docstring (`python3 scripts/param_sweep.py --help`) for full details.

```bash
python3 scripts/param_sweep.py \
  --video /path/to/ModelV3_2.mp4 \
  --graph patient_airway_graph/airway_graph_ModelV3.json \
  --weights /path/to/your_checkpoint.pt \
  --repo-root /path/to/bronchotrack_pipeline \
  --device cuda \
  --out-dir sweep_out \
  --results sweep_results.jsonl
```

Safe to Ctrl+C and re-run — it skips runs already recorded in
`sweep_results.jsonl`. Rank results afterward:

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('sweep_results.jsonl')]
rows = [r for r in rows if r.get('metrics')]
rows.sort(key=lambda r: r['metrics']['confirmed_rate'], reverse=True)
for r in rows[:5]:
    print(r['run_id'], r['metrics'])
"
```

---

## Best result so far (for reference)

`yolo11s` trained on the merged 3-dataset pool, run at `--conf-threshold
0.1` (paper default) on `ModelV3_2.mp4`: 83.9% coverage, 77.5% live-vote
rate, 66.0% confirmed-dot rate. Every higher `--conf-threshold` value we
tested (0.15, 0.20, 0.25) scored worse on confirmed-dot rate, so 0.1 is the
current recommended default.
