#!/usr/bin/env python3
"""Train a YOLOv12 checkpoint on a split dataset (the output of
split_yolo_dataset.py), for use as a bronchotrack `--weights` checkpoint.

Meant to run on your own machine (GPU strongly recommended -- this cloud
session has none to train on). Thin wrapper around `ultralytics.YOLO(...)
.train(...)`, with the same modeling choices the project's existing
checkpoints already use (imgsz=640 by default, matching best2.pt's own
training config) rather than reinventing hyperparameters.

Auto-detects detect vs. segment from data.yaml's sibling train/labels
folder (same sniff `split_yolo_dataset.py` reports at split time: 5
values/row -> bounding box -> `yolo12n.pt`; more -> polygon ->
`yolo12n-seg.pt`), so this defaults correctly whichever kind of
annotation you exported from Roboflow. Override with --model if you want
a different size (yolo12s/m/l/x) or an explicit checkpoint path to
resume/fine-tune from instead of training from the stock pretrained base.

Usage
-----
    python3 train_yolov12.py --data /path/to/split_dataset/data.yaml

    # override model size / epochs / image size
    python3 train_yolov12.py --data .../data.yaml --model yolo12s-seg.pt --epochs 150 --imgsz 640

    # resume/fine-tune from an existing checkpoint instead of the stock pretrained base
    python3 train_yolov12.py --data .../data.yaml --model /path/to/existing.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def detect_task(data_yaml: Path) -> str:
    """'segment' if the paired train/labels folder's annotations look like
    polygons (>5 values/row), else 'detect'. Falls back to 'detect' (the
    safer/more common default) if nothing usable is found to sniff --
    never raises, since this is only ever a convenience default that
    --model/--task can override."""
    try:
        import yaml

        cfg = yaml.safe_load(data_yaml.read_text())
        train_images = Path(cfg["train"])
    except Exception:
        return "detect"

    # train_images is ".../train/images" -> its sibling ".../train/labels"
    labels_dir = train_images.parent / "labels" if train_images.name == "images" else None
    if labels_dir is None or not labels_dir.is_dir():
        return "detect"

    for label_file in labels_dir.glob("*.txt"):
        text = label_file.read_text().strip()
        if not text:
            continue
        n = len(text.splitlines()[0].split())
        return "segment" if n > 5 else "detect"
    return "detect"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="Path to data.yaml (from split_yolo_dataset.py)")
    p.add_argument(
        "--model",
        default=None,
        help="Base checkpoint to start from: a stock pretrained name (e.g. yolo12n.pt, "
        "yolo12s-seg.pt) or a path to an existing .pt to resume/fine-tune from. "
        "Default: auto-picked as yolo12n.pt or yolo12n-seg.pt based on whether your "
        "labels look like boxes or polygons (see module docstring).",
    )
    p.add_argument("--task", choices=["detect", "segment"], default=None, help="Override task auto-detection")
    p.add_argument("--epochs", type=int, default=150, help="Default 150, matching this project's existing checkpoints")
    p.add_argument("--imgsz", type=int, default=640, help="Default 640, matching this project's existing checkpoints")
    p.add_argument("--batch", type=int, default=-1, help="Batch size; -1 = Ultralytics auto-selects from free VRAM")
    p.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu', or None to auto-select (default)")
    p.add_argument("--project", default="runs", help="Ultralytics output root (default: ./runs)")
    p.add_argument("--name", default="bronchotrack_yolov12", help="Run name under --project")
    p.add_argument("--patience", type=int, default=50, help="Early-stopping patience in epochs (default: 50)")
    p.add_argument("--resume", action="store_true", help="Resume the --name run's last checkpoint instead of starting fresh")
    args = p.parse_args(argv)

    if not args.data.is_file():
        p.error(f"--data {args.data} not found -- point this at the data.yaml split_yolo_dataset.py wrote")

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "error: ultralytics is not installed. On this machine, run:\n"
            "    pip install ultralytics\n",
            file=sys.stderr,
        )
        return 1

    task = args.task or detect_task(args.data)
    model_name = args.model or (f"yolo12n-seg.pt" if task == "segment" else "yolo12n.pt")
    print(f"task: {task}  |  base model: {model_name}  |  data: {args.data}")
    print(
        "(If your installed `ultralytics` version predates YOLOv12 support, this will error "
        "on the download/load step below -- run `pip install -U ultralytics` first.)"
    )

    model = YOLO(model_name)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )

    metrics = model.val(data=str(args.data), split="test")
    print("\nFinal held-out test-split metrics:")
    print(metrics)

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest checkpoint: {best_path}")
    print("Point bronchotrack's --weights at this file to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
