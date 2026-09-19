#!/usr/bin/env python3
"""Split an unsplit Roboflow YOLO export into train/val/test folders.

Written for a Roboflow YOLOv12 export that was downloaded without
Roboflow's own train/valid/test split (i.e. everything landed in one
bucket) -- this rebuilds the standard Ultralytics three-way split
Ultralytics/Roboflow normally produce, so the result is a drop-in
replacement for a "normal" export.

Auto-detects the export's actual layout rather than assuming one:
  * <source>/images/*  + <source>/labels/*.txt          (flat export), or
  * <source>/train/images/* + <source>/train/labels/*.txt
    (Roboflow put everything under train/ when you skip its split step)
Label files are matched to images by basename (frame1.jpg <-> frame1.txt).
Images with no matching label file are still included by default (an
empty/background image is valid YOLO training data) -- pass
--require-labels to drop them instead.

Works for both bounding-box AND polygon/segmentation YOLO annotations --
the split only ever copies/moves whole label files, it never parses their
contents, so either format (5 values per box row vs. a variable-length
polygon row) passes through unchanged. It DOES sniff one line from one
label file purely to tell you which it looks like, for your own sanity
check in the summary printout.

Split is a random 85/10/5 (train/val/test) shuffle by default, seeded for
reproducibility (--seed, default 42) -- re-running with the same seed on
the same input reproduces the exact same split.

Usage
-----
    python3 split_yolo_dataset.py --source /path/to/roboflow_export --dest /path/to/split_dataset

    # preview counts without copying anything
    python3 split_yolo_dataset.py --source ... --dest ... --dry-run

    # move instead of copy (frees the original export's disk space)
    python3 split_yolo_dataset.py --source ... --dest ... --move

    # custom ratios (must sum to 1.0)
    python3 split_yolo_dataset.py --source ... --dest ... --ratios 0.8 0.15 0.05
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_layout(source: Path) -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir), auto-detecting which of the two
    supported layouts `source` actually uses. Raises SystemExit with a
    clear message if neither is found."""
    candidates = [
        (source / "images", source / "labels"),
        (source / "train" / "images", source / "train" / "labels"),
    ]
    for images_dir, labels_dir in candidates:
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir

    raise SystemExit(
        f"error: couldn't find an images/ + labels/ pair under {source}\n"
        f"Looked for:\n"
        + "\n".join(f"  {img} + {lbl}" for img, lbl in candidates)
        + "\nIf your export is laid out differently, point --source directly "
        "at the folder that directly contains images/ and labels/."
    )


def find_data_yaml(source: Path) -> Optional[Path]:
    """Roboflow YOLO exports always include a data.yaml with `nc`/`names`
    at (or near) the export root -- reused here so the new split's
    data.yaml carries the same class list forward instead of guessing it."""
    for candidate in [source / "data.yaml", source / "train" / "data.yaml", source.parent / "data.yaml"]:
        if candidate.is_file():
            return candidate
    return None


def sniff_label_format(labels_dir: Path) -> str:
    """Peek at one non-empty label file's first line to report whether
    this looks like plain bounding-box YOLO (5 values: class cx cy w h) or
    polygon/segmentation YOLO (class + an even number of x,y pairs, i.e.
    an odd total count > 5). Informational only -- never affects the
    split itself."""
    for label_file in labels_dir.glob("*.txt"):
        text = label_file.read_text().strip()
        if not text:
            continue
        first_line = text.splitlines()[0].split()
        n = len(first_line)
        if n == 5:
            return "bounding-box (5 values/row: class cx cy w h)"
        elif n > 5:
            return f"polygon/segmentation ({n} values on first row: class + {n - 1} coords)"
        else:
            return f"unrecognized ({n} values on first row)"
    return "unknown (no non-empty label file found to sniff)"


def collect_pairs(images_dir: Path, labels_dir: Path, require_labels: bool) -> List[Tuple[Path, Optional[Path]]]:
    pairs: List[Tuple[Path, Optional[Path]]] = []
    skipped_no_label = 0
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        if label.is_file():
            pairs.append((img, label))
        elif require_labels:
            skipped_no_label += 1
        else:
            pairs.append((img, None))

    if skipped_no_label:
        print(f"skipped {skipped_no_label} image(s) with no matching label file (--require-labels)")
    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Optional[Path]]], ratios: Tuple[float, float, float], seed: int
) -> Tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    # test gets the remainder rather than its own round() so rounding
    # error can't silently drop or duplicate an image across splits
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_split(
    name: str, pairs: list, dest: Path, transfer, ) -> None:
    images_out = dest / name / "images"
    labels_out = dest / name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for img, label in pairs:
        transfer(img, images_out / img.name)
        if label is not None:
            transfer(label, labels_out / label.name)


def write_data_yaml(dest: Path, source_yaml: Optional[Path]) -> None:
    """Writes the new split's data.yaml, carrying `names`/`nc` forward from
    the source export's own data.yaml -- via a real YAML parse, not text
    regex. (An earlier version of this function regex-matched the
    "names:" block as raw text, assuming its list items would be indented
    under the key; Roboflow actually writes them as a dash list at the
    SAME indentation as the key itself --

        names:
        - lumen
        nc: 1

    -- which that regex's lookahead treated as ending immediately after
    "names:" (a line starting with "-" still counts as "starts with a
    non-whitespace character"), silently producing an empty names: block
    and a "0-class dataset" error from ultralytics at training time. YAML
    has several other equally valid ways to write the same list --
    flow-style `names: ['lumen']`, a plain dict `{0: lumen}` -- that a
    hand-rolled regex would need to separately special-case; parsing it
    as actual YAML handles all of them uniformly instead.)"""
    import yaml

    names: dict = {0: "lumen"}
    nc = 1
    if source_yaml is not None:
        try:
            src = yaml.safe_load(source_yaml.read_text()) or {}
            src_names = src.get("names")
            if isinstance(src_names, dict) and src_names:
                names = {int(k): v for k, v in src_names.items()}
                nc = src.get("nc", len(names))
            elif isinstance(src_names, list) and src_names:
                names = {i: v for i, v in enumerate(src_names)}
                nc = src.get("nc", len(names))
            else:
                print(
                    f"warning: no usable 'names' list/dict found in {source_yaml} "
                    f"(got {src_names!r}) -- writing a 1-class placeholder; edit "
                    f"data.yaml's names:/nc: by hand before training."
                )
        except Exception as e:
            print(
                f"warning: couldn't parse {source_yaml} ({e}) -- writing a "
                f"1-class placeholder; edit data.yaml's names:/nc: by hand."
            )
    else:
        print("warning: no source data.yaml found -- writing a 1-class placeholder; edit names:/nc: by hand.")

    payload = {
        "train": str((dest / "train" / "images").resolve()),
        "val": str((dest / "val" / "images").resolve()),
        "test": str((dest / "test" / "images").resolve()),
        "nc": nc,
        "names": names,
    }
    header = "# Generated by split_yolo_dataset.py -- train/val/test split of a Roboflow export\n"
    (dest / "data.yaml").write_text(header + yaml.safe_dump(payload, sort_keys=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="Path to the unsplit Roboflow export")
    p.add_argument("--dest", required=True, type=Path, help="Path to write the split train/val/test dataset")
    p.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.85, 0.10, 0.05],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios, must sum to 1.0 (default: 0.85 0.10 0.05)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (frees the source export's disk space; "
        "irreversible without re-running from a backup)",
    )
    p.add_argument(
        "--require-labels",
        action="store_true",
        help="Drop images with no matching label file instead of keeping them as background images",
    )
    p.add_argument("--dry-run", action="store_true", help="Print split counts only, don't write anything")
    args = p.parse_args(argv)

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        p.error(f"--ratios must sum to 1.0, got {args.ratios} (sums to {sum(args.ratios)})")
    if not args.source.is_dir():
        p.error(f"--source {args.source} is not a directory")

    images_dir, labels_dir = find_layout(args.source)
    print(f"images: {images_dir}")
    print(f"labels: {labels_dir}")
    print(f"label format looks like: {sniff_label_format(labels_dir)}")

    pairs = collect_pairs(images_dir, labels_dir, args.require_labels)
    if not pairs:
        raise SystemExit("error: no images found -- check --source and the layout detected above")

    train, val, test = split_pairs(pairs, tuple(args.ratios), args.seed)
    total = len(pairs)
    print(
        f"\n{total} images total -> "
        f"train {len(train)} ({len(train)/total:.1%})  "
        f"val {len(val)} ({len(val)/total:.1%})  "
        f"test {len(test)} ({len(test)/total:.1%})"
    )

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if args.dest.exists() and any(args.dest.iterdir()):
        print(f"\nwarning: --dest {args.dest} already exists and is not empty; files may be overwritten.")

    transfer = shutil.move if args.move else shutil.copy2
    for name, split in [("train", train), ("val", val), ("test", test)]:
        write_split(name, split, args.dest, transfer)

    source_yaml = find_data_yaml(args.source)
    write_data_yaml(args.dest, source_yaml)

    print(f"\nDone. Split dataset written to {args.dest.resolve()}")
    print(f"data.yaml written to {(args.dest / 'data.yaml').resolve()}")
    if source_yaml is None:
        print("Remember to edit data.yaml's nc:/names: to match your actual classes before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
