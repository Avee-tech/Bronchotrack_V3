#!/usr/bin/env python3
"""Merge multiple Roboflow YOLO exports -- each annotating the same real-world
object under different class name(s) -- into one flat, unsplit YOLO pool that
is a drop-in --source for split_yolo_dataset.py.

Written for combining BronchoTrack's three separate Roboflow projects
(BronchoTrack V2, Airwaydetection, Aveeshas Workspace) into one training set.
Each project names its lumen/airway class differently (`Lumen`,
`Airwaydetection`, ...); one of them (BronchoTrack V2) additionally has a
separate, real Trachea class that needs to be dropped rather than merged in;
and one (Aveeshas Workspace) turned out, on inspection, to have zero real
annotations at all -- every one of its 224 label files is empty (nc: 0 in its
own data.yaml). This script handles all three situations explicitly rather
than silently doing the wrong thing with any of them:

  * Maps every source class name to one of your chosen UNIFIED class names via
    repeated --class-map "<source name>=<unified name>" (case-insensitive on
    the source side). Map a class to the literal target name DROP to discard
    it entirely (e.g. "Trachea=DROP") -- any label row using a dropped class
    is removed from its label file rather than remapped or kept.

  * Refuses to guess: every class name actually present in a source's
    data.yaml MUST appear in some --class-map, or the script exits before
    writing anything, naming the unmapped class(es). This is deliberate --
    it's exactly the kind of thing that would otherwise silently vanish or
    get merged into the wrong bucket.

  * Skips, by default, any source whose label files are ALL empty across
    every one of its splits. A source with literally zero annotated objects
    is far more likely to be an incomplete/broken export than genuine
    100%-background footage -- folding it in as "background" would actively
    teach the model that frames containing the real object have nothing in
    them, which is worse than not using that source at all. The script
    prints exactly which source(s) it skipped and why. Pass
    --include-empty-sources to force inclusion anyway, if you really do have
    a deliberately all-background source.

  * Converts any stray polygon/segmentation row found inside an otherwise
    bounding-box dataset into its bounding box (min/max of the polygon's
    points), rather than erroring out or silently corrupting the row -- so
    one inconsistently-annotated image doesn't break a detect-task training
    run. (This script's own dry-run against the real datasets found exactly
    one such row, in the Airwaydetection export's training split.)

  * Copies (never moves -- your original exports are left untouched) images
    and rewritten label files into <dest>/images/ and <dest>/labels/, each
    under a per-source-prefixed filename so identical basenames from
    different Roboflow projects can never collide, and writes
    <dest>/data.yaml with your unified class list -- ready to feed straight
    into split_yolo_dataset.py's --source.

Auto-detects each source's own layout: train/valid/test subfolders (what
Roboflow writes when you DO use its split step) or a flat images/+labels/
pair (what you get when you skip it) -- any mix of the two across your
--source arguments is fine, each is detected independently.

Usage
-----
    python3 merge_yolo_datasets.py \\
      --source /path/to/BronchoTrack_V2 \\
      --source /path/to/Airwaydetection \\
      --source /path/to/Aveeshas_Workspace \\
      --class-map "Lumen=lumen" \\
      --class-map "Airwaydetection=lumen" \\
      --class-map "Trachea=DROP" \\
      --dest merged_pool

    # then, as usual:
    python3 split_yolo_dataset.py --source merged_pool --dest split_dataset
    python3 train_yolov12.py --data split_dataset/data.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DROP_TARGET = "DROP"


def find_data_yaml(source: Path) -> Optional[Path]:
    for candidate in [source / "data.yaml", source / "train" / "data.yaml"]:
        if candidate.is_file():
            return candidate
    return None


def load_class_names(source: Path) -> List[str]:
    """Returns the source's names list, indexed by class id (names[i] is the
    name for class id i). Empty list if no usable data.yaml / names found."""
    import yaml

    data_yaml = find_data_yaml(source)
    if data_yaml is None:
        return []
    src = yaml.safe_load(data_yaml.read_text()) or {}
    names = src.get("names")
    if isinstance(names, dict) and names:
        return [names[i] for i in sorted(names, key=int)]
    if isinstance(names, list) and names:
        return list(names)
    return []


def find_splits(source: Path) -> List[Tuple[str, Path, Path]]:
    """Returns [(split_name, images_dir, labels_dir), ...] for every split
    this source actually has. Supports train/valid(or val)/test subfolders
    and a flat images/+labels/ pair at the source root."""
    splits = []
    for split_name, val_name in [("train", "train"), ("valid", "valid"), ("val", "val"), ("test", "test")]:
        images_dir = source / val_name / "images"
        labels_dir = source / val_name / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            splits.append((val_name, images_dir, labels_dir))
    if not splits:
        images_dir, labels_dir = source / "images", source / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            splits.append(("flat", images_dir, labels_dir))
    return splits


def parse_class_map(pairs: List[str]) -> Dict[str, str]:
    """--class-map "Lumen=lumen" -> {"lumen": "lumen"} (source side lowered
    for case-insensitive lookup; target side kept as given, except DROP is
    normalized to the DROP sentinel regardless of case)."""
    mapping = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"error: --class-map {pair!r} must look like 'Source Name=unified_name' or 'Source Name=DROP'")
        src_name, target = pair.split("=", 1)
        target = DROP_TARGET if target.strip().upper() == DROP_TARGET else target.strip()
        mapping[src_name.strip().lower()] = target
    return mapping


def build_source_id_map(
    source: Path, names: List[str], class_map: Dict[str, str], unified_names: List[str]
) -> Dict[int, Optional[int]]:
    """Maps this source's class id -> unified class id (or None to drop),
    mutating `unified_names` in place to register any new unified class
    names it encounters (so unified ids stay stable and shared across all
    sources processed so far)."""
    id_map: Dict[int, Optional[int]] = {}
    unmapped = []
    for class_id, name in enumerate(names):
        target = class_map.get(name.strip().lower())
        if target is None:
            unmapped.append(name)
            continue
        if target == DROP_TARGET:
            id_map[class_id] = None
            continue
        if target not in unified_names:
            unified_names.append(target)
        id_map[class_id] = unified_names.index(target)

    if unmapped:
        raise SystemExit(
            f"error: {source} has class(es) with no --class-map entry: {unmapped!r}\n"
            f"Every class name in this source's data.yaml must be mapped to a unified "
            f"name or to DROP -- add e.g. --class-map \"{unmapped[0]}=lumen\" (or "
            f"--class-map \"{unmapped[0]}=DROP\") and re-run."
        )
    return id_map


def convert_polygon_to_bbox(coords: List[float]) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin


def remap_label_file(text: str, id_map: Dict[int, Optional[int]]) -> Tuple[List[str], int, int]:
    """Returns (new_lines, n_kept, n_dropped_as_mapped_out)."""
    new_lines = []
    n_kept = n_dropped = 0
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue  # malformed row, skip rather than crash the whole merge
        unified_id = id_map.get(class_id)
        if unified_id is None:
            n_dropped += 1
            continue
        if len(values) == 4:
            cx, cy, w, h = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            cx, cy, w, h = convert_polygon_to_bbox(values)
        else:
            continue  # unrecognized row shape, skip rather than crash
        new_lines.append(f"{unified_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        n_kept += 1
    return new_lines, n_kept, n_dropped


def source_has_any_annotations(splits: List[Tuple[str, Path, Path]]) -> bool:
    for _, _, labels_dir in splits:
        for lf in labels_dir.glob("*.txt"):
            if lf.read_text().strip():
                return True
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True, type=Path, help="A Roboflow YOLO export root; repeat for each dataset to merge")
    p.add_argument("--class-map", action="append", required=True, metavar="SRC_NAME=UNIFIED_NAME", help='e.g. "Lumen=lumen" or "Trachea=DROP"; repeat for every class name across all sources')
    p.add_argument("--dest", required=True, type=Path, help="Where to write the merged flat pool (images/, labels/, data.yaml)")
    p.add_argument("--include-empty-sources", action="store_true", help="Include a source even if every one of its label files is empty (off by default -- see module docstring)")
    args = p.parse_args(argv)

    class_map = parse_class_map(args.class_map)
    unified_names: List[str] = []
    dest_images = args.dest / "images"
    dest_labels = args.dest / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    grand_kept = grand_dropped = grand_images = 0
    per_source_summary = []

    for source in args.source:
        if not source.is_dir():
            raise SystemExit(f"error: --source {source} is not a directory")
        prefix = source.name.replace(" ", "_")
        names = load_class_names(source)
        splits = find_splits(source)
        if not splits:
            raise SystemExit(f"error: couldn't find any train/valid/test or images/+labels/ layout under {source}")

        if not names:
            print(f"warning: {source} has no usable class names in its data.yaml (nc: 0) -- treating as having 0 mappable classes.")

        if not args.include_empty_sources and not source_has_any_annotations(splits):
            print(
                f"SKIPPING {source}: every label file across all its splits is empty "
                f"(0 real annotations found). This looks like an incomplete/broken "
                f"Roboflow export rather than genuine all-background footage -- fix "
                f"the export in Roboflow and re-run, or pass --include-empty-sources "
                f"to force-include it as background images."
            )
            per_source_summary.append((str(source), 0, 0, 0, "SKIPPED (no annotations)"))
            continue

        id_map = build_source_id_map(source, names, class_map, unified_names)

        n_images = n_kept = n_dropped = 0
        for split_name, images_dir, labels_dir in splits:
            for img in sorted(images_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_file = labels_dir / f"{img.stem}.txt"
                out_stem = f"{prefix}_{split_name}_{img.stem}"
                out_img = dest_images / f"{out_stem}{img.suffix.lower()}"
                out_img.write_bytes(img.read_bytes())
                n_images += 1

                new_lines: List[str] = []
                if label_file.is_file():
                    new_lines, kept, dropped = remap_label_file(label_file.read_text(), id_map)
                    n_kept += kept
                    n_dropped += dropped
                (dest_labels / f"{out_stem}.txt").write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

        grand_images += n_images
        grand_kept += n_kept
        grand_dropped += n_dropped
        per_source_summary.append((str(source), n_images, n_kept, n_dropped, "included"))

    print("\n=== merge summary ===")
    for src, n_img, kept, dropped, status in per_source_summary:
        print(f"  {src}: {status}  images={n_img}  objects_kept={kept}  objects_dropped_by_class_map={dropped}")
    print(f"\nunified classes ({len(unified_names)}): {unified_names}")
    print(f"total images written: {grand_images}")
    print(f"total objects kept: {grand_kept}   total objects dropped (mapped to DROP): {grand_dropped}")

    if not unified_names:
        raise SystemExit("error: no unified classes were produced -- check your --class-map arguments")

    import yaml

    payload = {
        "train": None,  # placeholder; split_yolo_dataset.py regenerates real paths
        "val": None,
        "test": None,
        "nc": len(unified_names),
        "names": unified_names,
    }
    # write with placeholders removed -- split_yolo_dataset.py's own
    # find_data_yaml/write_data_yaml only reads nc/names from this file, and
    # writes fresh train/val/test paths itself, so we omit the null path keys
    # rather than write something misleading.
    payload = {"nc": len(unified_names), "names": unified_names}
    header = "# Generated by merge_yolo_datasets.py -- merged class list for split_yolo_dataset.py to reuse\n"
    (args.dest / "data.yaml").write_text(header + yaml.safe_dump(payload, sort_keys=False))

    print(f"\nDone. Merged pool written to {args.dest.resolve()}")
    print(f"Next: python3 split_yolo_dataset.py --source {args.dest} --dest split_dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
