#!/usr/bin/env python3
"""
Convert a standard COCO-format dataset into a YOLO-format dataset.

Expected COCO input format
──────────────────────────
    coco_dir/
    ├── annotations/
    │   ├── instances_train.json      # pattern configurable via --ann-pattern
    │   ├── instances_val.json        #   (e.g. instances_train2017.json for real COCO)
    │   └── instances_test.json
    ├── train/    img001.jpg ...      # pattern configurable via --img-pattern
    ├── val/                          #   (e.g. train2017 for real COCO)
    └── test/

Each instances_<split>.json follows the standard COCO object-detection schema:

    {
      "images":      [{"id": int, "file_name": str, "width": int, "height": int}, ...],
      "annotations": [{"id": int, "image_id": int, "category_id": int,
                       "bbox": [x, y, w, h],        # absolute pixels, top-left corner
                       "area": float, "iscrowd": 0|1}, ...],
      "categories":  [{"id": int, "name": str, ...}, ...]
    }

Category ids may be non-contiguous (like real COCO's 1..90); they are mapped
to 0-indexed YOLO class ids by ascending category id. Annotations with
iscrowd == 1 are skipped. Images listed in the JSON but missing on disk are
skipped with a warning.

Produced YOLO output format
───────────────────────────
    out_dir/
    ├── data.yaml
    ├── images/
    │   ├── train/  img001.jpg ...    (copied or symlinked)
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/  img001.txt ...
        ├── val/
        └── test/

Each label .txt has one object per line:

    <class_id> <cx> <cy> <w> <h>

with class_id 0-indexed and cx, cy, w, h normalized to [0, 1]. An empty .txt
is written for background images (images with no annotations). data.yaml
contains path/train/val/test keys (only converted splits), nc and names.

Usage
─────
    python scripts/coco_to_yolo.py --coco-dir dataset_coco --out-dir dataset_yolo
    python scripts/coco_to_yolo.py --coco-dir coco --out-dir yolo --splits train val \
        --ann-pattern "annotations/instances_{split}2017.json" --img-pattern "{split}2017"
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from PIL import Image


def convert_split(ann_file: Path, img_src_dir: Path, out_dir: Path, split: str,
                  link: bool) -> tuple:
    """Convert one split; returns (names, n_images, n_boxes)."""
    with open(ann_file) as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: int(c["id"]))
    cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(cats)}
    names = [str(c["name"]) for c in cats]

    anns_by_img = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_images, n_boxes = 0, 0
    for info in coco["images"]:
        # Guard against file_name containing subdirectories (rare, but stems
        # must stay unique and flat in the YOLO layout).
        fname = Path(info["file_name"]).name
        src = img_src_dir / fname
        if not src.exists():
            print(f"  warning: image listed in JSON but missing on disk, skipping: {src}")
            continue

        width, height = info.get("width"), info.get("height")
        if not width or not height:
            with Image.open(src) as im:
                width, height = im.size

        lines = []
        for ann in anns_by_img.get(info["id"], []):
            cls = cat_id_to_idx.get(int(ann["category_id"]))
            if cls is None:
                print(f"  warning: unknown category_id {ann['category_id']} "
                      f"(ann id {ann.get('id')}) — skipping")
                continue
            x, y, w, h = ann["bbox"]
            # Clamp to the image; skip boxes that end up degenerate.
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2, y2 = min(x + w, width), min(y + h, height)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1

        # Empty file for background images: explicit "no objects".
        (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))

        dst = img_out / fname
        if not dst.exists():
            if link:
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
        n_images += 1

    return names, n_images, n_boxes


def main():
    ap = argparse.ArgumentParser(description="Convert a COCO dataset to YOLO format")
    ap.add_argument("--coco-dir", required=True, help="COCO dataset root directory")
    ap.add_argument("--out-dir", required=True, help="output YOLO dataset directory")
    ap.add_argument("--splits", nargs="+", default=None, choices=["train", "val", "test"],
                    help="splits to convert (default: every split whose annotation JSON exists)")
    ap.add_argument("--ann-pattern", default="annotations/instances_{split}.json",
                    help="annotation JSON path pattern relative to --coco-dir")
    ap.add_argument("--img-pattern", default="{split}",
                    help="image dir pattern relative to --coco-dir")
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying (saves disk space)")
    args = ap.parse_args()

    coco_dir, out_dir = Path(args.coco_dir), Path(args.out_dir)
    splits = args.splits or ["train", "val", "test"]

    names_ref, converted = None, []
    for split in splits:
        ann_file = coco_dir / args.ann_pattern.format(split=split)
        img_dir = coco_dir / args.img_pattern.format(split=split)
        if not ann_file.exists():
            if args.splits:
                raise SystemExit(f"Annotation file not found for split '{split}': {ann_file}")
            continue
        if not img_dir.is_dir():
            raise SystemExit(f"Image dir not found for split '{split}': {img_dir}")

        names, n_img, n_box = convert_split(ann_file, img_dir, out_dir, split, args.link)
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise SystemExit(f"Categories in {ann_file} disagree with previous splits — "
                             f"all split JSONs must share the same category list")
        print(f"[{split}] {n_img} images, {n_box} boxes -> {out_dir}/labels/{split}")
        converted.append(split)

    if not converted:
        raise SystemExit(f"No annotation JSONs found under {coco_dir} "
                         f"(pattern: {args.ann_pattern})")

    data_yaml = {"path": ".", **{s: f"images/{s}" for s in converted},
                 "nc": len(names_ref), "names": names_ref}
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, default_flow_style=None, sort_keys=False)
    print(f"Done. YOLO dataset written to {out_dir} (nc={len(names_ref)}, data.yaml created)")


if __name__ == "__main__":
    main()
