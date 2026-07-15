#!/usr/bin/env python3
"""
Convert a YOLO-format dataset into a COCO-format dataset.

Expected YOLO input format
──────────────────────────
A dataset described by a data.yaml (standard Ultralytics practice):

    dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/  img001.jpg ...
    │   ├── val/    ...
    │   └── test/   ...
    └── labels/
        ├── train/  img001.txt ...
        ├── val/    ...
        └── test/   ...

    # data.yaml
    path: .                 # optional dataset root
    train: images/train
    val: images/val
    test: images/test       # optional
    nc: 3                   # optional if names given
    names: ['cat', 'dog', 'bird']   # list or {0: 'cat', ...} dict

Each label .txt has one object per line, whitespace-separated:

    <class_id> <cx> <cy> <w> <h>

where class_id is a 0-indexed integer and cx, cy, w, h are the box center,
width, and height normalized to [0, 1] by image width/height. A missing or
empty label file means the image has no objects (background image).

Produced COCO output format
───────────────────────────
    out_dir/
    ├── annotations/
    │   ├── instances_train.json
    │   ├── instances_val.json
    │   └── instances_test.json
    ├── train/   (images copied or symlinked)
    ├── val/
    └── test/

Each instances_<split>.json follows the standard COCO object-detection schema:

    {
      "images":      [{"id": int (1-based, sorted filename order),
                       "file_name": str, "width": int, "height": int}, ...],
      "annotations": [{"id": int, "image_id": int,
                       "category_id": int,          # yolo class_id + 1
                       "bbox": [x, y, w, h],        # absolute pixels, top-left
                       "area": float, "iscrowd": 0}, ...],
      "categories":  [{"id": i+1, "name": names[i],
                       "supercategory": "object"}, ...]
    }

Background images appear in "images" with zero annotations.

Usage
─────
    python scripts/yolo_to_coco.py --yaml dataset/data.yaml --out-dir dataset_coco
    python scripts/yolo_to_coco.py --yaml data.yaml --out-dir out --splits train val --link
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

from data.dataset import IMG_EXTS, find_label_path, parse_yolo_label, parse_yolo_yaml


def resolve_split_dir(yaml_path: str, split: str) -> str:
    """
    Resolve the images dir for an arbitrary split key (train/val/test).

    parse_yolo_yaml only resolves train/val, so this replicates its candidate
    root order: yaml_dir/path/rel, cwd/path/rel, yaml_dir/rel, cwd/rel.
    Returns "" if the split key is absent from the yaml.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    val = data.get(split)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    if not val:
        return ""
    rel = Path(val)
    if rel.is_absolute():
        return str(rel.resolve())
    yaml_dir = yaml_path.parent.resolve()
    cwd = Path.cwd()
    candidates = []
    root = data.get("path")
    if root:
        candidates += [yaml_dir / root / rel, cwd / root / rel]
    candidates += [yaml_dir / rel, cwd / rel]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(candidates[0].resolve())


def convert_split(img_dir: str, names: list, nc: int, out_img_dir: Path,
                  out_json: Path, link: bool) -> tuple:
    """Convert one split; returns (n_images, n_annotations)."""
    imgs = sorted(p for p in Path(img_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)

    # Prediction files are looked up by filename stem downstream, so stems
    # must be unique within a split.
    stems = [p.stem for p in imgs]
    if len(set(stems)) != len(stems):
        dupes = sorted({s for s in stems if stems.count(s) > 1})
        raise SystemExit(f"Duplicate image stems in {img_dir}: {dupes[:5]} ...")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    images, annotations, ann_id = [], [], 1

    for img_id, img_path in enumerate(imgs, start=1):
        with Image.open(img_path) as im:
            width, height = im.size
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        dst = out_img_dir / img_path.name
        if not dst.exists():
            if link:
                os.symlink(img_path.resolve(), dst)
            else:
                shutil.copy2(img_path, dst)

        labels = parse_yolo_label(find_label_path(str(img_path)), nc)
        for cls, cx, cy, w, h in labels.astype(float).tolist():
            bw, bh = w * width, h * height
            x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,  # COCO IDs are 1-indexed
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": str(n), "supercategory": "object"}
                       for i, n in enumerate(names)],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    return len(images), len(annotations)


def main():
    ap = argparse.ArgumentParser(description="Convert a YOLO dataset (data.yaml) to COCO format")
    ap.add_argument("--yaml", required=True, help="path to the YOLO data.yaml")
    ap.add_argument("--out-dir", required=True, help="output COCO dataset directory")
    ap.add_argument("--splits", nargs="+", default=None, choices=["train", "val", "test"],
                    help="splits to convert (default: every split present in the yaml)")
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying (saves disk space)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing annotation JSONs in out-dir")
    args = ap.parse_args()

    cfg = parse_yolo_yaml(args.yaml)
    names, nc = cfg["names"], cfg["nc"]
    if not names:
        names = [str(i) for i in range(nc)]

    splits = args.splits or ["train", "val", "test"]
    out_dir = Path(args.out_dir)

    converted = 0
    for split in splits:
        img_dir = resolve_split_dir(args.yaml, split)
        if not img_dir or not Path(img_dir).is_dir():
            if args.splits:  # explicitly requested → error, else silently skip
                raise SystemExit(f"Split '{split}' images dir not found: {img_dir or '(no yaml key)'}")
            continue
        out_json = out_dir / "annotations" / f"instances_{split}.json"
        if out_json.exists() and not args.overwrite:
            raise SystemExit(f"{out_json} already exists (use --overwrite to replace)")
        n_img, n_ann = convert_split(img_dir, names, nc, out_dir / split, out_json, args.link)
        print(f"[{split}] {n_img} images, {n_ann} annotations -> {out_json}")
        converted += 1

    if not converted:
        raise SystemExit("No splits found to convert — check the data.yaml train/val/test keys")
    print(f"Done. COCO dataset written to {out_dir} ({len(names)} categories)")


if __name__ == "__main__":
    main()
