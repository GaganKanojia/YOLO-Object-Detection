#!/usr/bin/env python3
"""
Convert a folder of YOLO-format predictions into COCO detection-results JSON.

Input format
────────────
Predictions: a folder of .txt files, one per image, named after the image's
filename stem (predictions for img001.jpg live in <pred-dir>/img001.txt).
Each line is one detection:

    <class_id> <cx> <cy> <w> <h> <confidence>

with class_id 0-indexed, cx/cy/w/h normalized to [0, 1] (Ultralytics
--save-txt --save-conf format) and confidence in [0, 1].

--gt-json is the COCO ground-truth annotation file for the same split (as
produced by scripts/yolo_to_coco.py). It supplies each image's COCO image_id
and width/height, guaranteeing the ids match what pycocotools will index —
no ids are re-derived here and no image files are read.

Output format
─────────────
The standard COCO detection-results array accepted by
pycocotools COCO.loadRes / COCOeval:

    [
      {"image_id": int, "category_id": int,   # yolo class_id + 1
       "bbox": [x, y, w, h],                  # absolute pixels, top-left
       "score": float},
      ...
    ]

Usage
─────
    python scripts/yolo_predictions_to_coco.py --pred-dir runs/preds/val \
        --gt-json dataset_coco/annotations/instances_val.json \
        --out predictions_val.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Convert YOLO-format predictions (class cx cy w h conf) to COCO results JSON")
    ap.add_argument("--pred-dir", required=True,
                    help="folder of per-image prediction .txt files (named by image stem)")
    ap.add_argument("--gt-json", required=True,
                    help="COCO GT annotations for the same split (supplies image ids and sizes)")
    ap.add_argument("--out", required=True, help="output COCO results JSON path")
    args = ap.parse_args()

    with open(args.gt_json) as f:
        gt = json.load(f)
    stem_to_img = {Path(im["file_name"]).stem: (im["id"], im["width"], im["height"])
                   for im in gt["images"]}
    valid_cats = {int(c["id"]) for c in gt["categories"]}

    pred_dir = Path(args.pred_dir)
    txts = sorted(pred_dir.glob("*.txt"))
    if not txts:
        raise SystemExit(f"No .txt prediction files found in {pred_dir}")

    results, skipped_files, covered = [], 0, set()
    for txt in txts:
        info = stem_to_img.get(txt.stem)
        if info is None:
            print(f"warning: {txt.name} matches no image in {args.gt_json} — skipping "
                  f"(loadRes fails on unknown image ids)")
            skipped_files += 1
            continue
        img_id, width, height = info
        covered.add(txt.stem)

        for ln, line in enumerate(txt.read_text().strip().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                print(f"warning: {txt}:{ln}: expected 6 values "
                      f"(class cx cy w h conf), got {len(parts)} — skipping")
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h, conf = (float(v) for v in parts[1:6])
            except ValueError:
                print(f"warning: {txt}:{ln}: non-numeric value — skipping")
                continue
            cat_id = cls + 1  # COCO IDs are 1-indexed
            if cat_id not in valid_cats:
                print(f"warning: {txt}:{ln}: class_id {cls} has no category "
                      f"{cat_id} in the GT JSON — skipping")
                continue
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round((cx - w / 2) * width, 2), round((cy - h / 2) * height, 2),
                         round(w * width, 2), round(h * height, 2)],
                "score": round(min(max(conf, 0.0), 1.0), 5),
            })

    with open(args.out, "w") as f:
        json.dump(results, f)

    uncovered = len(stem_to_img) - len(covered)
    print(f"Detections written : {len(results)} -> {args.out}")
    print(f"Images covered     : {len(covered)}/{len(stem_to_img)} "
          f"({uncovered} GT images without a prediction file)")
    if skipped_files:
        print(f"Skipped pred files : {skipped_files} (no matching image in GT)")
    if not results:
        print("WARNING: no detections written — pycocotools loadRes will fail on an "
              "empty results array", file=sys.stderr)


if __name__ == "__main__":
    main()
