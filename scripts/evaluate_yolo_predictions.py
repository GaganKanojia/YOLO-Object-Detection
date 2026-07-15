#!/usr/bin/env python3
"""
Evaluate a folder of YOLO-format predictions against a YOLO dataset split.

Inputs
──────
Ground truth: a YOLO dataset described by data.yaml (see scripts/yolo_to_coco.py
for the full layout). The split to evaluate (train/val/test) is selected with
--split; ground-truth labels are found via the standard images/ -> labels/
convention.

Predictions: a folder of .txt files, one per image, named after the image's
filename stem (predictions for images/val/img001.jpg live in
<pred-dir>/img001.txt). Each line is one detection:

    <class_id> <cx> <cy> <w> <h> <confidence>

with class_id 0-indexed, cx/cy/w/h normalized to [0, 1] (Ultralytics
--save-txt --save-conf format) and confidence in [0, 1]. A missing prediction
file means "no detections for that image" — the image still counts (its GT
boxes become false negatives).

Metrics
───────
Always reported, using the repo's Ultralytics-aligned pipeline
(utils/metrics.py — 101-point COCO interpolation in compute_ap):
  - mAP@50, mAP@50:95 (IoU 0.50:0.05:0.95)
  - overall precision / recall (at the confidence maximizing mean F1)
  - per-class AP@50, AP@50:95, precision, recall

Optionally, with --class-id, --iou-thresholds and --conf-thresholds (all three
required together): for every (IoU threshold, confidence) pair, TP / FP / FN /
TN / precision / recall for that single class.
  - TP/FP/FN are box-level: detections of the class with conf >= threshold are
    greedily matched (confidence-descending) to unmatched same-class GT boxes
    at IoU >= threshold.
  - TN is image-level: an image counts as one TN when it has no GT box of the
    class AND no kept detection of the class (box-level TN is undefined in
    object detection).

Usage
─────
    python scripts/evaluate_yolo_predictions.py --yaml dataset/data.yaml \
        --split val --pred-dir runs/preds/val

    python scripts/evaluate_yolo_predictions.py --yaml dataset/data.yaml \
        --split val --pred-dir runs/preds/val \
        --class-id 0 --iou-thresholds 0.5 0.75 --conf-thresholds 0.25 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from data.dataset import IMG_EXTS, find_label_path, parse_yolo_label, parse_yolo_yaml
from utils.bbox import bbox_iou, xywh2xyxy
from utils.metrics import ap_per_class, DetectionMetrics


def resolve_split_dir(yaml_path: str, split: str) -> str:
    """
    Resolve the images dir for an arbitrary split key (train/val/test).
    Replicates parse_yolo_yaml's candidate-root order (which only handles
    train/val): yaml_dir/path/rel, cwd/path/rel, yaml_dir/rel, cwd/rel.
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


def load_predictions(txt_path: Path, nc: int) -> np.ndarray:
    """
    Parse one prediction .txt (lines: class cx cy w h conf, normalized).
    Returns [N, 6] float32 (cls, cx, cy, w, h, conf); (0, 6) if missing/empty.
    Validation mirrors data.dataset.parse_yolo_label.
    """
    empty = np.zeros((0, 6), dtype=np.float32)
    if not txt_path.exists():
        return empty
    rows = []
    for ln, line in enumerate(txt_path.read_text().strip().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 6:
            print(f"  warning: {txt_path}:{ln}: expected 6 values "
                  f"(class cx cy w h conf), got {len(parts)} — skipping")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h, conf = (float(v) for v in parts[1:6])
        except ValueError:
            print(f"  warning: {txt_path}:{ln}: non-numeric value — skipping")
            continue
        if cls < 0 or cls >= nc:
            print(f"  warning: {txt_path}:{ln}: class_id {cls} out of range [0,{nc}) — skipping")
            continue
        if w <= 0 or h <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(w, 1.0), min(h, 1.0)
        conf = min(max(conf, 0.0), 1.0)
        rows.append([float(cls), cx, cy, w, h, conf])
    return np.array(rows, dtype=np.float32) if rows else empty


def collect_per_image(img_dir: str, pred_dir: Path, nc: int) -> list:
    """
    Returns a list of (gt, pred) per image, both in normalized xyxy:
      gt:   [M, 5] (cls, x1, y1, x2, y2)
      pred: [N, 6] (x1, y1, x2, y2, conf, cls)

    Note: no image files are read — IoU is invariant to per-image x/y
    normalization, so matching works directly in normalized coordinates.
    """
    imgs = sorted(p for p in Path(img_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"No images found in {img_dir}")

    stems = {p.stem for p in imgs}
    orphans = [t.name for t in pred_dir.glob("*.txt") if t.stem not in stems]
    if orphans:
        print(f"warning: {len(orphans)} prediction file(s) match no image in the split "
              f"(wrong --split or --pred-dir?), e.g. {orphans[:3]}")

    per_image = []
    for img_path in imgs:
        raw_gt = parse_yolo_label(find_label_path(str(img_path)), nc)  # [M,5] cls,cxcywh
        gt = np.zeros((0, 5), dtype=np.float32)
        if len(raw_gt):
            gt = np.concatenate([raw_gt[:, :1], xywh2xyxy(raw_gt[:, 1:5])], axis=1)

        raw_pred = load_predictions(pred_dir / (img_path.stem + ".txt"), nc)  # [N,6] cls,cxcywh,conf
        pred = np.zeros((0, 6), dtype=np.float32)
        if len(raw_pred):
            pred = np.concatenate([xywh2xyxy(raw_pred[:, 1:5]),
                                   raw_pred[:, 5:6], raw_pred[:, :1]], axis=1)
        per_image.append((gt, pred))
    return per_image


def global_metrics(per_image: list, nc: int, names: list) -> dict:
    """mAP@50, mAP@50:95, P, R and per-class AP via the repo's Ultralytics-style pipeline."""
    metrics = DetectionMetrics(nc, names={i: n for i, n in enumerate(names)})
    for gt, pred in per_image:
        targets = torch.zeros((len(gt), 6))
        if len(gt):
            targets[:, 1:] = torch.from_numpy(gt)  # batch_idx stays 0
        metrics.update([torch.from_numpy(pred)], targets)

    empty = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0, "per_class": {}}
    if not metrics.stats:
        return empty
    tp = torch.cat([s[0] for s in metrics.stats]).numpy()
    conf = torch.cat([s[1] for s in metrics.stats]).numpy()
    pred_cls = torch.cat([s[2] for s in metrics.stats]).numpy()
    target_cls = torch.cat([s[3] for s in metrics.stats]).numpy()
    if tp.shape[0] == 0 or target_cls.shape[0] == 0:
        return empty

    p, r, ap, unique_classes = ap_per_class(tp, conf, pred_cls, target_cls)
    per_class = {}
    for ci, c in enumerate(unique_classes):
        per_class[names[int(c)]] = {
            "ap50": float(ap[ci, 0]),
            "ap50_95": float(ap[ci].mean()),
            "precision": float(p[ci]),
            "recall": float(r[ci]),
        }
    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50_95": float(ap.mean()),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
        "per_class": per_class,
    }


def confusion_for_class(per_image: list, class_id: int, iou_t: float, conf_t: float) -> dict:
    """
    Box-level TP/FP/FN and image-level TN for one class at one (IoU, conf) pair.
    Detections are matched greedily in confidence-descending order; each takes
    the unmatched GT with the highest IoU >= iou_t.
    """
    tp = fp = fn = tn = 0
    for gt, pred in per_image:
        g = gt[gt[:, 0] == class_id][:, 1:5]  # [M,4] xyxy
        d = pred[(pred[:, 5] == class_id) & (pred[:, 4] >= conf_t)]
        d = d[np.argsort(-d[:, 4])][:, :4]    # [N,4] xyxy, conf desc

        if len(g) == 0 and len(d) == 0:
            tn += 1
            continue
        matched = 0
        if len(g) and len(d):
            iou = bbox_iou(torch.from_numpy(d).unsqueeze(1),
                           torch.from_numpy(g).unsqueeze(0), xywh=False).numpy()  # [N,M]
            taken = np.zeros(len(g), dtype=bool)
            for di in range(len(d)):
                cand = np.where((iou[di] >= iou_t) & ~taken)[0]
                if len(cand):
                    taken[cand[np.argmax(iou[di][cand])]] = True
                    matched += 1
        tp += matched
        fp += len(d) - matched
        fn += len(g) - matched
    return {
        "iou_threshold": iou_t, "conf_threshold": conf_t,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate YOLO-format predictions (class cx cy w h conf) against a YOLO dataset split")
    ap.add_argument("--yaml", required=True, help="path to the YOLO data.yaml")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"],
                    help="dataset split the predictions were run on")
    ap.add_argument("--pred-dir", required=True,
                    help="folder of per-image prediction .txt files (named by image stem)")
    ap.add_argument("--class-id", type=int, default=None,
                    help="0-indexed class for the per-(IoU, conf) confusion analysis")
    ap.add_argument("--iou-thresholds", type=float, nargs="+", default=None,
                    help="IoU thresholds for the confusion analysis, e.g. 0.5 0.75")
    ap.add_argument("--conf-thresholds", type=float, nargs="+", default=None,
                    help="confidence thresholds for the confusion analysis, e.g. 0.25 0.5")
    ap.add_argument("--json-out", default=None, help="optionally dump all results to a JSON file")
    args = ap.parse_args()

    confusion_args = (args.class_id, args.iou_thresholds, args.conf_thresholds)
    if any(a is not None for a in confusion_args) and not all(a is not None for a in confusion_args):
        ap.error("--class-id, --iou-thresholds and --conf-thresholds must be given together")

    cfg = parse_yolo_yaml(args.yaml)
    nc, names = cfg["nc"], cfg["names"] or [str(i) for i in range(cfg["nc"])]
    if args.class_id is not None and not 0 <= args.class_id < nc:
        ap.error(f"--class-id must be in [0, {nc})")

    img_dir = resolve_split_dir(args.yaml, args.split)
    if not img_dir or not Path(img_dir).is_dir():
        raise SystemExit(f"Images dir for split '{args.split}' not found: {img_dir or '(no yaml key)'}")
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise SystemExit(f"Prediction dir not found: {pred_dir}")

    per_image = collect_per_image(img_dir, pred_dir, nc)
    n_gt = sum(len(g) for g, _ in per_image)
    n_pred = sum(len(p) for _, p in per_image)
    print(f"Evaluating {len(per_image)} images ({n_gt} GT boxes, {n_pred} predictions)\n")

    results = global_metrics(per_image, nc, names)
    print(f"{'mAP@50':>12}: {results['mAP50']:.4f}")
    print(f"{'mAP@50:95':>12}: {results['mAP50_95']:.4f}")
    print(f"{'precision':>12}: {results['precision']:.4f}")
    print(f"{'recall':>12}: {results['recall']:.4f}")

    if results["per_class"]:
        w = max(len(n) for n in results["per_class"]) + 2
        print(f"\nPer-class AP (101-point COCO interpolation):")
        print(f"{'class':<{w}} {'AP@50':>8} {'AP@50:95':>9} {'P':>8} {'R':>8}")
        for name, m in results["per_class"].items():
            print(f"{name:<{w}} {m['ap50']:>8.4f} {m['ap50_95']:>9.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f}")

    if args.class_id is not None:
        rows = [confusion_for_class(per_image, args.class_id, iou_t, conf_t)
                for iou_t in sorted(args.iou_thresholds)
                for conf_t in sorted(args.conf_thresholds)]
        results["confusion"] = {"class_id": args.class_id, "class_name": names[args.class_id],
                                "rows": rows}
        print(f"\nConfusion analysis for class {args.class_id} ('{names[args.class_id]}')")
        print("(TP/FP/FN are box-level; TN is image-level: images with no GT and no "
              "prediction of this class)")
        print(f"{'IoU':>6} {'conf':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} "
              f"{'precision':>10} {'recall':>8}")
        for r in rows:
            print(f"{r['iou_threshold']:>6.2f} {r['conf_threshold']:>6.2f} "
                  f"{r['TP']:>6d} {r['FP']:>6d} {r['FN']:>6d} {r['TN']:>6d} "
                  f"{r['precision']:>10.4f} {r['recall']:>8.4f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
