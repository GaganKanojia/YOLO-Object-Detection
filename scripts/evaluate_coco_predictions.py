#!/usr/bin/env python3
"""
Evaluate COCO-format detection results with pycocotools.

Inputs
──────
--gt-json:   COCO ground-truth annotations (e.g. instances_val.json from
             scripts/yolo_to_coco.py).
--pred-json: COCO detection-results array (from scripts/yolo_predictions_to_coco.py):
             [{"image_id", "category_id", "bbox": [x, y, w, h], "score"}, ...]

Metrics
───────
Always reported, via COCOeval with default parameters:
  - mAP@50:95 (stats[0]), mAP@50 (stats[1]), AR@100 (stats[8])
  - mean precision over the accumulated PR surface
  - per-class AP@50, AP@50:95 and recall@50:95

Note on conventions: precision/recall here are COCO PR-curve averages, which
intentionally differ from scripts/evaluate_yolo_predictions.py's Ultralytics
max-F1-confidence convention. mAP also differs slightly by construction:
Ultralytics integrates a continuously interpolated PR curve (trapz over the
precision envelope), while pycocotools samples 101 fixed recall points with
zero precision beyond the maximum achieved recall. Ultralytics mAP therefore
reads a bit higher, and the gap grows on small datasets with low per-class
recall. The per-(IoU, conf) confusion counts below match
evaluate_yolo_predictions.py exactly — matching semantics are identical.

Optionally, with --class-id, --iou-thresholds and --conf-thresholds (all three
required together): for every (IoU threshold, confidence) pair, TP / FP / FN /
TN / precision / recall for that single class, derived from pycocotools'
matching (COCOeval.evalImgs). --class-id is the 0-indexed YOLO class id and is
mapped internally to COCO category_id = class_id + 1.

A single COCOeval.evaluate() run (with params.iouThrs set to the requested IoU
thresholds) is post-filtered per confidence value. This is exact: COCOeval
matches detections greedily in descending score order, so removing a
low-score suffix of the detection list cannot change higher-scored matches.
  - TP/FP/FN are box-level (from dtMatches/gtIgnore per image).
  - TN is image-level: an image counts as one TN when it has no GT box of the
    class AND no kept detection of the class (box-level TN is undefined in
    object detection).

Usage
─────
    python scripts/evaluate_coco_predictions.py \
        --gt-json dataset_coco/annotations/instances_val.json \
        --pred-json predictions_val.json \
        --class-id 0 --iou-thresholds 0.5 0.75 --conf-thresholds 0.25 0.5
"""

import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def standard_metrics(coco_gt: COCO, coco_dt: COCO) -> dict:
    """mAP / AR / per-class AP from a default-parameter COCOeval run."""
    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.evaluate()
    E.accumulate()
    E.summarize()

    # precision: [T=10 iou, R=101 recall, K classes, A=4 areas, M=3 maxDets]
    precision = E.eval["precision"]
    recall = E.eval["recall"]  # [T, K, A, M]
    valid = precision[precision > -1]

    per_class = {}
    for k, cat_id in enumerate(E.params.catIds):
        name = coco_gt.cats[cat_id]["name"]
        pr = precision[:, :, k, 0, -1]     # area='all', maxDets=100
        pr50 = pr[0][pr[0] > -1]
        rc = recall[:, k, 0, -1]
        per_class[name] = {
            "ap50": float(pr50.mean()) if pr50.size else float("nan"),
            "ap50_95": float(pr[pr > -1].mean()) if (pr > -1).any() else float("nan"),
            "recall50_95": float(rc[rc > -1].mean()) if (rc > -1).any() else float("nan"),
        }
    return {
        "mAP50": float(E.stats[1]),
        "mAP50_95": float(E.stats[0]),
        "precision": float(valid.mean()) if valid.size else 0.0,  # PR-surface average
        "recall": float(E.stats[8]),                              # AR@100
        "per_class": per_class,
    }


def confusion_for_class(coco_gt: COCO, coco_dt: COCO, class_id: int,
                        iou_thresholds: list, conf_thresholds: list) -> list:
    """
    TP/FP/FN (box-level) and TN (image-level) per (IoU, conf) pair for one
    class, from a single COCOeval.evaluate() run parsed via evalImgs.
    """
    cat_id = class_id + 1
    iou_thresholds = sorted(iou_thresholds)

    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.params.catIds = [cat_id]
    E.params.iouThrs = np.array(iou_thresholds)
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()

    # evalImgs is ordered [catIdx][areaIdx][imgIdx]; with one catId, the
    # area='all' slice is the first len(imgIds) entries.
    n_img = len(E.params.imgIds)
    max_det = E.params.maxDets[-1]
    entries = E.evalImgs[:n_img]

    rows = []
    for ti, iou_t in enumerate(iou_thresholds):
        for conf_t in sorted(conf_thresholds):
            tp = fp = fn = tn = 0
            for e in entries:
                if e is None:  # no GT and no detections of this class in the image
                    tn += 1
                    continue
                gt_ig = np.array(e["gtIgnore"], dtype=bool)
                n_gt = int((~gt_ig).sum())
                scores = np.array(e["dtScores"][:max_det])
                keep = scores >= conf_t
                dtm = np.array(e["dtMatches"])[ti, :max_det][keep]
                dt_ig = np.array(e["dtIgnore"], dtype=bool)[ti, :max_det][keep]
                tp_i = int(((dtm > 0) & ~dt_ig).sum())
                fp_i = int(((dtm == 0) & ~dt_ig).sum())
                tp += tp_i
                fp += fp_i
                # Recompute FN from GT count so GTs whose only match fell
                # below the confidence threshold are counted as missed.
                fn += n_gt - tp_i
                if n_gt == 0 and not keep.any():
                    tn += 1
            rows.append({
                "iou_threshold": iou_t, "conf_threshold": conf_t,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate COCO-format detection results with pycocotools")
    ap.add_argument("--gt-json", required=True, help="COCO ground-truth annotations JSON")
    ap.add_argument("--pred-json", required=True, help="COCO detection-results JSON")
    ap.add_argument("--class-id", type=int, default=None,
                    help="0-indexed YOLO class for the per-(IoU, conf) confusion analysis "
                         "(mapped to COCO category_id = class_id + 1)")
    ap.add_argument("--iou-thresholds", type=float, nargs="+", default=None,
                    help="IoU thresholds for the confusion analysis, e.g. 0.5 0.75")
    ap.add_argument("--conf-thresholds", type=float, nargs="+", default=None,
                    help="confidence thresholds for the confusion analysis, e.g. 0.25 0.5")
    ap.add_argument("--json-out", default=None, help="optionally dump all results to a JSON file")
    args = ap.parse_args()

    confusion_args = (args.class_id, args.iou_thresholds, args.conf_thresholds)
    if any(a is not None for a in confusion_args) and not all(a is not None for a in confusion_args):
        ap.error("--class-id, --iou-thresholds and --conf-thresholds must be given together")

    with open(args.pred_json) as f:
        preds = json.load(f)
    if not preds:
        print("Predictions JSON is empty — all metrics are 0.")
        return

    coco_gt = COCO(args.gt_json)
    if args.class_id is not None and (args.class_id + 1) not in coco_gt.cats:
        ap.error(f"class_id {args.class_id} (category_id {args.class_id + 1}) "
                 f"not in GT categories {sorted(coco_gt.cats)}")
    coco_dt = coco_gt.loadRes(args.pred_json)

    results = standard_metrics(coco_gt, coco_dt)
    print(f"\n{'mAP@50':>12}: {results['mAP50']:.4f}")
    print(f"{'mAP@50:95':>12}: {results['mAP50_95']:.4f}")
    print(f"{'precision':>12}: {results['precision']:.4f}  (mean over COCO PR surface)")
    print(f"{'recall':>12}: {results['recall']:.4f}  (AR@100)")

    if results["per_class"]:
        w = max(len(n) for n in results["per_class"]) + 2
        print(f"\nPer-class AP (pycocotools):")
        print(f"{'class':<{w}} {'AP@50':>8} {'AP@50:95':>9} {'R@50:95':>8}")
        for name, m in results["per_class"].items():
            print(f"{name:<{w}} {m['ap50']:>8.4f} {m['ap50_95']:>9.4f} {m['recall50_95']:>8.4f}")

    if args.class_id is not None:
        rows = confusion_for_class(coco_gt, coco_dt, args.class_id,
                                   args.iou_thresholds, args.conf_thresholds)
        name = coco_gt.cats[args.class_id + 1]["name"]
        results["confusion"] = {"class_id": args.class_id, "class_name": name, "rows": rows}
        print(f"\nConfusion analysis for class {args.class_id} ('{name}') via pycocotools")
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
