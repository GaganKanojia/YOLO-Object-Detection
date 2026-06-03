#!/usr/bin/env python3
"""
Validation entry point.

Example:
    # COCO format:
    python val.py \
        --weights runs/train/exp/best.pt \
        --model configs/model/yolov11n.yaml \
        --val-img-dir /data/coco/val2017 \
        --val-ann /data/coco/annotations/instances_val2017.json

    # YOLO format:
    python val.py \
        --weights runs/train/exp/best.pt \
        --model configs/model/yolov11n.yaml \
        --val-yaml dataset/coco128/data.yaml
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import torch
from models.yolov11 import load_model
from data.loaders import build_dataloader
from data.dataset import build_dataset
from engine.validator import Validator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model", default="configs/model/yolov11n.yaml")
    p.add_argument("--config", default=None,
                   help="Training-config YAML (COCO or YOLO) to read the val "
                        "dataset, model and hyperparameters from")
    p.add_argument("--val-img-dir", default=None, help="COCO format: validation images dir")
    p.add_argument("--val-ann", default=None, help="COCO format: validation JSON annotations")
    p.add_argument("--val-yaml", default=None, help="YOLO format: data.yaml descriptor")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # A --config training YAML (COCO or YOLO) can supply the model, dataset and
    # validation hyperparameters; explicit CLI flags still take precedence.
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    model_path = cfg.get("model", args.model)
    imgsz   = cfg.get("imgsz", args.imgsz)
    conf    = cfg.get("conf", args.conf)
    iou     = cfg.get("iou", args.iou)
    max_det = cfg.get("max_det", args.max_det)

    model = load_model(model_path, weights=args.weights, device=str(device))
    model.eval()

    # Pick dataset format: --config keys first, then explicit flags.
    val_yaml = cfg.get("val_yaml") or args.val_yaml
    val_img_dir = cfg.get("val_img_dir") or args.val_img_dir
    val_ann = cfg.get("val_ann_file") or cfg.get("val_ann") or args.val_ann
    if val_yaml:
        ds_args = {"val_yaml": val_yaml, "imgsz": imgsz}
    elif val_img_dir and val_ann:
        ds_args = {
            "val_img_dir": val_img_dir,
            "val_ann_file": val_ann,
            "imgsz": imgsz,
            "nc": cfg.get("nc", getattr(model, "nc", None)),
        }
    else:
        raise SystemExit(
            "Provide either --val-yaml / 'val_yaml' (YOLO format) or "
            "--val-img-dir + --val-ann / 'val_img_dir' + 'val_ann_file' (COCO format)."
        )

    dataset = build_dataset(ds_args, split="val", augment=False)
    loader, _ = build_dataloader(
        dataset,
        batch_size=args.batch,
        workers=args.workers,
        augment=False,
        shuffle=False,
    )

    validator = Validator(
        model=model,
        dataloader=loader,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
    metrics = validator.run()
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"mAP@50:    {metrics['mAP50']:.4f}")
    print(f"mAP@50:95: {metrics['mAP50_95']:.4f}")


if __name__ == "__main__":
    main()
