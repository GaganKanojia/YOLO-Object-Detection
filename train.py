#!/usr/bin/env python3
"""
Training entry point.

Two ways to invoke:

1. Single-config form (used by smoke tests / end-to-end runs):
    python train.py --config configs/training/smoke_test.yaml

   The YAML must contain dataset paths (train_img_dir, train_ann_file,
   val_img_dir, val_ann_file) and any training overrides.

2. Explicit CLI form (used for ad-hoc runs):
    python train.py \
        --model configs/model/yolov11n.yaml \
        --train-img-dir /data/coco/train2017 \
        --train-ann /data/coco/annotations/instances_train2017.json \
        --val-img-dir /data/coco/val2017 \
        --val-ann /data/coco/annotations/instances_val2017.json \
        --epochs 100 --batch 16 --device cuda:0
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yaml

from engine.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="Path to a single training-config YAML containing both dataset "
                        "paths and training hyperparameters")
    p.add_argument("--model", default="configs/model/yolov11n.yaml")
    p.add_argument("--cfg-train", default="configs/training/default.yaml")
    p.add_argument("--train-img-dir", default=None)
    p.add_argument("--train-ann", default=None)
    p.add_argument("--val-img-dir", default=None)
    p.add_argument("--val-ann", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    # Fine-tuning / transfer learning
    p.add_argument("--weights", type=str, default=None,
                   help="Path to pretrained .pt file for fine-tuning")
    p.add_argument("--strategy", type=str, default=None,
                   choices=["partial", "backbone_neck", "subset"],
                   help="Weight loading strategy (default: config value or 'partial')")
    p.add_argument("--nc", type=int, default=None,
                   help="Number of classes for fine-tuning")
    p.add_argument("--freeze", type=int, default=0,
                   help="Freeze the backbone for the first N epochs (0=none)")
    return p.parse_args()


def main():
    args = parse_args()

    # If --config is given, the YAML is the source of truth for both dataset and
    # training hyperparameters. The YAML overlays the defaults in --cfg-train.
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg_train_path = args.cfg_train
        model_path = cfg.get("model", args.model)
        train_img_dir = cfg.get("train_img_dir") or args.train_img_dir
        train_ann = cfg.get("train_ann_file") or cfg.get("train_ann") or args.train_ann
        val_img_dir = cfg.get("val_img_dir") or args.val_img_dir
        val_ann = cfg.get("val_ann_file") or cfg.get("val_ann") or args.val_ann
        device = cfg.get("device") or args.device or "cuda"
        # The single-config YAML may also carry every other training override.
        extra_overrides = {
            k: v for k, v in cfg.items()
            if k not in {
                "model", "train_img_dir", "train_ann_file", "train_ann",
                "val_img_dir", "val_ann_file", "val_ann", "device",
            }
        }
    else:
        cfg_train_path = args.cfg_train
        model_path = args.model
        train_img_dir = args.train_img_dir
        train_ann = args.train_ann
        val_img_dir = args.val_img_dir
        val_ann = args.val_ann
        device = args.device
        extra_overrides = {}

    # YOLO format is selected by a data.yaml (train_yaml/val_yaml) carried in the
    # config. In that mode the COCO dataset paths are not required.
    yolo_mode = bool(extra_overrides.get("train_yaml") or extra_overrides.get("val_yaml"))

    if yolo_mode:
        overrides = {}
    else:
        missing = [
            ("--train-img-dir / train_img_dir", train_img_dir),
            ("--train-ann / train_ann_file",    train_ann),
            ("--val-img-dir / val_img_dir",     val_img_dir),
            ("--val-ann / val_ann_file",        val_ann),
        ]
        missing_keys = [name for name, val in missing if not val]
        if missing_keys:
            raise SystemExit(
                f"Missing required dataset paths: {missing_keys}\n"
                f"  Provide COCO paths (train_img_dir + train_ann_file + val_*) "
                f"or a YOLO 'train_yaml' in the config."
            )
        overrides = {
            "train_img_dir": train_img_dir,
            "train_ann": train_ann,
            "val_img_dir": val_img_dir,
            "val_ann": val_ann,
        }
    overrides.update(extra_overrides)

    # CLI flags override config values when explicitly given.
    for k in ("epochs", "batch", "imgsz", "workers", "project", "name"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v

    # Fine-tuning / transfer-learning CLI overrides.
    if args.weights is not None:
        overrides["pretrained_weights"] = args.weights
    if args.strategy is not None:
        overrides["strategy"] = args.strategy
    if args.nc is not None:
        overrides["nc"] = args.nc
    if args.freeze and args.freeze > 0:
        overrides["freeze_backbone"] = True
        overrides["freeze_backbone_epochs"] = args.freeze

    trainer = Trainer(
        cfg_model=model_path,
        cfg_train=cfg_train_path,
        data_cfg=overrides,
        device=device or "cuda",
    )

    if args.resume:
        trainer.resume = args.resume

    trainer.train()


if __name__ == "__main__":
    main()
