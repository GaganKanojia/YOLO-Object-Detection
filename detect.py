#!/usr/bin/env python3
"""
Inference entry point.

Example:
    python detect.py \
        --weights runs/train/exp/best.pt \
        --model configs/model/yolov11n.yaml \
        --source /path/to/image_or_dir \
        --conf 0.25 --iou 0.7
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import torch
import numpy as np
from pathlib import Path

from models.yolov11 import load_model
from data.augmentations import letterbox
from utils.nms import non_max_suppression
from utils.bbox import clip_boxes


IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model", default="configs/model/yolov11n.yaml")
    p.add_argument("--source", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-dir", "--save", dest="save_dir", default="runs/detect/exp",
                   help="Directory to write annotated images to (alias: --save)")
    p.add_argument("--hide-labels", action="store_true")
    p.add_argument("--hide-conf", action="store_true")
    return p.parse_args()


def preprocess(img_bgr, imgsz):
    img, ratio, (dw, dh) = letterbox(img_bgr, imgsz, auto=False)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)
    return img, ratio, (dw, dh)


def draw_boxes(img, dets, names, hide_labels=False, hide_conf=False):
    for *xyxy, conf, cls in dets:
        x1, y1, x2, y2 = map(int, xyxy)
        label = "" if hide_labels else (
            f"{names.get(int(cls), int(cls))}" + ("" if hide_conf else f" {conf:.2f}")
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, weights=args.weights, device=str(device))
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.source)
    if source.is_dir():
        paths = [p for p in source.iterdir() if p.suffix.lower() in IMG_FORMATS]
    elif source.suffix.lower() in IMG_FORMATS:
        paths = [source]
    else:
        raise ValueError(f"Unsupported source: {source}")

    for path in paths:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"Could not read {path}, skipping")
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        tensor, ratio, (dw, dh) = preprocess(img_bgr, args.imgsz)
        tensor = tensor.to(device)

        with torch.no_grad():
            out, _ = model(tensor)

        dets = non_max_suppression(
            out, conf_thres=args.conf, iou_thres=args.iou,
            max_det=args.max_det, nc=model.nc
        )[0]

        if dets.shape[0]:
            # Scale back to original image coords
            dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / ratio
            dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / ratio
            clip_boxes(dets[:, :4], (orig_h, orig_w))

        out_img = draw_boxes(img_bgr.copy(), dets.cpu().numpy(),
                              model.names, args.hide_labels, args.hide_conf)

        out_path = save_dir / path.name
        cv2.imwrite(str(out_path), out_img)
        print(f"{path.name}: {len(dets)} detections → {out_path}")


if __name__ == "__main__":
    main()
