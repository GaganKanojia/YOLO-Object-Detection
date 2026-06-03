# YOLOv11 — Pure PyTorch Implementation

A faithful reimplementation of the YOLOv11 object detection pipeline in pure PyTorch.

## Requirements

```bash
pip install -r requirements.txt
```

Supported Python ≥ 3.9, PyTorch ≥ 2.0.

## Project Structure

```
yolov11/
├── configs/
│   ├── model/           # yolov11n/s/m/l/x.yaml
│   └── training/        # default.yaml (all hyperparameters)
├── models/
│   ├── blocks.py        # Conv, C3k2, C2PSA, SPPF, DFL, PSABlock, ...
│   ├── backbone.py      # Backbone architecture documentation
│   ├── neck.py          # FPN+PAN architecture documentation
│   ├── head.py          # Detect head, make_anchors, dist2bbox
│   └── yolov11.py       # Full model assembly from YAML
├── data/
│   ├── dataset.py       # YOLODataset (COCO) + YOLOTxtDataset (YOLO) + build_dataset
│   ├── augmentations.py # Mosaic, MixUp, RandomPerspective, HSV, flip, letterbox
│   └── loaders.py       # DataLoader + collate_fn
├── loss/
│   ├── tal.py           # TaskAlignedAssigner (topk=10, alpha=0.5, beta=6.0)
│   └── loss.py          # DetectionLoss: BCE + CIoU + DFL
├── engine/
│   ├── trainer.py       # Full training loop with warmup, AMP, EMA, scheduler
│   └── validator.py     # Validation loop, NMS, mAP
├── utils/
│   ├── metrics.py       # mAP@50, mAP@50:95, confusion matrix
│   ├── nms.py           # Non-Maximum Suppression
│   ├── bbox.py          # IoU variants (CIoU, DIoU, GIoU), box conversions
│   └── general.py       # EMA, optimizer builder, LR schedules, logging
├── export/
│   └── exporter.py      # ONNX, TorchScript, TensorRT export
├── train.py             # CLI: training
├── val.py               # CLI: validation
└── detect.py            # CLI: inference
```

## Data Format

The pipeline supports **two dataset formats**, selected automatically from the
config keys you provide.

### 1. COCO JSON format

```
data/
├── train2017/           # Training images
├── val2017/             # Validation images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 2. YOLO TXT format (Roboflow / Ultralytics ecosystem)

```
dataset/
├── data.yaml            # dataset descriptor — the entry point
├── images/
│   ├── train/           # training images (.jpg/.png/...)
│   └── val/             # validation images
└── labels/
    ├── train/           # one .txt per image, same stem name
    └── val/
```

Each `labels/.../img.txt` has one row per object: `class_id cx cy w h`
(class 0-indexed, box coords normalized to `[0,1]`). An empty `.txt` is a valid
image with no objects. The `data.yaml`:

```yaml
path: ./dataset          # optional root
train: images/train      # relative to path
val: images/val
nc: 3                    # optional — inferred from names if absent
names: ['cat', 'dog', 'bird']   # list or {0: cat, 1: dog, 2: bird}
```

### Config keys side by side

| | COCO JSON | YOLO TXT |
|---|---|---|
| train images | `train_img_dir` | *(from data.yaml)* |
| train labels | `train_ann_file` (.json) | *(from data.yaml)* |
| val images | `val_img_dir` | *(from data.yaml)* |
| val labels | `val_ann_file` (.json) | *(from data.yaml)* |
| descriptor | — | `train_yaml` (+ optional `val_yaml`) |
| `nc` | required | inferred from `data.yaml` |

## Training

```bash
# COCO JSON format
python train.py \
    --model configs/model/yolov11n.yaml \
    --train-img-dir /data/coco/train2017 \
    --train-ann /data/coco/annotations/instances_train2017.json \
    --val-img-dir /data/coco/val2017 \
    --val-ann /data/coco/annotations/instances_val2017.json \
    --epochs 100 --batch 16 --imgsz 640 --device cuda:0
```

For **YOLO TXT format**, put the dataset descriptor in a single-file config and
run with `--config`:

```yaml
# configs/training/my_yolo.yaml
train_yaml : dataset/coco128/data.yaml
val_yaml   : dataset/coco128/data.yaml   # same file covers both splits
model      : configs/model/yolov11n.yaml
epochs     : 100
batch      : 16
imgsz      : 640
```

```bash
python train.py --config configs/training/my_yolo.yaml
# (a ready-made example lives at configs/training/smoke_test_yolo.yaml)
```

Resume training from checkpoint:
```bash
python train.py ... --resume runs/train/exp/last.pt
```

### Key Hyperparameters (configs/training/default.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final LR fraction (lr = lr0 × lrf at end) |
| `momentum` | 0.937 | SGD momentum / Adam β₁ |
| `weight_decay` | 0.0005 | L2 regularization |
| `warmup_epochs` | 3.0 | Warmup duration |
| `box` | 7.5 | Bounding box loss weight |
| `cls` | 0.5 | Classification loss weight |
| `dfl` | 1.5 | Distribution focal loss weight |
| `mosaic` | 1.0 | Mosaic augmentation probability |
| `close_mosaic` | 10 | Disable mosaic last N epochs |
| `fliplr` | 0.5 | Horizontal flip probability |

## Validation

```bash
# COCO JSON format
python val.py \
    --weights runs/train/exp/best.pt \
    --model configs/model/yolov11n.yaml \
    --val-img-dir /data/coco/val2017 \
    --val-ann /data/coco/annotations/instances_val2017.json \
    --conf 0.001 --iou 0.7

# YOLO TXT format
python val.py \
    --weights runs/train/exp/best.pt \
    --model configs/model/yolov11n.yaml \
    --val-yaml dataset/coco128/data.yaml \
    --conf 0.001 --iou 0.7
```

## Inference

```bash
python detect.py \
    --weights runs/train/exp/best.pt \
    --model configs/model/yolov11n.yaml \
    --source /path/to/images/ \
    --conf 0.25 --iou 0.7 \
    --save-dir runs/detect/exp
```

## Export

```python
from models.yolov11 import load_model
from export.exporter import export_onnx, export_torchscript

model = load_model("configs/model/yolov11n.yaml", weights="runs/train/exp/best.pt")
export_onnx(model, "yolov11n.onnx", imgsz=640)
export_torchscript(model, "yolov11n.torchscript", imgsz=640)
```

## Model Variants

| Variant | depth | width | Params | GFLOPs |
|---------|-------|-------|--------|--------|
| n | 0.50 | 0.25 | ~2.6M | ~6.6 |
| s | 0.50 | 0.50 | ~9.5M | ~21.7 |
| m | 0.50 | 1.00 | ~20.1M | ~68.5 |
| l | 1.00 | 1.00 | ~25.4M | ~87.6 |
| x | 1.00 | 1.50 | ~57.0M | ~196.0 |

## Architecture Overview

- **Backbone**: Conv→Conv→C3k2×5→SPPF→C2PSA (11 layers)
- **Neck**: FPN (top-down upsample + concat) + PAN (bottom-up downsample + concat)
- **Head**: Decoupled anchor-free Detect head with DFL box regression, 3 scales (P3/8, P4/16, P5/32)
- **Loss**: BCE (cls) + CIoU (box) + DFL, assigned via TaskAlignedAssigner (topk=10, α=0.5, β=6.0)
- **Training**: SGD with momentum, linear LR decay, 3-epoch warmup, AMP, EMA (decay=0.9999), gradient clipping (norm=10)
