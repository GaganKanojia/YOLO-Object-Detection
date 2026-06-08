# YOLO Object Detection — Pure PyTorch Implementation

A faithful reimplementation of the **YOLOv11** and **YOLOv9** object detection
pipelines in pure PyTorch. Both architectures share a single training, validation,
and inference entry point — the only difference is the `--model` flag.

## Requirements

```bash
pip install -r requirements.txt
```

Supported Python ≥ 3.9, PyTorch ≥ 2.0.

---

## Supported Architectures

### YOLOv11

| Variant | Params | GFLOPs |
|---------|--------|--------|
| n | ~2.6M | ~6.6 |
| s | ~9.5M | ~21.7 |
| m | ~20.1M | ~68.5 |
| l | ~25.4M | ~87.6 |
| x | ~57.0M | ~196.0 |

- **Backbone**: Conv → Conv → C3k2 × 5 → SPPF → C2PSA
- **Neck**: FPN top-down + PAN bottom-up
- **Head**: Decoupled anchor-free Detect with DFL, 3 scales (P3/8, P4/16, P5/32)

### YOLOv9 (GELAN)

| Variant | Params | GFLOPs |
|---------|--------|--------|
| t | ~2.1M | ~8.2 |
| s | ~7.3M | ~26.7 |
| m | ~20.2M | ~77.9 |
| c | ~25.6M | ~104.0 |
| e | ~58.2M | ~193.0 |

- **Backbone**: RepNCSPELAN4 GELAN blocks + AConv/ADown downsampling + SPPELAN
- **Neck/Head**: Fused into a single `head:` section in the YAML (no separate neck)
- **yolov9e only**: PGI (Programmable Gradient Information) via CBLinear/CBFuse feature injection — always active, including at inference
- **Detection head**: Legacy plain-Conv cv3 (`Conv→Conv→Conv2d`) matching the Ultralytics reference

Channel counts are hardcoded per variant (no `depth_multiple`/`width_multiple`).

---

## Architecture Selection

Both architectures are selected via the `--model` flag. The YAML's `arch:` key
drives dispatch inside `load_model`:

```
arch: yolov9   → YOLOv9  class (models/yolov9.py)
arch: yolov11  → YOLOv11 class (models/yolov11.py)   ← default when key absent
```

All callers (`train.py`, `val.py`, `detect.py`) use `load_model` — no other
changes are needed to switch between architectures.

---

## Project Structure

```
├── configs/
│   ├── model/
│   │   ├── yolov11{n,s,m,l,x}.yaml     # YOLOv11 variants
│   │   └── yolov9{t,s,m,c,e}.yaml      # YOLOv9 variants
│   └── training/
│       ├── default.yaml                 # all hyperparameters + defaults
│       ├── smoke_test_yolo.yaml         # quick YOLOv11 smoke test
│       └── smoke_test_yolov9.yaml       # quick YOLOv9 smoke test
├── models/
│   ├── blocks.py          # Conv, C3k2, C2PSA, SPPF, DFL, ... (YOLOv11 blocks)
│   ├── head.py            # Detect head (YOLOv11 / non-legacy cv3)
│   ├── yolov11.py         # YOLOv11 model + parse_model + load_model factory
│   ├── yolov9_blocks.py   # GELAN blocks: RepNCSPELAN4, ELAN1, AConv, ADown,
│   │                      #   SPPELAN, CBLinear, CBFuse, DetectV9
│   └── yolov9.py          # YOLOv9 model + parse_model_v9 + MODULE_MAP_V9
├── data/
│   ├── dataset.py         # YOLODataset (COCO JSON) + YOLOTxtDataset (YOLO TXT)
│   ├── augmentations.py   # Mosaic, MixUp, RandomPerspective, HSV, flip, letterbox
│   └── loaders.py         # DataLoader + collate_fn
├── loss/
│   ├── tal.py             # TaskAlignedAssigner (topk=10, alpha=0.5, beta=6.0)
│   └── loss.py            # DetectionLoss: BCE + CIoU + DFL (works for both archs)
├── engine/
│   ├── trainer.py         # Training loop: warmup, AMP, EMA, LR scheduler
│   └── validator.py       # Validation: NMS, per-class mAP
├── utils/
│   ├── metrics.py         # mAP@50, mAP@50:95, per-class AP
│   ├── nms.py             # Non-Maximum Suppression
│   ├── bbox.py            # IoU variants (CIoU, DIoU, GIoU), box conversions
│   ├── general.py         # EMA, optimizer builder, LR schedules, logging
│   └── transfer.py        # Pretrained weight loading (partial / backbone_neck / subset)
├── export/
│   └── exporter.py        # ONNX, TorchScript, TensorRT export
├── train.py               # CLI: training
├── val.py                 # CLI: validation
└── detect.py              # CLI: inference
```

---

## Data Format

Two dataset formats are supported, selected automatically from the config keys provided.

### 1. COCO JSON format

```
data/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 2. YOLO TXT format (Roboflow / Ultralytics ecosystem)

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/        # one .txt per image: "class_id cx cy w h" per line
    └── val/
```

`data.yaml` example:
```yaml
path: ./dataset
train: images/train
val:   images/val
nc: 3
names: ['cat', 'dog', 'bird']
```

### Config keys

| | COCO JSON | YOLO TXT |
|---|---|---|
| train images | `train_img_dir` | *(from data.yaml)* |
| train labels | `train_ann_file` (.json) | *(from data.yaml)* |
| val images | `val_img_dir` | *(from data.yaml)* |
| val labels | `val_ann_file` (.json) | *(from data.yaml)* |
| descriptor | — | `train_yaml` (+ optional `val_yaml`) |
| `nc` | required | inferred from `data.yaml` |

---

## Training

```bash
# Single-config form (recommended)
python train.py --config configs/training/smoke_test_yolo.yaml

# YOLOv9 — same interface, different model key
python train.py --config configs/training/smoke_test_yolov9.yaml

# Explicit CLI form
python train.py \
    --model configs/model/yolov9c.yaml \
    --train-img-dir /data/coco/train2017 \
    --train-ann /data/coco/annotations/instances_train2017.json \
    --val-img-dir /data/coco/val2017 \
    --val-ann /data/coco/annotations/instances_val2017.json \
    --epochs 100 --batch 16 --imgsz 640 --device cuda:0

# Resume from checkpoint
python train.py --config configs/training/my_run.yaml --resume runs/train/exp/last.pt
```

### Fine-tuning from pretrained weights

```yaml
# In your training config YAML:
pretrained_weights : yolov9t.pt    # Ultralytics checkpoint or our own .pt
strategy           : partial       # partial | backbone_neck | subset
```

`strategy=partial` loads every weight whose shape matches and reinitializes the
rest (e.g. the classification head when `nc` changes). This means you can
fine-tune with a **different number of classes** than the pretrained model —
the pipeline handles nc mismatches automatically.

```bash
python train.py --config configs/training/finetune_yolov9t_8cls.yaml
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final LR fraction |
| `momentum` | 0.937 | SGD momentum / Adam β₁ |
| `weight_decay` | 0.0005 | L2 regularization |
| `warmup_epochs` | 3.0 | Warmup duration |
| `box` | 7.5 | Box loss weight |
| `cls` | 0.5 | Classification loss weight |
| `dfl` | 1.5 | Distribution focal loss weight |
| `mosaic` | 1.0 | Mosaic augmentation probability |
| `close_mosaic` | 10 | Disable mosaic last N epochs |
| `nbs` | 64 | Nominal batch size (gradient accumulation target) |
| `amp` | true | Mixed-precision training |

---

## Validation

```bash
# YOLO TXT format
python val.py \
    --weights runs/train/exp/best.pt \
    --model configs/model/yolov9t.yaml \
    --val-yaml datasets/coco128/data.yaml \
    --conf 0.001 --iou 0.7

# COCO JSON format
python val.py \
    --weights runs/train/exp/best.pt \
    --model configs/model/yolov11n.yaml \
    --val-img-dir /data/coco/val2017 \
    --val-ann /data/coco/annotations/instances_val2017.json \
    --conf 0.001 --iou 0.7
```

The validator returns mAP@50, mAP@50:95, precision, recall, and **per-class AP@50**.

### Loading Ultralytics pretrained weights

Both `yolov9t.pt` (and other official Ultralytics checkpoints) can be loaded
directly — `load_model` unwraps the `DetectionModel` object automatically:

```bash
python val.py --weights yolov9t.pt --model configs/model/yolov9t.yaml \
    --val-yaml datasets/coco128/data.yaml
```

---

## Inference

```bash
python detect.py \
    --weights runs/train/exp/best.pt \
    --model configs/model/yolov9c.yaml \
    --source /path/to/images/ \
    --conf 0.25 --iou 0.7 \
    --save-dir runs/detect/exp
```

---

## Export

```python
from models.yolov11 import load_model
from export.exporter import export_onnx, export_torchscript

# Works identically for both architectures
model = load_model("configs/model/yolov9c.yaml", weights="best.pt")
export_onnx(model, "yolov9c.onnx", imgsz=640)
export_torchscript(model, "yolov9c.torchscript", imgsz=640)
```

---

## Accuracy Reference (COCO128, same Ultralytics pretrained weights)

| Model | mAP@50 | mAP@50:95 |
|-------|--------|-----------|
| YOLOv9t (our impl) | 0.610 | 0.456 |
| YOLOv9t (Ultralytics) | 0.606 | 0.453 |
| YOLOv9t fine-tuned 20ep | 0.782 | 0.610 |

Delta vs Ultralytics is <0.4% mAP — attributable to NMS implementation
differences and a 2-image dataset discrepancy (missing label files).

---

## Implementation Notes

### YOLOv9 design decisions

- **No `depth_multiple` / `width_multiple`**: channel counts are hardcoded per
  variant in each YAML, matching the original paper.
- **PGI (yolov9e only)**: CBLinear/CBFuse layers are feature-enhancement blocks
  in the backbone — they are *not* an auxiliary training loss. They remain active
  at inference.
- **DetectV9**: YOLOv9 uses a legacy plain-Conv classification branch
  (`Conv(x,c3,3)→Conv(c3,c3,3)→Conv2d(c3,nc,1)`) instead of YOLOv11's
  DWConv-based branch. `DetectV9` handles this without touching the existing
  `Detect` class.
- **Ultralytics checkpoint compatibility**: key names and tensor shapes are
  identical to Ultralytics' `DetectionModel` state dict — weights load with
  `strict=True` equivalence.

### YOLOv11 path is unchanged

Every change is additive: new files (`yolov9.py`, `yolov9_blocks.py`, 5 YAMLs),
plus a 5-line dispatch branch in `load_model`. All existing YOLOv11 commands
produce identical results.
