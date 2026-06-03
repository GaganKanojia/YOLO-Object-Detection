"""
Step 5 — custom implementation validation on COCO128.
NMS params identical to Step 3: conf=0.5, iou=0.5, imgsz=640, max_det=300.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from models.yolov11 import YOLOv11
from data.dataset import YOLODataset
from data.loaders import build_dataloader
from engine.validator import Validator

# ── Fixed NMS params — MUST match Step 3 exactly ──────────────────────
CONF = 0.5
IOU = 0.5
IMGSZ = 640
MAX_DET = 300
# ──────────────────────────────────────────────────────────────────────

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NC = 80

print(f"NMS params: conf={CONF}  iou={IOU}  imgsz={IMGSZ}  max_det={MAX_DET}")
print(f"Device    : {DEVICE}")

# ── Load the verified custom model ────────────────────────────────────
model = YOLOv11('configs/model/yolov11s.yaml', nc=NC).to(DEVICE)
ckpt = torch.load('weights/yolov11s_custom_loaded.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()
print("Custom model loaded")

# ── Build val dataset and dataloader ──────────────────────────────────
val_dataset = YOLODataset(
    img_dir='dataset/coco128/images/train2017',
    ann_file='dataset/coco128/annotations.json',
    nc=NC,
    imgsz=IMGSZ,
    augment=False,   # no augmentation for inference
)
val_loader, _ = build_dataloader(
    val_dataset, batch_size=8, workers=2, augment=False, shuffle=False)
print(f"Val images : {len(val_dataset)}")

# COCO class names (idx -> name) so per-class keys align with the Ultralytics run.
names = val_dataset.names

# ── Run validation ────────────────────────────────────────────────────
validator = Validator(
    model=model,
    dataloader=val_loader,
    device=DEVICE,
    conf=CONF,
    iou=IOU,
    max_det=MAX_DET,
    names=names,
)
results = validator.run()

# ── Extract all metrics ───────────────────────────────────────────────
map50 = results['mAP50']
map5095 = results['mAP50_95']
precision = results['precision']
recall = results['recall']
per_class = results.get('per_class', {})

output = {
    'implementation': 'custom',
    'model': 'yolov11s',
    'dataset': 'coco128 (128 images)',
    'nms_params': {'conf': CONF, 'iou': IOU, 'imgsz': IMGSZ, 'max_det': MAX_DET},
    'summary': {
        'mAP50': round(float(map50), 6),
        'mAP50_95': round(float(map5095), 6),
        'precision': round(float(precision), 6),
        'recall': round(float(recall), 6),
    },
    'per_class': per_class,
}

json.dump(output, open('results/custom_metrics.json', 'w'), indent=2)

print("\n" + "=" * 50)
print("CUSTOM IMPLEMENTATION RESULTS")
print("=" * 50)
print(f"  mAP@50      : {map50:.4f}")
print(f"  mAP@50:95   : {map5095:.4f}")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"\nPer-class results saved ({len(per_class)} classes)")
print("Saved: results/custom_metrics.json")
