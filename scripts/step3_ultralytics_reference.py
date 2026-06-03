"""
Step 3 — Ultralytics reference run on COCO128. Ground-truth metrics.
NMS: conf=0.5, iou=0.5, imgsz=640, max_det=300 (fixed).
rect=False so the input is square-640 letterboxed, matching the custom pipeline.
"""
import json
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── Fixed NMS params ──────────────────────────────────────────────────
CONF = 0.5
IOU = 0.5
IMGSZ = 640
MAX_DET = 300
# ──────────────────────────────────────────────────────────────────────

# Write an absolute-path data.yaml so Ultralytics resolves it against the
# real location (it resolves relative `path:` against its datasets_dir, not CWD).
COCO_ROOT = Path('dataset/coco128').resolve()
abs_yaml = COCO_ROOT / 'data_abs.yaml'
with open('dataset/coco128/data.yaml') as f:
    dy = yaml.safe_load(f)
dy['path'] = str(COCO_ROOT)
with open(abs_yaml, 'w') as f:
    yaml.safe_dump(dy, f, sort_keys=False)
print(f"Using data yaml: {abs_yaml} (path={dy['path']})")

print(f"NMS params: conf={CONF}  iou={IOU}  imgsz={IMGSZ}  max_det={MAX_DET}")
print("Loading official YOLOv11s...")
model = YOLO('weights/yolov11s_official.pt')

print("Running validation on COCO128...")
metrics = model.val(
    data=str(abs_yaml),
    imgsz=IMGSZ,
    conf=CONF,
    iou=IOU,
    max_det=MAX_DET,
    split='train',
    rect=False,      # square-640 letterbox to match the custom pipeline
    half=False,      # full precision for determinism
    augment=False,
    verbose=True,
    plots=False,
)

# ── Top-level metrics ─────────────────────────────────────────────────
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)
precision = float(metrics.box.mp)
recall = float(metrics.box.mr)

# ── Per-class metrics ─────────────────────────────────────────────────
class_names = model.names  # {0:'person', ...}
per_class = {}
ap50_arr = metrics.box.ap50 if hasattr(metrics.box, 'ap50') else []
p_arr = metrics.box.p if hasattr(metrics.box, 'p') else []
r_arr = metrics.box.r if hasattr(metrics.box, 'r') else []
ap_class_index = list(metrics.box.ap_class_index) if hasattr(metrics.box, 'ap_class_index') else list(range(len(ap50_arr)))

# Ultralytics arrays are indexed positionally; ap_class_index maps row → class id.
for row, cls_id in enumerate(ap_class_index):
    name = class_names.get(int(cls_id), f'class_{cls_id}')
    per_class[name] = {
        'ap50': round(float(ap50_arr[row]), 6),
        'precision': round(float(p_arr[row]), 6) if row < len(p_arr) else None,
        'recall': round(float(r_arr[row]), 6) if row < len(r_arr) else None,
    }

output = {
    'implementation': 'ultralytics',
    'model': 'yolov11s',
    'dataset': 'coco128 (128 images, split=train)',
    'nms_params': {'conf': CONF, 'iou': IOU, 'imgsz': IMGSZ, 'max_det': MAX_DET},
    'summary': {
        'mAP50': round(map50, 6),
        'mAP50_95': round(map5095, 6),
        'precision': round(precision, 6),
        'recall': round(recall, 6),
    },
    'per_class': per_class,
}

json.dump(output, open('results/ultralytics_metrics.json', 'w'), indent=2)

print("\n" + "=" * 50)
print("ULTRALYTICS REFERENCE RESULTS")
print("=" * 50)
print(f"  mAP@50      : {map50:.4f}")
print(f"  mAP@50:95   : {map5095:.4f}")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"\nPer-class results saved ({len(per_class)} classes)")
print("Saved: results/ultralytics_metrics.json")
