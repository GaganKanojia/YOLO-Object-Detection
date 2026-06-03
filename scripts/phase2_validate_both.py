"""Validate both finetuned models with identical NMS params (conf=0.5, iou=0.5)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import glob
import torch
from pathlib import Path
from ultralytics import YOLO

CONF, IOU, IMGSZ, MAX_DET, NC = 0.5, 0.5, 640, 300, 3
NAMES = ['person', 'car', 'dog']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── CUSTOM finetuned model ────────────────────────────────────────────
print("=" * 55)
print("VALIDATING CUSTOM FINETUNED MODEL")
print("=" * 55)
from models.yolov11 import YOLOv11
from data.dataset import YOLODataset
from data.loaders import build_dataloader
from engine.validator import Validator

# Newest custom run dir (Trainer may append a number via increment_path).
cand = sorted(glob.glob('runs/finetune_custom_subset3*/weights/best.pt'))
custom_best = cand[-1]
print(f"Loading: {custom_best}")
model_c = YOLOv11('configs/model/yolov11s.yaml', nc=NC).to(DEVICE)
ckpt_c = torch.load(custom_best, map_location='cpu', weights_only=False)
state_c = ckpt_c.get('ema') or ckpt_c['model']
model_c.load_state_dict(state_c)
model_c.eval()

val_ds_c = YOLODataset(img_dir='dataset/subset3/images/val',
                       ann_file='dataset/subset3/annotations_val.json',
                       nc=NC, imgsz=IMGSZ, augment=False)
loader_c, _ = build_dataloader(val_ds_c, batch_size=4, workers=2, augment=False, shuffle=False)

validator_c = Validator(model=model_c, dataloader=loader_c, device=DEVICE,
                        conf=CONF, iou=IOU, max_det=MAX_DET,
                        names={i: n for i, n in enumerate(NAMES)})
res_c = validator_c.run()
custom_metrics = {
    'mAP50': round(float(res_c['mAP50']), 4),
    'mAP50_95': round(float(res_c['mAP50_95']), 4),
    'precision': round(float(res_c['precision']), 4),
    'recall': round(float(res_c['recall']), 4),
    'per_class': res_c.get('per_class', {}),
}
json.dump({'implementation': 'custom', 'metrics': custom_metrics},
          open('results/phase2_custom_metrics.json', 'w'), indent=2)
for k in ['mAP50', 'mAP50_95', 'precision', 'recall']:
    print(f"  {k:<10}: {custom_metrics[k]}")

# ── ULTRALYTICS finetuned model ───────────────────────────────────────
print("\n" + "=" * 55)
print("VALIDATING ULTRALYTICS FINETUNED MODEL")
print("=" * 55)
# Ultralytics may write under its own runs tree (e.g. /ultralytics/runs/detect/...).
ult_cands = []
for root in ['runs', '/ultralytics/runs', Path.home() / 'runs']:
    ult_cands += list(Path(root).rglob('finetune_ultra_subset3*/weights/best.pt'))
ultra_best = sorted(ult_cands, key=lambda p: p.stat().st_mtime)[-1]
print(f"Loading: {ultra_best}")
model_u = YOLO(str(ultra_best))
metrics_u = model_u.val(data='dataset/subset3/data.yaml', imgsz=IMGSZ, conf=CONF,
                        iou=IOU, max_det=MAX_DET, split='val', rect=False,
                        half=False, plots=False, verbose=True)

idx = list(metrics_u.box.ap_class_index)
ap50 = metrics_u.box.ap50
p = metrics_u.box.p
r = metrics_u.box.r
per_class = {}
for row, cls_id in enumerate(idx):
    name = NAMES[int(cls_id)] if int(cls_id) < len(NAMES) else f'class_{cls_id}'
    per_class[name] = {
        'ap50': round(float(ap50[row]), 4),
        'precision': round(float(p[row]), 4),
        'recall': round(float(r[row]), 4),
    }
ultra_metrics = {
    'mAP50': round(float(metrics_u.box.map50), 4),
    'mAP50_95': round(float(metrics_u.box.map), 4),
    'precision': round(float(metrics_u.box.mp), 4),
    'recall': round(float(metrics_u.box.mr), 4),
    'per_class': per_class,
}
json.dump({'implementation': 'ultralytics', 'metrics': ultra_metrics},
          open('results/phase2_ultra_metrics.json', 'w'), indent=2)
for k in ['mAP50', 'mAP50_95', 'precision', 'recall']:
    print(f"  {k:<10}: {ultra_metrics[k]}")
