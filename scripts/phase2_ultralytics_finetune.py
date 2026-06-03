"""Finetune YOLOv11s with the official Ultralytics API on the 3-class subset."""
import yaml
from ultralytics import YOLO

cfg = yaml.safe_load(open('configs/training/finetune_subset3.yaml'))
model = YOLO('weights/yolov11s_official.pt')

results = model.train(
    data='dataset/subset3/data.yaml',
    epochs=cfg['epochs'], batch=cfg['batch'], imgsz=cfg['imgsz'],
    lr0=cfg['lr0'], lrf=cfg['lrf'], momentum=cfg['momentum'],
    weight_decay=cfg['weight_decay'],
    warmup_epochs=cfg['warmup_epochs'], warmup_momentum=cfg['warmup_momentum'],
    warmup_bias_lr=cfg['warmup_bias_lr'],
    box=cfg['box'], cls=cfg['cls'], dfl=cfg['dfl'],
    mosaic=cfg['mosaic'], close_mosaic=cfg['close_mosaic'],
    hsv_h=cfg['hsv_h'], hsv_s=cfg['hsv_s'], hsv_v=cfg['hsv_v'],
    fliplr=cfg['fliplr'], translate=cfg['translate'], scale=cfg['scale'],
    optimizer=cfg['optimizer'], amp=cfg['amp'], seed=cfg['seed'],
    project='runs', name='finetune_ultra_subset3', exist_ok=True, verbose=True,
)
print("Ultralytics finetuning complete")
print(f"Best mAP50   : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
