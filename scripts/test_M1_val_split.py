import sys, torch
from data.dataset import YOLODataset, YOLOTxtDataset
from data.loaders import build_dataloader

results = {}
def check(name, cond, detail=""):
    results[name] = cond
    status = "\033[92mPASS\033[0m" if cond else "\033[91mFAIL\033[0m"
    print(f"  {status} {name}" + (f": {detail}" if detail else ""))

print("="*60); print("T1.2 — VAL SPLIT"); print("="*60)

# COCO val split
try:
    ds_cv = YOLODataset(img_dir='dataset/coco128/images/train2017',
                         ann_file='dataset/coco128/annotations_val.json',
                         nc=80, imgsz=640, augment=False)
    img, lbl = ds_cv[0]
    check("coco_val_loads",    len(ds_cv)>0, f"{len(ds_cv)} images")
    check("coco_val_no_aug",   img.shape==torch.Size([3,640,640]), str(img.shape))
    loader, _ = build_dataloader(ds_cv, batch_size=4, workers=0, shuffle=False)
    batch = next(iter(loader))
    check("coco_val_loader",   batch['img'].shape[0]<=4, "")
except Exception as e:
    check("coco_val_loads", False, str(e))

# YOLO val split
try:
    ds_yv = YOLOTxtDataset(yaml_file='dataset/coco128/data.yaml',
                            split='val', nc=None, imgsz=640, augment=False)
    img, lbl = ds_yv[0]
    check("yolo_val_loads",    len(ds_yv)>0, f"{len(ds_yv)} images")
    check("yolo_val_no_aug",   img.shape==torch.Size([3,640,640]), str(img.shape))
    loader, _ = build_dataloader(ds_yv, batch_size=4, workers=0, shuffle=False)
    batch = next(iter(loader))
    check("yolo_val_loader",   batch['img'].shape[0]<=4, "")
except Exception as e:
    check("yolo_val_loads", False, str(e))

passed = sum(results.values()); total = len(results)
print(f"\nT1.2 Result: {passed}/{total} passed")
if passed < total: sys.exit(1)
