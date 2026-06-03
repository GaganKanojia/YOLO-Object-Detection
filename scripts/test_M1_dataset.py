import sys, torch, numpy as np
from data.dataset import YOLODataset, YOLOTxtDataset, build_dataset
from data.loaders import build_dataloader

PASS = "\033[92mPASS\033[0m"; FAIL = "\033[91mFAIL\033[0m"
results = {}

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    results[name] = cond
    print(f"  {status} {name}" + (f": {detail}" if detail else ""))
    return cond

print("="*60)
print("MODULE 1 — DATASET TESTS")
print("="*60)

# ── COCO FORMAT ──────────────────────────────────────────────────────
print("\n[COCO] Loading dataset...")
try:
    ds_coco = YOLODataset(
        img_dir  = 'dataset/coco128/images/train2017',
        ann_file = 'dataset/coco128/annotations_train.json',
        nc=80, imgsz=640, augment=False)
    check("coco_load",       len(ds_coco) > 0, f"size={len(ds_coco)}")
    img, lbl = ds_coco[0]
    check("coco_img_shape",  img.shape  == torch.Size([3,640,640]), str(img.shape))
    check("coco_img_dtype",  img.dtype  == torch.float32,           str(img.dtype))
    check("coco_img_range",  img.min()>=0 and img.max()<=1.0,      f"[{img.min():.3f},{img.max():.3f}]")
    check("coco_lbl_cols",   lbl.ndim==2 and lbl.shape[1]==6,      str(lbl.shape))
    check("coco_no_nan",     not torch.isnan(img).any(),            "")
    if lbl.numel()>0:
        check("coco_cls_range", (lbl[:,1]>=0).all() and (lbl[:,1]<80).all(), "")
        check("coco_box_range", (lbl[:,2:6]>=0).all() and (lbl[:,2:6]<=1).all(), "")
except Exception as e:
    check("coco_load", False, str(e)); print(f"  CRASH: {e}")

# ── YOLO FORMAT ──────────────────────────────────────────────────────
print("\n[YOLO] Loading dataset...")
try:
    ds_yolo = YOLOTxtDataset(
        yaml_file='dataset/coco128/data.yaml',
        split='train', nc=None, imgsz=640, augment=False)
    check("yolo_load",       len(ds_yolo) > 0, f"size={len(ds_yolo)}")
    img, lbl = ds_yolo[0]
    check("yolo_img_shape",  img.shape  == torch.Size([3,640,640]), str(img.shape))
    check("yolo_img_dtype",  img.dtype  == torch.float32,           str(img.dtype))
    check("yolo_img_range",  img.min()>=0 and img.max()<=1.0,      f"[{img.min():.3f},{img.max():.3f}]")
    check("yolo_lbl_cols",   lbl.ndim==2 and lbl.shape[1]==6,      str(lbl.shape))
    check("yolo_no_nan",     not torch.isnan(img).any(),            "")
    if lbl.numel()>0:
        check("yolo_cls_range", (lbl[:,1]>=0).all() and (lbl[:,1]<80).all(), "")
        check("yolo_box_range", (lbl[:,2:6]>=0).all() and (lbl[:,2:6]<=1).all(), "")
except Exception as e:
    check("yolo_load", False, str(e)); print(f"  CRASH: {e}")

# ── AUGMENTATION ─────────────────────────────────────────────────────
print("\n[AUG] Augmented samples (5 samples, check no crash/NaN)...")
for fmt, ds_cls, kwargs in [
    ("coco_aug", YOLODataset,    dict(img_dir='dataset/coco128/images/train2017',
                                      ann_file='dataset/coco128/annotations_train.json',
                                      nc=80,imgsz=640,augment=True)),
    ("yolo_aug", YOLOTxtDataset, dict(yaml_file='dataset/coco128/data.yaml',
                                      split='train',nc=None,imgsz=640,augment=True)),
]:
    try:
        ds_a = ds_cls(**kwargs)
        ok = True
        for i in range(5):
            img_a, lbl_a = ds_a[i % len(ds_a)]
            if img_a.shape != torch.Size([3,640,640]): ok=False; break
            if torch.isnan(img_a).any(): ok=False; break
        check(f"{fmt}_5samples", ok, "")
    except Exception as e:
        check(f"{fmt}_5samples", False, str(e))

# ── DATALOADER + COLLATE ─────────────────────────────────────────────
print("\n[LOADER] DataLoader batch...")
for fmt, ds in [("coco", ds_coco), ("yolo", ds_yolo)]:
    try:
        loader, _ = build_dataloader(ds, batch_size=4, workers=0, shuffle=True)
        batch = next(iter(loader))
        check(f"{fmt}_batch_img",  batch['img'].shape==torch.Size([4,3,640,640]), str(batch['img'].shape))
        check(f"{fmt}_batch_bbox", batch['bboxes'].shape[1]==4, str(batch['bboxes'].shape))
        check(f"{fmt}_batch_idx",  'batch_idx' in batch, str(list(batch.keys())))
    except Exception as e:
        check(f"{fmt}_loader", False, str(e))

# ── build_dataset FACTORY ────────────────────────────────────────────
print("\n[FACTORY] build_dataset routing...")
try:
    ds_f1 = build_dataset({'train_yaml':'dataset/coco128/data.yaml','imgsz':640,'nc':None}, 'train', False)
    check("factory_yolo_mode", type(ds_f1).__name__=='YOLOTxtDataset', type(ds_f1).__name__)
except Exception as e:
    check("factory_yolo_mode", False, str(e))

try:
    ds_f2 = build_dataset({'train_img_dir':'dataset/coco128/images/train2017',
                            'train_ann_file':'dataset/coco128/annotations_train.json',
                            'nc':80,'imgsz':640}, 'train', False)
    check("factory_coco_mode", type(ds_f2).__name__=='YOLODataset', type(ds_f2).__name__)
except Exception as e:
    check("factory_coco_mode", False, str(e))

# ── SUMMARY ──────────────────────────────────────────────────────────
passed = sum(results.values()); total = len(results)
print(f"\nM1 Result: {passed}/{total} passed")
failed = [k for k,v in results.items() if not v]
if failed: print("FAILED:", failed); sys.exit(1)
