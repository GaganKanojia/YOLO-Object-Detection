import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import parse_yolo_yaml, YOLOTxtDataset

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(cond, msg):
    if not cond:
        errors.append(msg)
        print(f"  {FAIL}: {msg}")
    else:
        print(f"  {PASS}")

# T1: parse data.yaml
print("T1 — parse_yolo_yaml")
cfg = parse_yolo_yaml('dataset/coco128/data.yaml')
print(f"  nc    = {cfg['nc']}")
print(f"  names = {cfg['names'][:5]} ...")
print(f"  train = {cfg['train_img_dir']}")
print(f"  val   = {cfg['val_img_dir']}")
check(cfg['nc'] == 80,                     f"nc={cfg['nc']} expected 80")
check(len(cfg['names']) == 80,             f"len(names)={len(cfg['names'])} expected 80")
check('person' in cfg['names'],            "'person' not in names")
check(cfg['train_img_dir'] != '',          "train_img_dir is empty")

# T2: dataset builds without error
print("T2 — YOLOTxtDataset builds")
import torch
ds = YOLOTxtDataset('dataset/coco128/data.yaml', split='train',
                    imgsz=640, augment=False)
print(f"  Dataset size: {len(ds)}")
check(len(ds) > 0, "Dataset is empty")

# T3: single sample shape and value range
print("T3 — Single sample shape and range")
img, labels = ds[0]
print(f"  img   : {img.shape}  dtype={img.dtype}  range=[{img.min():.3f},{img.max():.3f}]")
print(f"  labels: {labels.shape}")
check(img.shape    == torch.Size([3,640,640]),  f"img shape {img.shape}")
check(img.dtype    == torch.float32,            "img dtype not float32")
check(img.min()    >= 0.0,                      f"img min {img.min():.3f} < 0")
check(img.max()    <= 1.0,                      f"img max {img.max():.3f} > 1")
check(labels.ndim  == 2,                        f"labels ndim {labels.ndim}")
check(labels.shape[1] == 6,                     f"labels cols {labels.shape[1]} != 6")
if labels.numel() > 0:
    cls_ids = labels[:,1].long()
    check((cls_ids >= 0).all() and (cls_ids < cfg['nc']).all(),
          f"class IDs out of [0,{cfg['nc']})")
    cx, cy, w, h = labels[:,2], labels[:,3], labels[:,4], labels[:,5]
    check((cx>=0).all() and (cx<=1).all(), "cx out of [0,1]")
    check((cy>=0).all() and (cy<=1).all(), "cy out of [0,1]")
    check((w>0).all()  and (w<=1).all(),   "w out of (0,1]")
    check((h>0).all()  and (h<=1).all(),   "h out of (0,1]")

# T4: augmented sample
print("T4 — Augmented sample")
ds_aug = YOLOTxtDataset('dataset/coco128/data.yaml', split='train',
                         imgsz=640, augment=True)
img_a, lbl_a = ds_aug[0]
check(img_a.shape == torch.Size([3,640,640]), f"aug img shape {img_a.shape}")
check(not torch.isnan(img_a).any(), "NaN in augmented img")

# T5: DataLoader with collate_fn
print("T5 — DataLoader batch")
from data.loaders import build_dataloader
loader, _ = build_dataloader(ds, batch_size=4, workers=0, shuffle=True)
batch = next(iter(loader))
print(f"  keys   : {list(batch.keys())}")
print(f"  img    : {batch['img'].shape}")
print(f"  bboxes : {batch['bboxes'].shape}")
check(batch['img'].shape == torch.Size([4,3,640,640]), "batch img shape")
check(batch['bboxes'].shape[1] == 4, "bboxes cols != 4")

# T6: build_dataset factory — YOLO mode
print("T6 — build_dataset factory (YOLO mode)")
from data.dataset import build_dataset
args = {'train_yaml':'dataset/coco128/data.yaml', 'imgsz':640, 'nc':None}
ds_f = build_dataset(args, split='train', augment=False)
check(len(ds_f) > 0, "factory returned empty dataset")
check(type(ds_f).__name__ == 'YOLOTxtDataset', f"wrong type: {type(ds_f).__name__}")

# T7: build_dataset factory — COCO mode still works
print("T7 — build_dataset factory (COCO mode backward compat)")
# Only run if you have a COCO JSON from the previous smoke test
import os
if os.path.exists('dataset/coco128/annotations.json'):
    args_coco = {
        'train_img_dir' : 'dataset/coco128/images/train2017',
        'train_ann_file': 'dataset/coco128/annotations.json',
        'nc': 80, 'imgsz': 640
    }
    ds_c = build_dataset(args_coco, split='train', augment=False)
    check(len(ds_c) > 0, "COCO factory returned empty dataset")
    check(type(ds_c).__name__ == 'YOLODataset', f"wrong type: {type(ds_c).__name__}")
else:
    print("  SKIP (no COCO JSON available)")

# ── Summary ─────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[91m{len(errors)} FAILURES:\033[0m")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
else:
    print("\033[92mAll YOLO dataset tests PASSED.\033[0m")
