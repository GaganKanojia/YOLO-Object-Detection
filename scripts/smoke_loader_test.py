import sys, torch
from data.dataset import YOLODataset
from data.loaders  import build_dataloader

IMG_DIR  = 'dataset/coco128/images/train2017'
ANN_FILE = 'dataset/coco128/annotations.json'
NC       = 80

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(cond, msg):
    if not cond:
        errors.append(msg)
        print(f"  {FAIL}: {msg}")
    else:
        print(f"  {PASS}")

# ── T1: Dataset loads ────────────────────────────────────────────────
print("T1 — Dataset loads")
try:
    ds = YOLODataset(img_dir=IMG_DIR, ann_file=ANN_FILE, nc=NC,
                     imgsz=640, augment=False)
    print(f"  Dataset size: {len(ds)}")
    check(len(ds) > 0, "Dataset is empty")
except Exception as e:
    errors.append(f"Dataset load crashed: {e}")
    print(f"  {FAIL}: {e}")
    sys.exit(1)

# ── T2: Single sample shape and value range ──────────────────────────
print("T2 — Single sample")
img, labels = ds[0]
print(f"  img   shape={img.shape} dtype={img.dtype} "
      f"min={img.min():.3f} max={img.max():.3f}")
print(f"  labels shape={labels.shape}")
check(img.shape    == torch.Size([3, 640, 640]), f"img shape {img.shape} != [3,640,640]")
check(img.dtype    == torch.float32,             f"img dtype {img.dtype} != float32")
check(img.min()    >= 0.0,                       f"img min {img.min():.3f} < 0")
check(img.max()    <= 1.0,                       f"img max {img.max():.3f} > 1")
check(labels.ndim  == 2,                         f"labels ndim {labels.ndim} != 2")
check(labels.shape[1] == 6,                      f"labels cols {labels.shape[1]} != 6")

# ── T3: No NaN / Inf in sample ──────────────────────────────────────
print("T3 — No NaN/Inf in sample")
check(not torch.isnan(img).any(),   "NaN in img")
check(not torch.isinf(img).any(),   "Inf in img")
if labels.numel() > 0:
    check(not torch.isnan(labels).any(), "NaN in labels")
    check(not torch.isinf(labels).any(), "Inf in labels")
    cx, cy, w, h = labels[:,2], labels[:,3], labels[:,4], labels[:,5]
    check((cx >= 0).all() and (cx <= 1).all(), "cx out of [0,1]")
    check((cy >= 0).all() and (cy <= 1).all(), "cy out of [0,1]")
    check((w  >  0).all() and (w  <= 1).all(), "w out of (0,1]")
    check((h  >  0).all() and (h  <= 1).all(), "h out of (0,1]")

# ── T4: DataLoader batch collation ──────────────────────────────────
print("T4 — DataLoader batch")
loader, _ = build_dataloader(ds, batch_size=4, workers=0, shuffle=True)
batch = next(iter(loader))
print(f"  keys={list(batch.keys())}")
print(f"  img={batch['img'].shape}  "
      f"cls={batch['cls'].shape}  "
      f"bboxes={batch['bboxes'].shape}  "
      f"batch_idx unique={batch['batch_idx'].unique().tolist()}")
check(batch['img'].shape    == torch.Size([4,3,640,640]), f"batch img {batch['img'].shape}")
check(batch['bboxes'].shape[1] == 4,                      "bboxes cols != 4")
check(set(batch['batch_idx'].tolist()).issubset({0,1,2,3}),"batch_idx out of range")

# ── T5: Augmented sample ─────────────────────────────────────────────
print("T5 — Augmented sample")
ds_aug = YOLODataset(img_dir=IMG_DIR, ann_file=ANN_FILE, nc=NC,
                     imgsz=640, augment=True)
img_a, lbl_a = ds_aug[0]
check(img_a.shape == torch.Size([3,640,640]), f"aug img shape {img_a.shape}")
check(not torch.isnan(img_a).any(),           "NaN in augmented img")
print(f"  aug labels: {lbl_a.shape}")

# ── Summary ──────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[91m{len(errors)} FAILURES:\033[0m")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
else:
    print("\033[92mAll loader tests PASSED.\033[0m")
