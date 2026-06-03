"""
Filter COCO128 to person+car+dog only.
Produce:
  dataset/subset3/images/{train,val}/   ← copies
  dataset/subset3/labels/{train,val}/   ← filtered .txt (only 3 classes, re-indexed)
  dataset/subset3/data.yaml
  dataset/subset3/annotations_{train,val}.json   ← COCO JSON for custom pipeline
  dataset/subset3/split_record.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import shutil
from pathlib import Path
from PIL import Image

SRC_IMGS = Path('dataset/coco128/images/train2017')
SRC_LBLS = Path('dataset/coco128/labels/train2017')
OUT_DIR = Path('dataset/subset3')

# COCO128 source class indices for our 3 targets.
# NOTE (plan D1): COCO id 1 is *bicycle* (rare); car is id 2. The original task script
# mislabeled id 1 as car — corrected here to match the stated intent.
COCO_TARGET = {0: 'person', 2: 'car', 16: 'dog'}
REMAP = {0: 0, 2: 1, 16: 2}   # person->0, car->1, dog->2
NEW_NAMES = ['person', 'car', 'dog']
NC = 3

all_imgs = sorted(SRC_IMGS.glob('*.jpg'))
valid = []  # (img_path, filtered_lines)
for img_path in all_imgs:
    lbl_path = SRC_LBLS / (img_path.stem + '.txt')
    if not lbl_path.exists():
        continue
    lines = lbl_path.read_text().strip().splitlines()
    filtered = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        if cls_id in REMAP:
            new_id = REMAP[cls_id]
            filtered.append(f"{new_id} {' '.join(parts[1:])}")
    if filtered:
        valid.append((img_path, filtered))

print(f"Images with target classes: {len(valid)}")
assert len(valid) >= 20, f"Too few images ({len(valid)}) — need at least 20"

# Split: first 80% train, last 20% val (script-defined; coco128 has ~65 valid imgs).
split_idx = int(len(valid) * 0.8)
train_items = valid[:split_idx]
val_items = valid[split_idx:]
print(f"Train: {len(train_items)} images")
print(f"Val  : {len(val_items)}   images")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for split, items in [('train', train_items), ('val', val_items)]:
    img_out = OUT_DIR / 'images' / split
    lbl_out = OUT_DIR / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, filtered_lines in items:
        shutil.copy2(img_path, img_out / img_path.name)
        (lbl_out / (img_path.stem + '.txt')).write_text('\n'.join(filtered_lines))

yaml_content = f"""path: {OUT_DIR.resolve()}
train: images/train
val: images/val
nc: {NC}
names: {NEW_NAMES}
"""
(OUT_DIR / 'data.yaml').write_text(yaml_content)


def build_coco_json(items):
    images, annotations, ann_id = [], [], 1
    for img_id, (img_path, filtered_lines) in enumerate(items, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({'id': img_id, 'file_name': img_path.name, 'width': w, 'height': h})
        for line in filtered_lines:
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            annotations.append({
                'id': ann_id, 'image_id': img_id, 'category_id': cls_id + 1,
                'bbox': [round(x1, 2), round(y1, 2), round(bw * w, 2), round(bh * h, 2)],
                'area': round(bw * w * bh * h, 2), 'iscrowd': 0,
            })
            ann_id += 1
    return {
        'images': images, 'annotations': annotations,
        'categories': [{'id': i + 1, 'name': n, 'supercategory': 'object'}
                       for i, n in enumerate(NEW_NAMES)],
    }


for split, items in [('train', train_items), ('val', val_items)]:
    out = build_coco_json(items)
    path = OUT_DIR / f'annotations_{split}.json'
    json.dump(out, open(path, 'w'))
    print(f"COCO JSON {split}: {len(out['images'])} images, {len(out['annotations'])} annotations -> {path}")

# Per-class instance counts in each split (sanity for D1 — car must be non-empty).
for split, items in [('train', train_items), ('val', val_items)]:
    counts = {0: 0, 1: 0, 2: 0}
    for _, lines in items:
        for ln in lines:
            counts[int(ln.split()[0])] += 1
    print(f"  {split} instances: person={counts[0]} car={counts[1]} dog={counts[2]}")

split_record = {
    'train': [str(p) for p, _ in train_items],
    'val': [str(p) for p, _ in val_items],
    'nc': NC, 'names': NEW_NAMES,
    'remap': {str(k): v for k, v in REMAP.items()},
}
json.dump(split_record, open(OUT_DIR / 'split_record.json', 'w'), indent=2)
print(f"\nSubset ready: {OUT_DIR}")
print("Split record saved: dataset/subset3/split_record.json")
