import json
from pathlib import Path

ANN  = 'dataset/coco128/annotations.json'
IMGS = 'dataset/coco128/images/train2017'

ann = json.load(open(ANN))

print(f"Images      : {len(ann['images'])}")
print(f"Annotations : {len(ann['annotations'])}")
print(f"Categories  : {len(ann['categories'])}")
print(f"First image : {ann['images'][0]}")
print(f"First ann   : {ann['annotations'][0]}")
print(f"First cat   : {ann['categories'][0]}")

missing = []
for img in ann['images']:
    p = Path(IMGS) / img['file_name']
    if not p.exists():
        missing.append(img['file_name'])

if missing:
    print(f"\nMISSING FILES ({len(missing)}): {missing[:5]}")
else:
    print(f"\nAll {len(ann['images'])} image files exist on disk.")

bad = [a for a in ann['annotations']
       if a['bbox'][2] <= 0 or a['bbox'][3] <= 0]
print(f"Zero/negative boxes: {len(bad)}")

cat_ids = {c['id'] for c in ann['categories']}
bad_cats = [a for a in ann['annotations'] if a['category_id'] not in cat_ids]
print(f"Annotations with unknown category: {len(bad_cats)}")

assert len(missing)  == 0, "Missing image files!"
assert len(bad)      == 0, "Bad bounding boxes!"
assert len(bad_cats) == 0, "Unknown category IDs!"
print("\nJSON verification PASSED.")
