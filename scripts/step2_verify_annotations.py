"""Step 2 — verify the existing COCO-JSON annotations (already built; do not regenerate)."""
import json

ANN = 'dataset/coco128/annotations.json'
d = json.load(open(ANN))
n_img = len(d['images'])
n_ann = len(d['annotations'])
n_cat = len(d['categories'])
print(f"Annotations file : {ANN}")
print(f"Images           : {n_img}")
print(f"Annotations      : {n_ann}")
print(f"Categories       : {n_cat}")
print(f"First category   : {d['categories'][0]}")
print(f"Last  category id: {d['categories'][-1]['id']}")
assert n_img == 128, f"expected 128 images, got {n_img}"
assert n_ann > 500, f"expected >500 annotations, got {n_ann}"
assert n_cat == 80, f"expected 80 categories, got {n_cat}"
print("OK — reusing existing annotations.json")
