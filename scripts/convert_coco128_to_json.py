import json, os
from pathlib import Path
from PIL import Image

IMG_DIR   = 'dataset/coco128/images/train2017'
LABEL_DIR = 'dataset/coco128/labels/train2017'
OUT_FILE  = 'dataset/coco128/annotations.json'

CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]

images, annotations, ann_id = [], [], 1

for img_id, img_path in enumerate(sorted(Path(IMG_DIR).glob('*.jpg')), start=1):
    with Image.open(img_path) as im:
        w, h = im.size
    images.append({
        'id'       : img_id,
        'file_name': img_path.name,
        'width'    : w,
        'height'   : h
    })
    lbl_path = Path(LABEL_DIR) / (img_path.stem + '.txt')
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            cls_id, cx, cy, bw, bh = map(float, line.split())
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
            annotations.append({
                'id'         : ann_id,
                'image_id'   : img_id,
                'category_id': int(cls_id) + 1,   # COCO IDs are 1-indexed
                'bbox'       : [round(x1,2), round(y1,2), round(box_w,2), round(box_h,2)],
                'area'       : round(box_w * box_h, 2),
                'iscrowd'    : 0
            })
            ann_id += 1

coco_json = {
    'images'     : images,
    'annotations': annotations,
    'categories' : [{'id': i + 1, 'name': name, 'supercategory': 'object'}
                    for i, name in enumerate(CLASSES)]
}

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, 'w') as f:
    json.dump(coco_json, f)

print(f"Images      : {len(images)}")
print(f"Annotations : {len(annotations)}")
print(f"Categories  : {len(CLASSES)}")
print(f"Saved to    : {OUT_FILE}")
