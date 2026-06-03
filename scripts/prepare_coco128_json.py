import json, os
from pathlib import Path
from PIL import Image

IMG_DIR   = 'dataset/coco128/images/train2017'
LABEL_DIR = 'dataset/coco128/labels/train2017'
OUT_TRAIN = 'dataset/coco128/annotations_train.json'
OUT_VAL   = 'dataset/coco128/annotations_val.json'

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

all_imgs = sorted(Path(IMG_DIR).glob('*.jpg'))
# 80% train, 20% val split
split_idx = int(len(all_imgs) * 0.8)
splits = {'train': all_imgs[:split_idx], 'val': all_imgs[split_idx:]}

for split, img_paths in splits.items():
    images, annotations, ann_id = [], [], 1
    for img_id, img_path in enumerate(img_paths, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({'id': img_id, 'file_name': img_path.name, 'width': w, 'height': h})
        lbl = Path(LABEL_DIR) / (img_path.stem + '.txt')
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                if not line.strip(): continue
                cls, cx, cy, bw, bh = map(float, line.split())
                x1, y1 = (cx-bw/2)*w, (cy-bh/2)*h
                annotations.append({
                    'id': ann_id, 'image_id': img_id,
                    'category_id': int(cls)+1,
                    'bbox': [round(x1,2), round(y1,2), round(bw*w,2), round(bh*h,2)],
                    'area': round(bw*w*bh*h, 2), 'iscrowd': 0
                })
                ann_id += 1
    out = {'images': images, 'annotations': annotations,
           'categories': [{'id':i+1,'name':n} for i,n in enumerate(CLASSES)]}
    out_path = OUT_TRAIN if split == 'train' else OUT_VAL
    json.dump(out, open(out_path,'w'))
    print(f"{split}: {len(images)} images, {len(annotations)} annotations → {out_path}")
