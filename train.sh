
    python3 train.py \
        --model configs/model/yolov11s.yaml \
        --train-img-dir /data/coco/val2017 \
        --train-ann /data/coco/annotations/instances_val2017.json \
        --val-img-dir /data/coco/val2017 \
        --val-ann /data/coco/annotations/instances_val2017.json \
        --epochs 100 --batch 4 --device cuda:0