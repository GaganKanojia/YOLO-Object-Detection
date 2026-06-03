import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data.dataset import YOLODataset


def collate_fn(batch):
    """
    Collate images and labels into batched tensors.
    Each sample: (img [3,H,W], labels [N, 6])
    labels[:,0] will be set to batch index here.
    Returns dict with 'img', 'batch_idx', 'cls', 'bboxes'.
    """
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)

    batch_labels = []
    for i, lbl in enumerate(labels):
        if lbl.shape[0]:
            lbl[:, 0] = i  # set batch index
            batch_labels.append(lbl)

    if batch_labels:
        batch_labels = torch.cat(batch_labels, 0)
    else:
        batch_labels = torch.zeros((0, 6))

    return {
        "img": imgs,
        "batch_idx": batch_labels[:, 0],
        "cls": batch_labels[:, 1],
        "bboxes": batch_labels[:, 2:],
    }


def build_dataloader(
    img_dir=None,
    ann_file=None,
    imgsz=640,
    batch_size=16,
    workers=8,
    augment=True,
    shuffle=True,
    mosaic=1.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    copy_paste_mode="flip",
    bgr=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    close_mosaic=0,
    epoch=0,
):
    # Allow the first positional arg to be a pre-built Dataset instance
    # (YOLODataset, YOLOTxtDataset, or any torch Dataset).
    if isinstance(img_dir, Dataset):
        dataset = img_dir
    else:
        if img_dir is None or ann_file is None:
            raise TypeError(
                "build_dataloader requires either a YOLODataset as the first arg, "
                "or both img_dir and ann_file"
            )
        dataset = YOLODataset(
            img_dir=img_dir,
            ann_file=ann_file,
            imgsz=imgsz,
            augment=augment,
            mosaic=mosaic if augment else 0.0,
            mixup=mixup if augment else 0.0,
            cutmix=cutmix if augment else 0.0,
            copy_paste=copy_paste if augment else 0.0,
            copy_paste_mode=copy_paste_mode,
            bgr=bgr if augment else 0.0,
            hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
            degrees=degrees, translate=translate, scale=scale,
            shear=shear, perspective=perspective,
            flipud=flipud, fliplr=fliplr,
            close_mosaic=close_mosaic,
            epoch=epoch,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=augment,
        persistent_workers=workers > 0,
    )
    return loader, dataset
