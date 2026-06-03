import torch
import numpy as np
from tqdm import tqdm

from utils.nms import non_max_suppression
from utils.metrics import DetectionMetrics
from utils.bbox import xywh2xyxy


class Validator:
    def __init__(self, model, dataloader, device, conf=0.001, iou=0.7, max_det=300, names=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        # Class-name map for per-class metrics. Falls back to the model's names.
        self.names = names if names is not None else getattr(model, "names", None)

    @torch.no_grad()
    def run(self):
        self.model.eval()
        nc = self.model.nc
        metrics = DetectionMetrics(nc=nc, names=self.names)

        pbar = tqdm(self.dataloader, desc="Validating")
        for batch in pbar:
            imgs = batch["img"].to(self.device)
            batch_idx = batch["batch_idx"]
            cls = batch["cls"]
            bboxes = batch["bboxes"]
            bs = imgs.shape[0]
            imgsz = imgs.shape[2]

            # Inference
            preds_raw, _ = self.model(imgs)  # [bs, 4+nc, n_anchors]

            # NMS
            preds = non_max_suppression(
                preds_raw,
                conf_thres=self.conf,
                iou_thres=self.iou,
                max_det=self.max_det,
                nc=nc,
            )

            # Build ground truth tensor [M, 6]: batch_idx, cls, x1,y1,x2,y2 (pixel)
            targets = torch.cat([
                batch_idx.view(-1, 1),
                cls.view(-1, 1),
                bboxes  # normalized cx,cy,w,h
            ], dim=1)

            # Convert normalized xywh → pixel xyxy
            targets_xyxy = targets.clone()
            targets_xyxy[:, 2:] = xywh2xyxy(targets[:, 2:]) * imgsz

            metrics.update(preds, targets_xyxy.to(self.device))

        results = metrics.compute()
        self.model.train()
        return results
