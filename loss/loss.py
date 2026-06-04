import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.tal import TaskAlignedAssigner
from models.head import make_anchors, dist2bbox, bbox2dist
from utils.bbox import bbox_iou


class BboxLoss(nn.Module):
    """CIoU + DFL bounding box loss."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        if reg_max > 1:
            self.proj = torch.arange(reg_max, dtype=torch.float)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask):
        """
        pred_dist:    [bs, n_anchors, 4*reg_max]
        pred_bboxes:  [bs, n_anchors, 4]  xyxy
        anchor_points:[n_anchors, 2]
        target_bboxes:[bs, n_anchors, 4]  xyxy
        target_scores:[bs, n_anchors, nc]
        fg_mask:      [bs, n_anchors]     bool
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # [N, 1]

        # CIoU loss — `bbox_iou` returns shape [N]; force [N, 1] to avoid
        # broadcasting against `weight` ([N, 1]) into an [N, N] tensor whose
        # `.sum()` was inflating box loss by ~N×.
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        ).view(-1, 1)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.reg_max > 1:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss_dfl = self._dfl_loss(pred_dist[fg_mask], target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    def _dfl_loss(self, pred_dist, target):
        """
        pred_dist: [N, 4*reg_max]
        target:    [N, 4]  continuous bin values in [0, reg_max-1]
        """
        tl = target.long()
        tr = tl + 1
        wl = tr.float() - target
        wr = 1.0 - wl

        # pred_dist is [N, 4*reg_max] laid out coordinate-major (each coord's reg_max
        # bins are contiguous). Reshape directly to [N*4, reg_max] so each row holds one
        # coordinate's bins — matching tl/tr's [N*4] coord-major order. A previous version
        # inserted .transpose(1, 2) here, which scrambled bins across the 4 coordinates and
        # inflated the DFL loss (~4.8x); cross_entropy must see one coordinate per row.
        pred = pred_dist.reshape(-1, self.reg_max)  # [N*4, reg_max]

        loss = (
            F.cross_entropy(pred, tl.reshape(-1), reduction="none") * wl.reshape(-1) +
            F.cross_entropy(pred, tr.clamp(max=self.reg_max - 1).reshape(-1), reduction="none") * wr.reshape(-1)
        )
        return loss.view(-1, 4).mean(-1, keepdim=True)


class DetectionLoss(nn.Module):
    """
    YOLOv11 detection loss: BCE (cls) + CIoU (box) + DFL.
    Weights: box=7.5, cls=0.5, dfl=1.5
    """
    def __init__(self, model, box=7.5, cls=0.5, dfl=1.5):
        super().__init__()
        device = next(model.parameters()).device

        self.nc = model.nc
        # Read reg_max from the Detect head (matches Ultralytics' `m = model.model[-1]`)
        # instead of hardcoding, so a non-default reg_max in the model YAML is honored.
        self.reg_max = getattr(model.model[-1], "reg_max", 16)
        self.strides = model.stride.to(device)
        self.device = device

        self.box_w = box
        self.cls_w = cls
        self.dfl_w = dfl

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0,
            stride=self.strides.tolist(),
        )
        self.bbox_loss = BboxLoss(self.reg_max).to(device)

        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        targets: [M, 6]  batch_idx, cls, cx, cy, w, h  (normalized)
        Returns: [bs, max_gt, 5] padded,  and mask [bs, max_gt, 1]
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 1, 5, device=self.device)
            return out, torch.zeros(batch_size, 1, 1, dtype=torch.bool, device=self.device)

        i = targets[:, 0].long()
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for j in range(batch_size):
            matches = targets[i == j]
            n = matches.shape[0]
            if n:
                out[j, :n] = matches[:, 1:]
        out[..., 1:5] = out[..., 1:5] * scale_tensor
        mask = (out.sum(-1, keepdim=True) != 0)
        return out, mask

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted DFL distributions → xyxy bboxes."""
        b, a, _ = pred_dist.shape
        # [b, a, 4, reg_max] → softmax → integral → ltrb
        dist = pred_dist.view(b, a, 4, self.reg_max).softmax(-1)
        dist = dist.matmul(self.proj.to(device=dist.device, dtype=dist.dtype))
        return dist2bbox(dist, anchor_points, xywh=False)

    def forward(self, preds, batch):
        """
        preds:  list of feature tensors from Detect head (training mode)
        batch:  dict with keys 'img', 'batch_idx', 'cls', 'bboxes'
        """
        loss = torch.zeros(3, device=self.device)  # [box, cls, dfl]

        # Collect raw predictions from all scales
        feats = preds if isinstance(preds, (list, tuple)) else preds[1]
        pred_distri, pred_scores = torch.cat(
            [f.view(f.shape[0], self.nc + self.reg_max * 4, -1) for f in feats], dim=2
        ).split((self.reg_max * 4, self.nc), dim=1)

        # [bs, n_anchors, ...]
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        bs, n_anchors = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.strides[0]

        anchor_points, stride_tensor = make_anchors(feats, self.strides, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1
        ).to(self.device)

        scale = torch.cat([imgsz[[1, 0]], imgsz[[1, 0]]])  # w,h,w,h
        gt_bboxes, mask_gt = self.preprocess(targets, bs, scale)

        gt_labels = gt_bboxes[..., :1]    # [bs, max_gt, 1]
        gt_bboxes_xyxy = gt_bboxes[..., 1:]  # [bs, max_gt, 4] xywh → need xyxy

        # Convert GT from cx,cy,w,h → x1,y1,x2,y2
        gt_bboxes_xyxy = self._xywh2xyxy(gt_bboxes_xyxy)

        # stride_tensor: [n_anchors, 1]
        # Decode predictions to boxes (in image coords, scaled by stride)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)          # [bs, n_anchors, 4]
        pred_bboxes_scaled = pred_bboxes * stride_tensor.unsqueeze(0)       # [bs, n_anchors, 4]

        # Assign — cast preds to the GT dtype so AMP/half-precision predictions
        # don't clash with the float32 IoU results inside the assigner.
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid().type(gt_bboxes_xyxy.dtype),
            pred_bboxes_scaled.detach().type(gt_bboxes_xyxy.dtype),
            anchor_points * stride_tensor,                                   # [n_anchors, 2]
            gt_labels,
            gt_bboxes_xyxy,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1.0)

        # Classification loss (BCE)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # Target boxes in anchor (feature) space
            target_bboxes_norm = target_bboxes / stride_tensor.unsqueeze(0)
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes_norm,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.box_w
        loss[1] *= self.cls_w
        loss[2] *= self.dfl_w

        # Ultralytics scales final loss by batch size before backward
        return loss.sum() * bs, loss.detach()

    @staticmethod
    def _xywh2xyxy(x):
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
