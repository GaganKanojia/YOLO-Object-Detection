import torch
import torch.nn as nn

from utils.bbox import bbox_iou, xyxy2xywh, xywh2xyxy


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner (TAL).
    alignment_metric = cls_score^alpha * iou^beta
    Assigns top-k anchors per GT.
    """
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 stride=(8, 16, 32), eps=1e-9, topk2=None):
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 if topk2 is not None else topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = list(stride)
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pd_scores:  [bs, n_anchors, nc]
        pd_bboxes:  [bs, n_anchors, 4]  xyxy in image space
        anc_points: [n_anchors, 2]      anchor centers in image space
        gt_labels:  [bs, n_max_boxes, 1]
        gt_bboxes:  [bs, n_max_boxes, 4]  xyxy
        mask_gt:    [bs, n_max_boxes, 1]  (a 2D [bs, n_max_boxes] mask is also accepted)
        """
        # Accept a 2D validity mask [bs, n_max_boxes] by promoting it to the
        # canonical [bs, n_max_boxes, 1] used throughout the assigner.
        if mask_gt.ndim == 2:
            mask_gt = mask_gt.unsqueeze(-1)

        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Normalize scores by alignment metric
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Build positive mask: anchors that are inside GT, in top-k, and not padding."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """alignment_metric = cls_score^alpha * iou^beta, only at mask_gt positions."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long, device=pd_scores.device)
        ind[0] = torch.arange(self.bs, device=pd_scores.device).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long()
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Pick top-k positions per GT; dedupe collisions; return float count mask."""
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """
        Boolean mask of anchors whose center lies in each GT box.
        Tiny GTs (wh < smallest stride) are inflated to stride_val so they still receive anchors.
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """If an anchor is matched to multiple GTs, keep the one with highest IoU; optional topk2 filter."""
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)

        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Flat-indexed target assembly (matches Ultralytics)."""
        batch_ind = torch.arange(self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        target_labels.clamp_(0)

        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.zeros_like(target_scores))

        return target_labels, target_bboxes, target_scores
