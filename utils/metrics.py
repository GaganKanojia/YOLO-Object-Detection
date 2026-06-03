import numpy as np
import torch


def smooth(y, f=0.05):
    """Box filter of fraction f (matches Ultralytics utils.metrics.smooth)."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def compute_ap(recall, precision):
    """Compute Average Precision using COCO 101-point interpolation (Ultralytics 'interp')."""
    # Sentinel values (must match Ultralytics exactly — a single pad point each side).
    # NOTE: an earlier version added an extra (recall[-1], 0.0) sentinel which collapsed
    # the precision envelope early and systematically deflated AP — that was a bug.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Precision envelope: monotonically decreasing from right to left
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 101-point interpolation (COCO standard)
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute AP per class — mirrors Ultralytics ap_per_class.

    tp:        [N, n_iou] binary true-positive matrix (per IoU threshold)
    conf:      [N] confidence scores
    pred_cls:  [N] predicted class indices
    target_cls:[M] all GT class indices

    Returns:
        p   : [nc] precision per class at the max-mean-F1 confidence
        r   : [nc] recall    per class at the max-mean-F1 confidence
        ap  : [nc, n_iou] average precision per class per IoU threshold
        unique_classes : [nc] GT class indices that have data (ascending)
    """
    # Sort detections by confidence descending
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        idx = pred_cls == c
        n_l = nt[ci]            # number of GT labels for this class
        n_p = idx.sum()         # number of predictions for this class
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[idx]).cumsum(0)
        tpc = tp[idx].cumsum(0)

        # Recall / precision curves interpolated over confidence (xp decreases → negate)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[idx], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[idx], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Precision/recall reported at the confidence that maximizes mean F1 (Ultralytics).
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    if nc:
        k = smooth(f1_curve.mean(0), 0.1).argmax()
        p = p_curve[:, k]
        r = r_curve[:, k]
    else:
        p = np.zeros(0)
        r = np.zeros(0)
    return p, r, ap, unique_classes.astype(int)


class DetectionMetrics:
    """Accumulate TP/FP/conf across batches, compute mAP at end (Ultralytics-aligned)."""

    def __init__(self, nc, iou_thres=None, names=None):
        self.nc = nc
        self.iou_thres = iou_thres if iou_thres is not None else np.linspace(0.5, 0.95, 10)
        self.names = names if names is not None else {i: str(i) for i in range(nc)}
        self.reset()

    def reset(self):
        self.stats = []

    def update(self, preds, targets):
        """
        preds:   list of [N, 6] tensors [x1,y1,x2,y2,conf,cls] per image
        targets: [M, 6] tensor [batch_idx, cls, x1, y1, x2, y2] (abs coords)
        """
        for si, pred in enumerate(preds):
            gt = targets[targets[:, 0] == si][:, 1:]  # [K, 5] cls, x1y1x2y2
            n_gt = len(gt)
            n_pred = len(pred)

            if n_pred == 0:
                if n_gt:
                    self.stats.append((
                        torch.zeros(0, len(self.iou_thres), dtype=torch.bool),
                        torch.zeros(0),
                        torch.zeros(0),
                        gt[:, 0].cpu()
                    ))
                continue

            if n_gt == 0:
                self.stats.append((
                    torch.zeros(n_pred, len(self.iou_thres), dtype=torch.bool),
                    pred[:, 4].cpu(),
                    pred[:, 5].cpu(),
                    torch.zeros(0)
                ))
                continue

            tp = self._match(pred, gt)
            self.stats.append((tp.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), gt[:, 0].cpu()))

    def _match(self, pred, gt):
        """
        Match predictions to GTs using IoU; return TP matrix [N_pred, n_iou].
        Mirrors Ultralytics match_predictions: zero out wrong-class IoU pairs, then per
        threshold take matches with IoU>=t, sort by IoU descending, keep unique det then
        unique GT.
        """
        from utils.bbox import bbox_iou
        n_iou = len(self.iou_thres)
        tp = torch.zeros(len(pred), n_iou, dtype=torch.bool)
        gt_boxes = gt[:, 1:].to(pred.device)  # [K, 4] xyxy
        gt_cls = gt[:, 0]

        # [N_pred, K_gt] IoU, with wrong-class pairs zeroed out.
        iou = bbox_iou(pred[:, :4].unsqueeze(1), gt_boxes.unsqueeze(0), xywh=False)
        correct_class = (pred[:, 5].unsqueeze(1) == gt_cls.unsqueeze(0))  # [N_pred, K_gt]
        iou = (iou * correct_class).cpu().numpy()

        for ti, thresh in enumerate(self.iou_thres):
            matches = np.nonzero(iou >= thresh)          # (pred_idx[], gt_idx[])
            matches = np.array(matches).T                # [M, 2] = (pred_idx, gt_idx)
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    order = iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
                    matches = matches[order]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # unique det
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # unique gt
                tp[matches[:, 0].astype(int), ti] = True
        return tp

    def compute(self):
        empty = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0, "per_class": {}}
        if not self.stats:
            return empty

        tp_all = torch.cat([s[0] for s in self.stats]).numpy()
        conf_all = torch.cat([s[1] for s in self.stats]).numpy()
        pred_cls_all = torch.cat([s[2] for s in self.stats]).numpy()
        target_cls_all = torch.cat([s[3] for s in self.stats]).numpy()

        if tp_all.shape[0] == 0 or target_cls_all.shape[0] == 0:
            return empty

        p, r, ap, unique_classes = ap_per_class(
            tp_all, conf_all, pred_cls_all, target_cls_all
        )

        ap50 = ap[:, 0] if ap.size else np.zeros(0)
        mAP50 = float(ap50.mean()) if ap50.size else 0.0
        mAP50_95 = float(ap.mean()) if ap.size else 0.0
        precision = float(p.mean()) if p.size else 0.0
        recall = float(r.mean()) if r.size else 0.0

        per_class = {}
        for ci, c in enumerate(unique_classes):
            name = self.names.get(int(c), str(int(c)))
            per_class[name] = {
                "ap50": round(float(ap50[ci]), 6),
                "precision": round(float(p[ci]), 6),
                "recall": round(float(r[ci]), 6),
            }

        return {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
            "per_class": per_class,
        }


class ConfusionMatrix:
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        detections: [N, 6] x1,y1,x2,y2,conf,cls
        labels:     [M, 5] cls,x1,y1,x2,y2
        """
        from utils.bbox import bbox_iou
        gt_classes = labels[:, 0].int()
        det = detections[detections[:, 4] > self.conf]

        if det.shape[0] == 0:
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        iou = bbox_iou(det[:, :4].unsqueeze(1), labels[:, 1:].unsqueeze(0), xywh=False)
        x = torch.where(iou > self.iou_thres)
        matched_gt = set()
        matched_pred = set()

        if x[0].shape[0]:
            matches = torch.stack(x, 1).cpu().numpy()
            for pi, gi in matches:
                if pi in matched_pred or gi in matched_gt:
                    continue
                self.matrix[det[pi, 5].int(), gt_classes[gi]] += 1
                matched_pred.add(pi)
                matched_gt.add(gi)

        for gi in range(len(labels)):
            if gi not in matched_gt:
                self.matrix[self.nc, gt_classes[gi]] += 1

        for pi in range(len(det)):
            if pi not in matched_pred:
                self.matrix[det[pi, 5].int(), self.nc] += 1
