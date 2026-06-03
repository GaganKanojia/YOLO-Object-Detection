import torch


def box_area(box):
    return (box[..., 2] - box[..., 0]).clamp(0) * (box[..., 3] - box[..., 1]).clamp(0)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Compute IoU variants between box1 and box2.
    box1: [..., 4], box2: [..., 4]
    xywh=True: inputs are [cx, cy, w, h]; False: [x1, y1, x2, y2]
    """
    if xywh:
        (x1, y1, w1, h1) = box1.unbind(-1)
        (x2, y2, w2, h2) = box2.unbind(-1)
        b1_x1, b1_y1 = x1 - w1 / 2, y1 - h1 / 2
        b1_x2, b1_y2 = x1 + w1 / 2, y1 + h1 / 2
        b2_x1, b2_y1 = x2 - w2 / 2, y2 - h2 / 2
        b2_x2, b2_y2 = x2 + w2 / 2, y2 + h2 / 2
        w1, h1, w2, h2 = w1, h1, w2, h2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2

        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area

    return iou


def xywh2xyxy(x):
    """Convert nx4 boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [cx, cy, w, h]."""
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def clip_boxes(boxes, shape):
    """Clip boxes to image boundaries [h, w]."""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])
    return boxes
