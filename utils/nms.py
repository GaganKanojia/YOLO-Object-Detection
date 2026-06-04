import torch
import torchvision


def non_max_suppression(
    prediction,
    conf_thres=0.001,
    iou_thres=0.7,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    nc=80,
):
    """
    NMS on model output.

    prediction: [batch, 4+nc, num_anchors] (xywh + class scores, already sigmoid'd).
                This is the eval-mode head output; it is transposed internally to
                [batch, num_anchors, 4+nc].
    multi_label: if True (and nc>1), a box emits one detection per class whose score
                exceeds conf_thres (matches Ultralytics validation). If False, only the
                single best class per box is kept.
    Returns: list of [N, 6] tensors per image: [x1, y1, x2, y2, conf, cls]
    """
    # Head emits [batch, no, anchors]; bring to [batch, anchors, no] (matches Ultralytics).
    prediction = prediction.transpose(-1, -2)

    bs = prediction.shape[0]
    nc_actual = prediction.shape[2] - 4
    nm = 0  # extra mask dims if any
    mi = 4 + nc_actual
    multi_label &= nc_actual > 1  # multiple labels per box only make sense for nc>1

    xc = prediction[..., 4:mi].amax(2) > conf_thres  # confidence mask

    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # filter by confidence

        if not x.shape[0]:
            continue

        box = x[:, :4]  # cx, cy, w, h
        cls = x[:, 4:mi]

        if multi_label:
            # One row per (box, class) pair whose score exceeds the threshold.
            i, j = (cls > conf_thres).nonzero(as_tuple=True)
            x = torch.cat((box[i], cls[i, j, None], j[:, None].float()), 1)
        else:
            # Best class only.
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > 30000:
            x = x[x[:, 4].argsort(descending=True)[:30000]]

        # Convert cx,cy,w,h → x1,y1,x2,y2 for torchvision NMS
        boxes_xyxy = _xywh2xyxy(x[:, :4])
        scores = x[:, 4]
        cls_ids = x[:, 5]

        if agnostic:
            i = torchvision.ops.nms(boxes_xyxy, scores, iou_thres)
        else:
            # Class-aware NMS via offset trick
            offset = cls_ids * 7680  # max image size offset
            i = torchvision.ops.nms(boxes_xyxy + offset.unsqueeze(1), scores, iou_thres)

        i = i[:max_det]
        output[xi] = torch.cat([boxes_xyxy[i], scores[i].unsqueeze(1), cls_ids[i].unsqueeze(1)], dim=1)

    return output


def _xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
