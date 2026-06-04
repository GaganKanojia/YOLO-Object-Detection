import math
import torch
import torch.nn as nn
from models.blocks import Conv, DWConv, DFL


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points from feature maps."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Convert ltrb distance predictions to bounding boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    """Convert xyxy bounding boxes to ltrb distance format for DFL."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


class Detect(nn.Module):
    """YOLOv11 anchor-free detection head."""
    stride = None

    def __init__(self, nc=80, ch=(), reg_max=16):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # Box regression branch: Conv→Conv→Conv2d (matches Ultralytics non-legacy)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        # Classification branch: DWConv→Conv→DWConv→Conv→Conv2d (matches Ultralytics non-legacy)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        # Inference: decode predictions
        shape = x[0].shape
        anchors, strides = make_anchors(x, self.stride, 0.5)
        anchors, strides = anchors.transpose(0, 1), strides.transpose(0, 1)

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        return torch.cat((dbox, cls.sigmoid()), 1), x

    def bias_init(self):
        """Initialize biases for detection head."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
