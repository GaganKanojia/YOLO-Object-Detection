"""
YOLOv9 building blocks.

New modules introduced by the YOLOv9 / GELAN family that do not exist in the
YOLOv11 block set.  All blocks follow the same style conventions as
models/blocks.py: Conv-BN-SiLU as the base unit, no global state, all
configuration through __init__ parameters.

None of these blocks are mode-sensitive (train vs eval) except via the BatchNorm
layers they contain (which switch between batch statistics and running statistics
as normal).  CBLinear is specifically noted because it returns a Python list of
tensors rather than a single tensor — callers must account for this.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import Conv, Bottleneck
from models.head import Detect


# ---------------------------------------------------------------------------
# Reparameterisable convolution (training form only; no deploy fuse needed)
# ---------------------------------------------------------------------------

class RepConv(nn.Module):
    """Reparametrisable 3×3 convolution used inside RepBottleneck.

    During training the forward pass sums a 3×3 conv and a 1×1 conv.  A
    batch-norm identity branch is optionally added when bn=True and c1==c2
    and stride==1, but RepBottleneck calls this with bn=False so only the
    two conv branches are active.  The result is mathematically equivalent
    to a single 3×3 conv (after fusing), so there is no accuracy difference.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 p: int = 1, g: int = 1, act: bool = True, bn: bool = False):
        super().__init__()
        assert k == 3 and p == 1
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # Identity BN branch (only when input == output channels and stride == 1)
        self.bn = nn.BatchNorm2d(c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


# ---------------------------------------------------------------------------
# RepBottleneck / RepCSP — reparametrisable CSP building blocks
# ---------------------------------------------------------------------------

class RepBottleneck(Bottleneck):
    """Bottleneck whose first 3×3 conv is replaced by RepConv."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, k: tuple = (3, 3), e: float = 0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        # Override cv1 with the reparametrisable conv (bn=False → two-branch form)
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(nn.Module):
    """Cross Stage Partial module with RepBottleneck blocks.

    Structurally identical to C3 but uses RepBottleneck instead of Bottleneck.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ---------------------------------------------------------------------------
# RepNCSPELAN4 — GELAN core block
# ---------------------------------------------------------------------------

class RepNCSPELAN4(nn.Module):
    """GELAN block: CSP-ELAN with reparametrisable bottleneck layers.

    Architecture:
        cv1: c1 → c3  (1×1 conv)
        Split cv1 output into two c3//2 halves.
        cv2: c3//2 → c4  (RepCSP + Conv)
        cv3: c4    → c4  (RepCSP + Conv)
        cv4: c3 + 2*c4 → c2  (1×1 conv)
    Output channels: c2.
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# ELAN1 — simplified GELAN (plain Conv instead of RepCSP)
# ---------------------------------------------------------------------------

class ELAN1(RepNCSPELAN4):
    """ELAN1: like RepNCSPELAN4 but uses plain Conv instead of RepCSP.

    Used in the smaller YOLOv9 variants (yolov9t, yolov9s) where RepCSP
    would be over-parameterised.  Inherits RepNCSPELAN4.forward unchanged.
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        # Call grandparent nn.Module.__init__ — we rebuild all submodules below.
        nn.Module.__init__(self)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)   # plain Conv, no RepCSP
        self.cv3 = Conv(c4, c4, 3, 1)          # plain Conv
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)


# ---------------------------------------------------------------------------
# AConv / ADown — stride-2 downsampling blocks
# ---------------------------------------------------------------------------

class AConv(nn.Module):
    """Lightweight stride-2 downsampling: AvgPool(2,1) → Conv(3,2).

    Net spatial effect: H → H//2, W → W//2.
    Used in yolov9t, yolov9s, yolov9m as the downsampling block.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AvgPool with kernel=2, stride=1 softens edges before strided conv.
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """Dual-path stride-2 downsampling: AvgPool → split → [Conv3×3, MaxPool+Conv1×1].

    Output channels: c2 (c2//2 from each path).
    Used in yolov9c and yolov9e as the downsampling block.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


# ---------------------------------------------------------------------------
# SPPELAN — SPP-ELAN (spatial pyramid pooling, ELAN variant)
# ---------------------------------------------------------------------------

class SPPELAN(nn.Module):
    """Spatial Pyramid Pooling – ELAN variant.

    Architecture:
        cv1: c1 → c3  (1×1 conv)
        Three successive MaxPool(k, 1, k//2) with cumulative feature accumulation
        cv5: 4*c3 → c2  (1×1 conv, concatenating the four feature levels)
    Output channels: c2.
    """

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# CBLinear / CBFuse — PGI gradient information blocks (yolov9e only)
# ---------------------------------------------------------------------------

class CBLinear(nn.Module):
    """Projects a feature map to multiple channel-width outputs simultaneously.

    Returns a Python LIST of tensors (one per element of c2s).  This is the
    only block in the codebase whose forward() does not return a single tensor.
    The channel-tracking logic in parse_model_v9 stores the list [c2s] in
    ch_map for this layer index so that CBFuse can resolve the right width.

    Used exclusively in yolov9e as part of PGI feature-injection.
    """

    def __init__(self, c1: int, c2s: list, k: int = 1, s: int = 1,
                 p: int = None, g: int = 1):
        super().__init__()
        self.c2s = c2s
        from models.blocks import autopad
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list:
        """Returns list of tensors split along the channel dimension."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """Fuses multi-scale CBLinear outputs with the current feature map.

    Receives xs = [cblinear_out_0, cblinear_out_1, ..., current_tensor] where
    each cblinear_out_i is a list of tensors (from CBLinear.forward).
    self.idx[i] selects which element of cblinear_out_i to use for this fuse.
    All selected tensors are interpolated to the spatial size of xs[-1] and
    summed element-wise, enhancing the current feature with multi-scale context.

    PGI branches remain active at inference time (they are not removed at eval).
    """

    def __init__(self, idx: list):
        super().__init__()
        self.idx = idx

    def forward(self, xs: list) -> torch.Tensor:
        """
        xs: list where xs[:-1] are CBLinear outputs (each a list of tensors)
            and xs[-1] is the current-level feature tensor.
        Returns: fused tensor with the same shape as xs[-1].
        """
        target_size = xs[-1].shape[2:]
        res = [
            F.interpolate(x[self.idx[i]], size=target_size, mode="nearest")
            for i, x in enumerate(xs[:-1])
        ]
        return torch.sum(torch.stack(res + [xs[-1]]), dim=0)


# ---------------------------------------------------------------------------
# DetectV9 — YOLOv9 detection head (legacy classification branch)
# ---------------------------------------------------------------------------

class DetectV9(Detect):
    """YOLOv9 detection head with legacy plain-Conv classification branch.

    Inherits everything from Detect (cv2, dfl, forward, bias_init) but replaces
    cv3 with the legacy structure used by Ultralytics YOLOv9 (legacy=True):
        cv3: Conv(x→c3, 3×3) → Conv(c3→c3, 3×3) → Conv2d(c3→nc, 1×1)
    vs YOLOv11 non-legacy:
        cv3: DWConv(x→x,3)+Conv(x→c3,1) → DWConv(c3→c3,3)+Conv(c3→c3,1) → Conv2d(c3→nc,1)
    The plain-Conv branch carries more parameters but matches the original paper.
    """

    def __init__(self, nc: int = 80, ch: tuple = (), reg_max: int = 16):
        super().__init__(nc, ch, reg_max)
        c3 = max(ch[0], min(self.nc, 100))
        # Replace the DWConv-based cv3 from parent with legacy plain-Conv version.
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )

    def bias_init(self):
        """Re-initialise biases after cv3 replacement."""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
