import math
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from models.blocks import (
    Conv, C3k2, SPPF, C2PSA, Concat, Upsample, DFL
)
from models.head import Detect, make_anchors, dist2bbox


def make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def initialize_weights(model):
    """Match Ultralytics initialize_weights: BN eps=1e-3, momentum=0.03, inplace activations."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True


MODULE_MAP = {
    "Conv": Conv,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Upsample": Upsample,
    "Detect": Detect,
    "nn.Upsample": Upsample,
}


def parse_model(cfg: dict, ch_in: int = 3):
    """Parse YAML config and build model layers."""
    depth_mul = cfg["depth_multiple"]
    width_mul = cfg["width_multiple"]
    max_ch = cfg["max_channels"]
    nc = cfg["nc"]
    reg_max = cfg.get("reg_max", 16)
    scale = str(cfg.get("scale", "")).lower()

    def scale_ch(c):
        # Width-scale output channels, but — like Ultralytics — skip scaling when
        # the count already equals nc (e.g. a head whose width is the class count).
        return c if c == nc else make_divisible(min(c, max_ch) * width_mul, 8)

    def maybe_repeat(build, repeats):
        # Ultralytics wraps a non-repeat base module in nn.Sequential when n>1
        # (modules that consume `n` internally, like C3k2/C2PSA, are built once).
        return nn.Sequential(*(build() for _ in range(repeats))) if repeats > 1 else build()

    layers = []
    save = []
    # ch_map[i] = output channels of layer i; -1 = initial input
    ch_map = {-1: ch_in}
    prev_c = ch_in  # tracks previous layer's output channels

    model_def = cfg.get("backbone", []) + cfg.get("neck", []) + cfg.get("head", [])

    for i, (f, n, m_name, args) in enumerate(model_def):
        m_cls = MODULE_MAP.get(m_name)
        if m_cls is None:
            raise ValueError(f"Unknown module: {m_name}")

        n = max(round(n * depth_mul), 1) if n > 1 else n

        # Determine input channels
        if f == -1:
            c1 = prev_c
        elif isinstance(f, list):
            c1 = sum(ch_map[j] if j != -1 else prev_c for j in f)
        else:
            c1 = ch_map[f]

        # Build module and compute output channels
        if m_cls is Conv:
            c2 = scale_ch(args[0])
            m = maybe_repeat(lambda: m_cls(c1, c2, *args[1:]), n)

        elif m_cls is C3k2:
            c2 = scale_ch(args[0])
            c3k = args[1] if len(args) > 1 else False
            e = args[2] if len(args) > 2 else 0.5
            # Ultralytics forces c3k=True at M/L/X scales (parse_model line 1759)
            if scale in "mlx":
                c3k = True
            m = m_cls(c1, c2, n, c3k=c3k, e=e)

        elif m_cls is SPPF:
            c2 = scale_ch(args[0])
            m = maybe_repeat(lambda: m_cls(c1, c2, *args[1:]), n)

        elif m_cls is C2PSA:
            c2 = scale_ch(args[0])
            m = m_cls(c1, c2, n)

        elif m_cls is Concat:
            c2 = c1  # c1 already summed over inputs
            m = m_cls(*args)

        elif m_cls is Upsample:
            c2 = c1
            scale_factor = args[1] if len(args) > 1 else 2
            mode = args[2] if len(args) > 2 else "nearest"
            m = m_cls(size=args[0], scale_factor=scale_factor, mode=mode)

        elif m_cls is Detect:
            in_chs = tuple(ch_map[x] for x in f)
            m = m_cls(nc=nc, ch=in_chs, reg_max=reg_max)
            c2 = None

        else:
            c2 = c1
            m = maybe_repeat(lambda: m_cls(*args), n)

        m.i = i
        m.f = f
        m.type = m_name

        layers.append(m)
        ch_map[i] = c2 if c2 is not None else prev_c
        prev_c = ch_map[i]

        if isinstance(f, list):
            save.extend(x for x in f if x != -1)

    return nn.ModuleList(layers), sorted(set(save))


class YOLOv11(nn.Module):
    def __init__(self, cfg_path: str, ch=3, nc=None):
        super().__init__()
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        if nc is not None:
            self.cfg["nc"] = nc

        self.model, self.save = parse_model(self.cfg, ch_in=ch)
        self.nc = self.cfg["nc"]
        self.stride = self._get_strides()
        self._init_head_strides()
        self.names = {i: str(i) for i in range(self.nc)}
        initialize_weights(self)

    def _get_strides(self):
        # Compute strides by running dummy forward in training mode.
        # Detect.forward() in training mode returns the list of raw feature maps.
        imgsz = 256
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        self.train()
        with torch.no_grad():
            feats = self._forward_once(dummy)
        # feats is a list of tensors [P3, P4, P5]
        if isinstance(feats, (list, tuple)) and isinstance(feats[0], torch.Tensor):
            strides = torch.tensor([imgsz / f.shape[-2] for f in feats], dtype=torch.float32)
        else:
            strides = torch.tensor([8.0, 16.0, 32.0])
        return strides

    def _init_head_strides(self):
        for m in self.model:
            if isinstance(m, Detect):
                m.stride = self.stride
                m.bias_init()

    def _forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, list):
                    # -1 means "previous layer output" (current x); positive = saved index
                    x = [x if j == -1 else y[j] for j in m.f]
                else:
                    x = y[m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def forward(self, x):
        return self._forward_once(x)

    def info(self):
        n_params = sum(p.numel() for p in self.parameters())
        print(f"YOLOv11 | scale={self.cfg.get('scale','?')} | params={n_params/1e6:.1f}M")


def load_model(cfg_path: str, weights: str = None, nc: int = None, device="cpu"):
    with open(cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    if _cfg.get("arch") == "yolov9":
        from models.yolov9 import YOLOv9
        model = YOLOv9(cfg_path, nc=nc)
    else:
        model = YOLOv11(cfg_path, nc=nc)
    if weights:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        # Prefer EMA (smoothed) weights for evaluation/inference, then fall back to
        # the raw model weights, then to a bare state_dict.
        if isinstance(ckpt, dict):
            state = ckpt.get("ema") or ckpt.get("model") or ckpt
        else:
            state = ckpt
        # Ultralytics checkpoints store the full model object under "model"; unwrap it.
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        model.load_state_dict(state, strict=False)
    return model.to(device)
