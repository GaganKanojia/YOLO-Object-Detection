"""
YOLOv9 model assembly.

Parses a YOLOv9 YAML config file and constructs the model.  The GELAN
architecture (RepNCSPELAN4-based) differs from YOLOv11 in several ways:
  - No depth_multiple / width_multiple scaling — channel counts are hardcoded
    per variant in each YAML.
  - No separate 'neck' section — FPN + PAN are embedded in the 'head' section.
  - CBLinear layers (yolov9e only) return a Python list of tensors; the
    forward pass and channel-tracking logic handle this correctly.
  - PGI feature enhancement (yolov9e) is always active, including at inference.

The YOLOv9 class exposes the same interface as YOLOv11:
  model.nc, model.stride, model.names, model.model (nn.ModuleList), model.save
"""

import yaml
import torch
import torch.nn as nn
from pathlib import Path

from models.blocks import Conv, Concat, Upsample, DFL
from models.head import Detect, make_anchors, dist2bbox
from models.yolov9_blocks import (
    RepNCSPELAN4, ELAN1, AConv, ADown, SPPELAN, CBLinear, CBFuse, DetectV9
)


# ---------------------------------------------------------------------------
# Module registry for YOLOv9 YAML strings → Python classes
# ---------------------------------------------------------------------------

MODULE_MAP_V9 = {
    "Conv":           Conv,
    "AConv":          AConv,
    "ADown":          ADown,
    "RepNCSPELAN4":   RepNCSPELAN4,
    "ELAN1":          ELAN1,
    "SPPELAN":        SPPELAN,
    "CBLinear":       CBLinear,
    "CBFuse":         CBFuse,
    "Concat":         Concat,
    # Both "Upsample" and "nn.Upsample" map to our wrapper which handles
    # size=None correctly (consistent with how yolov11.py treats "nn.Upsample").
    "Upsample":       Upsample,
    "nn.Upsample":    Upsample,
    "nn.Identity":    torch.nn.Identity,
    # YOLOv9 uses the legacy plain-Conv classification branch in the detect head.
    "Detect":         DetectV9,
}


def _resolve_c1(f, ch_map, prev_c):
    """Single-input channel lookup — raises if the source is a CBLinear (list)."""
    if f == -1:
        return prev_c
    ch = ch_map[f]
    if isinstance(ch, list):
        raise ValueError(
            f"Layer with f={f} tries to use a CBLinear output as a direct "
            f"single-tensor input.  Only CBFuse should reference CBLinear outputs."
        )
    return ch


def _sum_channels(f_list, ch_map, prev_c):
    """Sum channels across a list of source indices (for Concat)."""
    total = 0
    for j in f_list:
        ch = prev_c if j == -1 else ch_map[j]
        if isinstance(ch, list):
            total += sum(ch)
        else:
            total += ch
    return total


def parse_model_v9(cfg: dict, ch_in: int = 3):
    """Parse a YOLOv9 YAML config dictionary and build the layer list.

    Unlike parse_model in yolov11.py, this function:
      - Does not apply depth/width scaling.
      - Handles CBLinear (list output) and CBFuse (list input) correctly.
      - Maps 'backbone' + 'head' sections; there is no 'neck' section in v9.

    Returns:
        layers: nn.ModuleList of all model layers.
        save:   sorted list of layer indices whose outputs must be cached.
    """
    nc = cfg["nc"]
    reg_max = cfg.get("reg_max", 16)

    layers = []
    save = []
    # ch_map[i] = output channels of layer i.
    # For CBLinear layers this is a Python list of ints (one per output slice).
    # For all other layers it is a plain int.
    ch_map = {-1: ch_in}
    prev_c = ch_in  # channels of the most-recently-processed non-list layer

    model_def = cfg.get("backbone", []) + cfg.get("head", [])

    for i, (f, n, m_name, args) in enumerate(model_def):
        m_cls = MODULE_MAP_V9.get(m_name)
        if m_cls is None:
            raise ValueError(
                f"YOLOv9 parser: unknown module '{m_name}' at layer {i}. "
                f"Add it to MODULE_MAP_V9 in models/yolov9.py."
            )

        # ---- build the module ------------------------------------------------
        if m_cls is Conv:
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2 = args[0]
            m = m_cls(c1, c2, *args[1:])

        elif m_cls is AConv or m_cls is ADown:
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2 = args[0]
            m = m_cls(c1, c2)

        elif m_cls is RepNCSPELAN4 or m_cls is ELAN1:
            # YAML args: [c2, c3, c4, n?]  c1 comes from previous layer.
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2 = args[0]
            m = m_cls(c1, *args)

        elif m_cls is SPPELAN:
            # YAML args: [c2, c3]
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2 = args[0]
            m = m_cls(c1, *args)

        elif m_cls is Concat:
            # f is always a list for Concat.
            c1 = _sum_channels(f, ch_map, prev_c)
            c2 = c1
            m = m_cls(*args)

        elif m_cls is Upsample:
            # "Upsample" and "nn.Upsample" both map to our Upsample wrapper.
            # YAML args: [size, scale_factor, mode] — size may be None (YAML null).
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2 = c1
            size = args[0]
            scale_factor = args[1] if len(args) > 1 else 2
            mode = args[2] if len(args) > 2 else "nearest"
            m = m_cls(size=size, scale_factor=scale_factor, mode=mode)

        elif m_cls is torch.nn.Identity:
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2 = c1
            m = m_cls()

        elif m_cls is CBLinear:
            # YAML args: [list_of_channel_widths]
            # f is a single int referencing a regular (non-CBLinear) layer.
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2s = args[0]           # a list, e.g. [64] or [64, 128]
            c2 = c2s                # stored as list in ch_map
            m = m_cls(c1, c2s)

        elif m_cls is CBFuse:
            # YAML args: [idx_list]
            # f is a list: [...CBLinear_indices..., current_or_explicit_index]
            # Output channels = channels of last input (the current-level tensor).
            last_f = f[-1]
            c2 = prev_c if last_f == -1 else ch_map[last_f]
            if isinstance(c2, list):
                raise ValueError(
                    f"CBFuse at layer {i}: last input f[-1]={last_f} points to a "
                    f"CBLinear output (list).  The last element of CBFuse.f must be "
                    f"a regular tensor source."
                )
            c1 = c2
            m = m_cls(args[0])      # CBFuse(idx)

        elif isinstance(m_cls, type) and issubclass(m_cls, Detect):
            # f is a list of source indices (all must be regular tensor layers).
            in_chs = []
            for x in f:
                ch = ch_map[x]
                if isinstance(ch, list):
                    raise ValueError(
                        f"Detect at layer {i}: source {x} is a CBLinear output "
                        f"(list).  Detect must receive regular feature maps."
                    )
                in_chs.append(ch)
            m = m_cls(nc=nc, ch=tuple(in_chs), reg_max=reg_max)
            c2 = None

        else:
            c1 = _resolve_c1(f, ch_map, prev_c)
            c2 = c1
            m = m_cls(*args) if args else m_cls()

        # ---- bookkeeping -----------------------------------------------------
        m.i = i
        m.f = f
        m.type = m_name

        layers.append(m)

        # Update channel map and prev_c.
        if c2 is None:
            ch_map[i] = prev_c
        else:
            ch_map[i] = c2
        # Only update prev_c when the output is a concrete int (not a list).
        if c2 is not None and not isinstance(c2, list):
            prev_c = c2

        # Collect save indices: any layer whose output is referenced by a
        # future layer via an explicit (non -1) index.
        if isinstance(f, list):
            save.extend(x for x in f if x != -1)
        elif f != -1:
            save.append(f)

    return nn.ModuleList(layers), sorted(set(save))


def _initialize_weights(model):
    """Match Ultralytics initialize_weights: BN eps=1e-3, momentum=0.03."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True


# ---------------------------------------------------------------------------
# YOLOv9 model class
# ---------------------------------------------------------------------------

class YOLOv9(nn.Module):
    """YOLOv9 detection model.

    Exposes the same external interface as YOLOv11:
        .nc           number of detection classes
        .stride       tensor of feature-map strides (e.g. [8, 16, 32])
        .names        dict {class_id: name}
        .model        nn.ModuleList of all layers
        .save         list of layer indices whose outputs are cached
    """

    def __init__(self, cfg_path: str, ch: int = 3, nc: int = None):
        super().__init__()
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        if nc is not None:
            self.cfg["nc"] = nc

        self.model, self.save = parse_model_v9(self.cfg, ch_in=ch)
        self.nc = self.cfg["nc"]
        self.stride = self._get_strides()
        self._init_head_strides()
        self.names = {i: str(i) for i in range(self.nc)}
        _initialize_weights(self)

    def _get_strides(self):
        """Compute output strides by running a dummy forward in training mode."""
        imgsz = 256
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        self.train()
        with torch.no_grad():
            feats = self._forward_once(dummy)
        # Training mode: Detect returns list of [P3, P4, P5] feature tensors.
        if isinstance(feats, (list, tuple)) and isinstance(feats[0], torch.Tensor):
            strides = torch.tensor(
                [imgsz / f.shape[-2] for f in feats], dtype=torch.float32
            )
        else:
            strides = torch.tensor([8.0, 16.0, 32.0])
        return strides

    def _init_head_strides(self):
        for m in self.model:
            if isinstance(m, Detect):
                m.stride = self.stride
                m.bias_init()

    def _forward_once(self, x):
        """Forward pass through the model.

        For yolov9e, CBLinear layers store Python lists of tensors in `y`.
        CBFuse layers receive a mixed list [list, list, ..., tensor] and return
        a single fused tensor.  All other layers operate on single tensors.
        The -1 index in a layer's `f` list means 'use the current running x'.
        """
        y = []
        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, list):
                    # -1 → use current x; positive → fetch saved output
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
        print(
            f"YOLOv9 | scale={self.cfg.get('scale', '?')} | "
            f"params={n_params / 1e6:.1f}M"
        )


def load_yolov9(cfg_path: str, weights: str = None, nc: int = None, device: str = "cpu"):
    """Convenience loader — builds YOLOv9 and optionally loads checkpoint weights."""
    model = YOLOv9(cfg_path, nc=nc)
    if weights:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("ema") or ckpt.get("model") or ckpt
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
    return model.to(device)
