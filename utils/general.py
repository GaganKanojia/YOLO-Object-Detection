import os
import math
import yaml
import torch
import logging
from pathlib import Path
from copy import deepcopy


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def setup_logging(name="yolov11"):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(name)


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        stem = path.stem
        parent = path.parent
        for n in range(2, 9999):
            p = parent / f"{stem}{sep}{n}{suffix}"
            if not p.exists():
                path = p
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Cosine annealing from y1 to y2 over steps."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def linear_lr(lrf, epochs):
    return lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf


class ModelEMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay_base = decay
        self.tau = tau
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def decay(self):
        """Current EMA decay for self.updates — ramps from 0 toward decay_base."""
        return self.decay_base * (1 - math.exp(-self.updates / self.tau))

    def update(self, model):
        self.updates += 1
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        for k, v in model.__dict__.items():
            if (not include or k in include) and k not in exclude and not k.startswith("_"):
                setattr(self.ema, k, v)


def build_optimizer(model, name="auto", lr=0.01, momentum=0.937, decay=0.0005, iterations=1e5):
    """Build optimizer with parameter groups (bias/BN separate from weights)."""
    g = [], [], []  # weights, biases, BN weights
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)

    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param_name == "bias":
                g[1].append(param)
            elif param_name == "weight" and isinstance(module, bn):
                g[2].append(param)
            else:
                g[0].append(param)

    if name == "auto":
        # Use AdamW for short runs, SGD otherwise
        if iterations > 10000:
            name = "SGD"
        else:
            name = "AdamW"

    if name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    elif name in ("Adam", "AdamW"):
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer


def strip_optimizer(ckpt_path, output_path=None):
    """Strip optimizer from checkpoint to reduce file size."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt.pop("optimizer", None)
    ckpt.pop("scaler", None)
    output_path = output_path or ckpt_path
    torch.save(ckpt, output_path)
    mb = Path(output_path).stat().st_size / 1e6
    print(f"Stripped optimizer → {output_path} ({mb:.1f}MB)")
