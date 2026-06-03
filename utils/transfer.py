"""
Transfer-learning / pretrained-weight loading for YOLOv11.

================================================================================
ARCHITECTURE ANALYSIS (verified against this codebase, not assumed)
================================================================================
State-dict keys here are `model.<idx>.…`. The Detect head is element 23 of the
`YOLOv11.model` ModuleList — there is NO `model.detect` attribute. All head access
goes through `get_detect(model)` and the index is discovered dynamically.

Layer-index → stage map (yolov11n.yaml; scale-independent layer count):
    backbone : model.0  … model.10   (len(cfg['backbone']) layers)
    neck     : model.11 … model.22
    head     : model.23 (Detect)

Detect head (models/head.py):
  cv2[i] (box):  Conv(x,c2,3) → Conv(c2,c2,3) → Conv2d(c2, 4*reg_max, 1)
                 c2 = max(16, ch[0]//4, reg_max*4). NO layer depends on nc.
  cv3[i] (cls):  [DWConv(x,x,3), Conv(x,c3,1)] →
                 [DWConv(c3,c3,3), Conv(c3,c3,1)] → Conv2d(c3, nc, 1)
                 c3 = max(ch[0], min(nc, 100)).
  dfl: fixed conv, requires_grad=False, zero class-dependent params.

⚠️ CRITICAL: `c3` DEPENDS ON nc. For yolov11n ch[0]=64, so:
       nc<=64  → c3=64  (only the final Conv2d `cv3.i.2` changes with nc)
       64<nc<=100 → c3=nc (the WHOLE cv3 branch — intermediates + final — changes)
   Therefore "only the final cls Conv2d changes" is FALSE in general. Loading must
   be SHAPE-AWARE: load every key whose shape matches, reinitialize the rest.

================================================================================
STRATEGY SELECTION
================================================================================
  'partial'        Load every shape-compatible key; reinitialize shape-mismatched
                   ones (the cls branch when nc/ c3 differ). Recommended default.
  'backbone_neck'  Load backbone+neck only; reinitialize the entire Detect head
                   (cv2+cv3+dfl). Use when the head's domain is very different.
  'subset'         Like 'partial', but the final cls Conv2d rows are copied from
                   chosen pretrained class indices (nc_subset_map={new:old}).
                   Only meaningful when the pretrained final-cls input dim (c3)
                   matches the new model's; otherwise the overlapping input slice
                   is copied and a warning is logged.

================================================================================
RECOMMENDED FINE-TUNING LR RATIOS  (backbone : neck : head_cls_new = 1 : 1 : 100)
    lr_backbone = lr_neck = 1e-4,  lr_head_box = lr_head_cls_ft = 1e-3,
    lr_head_cls_new = 1e-2   (final cls is randomly initialized → needs high LR)
Bias and BatchNorm params get weight_decay=0 (separate groups).

FREEZE SCHEDULE: freeze the backbone for ~5–10% of total epochs, then unfreeze and
rebuild the optimizer so the now-trainable backbone params join with fresh state.

================================================================================
WORKED EXAMPLES
================================================================================
  partial (COCO80 → custom 3-class):
      load_pretrained_weights(model, 'best.pt', nc_new=3, strategy='partial')
      config: configs/training/finetune_partial.yaml
  backbone_neck (extreme domain shift):
      load_pretrained_weights(model, 'best.pt', nc_new=5, strategy='backbone_neck')
      config: configs/training/finetune_backbone_neck.yaml
  subset (person+car out of COCO80):
      load_pretrained_weights(model, 'best.pt', nc_new=2, strategy='subset',
                              nc_subset_map={0: 0, 1: 2})
      config: configs/training/finetune_subset.yaml
"""
import re
import torch
import torch.nn as nn

from models.head import Detect

# Final cls Conv2d inside cv3 is sub-module index 2: e.g. `...cv3.0.2.weight`.
_CLS_FINAL_RE = re.compile(r"cv3\.\d+\.2\.(weight|bias)$")


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────
def get_detect(model: nn.Module) -> Detect:
    """Return the Detect head module of a YOLOv11 model."""
    inner = getattr(model, "model", model)
    for m in inner:
        if isinstance(m, Detect):
            return m
    raise ValueError("No Detect head found in model")


def _detect_index(model: nn.Module) -> int:
    """Layer index of the Detect head within model.model (e.g. 23)."""
    inner = getattr(model, "model", model)
    for i, m in enumerate(inner):
        if isinstance(m, Detect):
            return getattr(m, "i", i)
    raise ValueError("No Detect head found in model")


def _detect_prefix(model: nn.Module) -> str:
    return f"model.{_detect_index(model)}."


def _stage_bounds(model: nn.Module):
    """(n_backbone, detect_idx): backbone=[0,n_backbone), neck=[n_backbone,detect_idx), head=detect_idx."""
    detect_idx = _detect_index(model)
    n_backbone = len(model.cfg.get("backbone", [])) if hasattr(model, "cfg") else detect_idx
    return n_backbone, detect_idx


def _layer_index(param_name: str):
    """Parse the integer layer index from a key like 'model.11.cv1.conv.weight'."""
    parts = param_name.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def _load_state_from_ckpt(weights_path: str):
    """Load a checkpoint and return (state_dict, raw_ckpt, nc_source)."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    nc_source = "model"
    if isinstance(ckpt, dict):
        state = ckpt.get("ema")
        if state is not None:
            nc_source = "ema"
        else:
            state = ckpt.get("model")
        if state is None:
            state = ckpt  # bare state_dict
    else:
        state = ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    # Cast to float for clean comparison/loading (checkpoints are saved fp16).
    state = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in state.items()}
    return state, (ckpt if isinstance(ckpt, dict) else {}), nc_source


# ──────────────────────────────────────────────────────────────────────────
# 2.1  inspect_checkpoint
# ──────────────────────────────────────────────────────────────────────────
def inspect_checkpoint(weights_path: str) -> dict:
    """Load a checkpoint and return metadata without building any model."""
    state, ckpt, nc_source = _load_state_from_ckpt(weights_path)

    cls_keys = sorted(k for k in state if _CLS_FINAL_RE.search(k) and k.endswith("weight"))
    cls_weight_shapes = [tuple(state[k].shape) for k in cls_keys]
    nc = cls_weight_shapes[0][0] if cls_weight_shapes else -1

    param_count = sum(v.numel() for v in state.values() if torch.is_tensor(v)) / 1e6
    arch = "unknown"
    if isinstance(ckpt, dict):
        arch = str(ckpt.get("train_args", {}).get("model", "unknown"))

    return {
        "nc": nc,
        "nc_pretrained": nc,          # alias — different call sites use either name
        "nc_source": nc_source,
        # Official checkpoints may store epoch/best_fitness as None — coerce safely.
        "epoch": int(ckpt.get("epoch") if isinstance(ckpt, dict) and ckpt.get("epoch") is not None else -1),
        "best_fitness": float(ckpt.get("best_fitness") if isinstance(ckpt, dict) and ckpt.get("best_fitness") is not None else 0.0),
        "arch": arch,
        "param_count": param_count,
        "cls_weight_shapes": cls_weight_shapes,
    }


# ──────────────────────────────────────────────────────────────────────────
# 2.3  is_cls_final_layer
# ──────────────────────────────────────────────────────────────────────────
def is_cls_final_layer(key: str, model: nn.Module) -> bool:
    """True if `key` is the final classification Conv2d (weight or bias) of the Detect head."""
    return key.startswith(_detect_prefix(model)) and bool(_CLS_FINAL_RE.search(key))


# ──────────────────────────────────────────────────────────────────────────
# 2.2  load_pretrained_weights
# ──────────────────────────────────────────────────────────────────────────
def load_pretrained_weights(
    model: nn.Module,
    weights_path: str,
    nc_new: int,
    strategy: str = "partial",
    nc_subset_map: dict = None,
    verbose: bool = True,
) -> dict:
    """Load pretrained weights into a model that may have a different nc. See module docstring."""
    state, _, _ = _load_state_from_ckpt(weights_path)
    info = inspect_checkpoint(weights_path)
    nc_pretrained = info["nc"]
    model_sd = model.state_dict()
    detect_pfx = _detect_prefix(model)

    loaded, skipped, reinitialized = [], [], []
    to_load = {}

    def shape_ok(k):
        return k in model_sd and model_sd[k].shape == state[k].shape

    if strategy == "partial":
        for k in model_sd:
            if shape_ok(k):
                to_load[k] = state[k]
                loaded.append(k)
            else:
                reinitialized.append(k)

    elif strategy == "backbone_neck":
        for k in model_sd:
            if k.startswith(detect_pfx):
                reinitialized.append(k)
            elif shape_ok(k):
                to_load[k] = state[k]
                loaded.append(k)
            else:
                reinitialized.append(k)

    elif strategy == "subset":
        if nc_subset_map is None:
            raise ValueError("strategy='subset' requires nc_subset_map={new_idx: old_idx}")
        # Load everything shape-compatible first (backbone/neck/cv2; reinit cv3 by shape).
        for k in model_sd:
            if shape_ok(k):
                to_load[k] = state[k]
                loaded.append(k)
            else:
                reinitialized.append(k)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    # Keys present in the checkpoint but absent from the model are genuine extras.
    skipped = [k for k in state if k not in model_sd]

    # Apply the shape-compatible tensors.
    model.load_state_dict(to_load, strict=False)

    # Re-seed the head appropriately.
    detect = get_detect(model)
    if strategy in ("partial", "backbone_neck"):
        detect.bias_init()

    elif strategy == "subset":
        detect.bias_init()
        new_sd = model.state_dict()
        for k in [c for c in model_sd if is_cls_final_layer(c, model)]:
            if k not in state:
                continue
            dst = new_sd[k].clone()
            src = state[k]
            for new_idx, old_idx in nc_subset_map.items():
                if new_idx >= dst.shape[0] or old_idx >= src.shape[0]:
                    continue
                if dst[new_idx].shape == src[old_idx].shape:
                    dst[new_idx] = src[old_idx]
                else:
                    # c3 differs between checkpoint and model — copy overlapping slice.
                    n = min(dst[new_idx].shape[0], src[old_idx].shape[0]) if dst.dim() > 1 else None
                    if n is not None:
                        dst[new_idx, :n] = src[old_idx, :n]
                    if verbose:
                        print(f"  [subset] {k}: c3 mismatch "
                              f"({tuple(src[old_idx].shape)} → {tuple(dst[new_idx].shape)}), "
                              f"copied overlapping slice")
            with torch.no_grad():
                dict(model.named_parameters())[k].copy_(dst)
            if k not in reinitialized:
                reinitialized.append(k)

    result = {
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "reinitialized": reinitialized,
        "nc_pretrained": nc_pretrained,
        "nc_new": nc_new,
        "strategy": strategy,
    }
    if verbose:
        print(f"  [transfer] strategy={strategy}  nc {nc_pretrained}→{nc_new}  "
              f"loaded={len(loaded)}  reinitialized={len(reinitialized)}  skipped={len(skipped)}")
    return result


# ──────────────────────────────────────────────────────────────────────────
# 2.4  build_param_groups
# ──────────────────────────────────────────────────────────────────────────
def build_param_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_neck: float,
    lr_head_box: float,
    lr_head_cls_ft: float,
    lr_head_cls_new: float,
    weight_decay: float,
) -> list:
    """Build per-stage optimizer param groups with differential LRs (see module docstring)."""
    n_backbone, detect_idx = _stage_bounds(model)
    bn_modules = {n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)}

    def role_lr(name):
        idx = _layer_index(name)
        if idx is None:
            return "neck", lr_neck  # safe fallback
        if idx < n_backbone:
            return "backbone", lr_backbone
        if idx < detect_idx:
            return "neck", lr_neck
        # head
        if "cv2" in name:
            return "head_box", lr_head_box
        if _CLS_FINAL_RE.search(name):
            return "head_cls_new", lr_head_cls_new
        if "cv3" in name:
            return "head_cls_ft", lr_head_cls_ft
        return "head_box", lr_head_box  # dfl & misc → box group (frozen / harmless)

    # Ordered buckets: (role, is_bias) → {params, lr}
    order = ["backbone", "neck", "head_box", "head_cls_ft", "head_cls_new"]
    buckets = {}
    for name, p in model.named_parameters():
        role, lr = role_lr(name)
        parent = name.rsplit(".", 1)[0]
        leaf = name.rsplit(".", 1)[1]
        is_bias = (leaf == "bias") or (parent in bn_modules)
        key = (role, is_bias)
        buckets.setdefault(key, {"params": [], "lr": lr})
        buckets[key]["params"].append(p)

    groups = []
    for role in order:
        for is_bias in (False, True):
            b = buckets.get((role, is_bias))
            if not b or not b["params"]:
                continue
            groups.append({
                "params": b["params"],
                "lr": b["lr"],
                "weight_decay": 0.0 if is_bias else weight_decay,
                "is_bias": is_bias,
                "name": f"{role}{'_bias_bn' if is_bias else '_weights'}",
            })
    return groups


# ──────────────────────────────────────────────────────────────────────────
# 2.5 / 2.6  freezing
# ──────────────────────────────────────────────────────────────────────────
def freeze_backbone(model: nn.Module, freeze: bool = True) -> list:
    """Freeze/unfreeze backbone parameters (layers before the neck). BN buffers untouched."""
    n_backbone, _ = _stage_bounds(model)
    affected = []
    for name, p in model.named_parameters():
        idx = _layer_index(name)
        if idx is not None and idx < n_backbone:
            p.requires_grad_(not freeze)
            affected.append(name)
    return affected


def freeze_layers_by_name(model: nn.Module, patterns: list, freeze: bool = True) -> list:
    """Freeze/unfreeze parameters whose names match any pattern (with friendly aliases)."""
    n_backbone, detect_idx = _stage_bounds(model)
    detect_pfx = _detect_prefix(model)

    def matches(name):
        idx = _layer_index(name)
        for pat in patterns:
            if pat == "backbone" and idx is not None and idx < n_backbone:
                return True
            if pat == "neck" and idx is not None and n_backbone <= idx < detect_idx:
                return True
            if pat in ("head", "detect") and name.startswith(detect_pfx):
                return True
            if pat.startswith("detect.") and name.startswith(detect_pfx + pat[len("detect."):]):
                return True
            if pat in name:
                return True
        return False

    affected = []
    for name, p in model.named_parameters():
        if matches(name):
            p.requires_grad_(not freeze)
            affected.append(name)
    return affected


# ──────────────────────────────────────────────────────────────────────────
# 2.8  build_optimizer_from_groups
# ──────────────────────────────────────────────────────────────────────────
def build_optimizer_from_groups(param_groups: list, name: str = "AdamW", momentum: float = 0.937):
    """Construct an optimizer from pre-built parameter groups."""
    if name in ("Adam", "AdamW"):
        return torch.optim.AdamW(param_groups, betas=(momentum, 0.999))
    if name == "SGD":
        return torch.optim.SGD(param_groups, momentum=momentum, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


# ──────────────────────────────────────────────────────────────────────────
# 2.7  rebuild_optimizer_after_unfreeze
# ──────────────────────────────────────────────────────────────────────────
def rebuild_optimizer_after_unfreeze(model: nn.Module, args: dict, old_opt: torch.optim.Optimizer):
    """
    Rebuild the optimizer after unfreezing the backbone, preserving optimizer state
    (momentum history) for params that already existed in old_opt. Newly-trainable
    backbone params start with fresh state.
    """
    groups = build_param_groups(
        model=model,
        lr_backbone=args["lr_backbone"],
        lr_neck=args["lr_neck"],
        lr_head_box=args["lr_head_box"],
        lr_head_cls_ft=args["lr_head_cls_ft"],
        lr_head_cls_new=args["lr_head_cls_new"],
        weight_decay=args["weight_decay"],
    )
    new_opt = build_optimizer_from_groups(
        groups, name=args.get("optimizer", "AdamW"), momentum=args.get("momentum", 0.937)
    )
    # Carry over state for params already tracked by the old optimizer.
    old_state = old_opt.state
    for g in new_opt.param_groups:
        for p in g["params"]:
            if p in old_state:
                new_opt.state[p] = old_state[p]
    return new_opt
