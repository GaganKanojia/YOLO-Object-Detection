#!/usr/bin/env python3
"""Verification tests for utils/transfer.py (transfer learning / pretrained loading)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.yolov11 import YOLOv11
from utils.transfer import (
    inspect_checkpoint, load_pretrained_weights,
    build_param_groups, freeze_backbone,
)

PASS = "\033[92mPASS\033[0m"; FAIL = "\033[91mFAIL\033[0m"
results = {}

def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(f"  {PASS if cond else FAIL} {name}" + (f": {detail}" if detail else ""))

print("=" * 60); print("TRANSFER LEARNING TESTS"); print("=" * 60)

CFG = "configs/model/yolov11n.yaml"

# ── SETUP: create a dummy pretrained checkpoint (nc=80) ──────────────
print("\nSetup: creating dummy nc=80 checkpoint...")
m80 = YOLOv11(CFG, nc=80)
dummy_ckpt = {
    "model": m80.state_dict(),
    "ema": m80.state_dict(),
    "epoch": 100,
    "best_fitness": 0.512,
    "updates": 10000,
}
torch.save(dummy_ckpt, "/tmp/dummy_pretrained.pt")

# T1 — inspect_checkpoint
print("\nT1 — inspect_checkpoint")
info = inspect_checkpoint("/tmp/dummy_pretrained.pt")
print(f"  {info}")
check("inspect_nc",      info["nc_pretrained"] == 80, f"nc={info['nc_pretrained']}")
check("inspect_epoch",   info["epoch"] == 100,        f"epoch={info['epoch']}")
check("inspect_fitness", info["best_fitness"] > 0,    f"fitness={info['best_fitness']}")
check("inspect_shapes",  len(info["cls_weight_shapes"]) == 3, f"{info['cls_weight_shapes']}")

# T2 — partial strategy: nc=80 → nc=3
print("\nT2 — partial strategy (nc=80 → nc=3)")
m3 = YOLOv11(CFG, nc=3)
result = load_pretrained_weights(m3, "/tmp/dummy_pretrained.pt", nc_new=3, strategy="partial")
print(f"  loaded={len(result['loaded_keys'])}  reinit={len(result['reinitialized'])}  skip={len(result['skipped_keys'])}")
check("partial_loaded_many",   len(result["loaded_keys"]) > 100,    f"{len(result['loaded_keys'])} keys")
check("partial_reinitialized", len(result["reinitialized"]) > 0,    f"{len(result['reinitialized'])} keys")
check("partial_skipped_few",   len(result["skipped_keys"]) < 20,    f"{len(result['skipped_keys'])} keys")
m3.eval()
with torch.no_grad():
    out = m3(torch.randn(1, 3, 640, 640))
decoded = out[0] if isinstance(out, (list, tuple)) else out
check("partial_forward_ok", decoded.shape == torch.Size([1, 7, 8400]), str(decoded.shape))  # 7 = 4+nc

# T3 — partial strategy: nc=80 → nc=80
print("\nT3 — partial strategy same nc (nc=80 → nc=80)")
m80b = YOLOv11(CFG, nc=80)
result80 = load_pretrained_weights(m80b, "/tmp/dummy_pretrained.pt", nc_new=80, strategy="partial")
check("same_nc_all_loaded", len(result80["reinitialized"]) == 0,
      f"reinitialized={len(result80['reinitialized'])}")

# T4 — backbone_neck strategy: nc=80 → nc=5
print("\nT4 — backbone_neck strategy (nc=80 → nc=5)")
m5 = YOLOv11(CFG, nc=5)
result5 = load_pretrained_weights(m5, "/tmp/dummy_pretrained.pt", nc_new=5, strategy="backbone_neck")
check("bn_loaded_backbone", len(result5["loaded_keys"]) > 50, f"{len(result5['loaded_keys'])} keys")
check("bn_reinit_head", len(result5["reinitialized"]) > len(result["reinitialized"]),
      f"reinit={len(result5['reinitialized'])}")
m5.eval()
with torch.no_grad():
    out5 = m5(torch.randn(1, 3, 640, 640))
d5 = out5[0] if isinstance(out5, (list, tuple)) else out5
check("bn_forward_ok", d5.shape == torch.Size([1, 9, 8400]), str(d5.shape))

# T5 — subset strategy: nc=80 → nc=2 (person+car)
# NOTE (adapted from strategy): here c3 differs between checkpoint (nc=80 → c3=80) and
# model (nc=2 → c3=64), so the final cls Conv2d input dim differs (80 vs 64). The subset
# loader copies the overlapping input slice [:min(c3)]; we verify that copied slice.
print("\nT5 — subset strategy (nc=80 → nc=2, person+car)")
m2 = YOLOv11(CFG, nc=2)
nc_map = {0: 0, 1: 2}
result2 = load_pretrained_weights(m2, "/tmp/dummy_pretrained.pt", nc_new=2,
                                  strategy="subset", nc_subset_map=nc_map)
m80_state = m80.state_dict()
m2_state = m2.state_dict()
cls_keys_m2 = [k for k in result2["reinitialized"] if k.endswith("cv3.0.2.weight")] or \
              [k for k in result2["reinitialized"] if "weight" in k]
if cls_keys_m2:
    k = cls_keys_m2[0]
    n = min(m2_state[k].shape[1], m80_state[k].shape[1])  # overlapping c3 input channels
    row0_diff = (m2_state[k][0, :n] - m80_state[k][0, :n]).abs().max().item()
    row1_diff = (m2_state[k][1, :n] - m80_state[k][2, :n]).abs().max().item()
    check("subset_row0_copied", row0_diff < 1e-6, f"diff={row0_diff:.8f}")
    check("subset_row1_copied", row1_diff < 1e-6, f"diff={row1_diff:.8f}")
else:
    check("subset_row0_copied", False, "no cls final key reinitialized")
m2.eval()
with torch.no_grad():
    out2 = m2(torch.randn(1, 3, 640, 640))
d2 = out2[0] if isinstance(out2, (list, tuple)) else out2
check("subset_forward_ok", d2.shape == torch.Size([1, 6, 8400]), str(d2.shape))

# T6 — Backbone weights preserved after partial load
print("\nT6 — Backbone weights identical after partial load")
m80c = YOLOv11(CFG, nc=80)
load_pretrained_weights(m80c, "/tmp/dummy_pretrained.pt", nc_new=80, strategy="partial")
m80_s = m80.state_dict(); m80c_s = m80c.state_dict()
backbone_keys = [k for k in m80_s if "model.23" not in k][:5]
all_match = True
for k in backbone_keys:
    if k in m80c_s:
        if (m80_s[k].float() - m80c_s[k].float()).abs().max().item() > 1e-6:
            all_match = False; break
check("backbone_preserved", all_match, f"checked {len(backbone_keys)} keys")

# T7 — Differential LR param groups
print("\nT7 — build_param_groups differential LRs")
m_pg = YOLOv11(CFG, nc=5)
groups = build_param_groups(
    model=m_pg, lr_backbone=1e-4, lr_neck=1e-4, lr_head_box=1e-3,
    lr_head_cls_ft=1e-3, lr_head_cls_new=1e-2, weight_decay=5e-4)
lrs = list({g["lr"] for g in groups})
check("param_groups_exist", len(groups) >= 3, f"{len(groups)} groups")
check("multiple_lrs",       len(lrs) >= 2, f"LRs={sorted(lrs)}")
total_params_in_groups = sum(len(g["params"]) for g in groups)
total_params_in_model = len(list(m_pg.parameters()))
check("all_params_covered", total_params_in_groups >= total_params_in_model,
      f"{total_params_in_groups} vs {total_params_in_model}")

# T8 — Backbone freeze/unfreeze
print("\nT8 — freeze_backbone")
m_fr = YOLOv11(CFG, nc=5)
frozen = freeze_backbone(m_fr, freeze=True)
check("freeze_runs", len(frozen) > 0, f"froze {len(frozen)} params")
frozen_params = [p for n, p in m_fr.named_parameters() if not p.requires_grad]
check("backbone_no_grad", len(frozen_params) > 0, f"{len(frozen_params)} frozen")
head_params = [p for n, p in m_fr.named_parameters() if "model.23" in n and p.requires_grad]
check("head_still_trains", len(head_params) > 0, f"{len(head_params)} trainable head params")
unfrozen = freeze_backbone(m_fr, freeze=False)
# NOTE (adapted from strategy): DFL.conv.weight is requires_grad_(False) by design
# (a fixed integral projection, never trained), so it stays non-trainable after
# unfreezing. We verify every *trainable-by-design* param is restored, excluding DFL.
non_trainable = [n for n, p in m_fr.named_parameters() if not p.requires_grad]
all_trainable = all("dfl" in n for n in non_trainable)
check("unfreeze_restores", all_trainable, f"remaining non-trainable={non_trainable}")

# T9 — Loss computation after partial load (sanity check)
print("\nT9 — Loss computation after weight loading")
from loss.loss import DetectionLoss
m_loss = YOLOv11(CFG, nc=3).train()
load_pretrained_weights(m_loss, "/tmp/dummy_pretrained.pt", nc_new=3, strategy="partial")
criterion = DetectionLoss(m_loss, box=7.5, cls=0.5, dfl=1.5)
dummy_img = torch.randn(2, 3, 640, 640)
batch = {
    "img": dummy_img,
    "cls": torch.randint(0, 3, (6, 1)).float(),
    "bboxes": torch.rand(6, 4).clamp(0.05, 0.95),
    "batch_idx": torch.tensor([0, 0, 0, 1, 1, 1]),
}
out = m_loss(dummy_img)
loss, items = criterion(out, batch)
check("loss_after_load_finite", not torch.isnan(loss).any() and not torch.isinf(loss).any(),
      f"loss={loss.item():.4f}")
loss.backward()
grads = [p.grad for p in m_loss.parameters() if p.grad is not None]
check("grads_after_load", len(grads) > 0, f"{len(grads)} grads")

# ── SUMMARY ──────────────────────────────────────────────────────────
print()
passed = sum(results.values()); total = len(results)
failed = [k for k, v in results.items() if not v]
print(f"Result: {passed}/{total} passed")
if failed:
    print(f"\033[91mFAILED: {failed}\033[0m"); sys.exit(1)
else:
    print("\033[92mAll transfer learning tests PASSED.\033[0m")
