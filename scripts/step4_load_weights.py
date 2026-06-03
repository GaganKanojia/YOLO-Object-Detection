"""
Step 4 — load official YOLOv11s weights into the custom implementation.
Verify: (1) param counts match, (2) loaded tensors numerically identical,
(3) forward pass produces correct shapes and finite values.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.yolov11 import YOLOv11

WEIGHTS = 'weights/yolov11s_official.pt'
ARCH = 'configs/model/yolov11s.yaml'
NC = 80
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 50)
print("STEP 4 — WEIGHT LOADING")
print("=" * 50)

# ── Build custom model ────────────────────────────────────────────────
model = YOLOv11(ARCH, nc=NC).to(DEVICE)
custom_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"\nCustom model params  : {custom_params:.4f}M")

# ── Load checkpoint ───────────────────────────────────────────────────
ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
state = ckpt.get('ema') or ckpt.get('model')
if hasattr(state, 'state_dict'):
    state = state.state_dict()

ckpt_params = sum(v.numel() for v in state.values()
                  if isinstance(v, torch.Tensor)) / 1e6
print(f"Checkpoint params    : {ckpt_params:.4f}M")
assert abs(custom_params - ckpt_params) < 0.05, \
    f"MISMATCH: custom={custom_params:.4f}M  ckpt={ckpt_params:.4f}M"
print("Parameter count      : MATCH (within 0.05M, buffers included on ckpt side)")

# ── Load state dict ───────────────────────────────────────────────────
missing, unexpected = model.load_state_dict(state, strict=False)

print(f"\nMissing keys  ({len(missing)}):")
for k in missing[:10]:
    print(f"  {k}")
if len(missing) > 10:
    print(f"  ... and {len(missing)-10} more")

print(f"\nUnexpected keys ({len(unexpected)}):")
for k in unexpected[:10]:
    print(f"  {k}")
if len(unexpected) > 10:
    print(f"  ... and {len(unexpected)-10} more")

assert len(missing) == 0, \
    f"FATAL: {len(missing)} missing keys — weights not fully loaded!\n" + \
    "\n".join(missing[:20])
print("\nAll keys loaded      : OK")

# ── Numerical verification ────────────────────────────────────────────
custom_state = model.state_dict()
all_keys = list(state.keys())
sample_keys = (all_keys[:3]
               + all_keys[len(all_keys) // 4:len(all_keys) // 4 + 2]
               + all_keys[len(all_keys) // 2:len(all_keys) // 2 + 2]
               + all_keys[-3:])

print("\nNumerical spot-checks (max abs diff between ckpt and loaded):")
all_match = True
for k in sample_keys:
    if k not in custom_state:
        print(f"  SKIP (not in custom): {k}")
        continue
    diff = (state[k].cpu().float() - custom_state[k].cpu().float()).abs().max().item()
    status = "OK" if diff < 1e-6 else "MISMATCH"
    print(f"  {status}  {k}: max_diff={diff:.2e}")
    if diff >= 1e-6:
        all_match = False

assert all_match, "Weight tensors do not match checkpoint — loading is incorrect!"
print("\nAll spot-checks passed")

# ── Forward pass ──────────────────────────────────────────────────────
model.eval()
dummy = torch.randn(1, 3, 640, 640).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
decoded = out[0] if isinstance(out, (list, tuple)) else out

print(f"\nForward pass output shape : {tuple(decoded.shape)}")
print(f"Expected                  : (1, 84, 8400)")
assert decoded.shape == torch.Size([1, 84, 8400]), f"Wrong output shape: {decoded.shape}"
assert not torch.isnan(decoded).any(), "NaN in model output!"
assert not torch.isinf(decoded).any(), "Inf in model output!"
print("Forward pass              : OK")

# ── Save the verified loaded model ────────────────────────────────────
torch.save({'model': model.state_dict(), 'nc': NC},
           'weights/yolov11s_custom_loaded.pt')
print("\nSaved verified model to: weights/yolov11s_custom_loaded.pt")
