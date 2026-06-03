"""Step 1 — inspect the official checkpoint before loading anything."""
import re
import torch

WEIGHTS = 'weights/yolov11s_official.pt'

ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
state = ckpt.get('ema') or ckpt.get('model')
if hasattr(state, 'state_dict'):
    state = state.state_dict()

# Infer nc from the final cls Conv2d of the Detect head: keys like
# `model.<idx>.cv3.<scale>.2.weight` with shape [nc, c3, 1, 1].
# NOTE: a loose `'cv3' in k` filter also matches C3k2 blocks' internal `cv3`
# convs (model.6/8/22...), so we anchor on the Detect head pattern instead.
_CLS_FINAL_RE = re.compile(r'cv3\.\d+\.2\.weight$')
final_cls_keys = [
    k for k in state
    if _CLS_FINAL_RE.search(k)
    and isinstance(state[k], torch.Tensor)
    and tuple(state[k].shape[2:]) == (1, 1)
]

print("Checkpoint keys  :", list(ckpt.keys()))
print("Epoch            :", ckpt.get('epoch', 'N/A'))
print("Best fitness     :", ckpt.get('best_fitness', 'N/A'))
print("Final cls keys   :", final_cls_keys)
for k in final_cls_keys:
    print(f"  {k}: {tuple(state[k].shape)}")

nc = state[final_cls_keys[0]].shape[0] if final_cls_keys else '?'
total_params = sum(v.numel() for v in state.values()
                   if isinstance(v, torch.Tensor)) / 1e6
print(f"\nnc inferred     : {nc}  (expected 80)")
print(f"Total params    : {total_params:.3f}M  (expected ~9.46M for yolov11s)")

print("\nAll state_dict key prefixes (first 3 levels):")
prefixes = sorted({'.'.join(k.split('.')[:3]) for k in state})
for p in prefixes:
    print(f"  {p}")
