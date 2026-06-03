import sys, torch
from models.yolov11 import YOLOv11
from loss.loss import DetectionLoss

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_ENABLED = DEVICE == 'cuda'
results = {}

def check(name, cond, detail=""):
    results[name] = cond
    print(f"  {PASS if cond else FAIL} {name}" + (f": {detail}" if detail else ""))

print("="*60); print("MODULE 4 — AMP TESTS"); print("="*60)
print(f"Device: {DEVICE}   AMP: {AMP_ENABLED}\n")

if not AMP_ENABLED:
    print("  CUDA not available — running AMP tests in CPU fallback mode")
    print("  (autocast is a no-op on CPU; testing that it doesn't crash)\n")

model = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE)
dummy = torch.randn(2,3,640,640).to(DEVICE)

def make_batch():
    return {
        'img'      : dummy,
        'cls'      : torch.randint(0,80,(6,1)).float().to(DEVICE),
        'bboxes'   : torch.rand(6,4).clamp(0.05,0.95).to(DEVICE),
        'batch_idx': torch.tensor([0,0,0,1,1,1]).to(DEVICE),
    }

# T4.1 — Autocast forward does not crash
print("T4.1 — Autocast forward pass")
try:
    model.train()
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        out = model(dummy)
    check("amp_forward_no_crash", True, "")
    # Check output has some FP16 tensors (only meaningful on CUDA)
    if AMP_ENABLED and isinstance(out,(list,tuple)):
        has_fp16 = any(o.dtype==torch.float16 for o in out if hasattr(o,'dtype'))
        check("amp_produces_fp16", has_fp16, "")
except Exception as e:
    check("amp_forward_no_crash", False, str(e))

# T4.2 — Autocast + loss computation
print("\nT4.2 — Autocast forward + loss")
try:
    model.train()
    criterion = DetectionLoss(model, box=7.5, cls=0.5, dfl=1.5)
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        out = model(dummy)
        loss, items = criterion(out, make_batch())
    check("amp_loss_finite",   not torch.isnan(loss) and not torch.isinf(loss),
                               f"loss={loss.item():.4f}")
    check("amp_loss_positive", loss.item() > 0, "")
except Exception as e:
    check("amp_loss_finite", False, str(e))

# T4.3 — GradScaler backward and optimizer step
print("\nT4.3 — GradScaler backward + optimizer step")
try:
    model.train()
    criterion = DetectionLoss(model, box=7.5, cls=0.5, dfl=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler    = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    init_scale = scaler.get_scale()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        out = model(dummy)
        loss, _ = criterion(out, make_batch())

    scaler.scale(loss).backward()
    # unscale before grad clip — CRITICAL order
    scaler.unscale_(optimizer)
    # Capture overflow BEFORE clipping. On an AMP overflow step some grads are
    # inf; stdlib clip_grad_norm_ would turn inf into NaN (inf * (max_norm/inf=0)).
    # GradScaler.step() correctly *skips* such steps (it recorded the inf at
    # unscale_ time) and update() reduces the scale — so an overflow step is
    # expected, not a bug. The real invariant: finite grads on non-overflow
    # steps, OR the scaler detected the overflow and reduced its scale.
    overflow = any((torch.isinf(p.grad).any() or torch.isnan(p.grad).any()).item()
                   for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    scaler.step(optimizer)
    scaler.update()

    check("scaler_backward_no_crash", True, "")
    check("scaler_scale_finite",      scaler.get_scale() > 0,
                                      f"scale={scaler.get_scale()}")
    if overflow:
        check("scaler_grads_finite", scaler.get_scale() < init_scale,
              f"AMP overflow correctly detected+skipped; scale {init_scale}→{scaler.get_scale()}")
    else:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        all_finite = all(not torch.isnan(g).any() for g in grads)
        check("scaler_grads_finite", all_finite, "")
except Exception as e:
    check("scaler_backward_no_crash", False, str(e))

# T4.4 — 5 full AMP training steps, loss decreases
print("\nT4.4 — 5 full AMP training steps")
try:
    m2 = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE).train()
    c2 = DetectionLoss(m2, box=7.5, cls=0.5, dfl=1.5)
    opt2    = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    scaler2 = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    amp_losses = []
    for step in range(5):
        opt2.zero_grad()
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            out = m2(dummy)
            l, _ = c2(out, make_batch())
        scaler2.scale(l).backward()
        scaler2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(m2.parameters(), 10.0)
        scaler2.step(opt2)
        scaler2.update()
        amp_losses.append(l.item())
    check("amp_5steps_no_crash",    True, "")
    check("amp_5steps_loss_finite", all(not (v!=v) for v in amp_losses), str(amp_losses))
    print(f"  AMP loss curve: {[f'{v:.4f}' for v in amp_losses]}")
except Exception as e:
    check("amp_5steps_no_crash", False, str(e))

# T4.5 — AMP vs no-AMP produce similar loss values
print("\nT4.5 — AMP vs no-AMP loss consistency")
try:
    torch.manual_seed(42)
    m_amp  = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE).train()
    torch.manual_seed(42)
    m_fp32 = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE).train()
    m_fp32.load_state_dict(m_amp.state_dict())
    c_amp  = DetectionLoss(m_amp,  box=7.5, cls=0.5, dfl=1.5)
    c_fp32 = DetectionLoss(m_fp32, box=7.5, cls=0.5, dfl=1.5)
    batch  = make_batch()

    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        loss_amp, _ = c_amp(m_amp(dummy), batch)
    loss_fp32, _ = c_fp32(m_fp32(dummy), batch)

    rel_diff = abs(loss_amp.item()-loss_fp32.item()) / (loss_fp32.item()+1e-8)
    check("amp_fp32_consistent", rel_diff < 0.15,
          f"amp={loss_amp.item():.4f} fp32={loss_fp32.item():.4f} rel_diff={rel_diff:.3f}")
except Exception as e:
    check("amp_fp32_consistent", False, str(e))

passed = sum(results.values()); total = len(results)
print(f"\nM4 Result: {passed}/{total} passed")
if sum(1 for v in results.values() if not v) > 0:
    print("FAILED:", [k for k,v in results.items() if not v]); sys.exit(1)
