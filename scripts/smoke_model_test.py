import sys, torch
from models.yolov11 import YOLOv11
from loss.loss      import DetectionLoss

NC     = 80
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP    = DEVICE == 'cuda'

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(cond, msg):
    if not cond:
        errors.append(msg)
        print(f"  {FAIL}: {msg}")
    else:
        print(f"  {PASS}")

print(f"Device : {DEVICE}")
print(f"AMP    : {AMP}")

# ── T1: Model instantiation ──────────────────────────────────────────
print("T1 — Build yolov11n")
try:
    model = YOLOv11('configs/model/yolov11n.yaml', nc=NC).to(DEVICE)
except Exception as e:
    print(f"  {FAIL}: {e}"); sys.exit(1)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Parameters: {params:.3f}M")
check(1.0 < params < 20.0, f"Suspicious param count {params:.3f}M")

# ── T2: Train-mode forward pass ──────────────────────────────────────
print("T2 — Train-mode forward")
model.train()
dummy = torch.randn(2, 3, 640, 640).to(DEVICE)
try:
    with torch.cuda.amp.autocast(enabled=AMP):
        out_train = model(dummy)
    print(f"  Output type : {type(out_train).__name__}")
    if isinstance(out_train, (list, tuple)):
        for i, o in enumerate(out_train):
            if hasattr(o, 'shape'):
                print(f"  out[{i}] shape: {o.shape}")
    check(out_train is not None, "train output is None")
except Exception as e:
    print(f"  {FAIL}: {e}"); sys.exit(1)

# ── T3: Eval-mode forward pass shape ────────────────────────────────
print("T3 — Eval-mode forward [2, 4+nc, 8400]")
model.eval()
try:
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=AMP):
            out_eval = model(dummy)
    decoded = out_eval[0] if isinstance(out_eval, (list, tuple)) else out_eval
    print(f"  Decoded shape : {decoded.shape}")
    check(decoded.shape == torch.Size([2, 4 + NC, 8400]),
          f"Expected [2,{4+NC},8400] got {decoded.shape}")
except Exception as e:
    print(f"  {FAIL}: {e}")
    errors.append(str(e))

# ── T4: Loss — no NaN, no Inf, loss > 0 ─────────────────────────────
print("T4 — Loss (no NaN / Inf / zero)")
model.train()
try:
    with torch.cuda.amp.autocast(enabled=AMP):
        out_train = model(dummy)

    criterion = DetectionLoss(model, box=7.5, cls=0.5, dfl=1.5)
    fake_batch = {
        'img'      : dummy,
        'cls'      : torch.tensor([[0],[1],[2],[3],[4],[5]],
                                   dtype=torch.float32).to(DEVICE),
        'bboxes'   : torch.tensor([
                         [0.30, 0.40, 0.20, 0.30],
                         [0.60, 0.55, 0.15, 0.20],
                         [0.50, 0.25, 0.35, 0.30],
                         [0.20, 0.70, 0.18, 0.22],
                         [0.75, 0.35, 0.20, 0.18],
                         [0.45, 0.60, 0.30, 0.25],
                     ]).to(DEVICE),
        'batch_idx': torch.tensor([0,0,0,1,1,1]).to(DEVICE),
    }
    loss, items = criterion(out_train, fake_batch)
    print(f"  loss={loss.item():.4f}  "
          f"box={items[0]:.4f}  cls={items[1]:.4f}  dfl={items[2]:.4f}")
    check(not torch.isnan(loss),  "NaN loss")
    check(not torch.isinf(loss),  "Inf loss")
    check(loss.item() > 0,        "Loss is zero")
    check(items[0] > 0,           "box loss is zero")
    check(items[1] > 0,           "cls loss is zero")
    check(items[2] > 0,           "dfl loss is zero")
except Exception as e:
    print(f"  {FAIL}: {e}")
    errors.append(str(e))

# ── T5: Backward pass — gradients exist and are finite ──────────────
print("T5 — Backward pass")
try:
    loss.backward()
    grads = [p.grad.norm().item()
             for p in model.parameters() if p.grad is not None]
    print(f"  Params with grads : {len(grads)}")
    print(f"  Mean grad norm    : {sum(grads)/len(grads):.6f}")
    print(f"  Max  grad norm    : {max(grads):.4f}")
    check(len(grads) > 0,        "No gradients computed")
    check(max(grads) < 1e5,      f"Exploding gradients (max={max(grads):.1f})")
    check(all(g == g for g in grads), "NaN in gradients")
except Exception as e:
    print(f"  {FAIL}: {e}")
    errors.append(str(e))

# ── T6: Two optimizer steps — loss decreases ────────────────────────
print("T6 — Two optimizer steps (loss should decrease)")
try:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = []
    for step in range(2):
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=AMP):
            out = model(dummy)
        l, _ = criterion(out, fake_batch)
        l.backward()
        opt.step()
        losses.append(l.item())
        print(f"  step {step+1} loss: {l.item():.4f}")
    check(losses[1] <= losses[0] * 1.5,
          f"Loss did not decrease in 2 steps: {losses[0]:.4f} → {losses[1]:.4f}")
except Exception as e:
    print(f"  {FAIL}: {e}")
    errors.append(str(e))

# ── Summary ──────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[91m{len(errors)} FAILURES:\033[0m")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
else:
    print("\033[92mAll model tests PASSED.\033[0m")
