import sys, gc, torch
from models.yolov11 import YOLOv11
from loss.loss import DetectionLoss

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
results = {}

def check(name, cond, detail=""):
    results[name] = cond
    print(f"  {PASS if cond else FAIL} {name}" + (f": {detail}" if detail else ""))

def mem_mb():
    if DEVICE == 'cuda':
        return torch.cuda.memory_allocated() / 1e6
    return 0.0

print("="*60); print("MODULE 9 — MEMORY LEAK TESTS"); print("="*60)

model     = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE).train()
criterion = DetectionLoss(model, box=7.5, cls=0.5, dfl=1.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))
dummy     = torch.randn(2,3,640,640).to(DEVICE)

def make_batch():
    return {
        'img': dummy,
        'cls': torch.randint(0,80,(6,1)).float().to(DEVICE),
        'bboxes': torch.rand(6,4).clamp(0.05,0.95).to(DEVICE),
        'batch_idx': torch.tensor([0,0,0,1,1,1]).to(DEVICE),
    }

# Warm up (1 step)
for _ in range(1):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
        out = model(dummy)
        loss, _ = criterion(out, make_batch())
    scaler.scale(loss).backward()
    scaler.step(optimizer); scaler.update()

gc.collect()
if DEVICE == 'cuda': torch.cuda.empty_cache()

# Measure memory over 10 steps
print("\nT9.1 — Memory stable over 10 training steps")
mem_start = mem_mb()
mems = []
for i in range(10):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
        out = model(dummy)
        loss, items = criterion(out, make_batch())
    scaler.scale(loss).backward()
    scaler.step(optimizer); scaler.update()
    # Detach loss for logging (critical — keeps graph from accumulating)
    _ = loss.detach().item()
    _ = [it.detach().item() for it in items]
    mems.append(mem_mb())

mem_end = mems[-1]
mem_growth = mem_end - mem_start
print(f"  start={mem_start:.1f}MB  end={mem_end:.1f}MB  growth={mem_growth:.1f}MB")
check("memory_stable", mem_growth < 50.0,
      f"growth={mem_growth:.1f}MB (< 50MB threshold)")

# T9.2 — No live graph references after step
print("\nT9.2 — Loss items returned as scalars (no graph leaks)")
optimizer.zero_grad()
with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
    out = model(dummy)
    loss, items = criterion(out, make_batch())
check("items_no_grad", not any(hasattr(it,'grad_fn') and it.grad_fn is not None
                                for it in items), "")
check("items_are_detached", all(not it.requires_grad for it in items), "")

# T9.3 — EMA update does not accumulate graph
print("\nT9.3 — EMA update does not hold graph references")
from utils.general import ModelEMA
ema = ModelEMA(model)
# Take a training step, then update EMA
optimizer.zero_grad()
with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
    out = model(dummy)
    loss, _ = criterion(out, make_batch())
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()
ema.update(model)
# EMA parameters should not have grad_fn
ema_has_graph = any(p.grad_fn is not None for p in ema.ema.parameters())
check("ema_no_graph_refs", not ema_has_graph, "")

passed = sum(results.values()); total = len(results)
print(f"\nM9 Result: {passed}/{total} passed")
if sum(1 for v in results.values() if not v) > 0:
    print("FAILED:", [k for k,v in results.items() if not v]); sys.exit(1)
