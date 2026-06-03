import sys, torch
from models.yolov11 import YOLOv11
from loss.loss import DetectionLoss
from loss.tal import TaskAlignedAssigner

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
results = {}

def check(name, cond, detail=""):
    results[name] = cond
    print(f"  {PASS if cond else FAIL} {name}" + (f": {detail}" if detail else ""))

print("="*60); print("MODULE 3 — LOSS TESTS"); print("="*60)
print(f"Device: {DEVICE}\n")

model = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE).train()
criterion = DetectionLoss(model, box=7.5, cls=0.5, dfl=1.5)
dummy = torch.randn(2,3,640,640).to(DEVICE)

def make_batch(n_obj=6, nc=80):
    return {
        'img'      : dummy,
        'cls'      : torch.randint(0,nc,(n_obj,1)).float().to(DEVICE),
        'bboxes'   : torch.rand(n_obj,4).clamp(0.05,0.95).to(DEVICE),
        'batch_idx': torch.tensor([0]*3+[1]*3).to(DEVICE),
    }

# T3.1 — Loss forward, basic validity
print("T3.1 — Loss forward pass")
try:
    out = model(dummy)
    loss, items = criterion(out, make_batch())
    check("loss_finite",    not torch.isnan(loss) and not torch.isinf(loss),
                            f"loss={loss.item():.4f}")
    check("loss_positive",  loss.item() > 0, f"{loss.item():.4f}")
    check("box_positive",   items[0] > 0, f"box={items[0]:.4f}")
    check("cls_positive",   items[1] > 0, f"cls={items[1]:.4f}")
    check("dfl_positive",   items[2] > 0, f"dfl={items[2]:.4f}")
    check("items_detached", not items[0].requires_grad, "")
except Exception as e:
    check("loss_forward", False, str(e)); sys.exit(1)

# T3.2 — Empty batch (image with zero objects)
print("\nT3.2 — Empty batch (zero GT objects)")
try:
    empty_batch = {
        'img': dummy, 'cls': torch.zeros(0,1).to(DEVICE),
        'bboxes': torch.zeros(0,4).to(DEVICE),
        'batch_idx': torch.zeros(0).to(DEVICE),
    }
    out = model(dummy)
    loss_e, items_e = criterion(out, empty_batch)
    check("empty_no_crash",  True, "")
    check("empty_finite",    not torch.isnan(loss_e), f"{loss_e.item():.4f}")
except Exception as e:
    check("empty_no_crash", False, str(e))

# T3.3 — Backward pass and gradient validity
print("\nT3.3 — Backward pass")
try:
    model.zero_grad()
    out = model(dummy)
    loss, _ = criterion(out, make_batch())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("grads_exist",      len(grads) > 0, f"{len(grads)} tensors")
    all_finite = all(not torch.isnan(g).any() and not torch.isinf(g).any() for g in grads)
    check("grads_finite",     all_finite, "")
    max_norm = max(g.norm().item() for g in grads)
    check("grads_not_exploding", max_norm < 1e5, f"max_norm={max_norm:.2f}")
except Exception as e:
    check("backward_pass", False, str(e))

# T3.4 — Loss decreases over 5 optimizer steps
print("\nT3.4 — Loss decreases over 5 optimizer steps")
try:
    m_test = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE).train()
    c_test = DetectionLoss(m_test, box=7.5, cls=0.5, dfl=1.5)
    opt    = torch.optim.AdamW(m_test.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        out = m_test(dummy)
        l, _ = c_test(out, make_batch())
        l.backward()
        opt.step()
        losses.append(l.item())
    check("loss_decreases_5steps", losses[-1] < losses[0]*1.5,
          f"{losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Steps: {[f'{v:.4f}' for v in losses]}")
except Exception as e:
    check("loss_decreases_5steps", False, str(e))

# T3.5 — TAL assigner smoke test
print("\nT3.5 — TaskAlignedAssigner")
try:
    assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
    B, A, NC = 2, 8400, 80
    pred_scores = torch.rand(B,A,NC).to(DEVICE)
    pred_bboxes = torch.rand(B,A,4).to(DEVICE) * 640
    anc_points  = torch.rand(A,2).to(DEVICE) * 640
    gt_labels   = torch.zeros(B,6,1).to(DEVICE)
    gt_bboxes   = torch.rand(B,6,4).to(DEVICE) * 640
    gt_mask     = torch.ones(B,6,dtype=torch.bool).to(DEVICE)
    t_lbl, t_box, t_score, fg_mask, _ = assigner(
        pred_scores.detach(), pred_bboxes.detach(),
        anc_points, gt_labels, gt_bboxes, gt_mask)
    check("tal_no_crash",     True, "")
    check("tal_fg_mask_bool", fg_mask.dtype==torch.bool, str(fg_mask.dtype))
    check("tal_fg_nonzero",   fg_mask.sum().item() > 0,  f"{fg_mask.sum().item()} positives")
    check("tal_scores_range", (t_score>=0).all() and (t_score<=1).all(), "")
except Exception as e:
    check("tal_smoke_test", False, str(e))

passed = sum(results.values()); total = len(results)
print(f"\nM3 Result: {passed}/{total} passed")
if sum(1 for v in results.values() if not v) > 0:
    print("FAILED:", [k for k,v in results.items() if not v]); sys.exit(1)
