import sys, os, torch, json
from pathlib import Path

run_dir = sys.argv[1]  # e.g. runs/test_coco
fmt     = sys.argv[2]  # 'coco' or 'yolo'

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"
results = {}

def check(name, cond, detail=""):
    results[name] = cond
    print(f"  {PASS if cond else FAIL} {fmt}_{name}" + (f": {detail}" if detail else ""))

print(f"\n=== Verifying {fmt} training run: {run_dir} ===")

# Check checkpoints exist
best  = Path(run_dir)/'weights'/'best.pt'
last  = Path(run_dir)/'weights'/'last.pt'
check("best_pt_exists",  best.exists(),  str(best))
check("last_pt_exists",  last.exists(),  str(last))

# Verify best.pt is loadable and has all required keys
if best.exists():
    try:
        ckpt = torch.load(best, map_location='cpu', weights_only=False)
        check("ckpt_has_model",   'model'   in ckpt, str(list(ckpt.keys())))
        check("ckpt_has_ema",     'ema'     in ckpt, "")
        check("ckpt_has_epoch",   'epoch'   in ckpt, f"epoch={ckpt.get('epoch')}")
        check("ckpt_has_fitness", 'best_fitness' in ckpt,
                                  f"fitness={ckpt.get('best_fitness',0):.4f}")
        check("ckpt_has_updates", 'updates' in ckpt,
                                  f"updates={ckpt.get('updates',0)}")
    except Exception as e:
        check("ckpt_loadable", False, str(e))

# Parse training log for loss trend
log_file = Path(run_dir)/'train_log.txt'
if log_file.exists():
    lines = log_file.read_text().splitlines()
    epoch_losses = []
    for line in lines:
        if 'loss' in line.lower() and 'epoch' in line.lower():
            try:
                parts = line.split()
                loss_val = float([p for p in parts if p.replace('.','').isdigit()][-1])
                epoch_losses.append(loss_val)
            except: pass
    if len(epoch_losses) >= 2:
        check("loss_trend", epoch_losses[-1] < epoch_losses[0] * 1.5,
              f"epoch1={epoch_losses[0]:.4f} → last={epoch_losses[-1]:.4f}")
    else:
        print(f"  WARN: Could not parse loss values from log")

# Verify EMA state in checkpoint
if best.exists() and 'ema' in ckpt:
    ema_state = ckpt['ema']
    has_bn = any('running_mean' in k for k in ema_state.keys())
    check("ema_has_bn_buffers", has_bn, "")
    all_finite = all(not torch.isnan(v).any() and not torch.isinf(v).any()
                     for v in ema_state.values() if isinstance(v, torch.Tensor))
    check("ema_weights_finite", all_finite, "")

passed = sum(results.values()); total = len(results)
print(f"\n{fmt} training verification: {passed}/{total} passed")
if sum(1 for v in results.values() if not v) > 0: sys.exit(1)
