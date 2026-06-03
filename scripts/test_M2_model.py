import sys, torch
from models.yolov11 import YOLOv11

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
results = {}

def check(name, cond, detail=""):
    results[name] = cond
    print(f"  {PASS if cond else FAIL} {name}" + (f": {detail}" if detail else ""))

print("="*60); print("MODULE 2 — MODEL TESTS"); print("="*60)
print(f"Device: {DEVICE}\n")

# T2.1 — All 5 scales build and have correct param counts
EXPECTED = {'n':2.62, 's':9.46, 'm':20.11, 'l':25.37, 'x':56.97}
print("T2.1 — All 5 model scales")
for scale, expected in EXPECTED.items():
    try:
        m = YOLOv11(f'configs/model/yolov11{scale}.yaml', nc=80)
        params = sum(p.numel() for p in m.parameters())/1e6
        ok = abs(params - expected) < 0.1
        check(f"yolov11{scale}_params", ok, f"{params:.3f}M (expected ~{expected}M)")
        del m
    except Exception as e:
        check(f"yolov11{scale}_params", False, str(e))

# T2.2 — Train-mode forward pass (yolov11n)
print("\nT2.2 — Train-mode forward pass")
model = YOLOv11('configs/model/yolov11n.yaml', nc=80).to(DEVICE)
model.train()
dummy = torch.randn(2,3,640,640).to(DEVICE)
try:
    out = model(dummy)
    check("train_forward_runs",   out is not None, "")
    check("train_output_is_list", isinstance(out,(list,tuple)), type(out).__name__)
except Exception as e:
    check("train_forward_runs", False, str(e)); sys.exit(1)

# T2.3 — Eval-mode forward pass → [B, 4+nc, 8400]
print("\nT2.3 — Eval-mode forward pass")
model.eval()
try:
    with torch.no_grad():
        out_eval = model(dummy)
    decoded = out_eval[0] if isinstance(out_eval,(list,tuple)) else out_eval
    check("eval_shape",   decoded.shape==torch.Size([2,84,8400]), str(decoded.shape))
    check("eval_no_nan",  not torch.isnan(decoded).any(), "")
    check("eval_no_inf",  not torch.isinf(decoded).any(), "")
    # Class scores should be probabilities after sigmoid
    cls_scores = decoded[:,4:,:]
    check("eval_cls_range", (cls_scores>=0).all() and (cls_scores<=1).all(),
          f"[{cls_scores.min():.3f},{cls_scores.max():.3f}]")
except Exception as e:
    check("eval_shape", False, str(e)); sys.exit(1)

# T2.4 — Different batch sizes
print("\nT2.4 — Batch size robustness")
model.eval()
for bs in [1, 2, 8]:
    try:
        x = torch.randn(bs,3,640,640).to(DEVICE)
        with torch.no_grad():
            o = model(x)
        d = o[0] if isinstance(o,(list,tuple)) else o
        check(f"batch_{bs}", d.shape==torch.Size([bs,84,8400]), str(d.shape))
    except Exception as e:
        check(f"batch_{bs}", False, str(e))

# T2.5 — Non-640 input size
print("\nT2.5 — Non-standard input sizes")
model.eval()
for sz in [320, 416, 512]:
    try:
        x = torch.randn(1,3,sz,sz).to(DEVICE)
        anchors = (sz//8)*(sz//8) + (sz//16)*(sz//16) + (sz//32)*(sz//32)
        with torch.no_grad():
            o = model(x)
        d = o[0] if isinstance(o,(list,tuple)) else o
        check(f"imgsz_{sz}", d.shape==torch.Size([1,84,anchors]), str(d.shape))
    except Exception as e:
        check(f"imgsz_{sz}", False, str(e))

# T2.6 — CPU/GPU consistency (only if CUDA available)
if torch.cuda.is_available():
    print("\nT2.6 — CPU vs GPU output consistency")
    model_cpu = YOLOv11('configs/model/yolov11n.yaml', nc=80).eval()
    model_gpu = YOLOv11('configs/model/yolov11n.yaml', nc=80).to('cuda').eval()
    # Copy weights
    model_gpu.load_state_dict(model_cpu.state_dict())
    x_cpu = torch.randn(1,3,320,320)
    with torch.no_grad():
        o_cpu = model_cpu(x_cpu)
        o_gpu = model_gpu(x_cpu.cuda())
    d_cpu = (o_cpu[0] if isinstance(o_cpu,(list,tuple)) else o_cpu).float()
    d_gpu = (o_gpu[0] if isinstance(o_gpu,(list,tuple)) else o_gpu).cpu().float()
    max_diff = (d_cpu - d_gpu).abs().max().item()
    check("cpu_gpu_consistency", max_diff < 1e-3, f"max_diff={max_diff:.6f}")

passed = sum(results.values()); total = len(results)
print(f"\nM2 Result: {passed}/{total} passed")
failed = [k for k,v in results.items() if not v]
if failed: print("FAILED:", failed); sys.exit(1)
