"""Compare finetuning metrics: PASS <= 0.02, WARN <= 0.05, FAIL > 0.05."""
import json

TOL_OK, TOL_WARN = 0.02, 0.05
custom = json.load(open('results/phase2_custom_metrics.json'))['metrics']
ultra = json.load(open('results/phase2_ultra_metrics.json'))['metrics']


def verdict(diff):
    return 'PASS' if diff <= TOL_OK else ('WARN' if diff <= TOL_WARN else 'FAIL')


print("=" * 68)
print("PHASE 2 — FINETUNING METRICS COMPARISON")
print("=" * 68)
print(f"Tolerance: PASS<={TOL_OK} | WARN<={TOL_WARN} | FAIL>{TOL_WARN}\n")
print(f"{'Metric':<14} {'Custom':>10} {'Ultralytics':>12} {'Diff':>10} {'Status':>8}")
print("-" * 58)
summary_flags = []
for m in ['mAP50', 'mAP50_95', 'precision', 'recall']:
    c_v, u_v = custom.get(m, 0.0), ultra.get(m, 0.0)
    diff = abs(c_v - u_v)
    summary_flags.append(verdict(diff))
    print(f"  {m:<12} {c_v:>10.4f} {u_v:>12.4f} {diff:>+10.4f}  {verdict(diff)}")

c_pc, u_pc = custom.get('per_class', {}), ultra.get('per_class', {})
class_flags = []
for title, field, collect in [("AP@50", 'ap50', True), ("Precision", 'precision', False),
                              ("Recall", 'recall', False)]:
    print(f"\n{'Class ' + title:<20} {'Custom':>10} {'Ultralytics':>12} {'Diff':>10} {'Status':>8}")
    print("-" * 64)
    for cls in ['person', 'car', 'dog']:
        c = (c_pc.get(cls) or {}).get(field, 0.0) or 0.0
        u = (u_pc.get(cls) or {}).get(field, 0.0) or 0.0
        diff = abs(c - u)
        if collect:
            class_flags.append(verdict(diff))
        print(f"  {cls:<18} {c:>10.4f} {u:>12.4f} {diff:>+10.4f}  {verdict(diff)}")

all_flags = summary_flags + class_flags
has_fail = 'FAIL' in all_flags
has_warn = 'WARN' in all_flags
print("\n" + "=" * 68)
print("OVERALL VERDICT")
print("=" * 68)
if not has_fail and not has_warn:
    print("  PASS — All metrics within +-0.02. Custom and Ultralytics finetuning are equivalent.")
elif not has_fail:
    print("  WARN — Some metrics in (0.02, 0.05]. Likely training stochasticity (tiny val set).")
else:
    print("  FAIL — Some metrics differ by > 0.05. Systematic difference; revisit LR/EMA/aug timing.")

verd = 'PASS' if not has_fail and not has_warn else ('WARN' if not has_fail else 'FAIL')
report = {
    'phase1_passed': json.load(open('results/phase1_loss_agreement.json'))['pass'],
    'phase2_verdict': verd,
    'tolerance_pass': TOL_OK, 'tolerance_warn': TOL_WARN,
    'summary': {m: {'custom': custom[m], 'ultra': ultra[m], 'diff': abs(custom[m] - ultra[m])}
                for m in ['mAP50', 'mAP50_95', 'precision', 'recall']},
}
json.dump(report, open('results/phase2_comparison.json', 'w'), indent=2)
print("\nResults saved: results/phase2_comparison.json")
