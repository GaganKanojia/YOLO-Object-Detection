"""
Side-by-side comparison of every metric between Ultralytics and custom.
Tolerance: +-0.003 on any metric.
"""
import json

TOLERANCE = 0.003

ultra = json.load(open('results/ultralytics_metrics.json'))
custom = json.load(open('results/custom_metrics.json'))

u_nms = ultra['nms_params']
c_nms = custom['nms_params']

print("=" * 65)
print("NMS PARAMETER CHECK")
print("=" * 65)
for key in ['conf', 'iou', 'imgsz', 'max_det']:
    match = u_nms[key] == c_nms[key]
    status = "OK" if match else "MISMATCH"
    print(f"  {key:<10}: ultralytics={u_nms[key]}  custom={c_nms[key]}  {status}")
if any(u_nms[k] != c_nms[k] for k in u_nms):
    print("\n  FATAL: NMS parameters differ — comparison is invalid.")
    exit(1)

print("\n" + "=" * 65)
print("SUMMARY METRICS COMPARISON")
print("=" * 65)
print(f"{'Metric':<15} {'Ultralytics':>12} {'Custom':>12} {'Diff':>10} {'Status':>10}")
print("-" * 65)
u_s = ultra['summary']
c_s = custom['summary']
summary_pass = True
for metric in ['mAP50', 'mAP50_95', 'precision', 'recall']:
    u_val = u_s.get(metric, 0.0)
    c_val = c_s.get(metric, 0.0)
    diff = abs(c_val - u_val)
    ok = diff <= TOLERANCE
    if not ok:
        summary_pass = False
    print(f"  {metric:<13} {u_val:>12.4f} {c_val:>12.4f} {diff:>+10.4f} {('PASS' if ok else 'FAIL'):>10}")

u_pc = ultra.get('per_class', {})
c_pc = custom.get('per_class', {})
all_classes = sorted(set(list(u_pc.keys()) + list(c_pc.keys())))


def compare_block(title, field):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)
    print(f"{'Class':<25} {'Ultralytics':>12} {'Custom':>12} {'Diff':>10} {'Status':>8}")
    print("-" * 65)
    mism = []
    ok_all = True
    for cls in all_classes:
        u_v = u_pc.get(cls, {}).get(field, 0.0)
        c_v = c_pc.get(cls, {}).get(field, 0.0)
        if u_v is None:
            u_v = 0.0
        if c_v is None:
            c_v = 0.0
        diff = abs(c_v - u_v)
        ok = diff <= TOLERANCE
        if not ok:
            ok_all = False
            mism.append((cls, u_v, c_v, diff))
        print(f"  {cls:<23} {u_v:>12.4f} {c_v:>12.4f} {diff:>+10.4f} {('OK' if ok else 'FAIL'):>8}")
    return ok_all, mism


class_pass, class_mismatches = compare_block("PER-CLASS AP@50 COMPARISON", 'ap50')
prec_pass, prec_mismatches = compare_block("PER-CLASS PRECISION COMPARISON", 'precision')
rec_pass, rec_mismatches = compare_block("PER-CLASS RECALL COMPARISON", 'recall')

print("\n" + "=" * 65)
print("FINAL VERDICT")
print("=" * 65)
overall_pass = summary_pass and class_pass and prec_pass and rec_pass
for label, passed in [
    ("Summary metrics (mAP50, mAP50:95, P, R)", summary_pass),
    ("Per-class AP@50", class_pass),
    ("Per-class Precision", prec_pass),
    ("Per-class Recall", rec_pass),
]:
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")

if overall_pass:
    print(f"\n  IMPLEMENTATIONS MATCH  (all diffs <= +-{TOLERANCE})")
else:
    print(f"\n  MISMATCH DETECTED  (tolerance +-{TOLERANCE})")
    for name, mism in [("AP@50", class_mismatches), ("Precision", prec_mismatches), ("Recall", rec_mismatches)]:
        if mism:
            print(f"\n  {name} mismatches ({len(mism)} classes):")
            for cls, u, c, d in sorted(mism, key=lambda x: -x[3])[:20]:
                print(f"    {cls:<25} ultra={u:.4f}  custom={c:.4f}  diff={d:+.4f}")

report = {
    'tolerance': TOLERANCE,
    'overall_pass': overall_pass,
    'nms_params': u_nms,
    'summary': {
        m: {
            'ultralytics': u_s[m],
            'custom': c_s[m],
            'diff': round(abs(c_s[m] - u_s[m]), 6),
            'pass': abs(c_s[m] - u_s[m]) <= TOLERANCE,
        } for m in ['mAP50', 'mAP50_95', 'precision', 'recall']
    },
    'per_class_ap50_mismatches': class_mismatches,
    'per_class_precision_mismatches': prec_mismatches,
    'per_class_recall_mismatches': rec_mismatches,
}
json.dump(report, open('results/comparison_report.json', 'w'), indent=2)
print(f"\n  Full report saved: results/comparison_report.json")
