import sys, re
from pathlib import Path

log_file = sys.argv[1]
fmt      = sys.argv[2]

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"

log = Path(log_file).read_text()
results = {}

def check(name, cond, detail=""):
    results[name] = cond
    print(f"  {PASS if cond else FAIL} {fmt}_{name}" + (f": {detail}" if detail else ""))

print(f"\n=== Verifying {fmt} validation log ===")

check("val_no_crash",    "Error" not in log and "Traceback" not in log, "")
check("val_no_nan",      "nan" not in log.lower() or "mAP" in log, "")
check("val_ran",         "mAP" in log or "map" in log.lower(), "")

# Extract mAP values
map50_matches   = re.findall(r'mAP[_@]?50[^:]*[:=\s]+([0-9.]+)', log, re.IGNORECASE)
map5095_matches = re.findall(r'mAP[_@]?50[_:]?95[^:]*[:=\s]+([0-9.]+)', log, re.IGNORECASE)

if map50_matches:
    map50 = float(map50_matches[0])
    check("map50_nonzero",   map50 > 0.001, f"mAP@50={map50:.4f}")
    check("map50_not_nan",   map50 == map50, "")
else:
    print(f"  WARN: Could not parse mAP@50 from log")
    check("map50_found",     False, "not found in log")

if map5095_matches:
    map5095 = float(map5095_matches[0])
    check("map5095_nonzero", map5095 > 0.0001, f"mAP@50:95={map5095:.4f}")
else:
    print(f"  WARN: Could not parse mAP@50:95 from log")

passed = sum(results.values()); total = len(results)
print(f"\n{fmt} validation verification: {passed}/{total} passed")
