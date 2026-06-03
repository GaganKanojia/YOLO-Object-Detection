import sys
from pathlib import Path

log_file = sys.argv[1]
fmt      = sys.argv[2]

PASS="\033[92mPASS\033[0m"; FAIL="\033[91mFAIL\033[0m"
log = Path(log_file).read_text()

def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {fmt}_{name}" + (f": {detail}" if detail else ""))
    return cond

print(f"\n=== Verifying {fmt} inference ===")
check("detect_no_crash",  "Error" not in log and "Traceback" not in log, "")
check("detect_ran",       "detect" in log.lower() or "box" in log.lower() or "saved" in log.lower(), "")
