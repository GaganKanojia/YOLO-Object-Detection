"""
Characterize Phase-2 run-to-run variance. The custom trainer is unseeded (plan D10) and
the val set is tiny (15 imgs; dog has 1 instance), so end-to-end metrics are expected to
swing. Run the custom finetune N times, validate each, and report the spread.
"""
import sys, os, glob, json, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
from models.yolov11 import YOLOv11
from data.dataset import YOLODataset
from data.loaders import build_dataloader
from engine.validator import Validator

N = int(os.environ.get('N_RUNS', '3'))
NC, NAMES = 3, ['person', 'car', 'dog']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
runs = []

base_cfg = yaml.safe_load(open('configs/training/finetune_subset3.yaml'))
for r in range(N):
    sd = f'runs/var_custom_{r}'
    shutil.rmtree(sd, ignore_errors=True)
    cfg = dict(base_cfg, save_dir=sd)   # unique save_dir per run (config key wins in Trainer)
    cfg_path = f'/tmp/var_cfg_{r}.yaml'
    yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False)
    cmd = ['python3', 'train.py', '--config', cfg_path,
           '--weights', 'weights/yolov11s_official.pt', '--strategy', 'partial']
    print(f"\n=== custom run {r+1}/{N} -> {sd} ===", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    best = sorted(glob.glob(f'{sd}*/weights/best.pt'))[-1]
    m = YOLOv11('configs/model/yolov11s.yaml', nc=NC).to(DEVICE)
    ck = torch.load(best, map_location='cpu', weights_only=False)
    m.load_state_dict(ck.get('ema') or ck['model']); m.eval()
    vds = YOLODataset('dataset/subset3/images/val', 'dataset/subset3/annotations_val.json',
                      nc=NC, imgsz=640, augment=False)
    vl, _ = build_dataloader(vds, batch_size=4, workers=2, augment=False, shuffle=False)
    res = Validator(m, vl, DEVICE, conf=0.5, iou=0.5, max_det=300,
                    names={i: n for i, n in enumerate(NAMES)}).run()
    pc = res.get('per_class', {})
    rec = {'run': r, 'mAP50': round(res['mAP50'], 4), 'mAP50_95': round(res['mAP50_95'], 4),
           'person_ap50': round(pc.get('person', {}).get('ap50', 0.0), 4),
           'car_ap50': round(pc.get('car', {}).get('ap50', 0.0), 4),
           'dog_ap50': round(pc.get('dog', {}).get('ap50', 0.0), 4)}
    runs.append(rec)
    print(rec, flush=True)

import statistics as st
def rng(key):
    vs = [r[key] for r in runs]
    return min(vs), max(vs), round(st.mean(vs), 4)

print("\n=== CUSTOM run-to-run variance (min / max / mean) ===")
for k in ['mAP50', 'mAP50_95', 'person_ap50', 'car_ap50', 'dog_ap50']:
    lo, hi, mu = rng(k)
    print(f"  {k:<12}: min={lo}  max={hi}  mean={mu}  spread={round(hi-lo,4)}")
json.dump({'n_runs': N, 'runs': runs,
           'ranges': {k: rng(k) for k in ['mAP50', 'mAP50_95', 'person_ap50', 'car_ap50', 'dog_ap50']}},
          open('results/phase2_variance.json', 'w'), indent=2)
print("\nSaved: results/phase2_variance.json")
