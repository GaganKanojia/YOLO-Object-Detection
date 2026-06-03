"""
data/dataset.py — YOLOv11 dataset classes

Supported dataset formats
─────────────────────────

1. COCO JSON format (original)
   Use when: your dataset has a single JSON annotation file.
   Config keys required:
     train_img_dir  : path to training images directory
     train_ann_file : path to COCO JSON annotation file
     val_img_dir    : path to validation images directory
     val_ann_file   : path to validation COCO JSON file
     nc             : number of classes

2. YOLO TXT format (new)
   Use when: your dataset has per-image .txt label files and a data.yaml.
   Config keys required:
     train_yaml : path to data.yaml (covers both train and val splits)
   Optional:
     val_yaml   : separate data.yaml for val (defaults to train_yaml)
   Note: nc is read from data.yaml automatically.

Dataset directory structure (YOLO format):
   dataset/
   ├── data.yaml
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/         # one .txt per image, same stem
       └── val/

Label file format (one line per object):
   class_id  cx  cy  w  h
   (all normalized to [0,1], class_id is 0-indexed integer)

Design decision (STEP 2)
────────────────────────
We use **Option 2 — a separate ``YOLOTxtDataset`` class** rather than overloading
``YOLODataset``. The choice is driven by what the rest of the pipeline expects:

  * ``data.yaml`` is the natural entry point for a YOLO dataset — it *contains*
    the train/val image directories. Forcing the YOLO path through
    ``YOLODataset(img_dir, ann_file)`` (as the old ``_load_yolo`` did) duplicates
    information and can't express "give me the val split of this yaml". A class
    keyed on ``(yaml_file, split)`` models the format faithfully.
  * Keeping COCO loading in ``YOLODataset`` untouched guarantees zero regression
    on the existing JSON path.
  * ``YOLOTxtDataset`` **subclasses** ``YOLODataset`` and reuses ``__getitem__``,
    ``load_image_and_labels`` and the augmentation machinery verbatim, so its
    output format is byte-for-byte identical to ``YOLODataset`` — collate_fn,
    loss and metrics see no difference. Only sample discovery differs.

``build_dataset(args, split, augment)`` is the factory that picks the right class
based on which config keys are present.
"""
import os
import cv2
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from data.augmentations import (
    letterbox, augment_hsv, random_perspective,
    Mosaic, MixUp, CutMix, CopyPaste, RandomBGR,
)
from utils.general import setup_logging


logger = setup_logging("dataset")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


# ─────────────────────────────────────────────────────────────────────────────
# YOLO format helpers (data.yaml parsing, label/image pairing, label parsing)
# ─────────────────────────────────────────────────────────────────────────────

def parse_yolo_yaml(yaml_path: str) -> dict:
    """
    Parse a YOLO data.yaml file and return a normalized config dict.

    Handles all common yaml variants:
      - path key present or absent
      - train/val as relative path strings or as lists of paths
      - names as list ['cat','dog'] or as dict {0:'cat', 1:'dog'}
      - nc present or inferred from len(names)

    Returns:
        {
          'train_img_dir' : str,       # absolute path to train images dir
          'val_img_dir'   : str,       # absolute path to val images dir
          'nc'            : int,
          'names'         : list[str], # class names, 0-indexed order
        }
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"YOLO data.yaml not found: {yaml_path}\n"
            f"  Expected a dataset descriptor like:\n"
            f"    path: ./dataset\n    train: images/train\n    val: images/val\n"
            f"    names: ['cat', 'dog', ...]"
        )
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    yaml_dir = yaml_path.parent.resolve()
    cwd = Path.cwd()

    # `path` may be absent, relative to the yaml dir, or (as in many real-world
    # yamls) relative to the project CWD. Resolve each split robustly by trying
    # several candidate roots and picking the first that actually exists.
    root = data.get("path")

    def _resolve_split(key):
        val = data.get(key)
        if val is None:
            return ""
        if isinstance(val, (list, tuple)):
            if len(val) > 1:
                logger.warning(
                    f"{yaml_path}: '{key}' has {len(val)} paths; using the first ({val[0]})"
                )
            val = val[0] if val else ""
        if not val:
            return ""
        rel = Path(val)
        if rel.is_absolute():
            return str(rel.resolve())
        candidates = []
        if root:
            root_p = Path(root)
            candidates.append((yaml_dir / root_p / rel))   # path rel to yaml dir
            candidates.append((cwd / root_p / rel))         # path rel to CWD
        candidates.append(yaml_dir / rel)                   # ignore path, rel to yaml dir
        candidates.append(cwd / rel)                        # ignore path, rel to CWD
        for c in candidates:
            if c.exists():
                return str(c.resolve())
        # Nothing exists yet — return the most likely candidate with a warning.
        best = candidates[0]
        logger.warning(
            f"{yaml_path}: could not locate '{key}' images dir; falling back to {best}"
        )
        return str(best.resolve())

    train_img_dir = _resolve_split("train")
    val_img_dir = _resolve_split("val")

    # names: list or dict → 0-indexed list of str
    names = data.get("names")
    if isinstance(names, dict):
        names = [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        names = [str(n) for n in names]
    elif names is None:
        names = []
    else:
        raise ValueError(
            f"{yaml_path}: 'names' must be a list or dict; got {type(names).__name__}"
        )

    nc = data.get("nc")
    if nc is None:
        if not names:
            raise ValueError(f"{yaml_path}: neither 'nc' nor 'names' is defined")
        nc = len(names)
    else:
        nc = int(nc)
        if names and nc != len(names):
            raise ValueError(
                f"{yaml_path}: nc ({nc}) disagrees with len(names) ({len(names)}). "
                f"Fix the yaml so they match."
            )

    return {
        "train_img_dir": train_img_dir,
        "val_img_dir": val_img_dir,
        "nc": nc,
        "names": names,
    }


def find_label_path(img_path: str):
    """
    Given an image path, return the corresponding .txt label path, or None if it
    does not exist (image has no objects).

    Standard YOLO convention:
      images/train/img001.jpg  →  labels/train/img001.txt
    """
    img_path = Path(img_path)
    posix = img_path.as_posix()

    if "/images/" in posix:
        # Replace the first occurrence of /images/ with /labels/.
        label_path = Path(posix.replace("/images/", "/labels/", 1)).with_suffix(".txt")
    else:
        # Fallback: a sibling 'labels' folder next to the image's parent folder.
        # e.g. .../images → .../labels  (or just put labels next to the parent dir)
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")

    if label_path.exists():
        return str(label_path)
    logger.warning(
        f"No label file for {img_path.name} (looked for {label_path}); "
        f"treating image as having no objects."
    )
    return None


def parse_yolo_label(txt_path: str, nc: int) -> np.ndarray:
    """
    Parse a YOLO .txt label file.

    Returns:
        np.ndarray of shape [N, 5]: (class_id, cx, cy, w, h)  (normalized)
        Returns np.zeros((0, 5)) for empty files or missing files.

    Validates:
        - Each line has exactly 5 values        → else skip with warning
        - class_id is integer in [0, nc-1]       → else skip with warning
        - cx, cy, w, h are float in [0, 1]       → clamp + warn (do not skip)
        - w > 0 and h > 0 (non-degenerate box)   → else skip
    """
    empty = np.zeros((0, 5), dtype=np.float32)
    if txt_path is None or not Path(txt_path).exists():
        return empty

    rows = []
    with open(txt_path) as f:
        lines = f.read().strip().splitlines()

    for ln, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            logger.warning(f"{txt_path}:{ln}: expected 5 values, got {len(parts)} — skipping")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            logger.warning(f"{txt_path}:{ln}: non-numeric value — skipping")
            continue
        if cls < 0 or cls >= nc:
            logger.warning(f"{txt_path}:{ln}: class_id {cls} out of range [0,{nc}) — skipping")
            continue
        if w <= 0 or h <= 0:
            logger.warning(f"{txt_path}:{ln}: degenerate box (w={w}, h={h}) — skipping")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            logger.warning(
                f"{txt_path}:{ln}: coords outside [0,1] (cx={cx},cy={cy},w={w},h={h}) — clamping"
            )
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
        rows.append([float(cls), cx, cy, w, h])

    if not rows:
        return empty
    return np.array(rows, dtype=np.float32)


class YOLODataset(Dataset):
    """
    Detection dataset supporting two annotation formats, auto-detected from
    the `ann_file` extension:

    - `.json`        — COCO format (images + annotations + categories).
    - `.yaml`/`.yml` — YOLO dataset.yaml describing `names`; .txt label files
                      are read from a sibling `labels/` directory (the canonical
                      Ultralytics layout: `.../images/X.jpg` ↔ `.../labels/X.txt`).
                      Each .txt line is `class cx cy w h` normalized to [0,1].

    Labels are stored internally as `[cls, x1, y1, x2, y2]` in pixel coords
    regardless of source format, so the rest of the pipeline doesn't care.
    """
    def __init__(
        self,
        img_dir,
        ann_file,
        imgsz=640,
        augment=True,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        bgr=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        close_mosaic=0,
        epoch=0,
        nc=None,
    ):
        self.img_dir = Path(img_dir)
        self._set_hyp(
            imgsz, augment, mosaic, mixup, hsv_h, hsv_s, hsv_v,
            degrees, translate, scale, shear, perspective,
            flipud, fliplr, close_mosaic, epoch,
        )

        ext = Path(str(ann_file)).suffix.lower()
        if ext in (".yaml", ".yml"):
            self._load_yolo(ann_file)
        else:
            self._load_coco(ann_file)

        if nc is not None and nc != self.nc:
            raise ValueError(
                f"nc mismatch: caller passed nc={nc} but annotations contain {self.nc} categories"
            )

        self._build_aug(imgsz, mixup, cutmix, copy_paste, copy_paste_mode, bgr)

    # ── Shared augmentation setup (used by YOLODataset and YOLOTxtDataset) ──
    def _set_hyp(self, imgsz, augment, mosaic, mixup, hsv_h, hsv_s, hsv_v,
                 degrees, translate, scale, shear, perspective,
                 flipud, fliplr, close_mosaic, epoch):
        """Set scalar hyperparameters used by __getitem__. No samples needed yet."""
        self.imgsz = imgsz
        self.augment = augment
        self.mosaic_prob = mosaic if epoch < (999999 - close_mosaic) else 0.0
        self.mixup_prob = mixup
        self.hsv_h, self.hsv_s, self.hsv_v = hsv_h, hsv_s, hsv_v
        self.degrees, self.translate, self.scale = degrees, translate, scale
        self.shear, self.perspective = shear, perspective
        self.flipud, self.fliplr = flipud, fliplr
        self.close_mosaic = close_mosaic

    def _build_aug(self, imgsz, mixup, cutmix, copy_paste, copy_paste_mode, bgr):
        """Build the augmentation objects. Must run after self.samples exists."""
        self.mosaic_aug = Mosaic(self, imgsz=imgsz, p=self.mosaic_prob)
        self.mixup_aug = MixUp(self, p=self.mixup_prob)
        self.cutmix_aug = CutMix(self, p=cutmix)
        self.copy_paste_aug = CopyPaste(self, p=copy_paste, mode=copy_paste_mode)
        self.bgr_aug = RandomBGR(p=bgr)

    # ── COCO JSON loader ────────────────────────────────────────────────
    def _load_coco(self, ann_file):
        with open(ann_file) as f:
            coco = json.load(f)

        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(coco["categories"])}
        self.nc = len(coco["categories"])
        self.names = {i: c["name"] for i, c in enumerate(coco["categories"])}

        img_info = {img["id"]: img for img in coco["images"]}
        anns_by_img = {}
        for ann in coco.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for img_id, info in img_info.items():
            fpath = self.img_dir / info["file_name"]
            if not fpath.exists():
                continue
            labels = []
            for ann in anns_by_img.get(img_id, []):
                if ann.get("iscrowd", 0):
                    continue
                cls = self.cat_id_to_idx[ann["category_id"]]
                x, y, w, h = ann["bbox"]
                labels.append([cls, x, y, x + w, y + h])
            labels = np.array(labels, dtype=np.float32).reshape(-1, 5)
            self.samples.append({"path": str(fpath), "labels": labels})

    # ── YOLO dataset.yaml + .txt labels loader ─────────────────────────
    def _load_yolo(self, ann_file):
        with open(ann_file) as f:
            data = yaml.safe_load(f) or {}

        names = data.get("names")
        if isinstance(names, list):
            names_map = {i: str(n) for i, n in enumerate(names)}
        elif isinstance(names, dict):
            names_map = {int(k): str(v) for k, v in names.items()}
        else:
            raise ValueError(
                f"{ann_file}: 'names' must be a list or dict; got {type(names).__name__}"
            )

        self.nc = len(names_map)
        self.names = names_map
        # YOLO .txt class indices are already 0-based, no remapping needed.
        self.cat_id_to_idx = {i: i for i in names_map}

        label_dir = self._yolo_label_dir(self.img_dir)
        if not label_dir.exists():
            raise FileNotFoundError(
                f"YOLO labels directory not found: {label_dir} "
                f"(derived from img_dir={self.img_dir})"
            )

        self.samples = []
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = label_dir / (img_path.stem + ".txt")
            with Image.open(img_path) as im:
                w_img, h_img = im.size
            labels = []
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw / 2) * w_img
                    y1 = (cy - bh / 2) * h_img
                    x2 = (cx + bw / 2) * w_img
                    y2 = (cy + bh / 2) * h_img
                    labels.append([cls, x1, y1, x2, y2])
            labels = np.array(labels, dtype=np.float32).reshape(-1, 5)
            self.samples.append({"path": str(img_path), "labels": labels})

    @staticmethod
    def _yolo_label_dir(img_dir):
        """Map `.../images[/split]` → `.../labels[/split]` (Ultralytics convention)."""
        parts = list(Path(img_dir).parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                return Path(*parts)
        # Fallback: a sibling 'labels' directory next to img_dir.
        return Path(img_dir).parent / "labels"

    def __len__(self):
        return len(self.samples)

    def load_image_and_labels(self, index):
        """Load raw image (BGR) and labels [N, 5] cls,x1,y1,x2,y2."""
        sample = self.samples[index]
        img = cv2.imread(sample["path"])
        assert img is not None, f"Image not found: {sample['path']}"
        labels = sample["labels"].copy()
        return img, labels

    def __getitem__(self, index):
        # Pipeline ordering matches Ultralytics v8_transforms:
        # Mosaic → CopyPaste → RandomPerspective(inside Mosaic) → MixUp → CutMix → HSV → Flips → BGR → Format
        if self.augment and self.mosaic_prob > 0:
            img, labels = self.mosaic_aug(index)
            img, labels = self.copy_paste_aug(img, labels, index)  # no-op for pure detection
            img, labels = self.mixup_aug(img, labels, index)
            img, labels = self.cutmix_aug(img, labels, index)
        else:
            img, labels = self.load_image_and_labels(index)
            # Letterbox resize
            img, ratio, (dw, dh) = letterbox(img, self.imgsz, auto=False)
            if labels.shape[0]:
                labels[:, 1] = labels[:, 1] * ratio + dw
                labels[:, 2] = labels[:, 2] * ratio + dh
                labels[:, 3] = labels[:, 3] * ratio + dw
                labels[:, 4] = labels[:, 4] * ratio + dh

        if self.augment:
            if self.mosaic_prob == 0:
                # Affine for non-mosaic path
                img, labels = random_perspective(
                    img, labels,
                    degrees=self.degrees, translate=self.translate,
                    scale=self.scale, shear=self.shear, perspective=self.perspective
                )
            augment_hsv(img, self.hsv_h, self.hsv_s, self.hsv_v)

            if self.flipud and np.random.random() < self.flipud:
                img = np.flipud(img)
                if labels.shape[0]:
                    # Swap so y1 < y2 after the flip (y1_new = H - y2, y2_new = H - y1)
                    y1 = img.shape[0] - labels[:, 4]
                    y2 = img.shape[0] - labels[:, 2]
                    labels[:, 2], labels[:, 4] = y1, y2

            if self.fliplr and np.random.random() < self.fliplr:
                img = np.fliplr(img)
                if labels.shape[0]:
                    # Swap so x1 < x2 after the flip (x1_new = W - x2, x2_new = W - x1)
                    x1 = img.shape[1] - labels[:, 3]
                    x2 = img.shape[1] - labels[:, 1]
                    labels[:, 1], labels[:, 3] = x1, x2

            # Optional BGR↔RGB swap (default p=0.0)
            img, labels = self.bgr_aug(img, labels)

        # Ensure image is imgsz×imgsz
        if img.shape[:2] != (self.imgsz, self.imgsz):
            img, _, _ = letterbox(img, self.imgsz, auto=False)

        # Convert BGR → RGB, HWC → CHW, normalize
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).float() / 255.0

        # Convert labels to normalized xywh
        nl = len(labels)
        if nl:
            # Clip to image
            labels[:, 1] = np.clip(labels[:, 1], 0, self.imgsz)
            labels[:, 2] = np.clip(labels[:, 2], 0, self.imgsz)
            labels[:, 3] = np.clip(labels[:, 3], 0, self.imgsz)
            labels[:, 4] = np.clip(labels[:, 4], 0, self.imgsz)
            # xyxy → cx,cy,w,h normalized
            labels_out = np.zeros((nl, 6), dtype=np.float32)
            labels_out[:, 1] = labels[:, 0]
            labels_out[:, 2] = ((labels[:, 1] + labels[:, 3]) / 2) / self.imgsz
            labels_out[:, 3] = ((labels[:, 2] + labels[:, 4]) / 2) / self.imgsz
            labels_out[:, 4] = (labels[:, 3] - labels[:, 1]) / self.imgsz
            labels_out[:, 5] = (labels[:, 4] - labels[:, 2]) / self.imgsz
        else:
            labels_out = np.zeros((0, 6), dtype=np.float32)

        labels_out = torch.from_numpy(labels_out)
        return img, labels_out


class YOLOTxtDataset(YOLODataset):
    """
    Dataset that reads YOLO .txt format labels and a data.yaml config.

    __init__ parameters mirror YOLODataset where they apply:
        yaml_file : str      — path to data.yaml
        split     : str      — 'train' or 'val'
        nc        : int      — overrides yaml nc if provided (must match)
        imgsz     : int      — target image size (default 640)
        augment   : bool     — whether to apply training augmentations
        cache     : bool     — cache images in RAM (default False)

    __getitem__ (inherited from YOLODataset) returns:
        img    : Tensor[3, imgsz, imgsz]  float32 [0,1]
        labels : Tensor[N, 6]             float32
                 columns: (batch_idx_placeholder, class_id, cx, cy, w, h)

    This output is IDENTICAL to YOLODataset.__getitem__ — the rest of the
    pipeline (collate_fn, loss, metrics) sees no difference.
    """

    def __init__(
        self,
        yaml_file,
        split="train",
        nc=None,
        imgsz=640,
        augment=True,
        cache=False,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        bgr=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        close_mosaic=0,
        epoch=0,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.yaml_file = str(yaml_file)
        self.split = split
        self.cache = cache

        self._set_hyp(
            imgsz, augment, mosaic, mixup, hsv_h, hsv_s, hsv_v,
            degrees, translate, scale, shear, perspective,
            flipud, fliplr, close_mosaic, epoch,
        )

        cfg = parse_yolo_yaml(yaml_file)
        self.nc = cfg["nc"]
        self.names = {i: n for i, n in enumerate(cfg["names"])}
        # YOLO .txt class indices are already 0-based — identity remap.
        self.cat_id_to_idx = {i: i for i in range(self.nc)}

        if nc is not None and nc != self.nc:
            raise ValueError(
                f"nc mismatch: caller passed nc={nc} but data.yaml defines {self.nc}"
            )

        img_dir = cfg["train_img_dir"] if split == "train" else cfg["val_img_dir"]
        if not img_dir:
            raise ValueError(
                f"{yaml_file}: no '{split}' images path defined in the data.yaml"
            )
        self.img_dir = Path(img_dir)
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"{split} image directory not found: {self.img_dir} "
                f"(resolved from {yaml_file})"
            )

        self._scan_samples()
        self._build_aug(imgsz, mixup, cutmix, copy_paste, copy_paste_mode, bgr)

    def _scan_samples(self):
        """Build self.samples as a list of {'path', 'labels'[N,5] cls,x1,y1,x2,y2 px}."""
        img_paths = sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        self.samples = []
        self.cache_imgs = [] if self.cache else None

        n_with_labels = 0
        n_background = 0
        n_missing = 0
        n_ann = 0
        class_counts = {}

        for img_path in img_paths:
            lbl_path = find_label_path(str(img_path))
            if lbl_path is None:
                n_missing += 1
                # Training: keep missing-label images as background (empty labels).
                # Validation (augment=False): skip them.
                if not self.augment:
                    continue
                labels_px = np.zeros((0, 5), dtype=np.float32)
            else:
                arr = parse_yolo_label(lbl_path, self.nc)  # [N,5] cls,cx,cy,w,h norm
                if arr.shape[0]:
                    with Image.open(img_path) as im:
                        w_img, h_img = im.size
                    labels_px = np.zeros((arr.shape[0], 5), dtype=np.float32)
                    labels_px[:, 0] = arr[:, 0]
                    cx, cy, bw, bh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
                    labels_px[:, 1] = (cx - bw / 2) * w_img
                    labels_px[:, 2] = (cy - bh / 2) * h_img
                    labels_px[:, 3] = (cx + bw / 2) * w_img
                    labels_px[:, 4] = (cy + bh / 2) * h_img
                    n_with_labels += 1
                    n_ann += arr.shape[0]
                    for c in arr[:, 0].astype(int):
                        class_counts[c] = class_counts.get(c, 0) + 1
                else:
                    # Label file present but empty → legitimate background image.
                    labels_px = np.zeros((0, 5), dtype=np.float32)
                    n_background += 1

            sample = {"path": str(img_path), "labels": labels_px}
            if self.cache:
                img = cv2.imread(str(img_path))
                assert img is not None, f"Failed to cache image: {img_path}"
                self.cache_imgs.append(img)
            self.samples.append(sample)

        # Log dataset stats at init.
        dist = ", ".join(
            f"{self.names.get(c, c)}:{class_counts[c]}"
            for c in sorted(class_counts)
        )
        logger.info(
            f"YOLOTxtDataset[{self.split}] {self.yaml_file}: "
            f"{len(self.samples)} images "
            f"({n_with_labels} with objects, {n_background} empty-label, "
            f"{n_missing} missing-label), {n_ann} annotations, nc={self.nc}"
        )
        if dist:
            logger.info(f"  class distribution → {dist}")

    def load_image_and_labels(self, index):
        """Load raw image (BGR) and labels [N,5] cls,x1,y1,x2,y2. Uses RAM cache if enabled."""
        if self.cache:
            img = self.cache_imgs[index].copy()
            labels = self.samples[index]["labels"].copy()
            return img, labels
        return super().load_image_and_labels(index)


def build_dataset(args: dict, split: str, augment: bool) -> Dataset:
    """
    Instantiate the correct dataset class based on config keys present.

    COCO mode: requires 'train_img_dir'/'val_img_dir' + 'train_ann_file'/'val_ann_file'
               (the trainer's remapped 'train_ann'/'val_ann' keys are also accepted)
    YOLO mode: requires 'train_yaml' or 'val_yaml' (single yaml covers both splits)

    Both modes require 'imgsz'. 'nc' is required for COCO, inferred for YOLO.
    """
    imgsz = args["imgsz"]
    nc = args.get("nc")

    # Augmentation hyperparameters, zeroed out when augment=False (val).
    aug = dict(
        imgsz=imgsz,
        augment=augment,
        mosaic=args.get("mosaic", 1.0) if augment else 0.0,
        mixup=args.get("mixup", 0.0) if augment else 0.0,
        cutmix=args.get("cutmix", 0.0) if augment else 0.0,
        copy_paste=args.get("copy_paste", 0.0) if augment else 0.0,
        copy_paste_mode=args.get("copy_paste_mode", "flip"),
        bgr=args.get("bgr", 0.0) if augment else 0.0,
        hsv_h=args.get("hsv_h", 0.015),
        hsv_s=args.get("hsv_s", 0.7),
        hsv_v=args.get("hsv_v", 0.4),
        degrees=args.get("degrees", 0.0),
        translate=args.get("translate", 0.1),
        scale=args.get("scale", 0.5),
        shear=args.get("shear", 0.0),
        perspective=args.get("perspective", 0.0),
        flipud=args.get("flipud", 0.0),
        fliplr=args.get("fliplr", 0.5),
        close_mosaic=args.get("close_mosaic", 0),
        epoch=args.get("epoch", 0),
    )

    if "train_yaml" in args or "val_yaml" in args:
        yaml_key = "train_yaml" if split == "train" else "val_yaml"
        # Fall back to train_yaml for val if val_yaml not specified.
        yaml_file = args.get(yaml_key) or args.get("train_yaml")
        return YOLOTxtDataset(
            yaml_file=yaml_file,
            split=split,
            nc=nc,
            cache=args.get("cache", False),
            **aug,
        )
    elif any(k in args for k in ("train_ann_file", "val_ann_file", "train_ann", "val_ann")):
        ann_key = "train_ann_file" if split == "train" else "val_ann_file"
        ann_alt = "train_ann" if split == "train" else "val_ann"
        img_key = "train_img_dir" if split == "train" else "val_img_dir"
        ann_file = args.get(ann_key) or args.get(ann_alt)
        return YOLODataset(
            img_dir=args[img_key],
            ann_file=ann_file,
            nc=nc,
            **aug,
        )
    else:
        raise ValueError(
            "Config must provide either 'train_yaml' (YOLO format) "
            "or 'train_ann_file' + 'train_img_dir' (COCO format)."
        )
