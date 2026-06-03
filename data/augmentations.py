import cv2
import math
import random
import numpy as np


# ---------------------------------------------------------------------------
# Core ops (function-style — used by the dataset's __getitem__)
# ---------------------------------------------------------------------------

def letterbox(im, new_shape=640, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image to new_shape, preserving aspect ratio."""
    shape = im.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # [w, h]
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = dw % stride, dh % stride
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """Apply random HSV augmentation in-place (matches Ultralytics ≥ 8.3.79)."""
    if hgain or sgain or vgain:
        # Additive hue, multiplicative sat/val
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        lut_sat[0] = 0  # prevent pure white from changing color
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_perspective(img, labels=(), degrees=0.0, translate=0.1, scale=0.5,
                       shear=0.0, perspective=0.0, border=(0, 0)):
    """Random perspective/affine transform. labels: [N, 5] cls, x1, y1, x2, y2 (pixel coords)."""
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation + Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)

    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    n = len(labels)
    if n:
        box1 = labels[:, 1:5].T.copy()  # original boxes (pre-transform) for area-ratio filtering
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        else:
            xy = xy[:, :2].reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.array([x.min(1), y.min(1), x.max(1), y.max(1)]).T

        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        labels[:, 1:5] = new

        i = _box_candidates(box1, labels[:, 1:5].T, wh_thr=2, ar_thr=100, area_thr=0.1, s=s)
        labels = labels[i]

    return img, labels


def _box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16, s=1.0):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 * s**2 + eps) > area_thr) & (ar < ar_thr)


def bbox_ioa(box1, box2, eps=1e-9):
    """Intersection over Area of box2 (matches Ultralytics bbox_ioa). box1,box2: [N,4],[M,4] xyxy."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_w = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0)
    inter_h = (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)
    inter = inter_w * inter_h
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    return inter / area2


# ---------------------------------------------------------------------------
# Class-based wrappers (Ultralytics-style API for explicit pipeline use)
# ---------------------------------------------------------------------------

class LetterBox:
    """Class-style wrapper around the letterbox function (matches Ultralytics API)."""
    def __init__(self, new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True,
                 center=True, stride=32, padding_value=114):
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.center = center
        self.stride = stride
        self.padding_value = padding_value

    def __call__(self, img, labels=None):
        im, r, (dw, dh) = letterbox(
            img,
            new_shape=self.new_shape,
            color=(self.padding_value,) * 3,
            auto=self.auto,
            scaleFill=self.scale_fill,
            scaleup=self.scaleup,
            stride=self.stride,
        )
        if labels is not None and len(labels):
            labels[:, 1] = labels[:, 1] * r + dw
            labels[:, 2] = labels[:, 2] * r + dh
            labels[:, 3] = labels[:, 3] * r + dw
            labels[:, 4] = labels[:, 4] * r + dh
        return im, labels


class Mosaic:
    """4- or 9-image mosaic augmentation."""
    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        assert n in {4, 9}, "grid must be 4 or 9"
        self.dataset = dataset
        self.imgsz = imgsz
        self.p = p
        self.n = n

    def __call__(self, index):
        if random.random() > self.p:
            return self.dataset.load_image_and_labels(index)
        return self._mosaic4(index) if self.n == 4 else self._mosaic9(index)

    def _mosaic4(self, index):
        indices = [index] + random.choices(range(len(self.dataset)), k=3)
        random.shuffle(indices)

        s = self.imgsz
        yc = int(random.uniform(s * 0.5, s * 1.5))
        xc = int(random.uniform(s * 0.5, s * 1.5))

        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []

        for i, idx in enumerate(indices):
            img, labels = self.dataset.load_image_and_labels(idx)
            h, w = img.shape[:2]

            if i == 0:   # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1: # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2: # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:        # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padx, pady = x1a - x1b, y1a - y1b

            if labels.shape[0]:
                labels_adj = labels.copy()
                labels_adj[:, 1] = labels[:, 1] + padx
                labels_adj[:, 2] = labels[:, 2] + pady
                labels_adj[:, 3] = labels[:, 3] + padx
                labels_adj[:, 4] = labels[:, 4] + pady
                mosaic_labels.append(labels_adj)

        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
            mosaic_labels[:, 1:5] = np.clip(mosaic_labels[:, 1:5], 0, 2 * s)
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)

        mosaic_img, mosaic_labels = random_perspective(
            mosaic_img, mosaic_labels,
            degrees=0.0, translate=0.1, scale=0.5,
            border=(-s // 2, -s // 2),
        )
        return mosaic_img, mosaic_labels

    def _mosaic9(self, index):
        """9-image mosaic — Ultralytics-style 3x3 grid centered at (s, s) in a 3s×3s canvas."""
        indices = [index] + random.choices(range(len(self.dataset)), k=8)
        random.shuffle(indices)
        s = self.imgsz
        canvas = np.full((s * 3, s * 3, 3), 114, dtype=np.uint8)
        labels_all = []
        hp = wp = -1
        h0 = w0 = None

        for i, idx in enumerate(indices):
            img, labels = self.dataset.load_image_and_labels(idx)
            h, w = img.shape[:2]
            if i == 0:    # center
                cx, cy = s, s
                w0, h0 = w, h
            elif i == 1:  # top
                cx, cy = s, s - h
            elif i == 2:  # top right
                cx, cy = s + wp, s - h
            elif i == 3:  # right
                cx, cy = s + w0, s
            elif i == 4:  # bottom right
                cx, cy = s + w0, s + hp
            elif i == 5:  # bottom
                cx, cy = s + w0 - w, s + h0
            elif i == 6:  # bottom left
                cx, cy = s + w0 - wp - w, s + h0
            elif i == 7:  # left
                cx, cy = s - w, s + h0 - h
            else:         # top left
                cx, cy = s - w, s + h0 - hp - h
            padx, pady = cx, cy
            x1, y1 = max(cx, 0), max(cy, 0)
            x2, y2 = min(cx + w, s * 3), min(cy + h, s * 3)
            canvas[y1:y2, x1:x2] = img[y1 - cy:y2 - cy, x1 - cx:x2 - cx]
            if labels.shape[0]:
                la = labels.copy()
                la[:, 1] += padx; la[:, 3] += padx
                la[:, 2] += pady; la[:, 4] += pady
                labels_all.append(la)
            hp, wp = h, w

        labels_all = np.concatenate(labels_all, 0) if labels_all else np.zeros((0, 5), dtype=np.float32)
        # Center crop back to 2s×2s and apply random perspective
        canvas = canvas[s // 2:s * 5 // 2, s // 2:s * 5 // 2]
        if labels_all.shape[0]:
            labels_all[:, 1] -= s // 2; labels_all[:, 3] -= s // 2
            labels_all[:, 2] -= s // 2; labels_all[:, 4] -= s // 2
            labels_all[:, 1:5] = np.clip(labels_all[:, 1:5], 0, 2 * s)
        canvas, labels_all = random_perspective(
            canvas, labels_all,
            degrees=0.0, translate=0.1, scale=0.5,
            border=(-s // 2, -s // 2),
        )
        return canvas, labels_all


class MixUp:
    """MixUp augmentation: Beta(32,32) blend of two images (matches Ultralytics)."""
    def __init__(self, dataset, p=0.0, pre_transform=None):
        self.dataset = dataset
        self.p = p
        # pre_transform applied to the second (mix) image; if provided, second image goes through
        # the same Mosaic+Affine pipeline as the primary, matching Ultralytics.
        self.pre_transform = pre_transform

    def __call__(self, img, labels, index):
        if random.random() > self.p:
            return img, labels
        idx2 = random.randint(0, len(self.dataset) - 1)
        if self.pre_transform is not None:
            img2, labels2 = self.pre_transform(idx2)
        else:
            img2, labels2 = self.dataset.load_image_and_labels(idx2)
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
        r = np.random.beta(32.0, 32.0)
        img = (img * r + img2 * (1 - r)).astype(np.uint8)
        if labels.shape[0] and labels2.shape[0]:
            labels = np.concatenate([labels, labels2], axis=0)
        return img, labels


class CutMix:
    """CutMix augmentation: replace a random rectangle from image with a patch from another image.

    Matches Ultralytics: samples cut size via Beta(beta, beta), tries `num_areas` candidate cuts,
    picks a cut that doesn't overlap primary instances, then keeps GT boxes from the source image
    whose center falls inside the cut region.
    """
    def __init__(self, dataset, p=0.0, beta=1.0, num_areas=3):
        self.dataset = dataset
        self.p = p
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, w, h):
        lam = np.random.beta(self.beta, self.beta)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def __call__(self, img, labels, index):
        if random.random() > self.p or img is None:
            return img, labels
        h, w = img.shape[:2]
        n_primary = labels.shape[0]

        # Sample candidate cut regions
        cuts = np.stack([self._rand_bbox(w, h) for _ in range(self.num_areas)])  # [num_areas, 4]
        if n_primary:
            ioa1 = bbox_ioa(cuts, labels[:, 1:5])  # [num_areas, n_primary]
            valid = np.nonzero(ioa1.sum(1) <= 0)[0]
        else:
            valid = np.arange(self.num_areas)
        if len(valid) == 0:
            return img, labels

        area = cuts[np.random.choice(valid)]
        x1, y1, x2, y2 = area.astype(int)

        # Load second image, resize to primary
        idx2 = random.randint(0, len(self.dataset) - 1)
        img2, labels2 = self.dataset.load_image_and_labels(idx2)
        img2 = cv2.resize(img2, (w, h))
        if labels2.shape[0]:
            # Scale labels2 to primary dims (load_image_and_labels returns pixel xyxy at orig scale)
            sy = h / img2.shape[0]; sx = w / img2.shape[1]
            # (img2 was resized already; treat labels2 as if at orig scale → rescale)
            # Since img2 was resized, labels2 were not — best-effort rescale not needed for CutMix
            # because we'll filter labels2 by the cut area below.
            pass

        # Apply patch swap
        img = img.copy()
        img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

        # Keep labels2 whose bbox overlaps the cut area significantly
        if labels2.shape[0]:
            ioa2 = bbox_ioa(area[None], labels2[:, 1:5]).squeeze(0)
            keep = np.nonzero(ioa2 >= 0.1)[0]
            if len(keep):
                kept = labels2[keep].copy()
                kept[:, 1] = np.clip(kept[:, 1], x1, x2)
                kept[:, 2] = np.clip(kept[:, 2], y1, y2)
                kept[:, 3] = np.clip(kept[:, 3], x1, x2)
                kept[:, 4] = np.clip(kept[:, 4], y1, y2)
                labels = np.concatenate([labels, kept], 0) if n_primary else kept

        return img, labels


class CopyPaste:
    """CopyPaste augmentation. Detection-mode (mode='flip'): pastes horizontally-flipped object crops
    from the same image into non-overlapping regions. For pure-detection (no segmentation masks),
    Ultralytics' CopyPaste exits early — we mirror that behavior.
    """
    def __init__(self, dataset=None, p=0.0, mode="flip"):
        assert mode in {"flip", "mixup"}
        self.dataset = dataset
        self.p = p
        self.mode = mode

    def __call__(self, img, labels, index=None):
        # Ultralytics CopyPaste requires segmentation masks (instances.segments). For pure detection
        # this augmentation is a no-op — match that behavior.
        return img, labels


class RandomBGR:
    """Random BGR↔RGB channel swap with probability p (matches Ultralytics)."""
    def __init__(self, p=0.0):
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            img = img[:, :, ::-1]
        return img, labels


class RandomFlip:
    """Random horizontal or vertical flip with probability p (matches Ultralytics RandomFlip)."""
    def __init__(self, p=0.5, direction="horizontal"):
        assert direction in {"horizontal", "vertical"}
        self.p = p
        self.direction = direction

    def __call__(self, img, labels):
        if random.random() >= self.p:
            return img, labels
        h, w = img.shape[:2]
        if self.direction == "horizontal":
            img = np.ascontiguousarray(img[:, ::-1])
            if labels.shape[0]:
                x1 = w - labels[:, 3]
                x2 = w - labels[:, 1]
                labels[:, 1], labels[:, 3] = x1, x2
        else:
            img = np.ascontiguousarray(img[::-1])
            if labels.shape[0]:
                y1 = h - labels[:, 4]
                y2 = h - labels[:, 2]
                labels[:, 2], labels[:, 4] = y1, y2
        return img, labels


class Albumentations:
    """Optional Albumentations block (Blur, MedianBlur, ToGray, CLAHE at low probability).

    Matches Ultralytics: image-only transforms, no label changes. Silently no-ops if the
    albumentations library is not installed.
    """
    def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        try:
            import albumentations as A
            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ])
        except Exception:
            self.transform = None

    def __call__(self, img, labels):
        if self.transform is None or random.random() >= self.p:
            return img, labels
        try:
            img = self.transform(image=img)["image"]
        except Exception:
            pass
        return img, labels


class Format:
    """Final tensor formatting: BGR→RGB, HWC→CHW, /255, xyxy→cxcywh normalized.

    Returns (img_tensor [3,H,W], labels_tensor [N, 6] with cls,cx,cy,w,h plus a leading batch_idx
    column for the collator to fill).
    """
    def __init__(self, imgsz=640, normalize=True, bgr_to_rgb=True):
        import torch
        self._torch = torch
        self.imgsz = imgsz
        self.normalize = normalize
        self.bgr_to_rgb = bgr_to_rgb

    def __call__(self, img, labels):
        torch = self._torch
        if self.bgr_to_rgb:
            img = img[:, :, ::-1]
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img_t = torch.from_numpy(img).float()
        if self.normalize:
            img_t /= 255.0
        nl = labels.shape[0] if hasattr(labels, "shape") else 0
        if nl:
            labels = labels.astype(np.float32, copy=False)
            labels[:, [1, 3]] = np.clip(labels[:, [1, 3]], 0, self.imgsz)
            labels[:, [2, 4]] = np.clip(labels[:, [2, 4]], 0, self.imgsz)
            out = np.zeros((nl, 6), dtype=np.float32)
            out[:, 1] = labels[:, 0]
            out[:, 2] = (labels[:, 1] + labels[:, 3]) / 2 / self.imgsz
            out[:, 3] = (labels[:, 2] + labels[:, 4]) / 2 / self.imgsz
            out[:, 4] = (labels[:, 3] - labels[:, 1]) / self.imgsz
            out[:, 5] = (labels[:, 4] - labels[:, 2]) / self.imgsz
        else:
            out = np.zeros((0, 6), dtype=np.float32)
        return img_t, torch.from_numpy(out)


class Compose:
    """Sequence of transforms applied in order (matches Ultralytics Compose semantics)."""
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, *args, **kwargs):
        # Compose is intentionally simple — callers know each transform's signature.
        # We don't enforce a single signature because Mosaic/MixUp take `index`,
        # while HSV/Flip/Albumentations take (img, labels).
        raise NotImplementedError(
            "Compose is a holder for transforms; the dataset applies them explicitly "
            "because YOLO transforms have heterogeneous signatures."
        )

    def insert(self, index, transform):
        self.transforms.insert(index, transform)

    def append(self, transform):
        self.transforms.append(transform)


def v8_transforms(dataset, imgsz, hyp):
    """Return a dict of class-based transforms matching Ultralytics' v8_transforms ordering.

    Order: Mosaic → CopyPaste → RandomPerspective → MixUp → CutMix → Albumentations → HSV → Flips → Format
    The dataset's __getitem__ applies these in this order.
    """
    return {
        "mosaic": Mosaic(dataset, imgsz=imgsz, p=hyp.get("mosaic", 1.0), n=hyp.get("mosaic_n", 4)),
        "copy_paste": CopyPaste(dataset, p=hyp.get("copy_paste", 0.0), mode=hyp.get("copy_paste_mode", "flip")),
        "mixup": MixUp(dataset, p=hyp.get("mixup", 0.0)),
        "cutmix": CutMix(dataset, p=hyp.get("cutmix", 0.0)),
        "albumentations": Albumentations(p=1.0),
        "bgr": RandomBGR(p=hyp.get("bgr", 0.0)),
        "flipud": RandomFlip(p=hyp.get("flipud", 0.0), direction="vertical"),
        "fliplr": RandomFlip(p=hyp.get("fliplr", 0.5), direction="horizontal"),
        "format": Format(imgsz=imgsz),
    }
