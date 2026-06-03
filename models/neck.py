# Neck (FPN + PAN) stages are built dynamically from YAML in models/yolov11.py.
# This module documents the neck architecture and connections.
#
# YOLOv11 Neck — FPN top-down path:
#   Layer 11: Upsample(P5, scale=2)                  → [bs, 256, 40, 40] (nano)
#   Layer 12: Concat([Layer11, Layer6])               → [bs, 384, 40, 40]
#   Layer 13: C3k2(384→256, n=2, attn=False)          → P4_neck
#
#   Layer 14: Upsample(P4_neck, scale=2)              → [bs, 256, 80, 80]
#   Layer 15: Concat([Layer14, Layer4])               → [bs, 384, 80, 80]
#   Layer 16: C3k2(384→128, n=2, attn=False)          → P3/8 output ✓
#
# YOLOv11 Neck — PAN bottom-up path:
#   Layer 17: Conv(P3, 128→128, k=3, s=2)             → [bs, 128, 40, 40]
#   Layer 18: Concat([Layer17, Layer13])              → [bs, 384, 40, 40]
#   Layer 19: C3k2(384→256, n=2, attn=False)          → P4/16 output ✓
#
#   Layer 20: Conv(P4/16, 256→256, k=3, s=2)          → [bs, 256, 20, 20]
#   Layer 21: Concat([Layer20, Layer10])              → [bs, 512, 20, 20]
#   Layer 22: C3k2(512→256, n=2, attn=True)           → P5/32 output ✓
#
# Detection head reads from layers [16, 19, 22] — strides [8, 16, 32].
# (Channel counts above are for yolov11n; scale by width_multiple for other variants.)
