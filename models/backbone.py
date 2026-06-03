# Backbone stages are built dynamically from YAML in models/yolov11.py.
# This module documents the backbone architecture and channel configuration.
#
# YOLOv11 Backbone (base channels before width scaling):
#   Layer 0:  Conv(3→64, k=3, s=2)          stride=2
#   Layer 1:  Conv(64→128, k=3, s=2)         stride=4
#   Layer 2:  C3k2(128→256, n=2, attn=False) stride=4
#   Layer 3:  Conv(256→256, k=3, s=2)        stride=8
#   Layer 4:  C3k2(256→512, n=2, attn=False) stride=8   ← P3 save
#   Layer 5:  Conv(512→512, k=3, s=2)        stride=16
#   Layer 6:  C3k2(512→512, n=2, attn=True)  stride=16  ← P4 save
#   Layer 7:  Conv(512→1024, k=3, s=2)       stride=32
#   Layer 8:  C3k2(1024→1024, n=2, attn=True) stride=32
#   Layer 9:  SPPF(1024→1024, k=5)           stride=32
#   Layer 10: C2PSA(1024→1024, n=2)          stride=32  ← P5 save
#
# Channel widths are scaled by width_multiple per variant:
#   n: 0.25 → [16, 32, 64, 64, 128, 128, 128, 256, 256, 256, 256]
#   s: 0.50 → [32, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512]
#   m: 1.00 → [64, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512] (capped at 512)
#   l: 1.00 → same as m, but depth_multiple=1.0 doubles repeat counts
#   x: 1.50 → [96, 192, 384, 384, 512, 512, 512, 512, 512, 512, 512] (capped at 512)
