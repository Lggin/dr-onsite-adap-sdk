"""depal_al — depalletizing vision: top-face pick-pose detection + active learning.

Two halves:
  * depal_al.runtime  — live RGB-D detection, top-face center/axis estimation,
    occlusion filtering, SAM labeling, visualization.
  * depal_al.pipeline — active-learning training loop: ingest labels, build a
    replay-mixed dataset, LoRA fine-tune, merge, evaluate.
"""

__version__ = "0.1.0"
