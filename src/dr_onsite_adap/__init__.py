"""dr_onsite_adap — on-site depalletizing vision: top-face pick-pose detection
+ active learning.

Two halves:
  * dr_onsite_adap.runtime  — live RGB-D detection, top-face center/axis
    estimation, occlusion filtering, SAM labeling, visualization.
  * dr_onsite_adap.pipeline — active-learning training loop: ingest labels,
    build a replay-mixed dataset, LoRA fine-tune, merge, evaluate.
"""

__version__ = "0.1.0"
