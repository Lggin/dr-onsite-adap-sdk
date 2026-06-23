"""dr_onsite_adap — on-site depalletizing vision: top-face pick-pose detection
+ active learning.

Two halves:
  * dr_onsite_adap.runtime  — live RGB-D detection, top-face center/axis
    estimation, occlusion filtering, SAM labeling, visualization.
  * dr_onsite_adap.pipeline — active-learning training loop: ingest labels,
    build a replay-mixed dataset, LoRA fine-tune, merge, evaluate.
"""

from ._version import __version__
from .exceptions import (
    DrOnsiteError,
    ValidationError,
    CameraError,
    ModelError,
    PipelineError,
)

# 상위 패키지가 공개로 보장하는 이름. 예외 계층은 가벼워서 top-level 에 노출하지만,
# 런타임/파이프라인 클래스는 무거운 의존성(torch 등)을 끌어오므로 강제 import 하지 않는다
# (서브패키지에서 직접 가져다 쓴다: from dr_onsite_adap.runtime.inference import SegModel).
__all__ = [
    "__version__",
    "DrOnsiteError",
    "ValidationError",
    "CameraError",
    "ModelError",
    "PipelineError",
]
