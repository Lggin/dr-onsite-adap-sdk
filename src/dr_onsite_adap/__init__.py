"""dr_onsite_adap — on-site depalletizing vision: top-face pick-pose detection
+ active learning.

Two halves:
  * dr_onsite_adap.runtime  — live RGB-D detection, top-face center/axis
    estimation, occlusion filtering, SAM labeling, visualization.
  * dr_onsite_adap.pipeline — active-learning training loop: ingest labels,
    build a replay-mixed dataset, LoRA fine-tune, merge, evaluate.
"""

import logging as _logging_mod

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
    # facade API (지연 로드 — 아래 __getattr__ 참고)
    "OnsiteDetector",      # low/mid-level: 프레임 넣고 결과 받기
    "OnsiteSession",       # high-level: 카메라+추론 소유, 콜백+폴링 (UI 팀용)
    "SceneResult",
    "PickPose",
    "Pallet",
    "Detection",
]

# 검출 facade 는 torch/rfdetr 를 끌어오므로, `import dr_onsite_adap` 만으로는 로드하지
# 않고 실제로 이름을 가져갈 때만 지연 import 한다(PEP 562). 그래서 가벼운 예외/버전
# 사용은 무거운 의존성 없이 되고, `from dr_onsite_adap import OnsiteDetector/OnsiteSession` 도 동작한다.
_LAZY_API = {"OnsiteDetector", "SceneResult", "PickPose", "Pallet", "Detection"}


def __getattr__(name):
    if name in _LAZY_API:
        from . import api
        return getattr(api, name)
    if name == "OnsiteSession":
        from .runtime.session import OnsiteSession
        return OnsiteSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 라이브러리 위생: 임포트만으로는 로그를 출력하지 않는다(임베딩 앱이 핸들러 제어).
# CLI/앱 진입점은 dr_onsite_adap._logging.setup_cli_logging() 으로 stdout 출력을 켠다.
_logging_mod.getLogger("dr_onsite_adap").addHandler(_logging_mod.NullHandler())
