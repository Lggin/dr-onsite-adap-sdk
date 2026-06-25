"""dr_onsite_adap 공개 facade API — 검출 + 윗면 픽 자세를 한 진입점으로.

런타임 내부(SegModel / run_detection / 윗면 기하 v1·v2 / 가림 필터)를 **재사용**해
한 클래스로 묶고, 결과를 supervision/내부 dict 대신 **깔끔한 dataclass**로 돌려준다.
앱/GUI 없이 라이브러리로 검출·픽자세를 쓰려는 모든 코드의 표준 진입점.

사용
----
    from dr_onsite_adap.api import OnsiteDetector
    import cv2

    det = OnsiteDetector(weights="continual_master_v9_full.pth", geom_algo="v2")

    # (A) 2D 검출만 — depth 불필요
    for d in det.detect(cv2.imread("frame.jpg")):
        print(d.class_name, round(d.confidence, 2), d.bbox)

    # (B) 검출 + 윗면 픽 자세 — RGB-D 필요
    scene = det.process(color_bgr, depth_m, intrinsics=(fx, fy, cx, cy), up_dir=(0,0,-1))
    for p in scene.pick_targets:
        print(p.class_name, p.center_3d, p.long_side_mm, p.short_side_mm)
    if scene.done:
        print("빈 팔레트 — 완료")

입력 규약
  * color_bgr : OpenCV BGR 이미지(HxWx3). cv2.imread / 카메라 프레임 그대로.
  * depth_m   : color 와 정렬된 미터 단위 depth(HxW). 0/범위밖은 무효로 처리됨.
  * intrinsics: (fx, fy, cx, cy) — depth=color 해상도 기준.
  * up_dir    : 카메라 프레임 기준 '위' 단위벡터(중력 반대). None 이면 config 기본값.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .runtime.detect import run_detection
from .runtime.inference import SegModel


def _name(cid: int) -> str:
    entry = config.CLASSES.get(cid)
    return entry[0] if entry else str(cid)


@dataclass
class Detection:
    """2D 검출 1건(픽 자세 없음)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]          # (x1, y1, x2, y2) 픽셀
    mask: np.ndarray | None = None           # HxW bool


@dataclass
class PickPose:
    """윗면 픽 대상 1건 — 가림 필터 통과 + 윗면 기하 추정 성공."""
    class_id: int
    class_name: str
    confidence: float
    center_3d: tuple[float, float, float]    # (x, y, z) m, 카메라 프레임
    center_2d: tuple[int, int]               # (u, v) 픽셀
    long_axis_3d: tuple[float, float, float]
    short_axis_3d: tuple[float, float, float]
    long_side_mm: float
    short_side_mm: float
    top_normal_3d: tuple[float, float, float] | None = None   # v2 만(그리퍼 접근축)
    mask: np.ndarray | None = None


@dataclass
class Pallet:
    """팔레트 검출(픽 대상 아님 — 존재/위치만)."""
    confidence: float
    center_2d: tuple[int, int] | None = None
    mask: np.ndarray | None = None


@dataclass
class SceneResult:
    """한 RGB-D 프레임 처리 결과."""
    pick_targets: list[PickPose] = field(default_factory=list)
    pallets: list[Pallet] = field(default_factory=list)
    item_count: int = 0          # 픽 대상(박스/토트/랩) 검출 수
    done: bool = False           # 팔레트는 보이는데 픽 대상 0 = 작업 완료
    counts: dict[int, int] = field(default_factory=dict)
    inference_time: float = 0.0


def _vec3(a) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float).ravel()
    return (float(a[0]), float(a[1]), float(a[2]))


class OnsiteDetector:
    """디팔레타이징 검출 + 윗면 픽 자세 facade.

    Parameters
    ----------
    weights : str | None
        가중치 경로. None 이면 config.WEIGHTS_PATH(env DEPAL_WEIGHTS).
    num_classes : int | None
        클래스 수. None 이면 config.NUM_CLASSES(기본 4). **가중치와 일치해야 함.**
    device : str | None
        "cuda"/"cpu". None 이면 config.DEVICE.
    geom_algo : str | None
        윗면 기하 알고리즘 "v1"/"v2". None 이면 config.GEOM_ALGO(env DEPAL_GEOM_ALGO).
        지정하면 이 인스턴스 사용 동안 config.GEOM_ALGO 를 그 값으로 둔다.
    conf_threshold : float | None
        신뢰도 임계. None 이면 config.CONF_THRES(기본 0.5).
    """

    def __init__(self, weights: str | None = None, num_classes: int | None = None,
                 device: str | None = None, geom_algo: str | None = None,
                 conf_threshold: float | None = None):
        if geom_algo is not None:
            config.GEOM_ALGO = geom_algo.lower()
        self.conf_threshold = (conf_threshold if conf_threshold is not None
                               else config.CONF_THRES)
        self.model = SegModel(
            weights or config.WEIGHTS_PATH,
            num_classes if num_classes is not None else config.NUM_CLASSES,
            device or config.DEVICE,
        )

    # ── (A) 2D 검출만 ──
    def detect(self, color_bgr: np.ndarray,
               conf_threshold: float | None = None) -> list[Detection]:
        """depth 없이 2D 검출만. Detection 리스트 반환(팔레트 포함, 가림필터 없음)."""
        thr = conf_threshold if conf_threshold is not None else self.conf_threshold
        preds = self.model.predict(color_bgr, conf_threshold=thr)
        xyxy = np.asarray(preds.xyxy)
        cls = np.asarray(preds.class_id)
        conf = np.asarray(preds.confidence)
        masks = getattr(preds, "mask", None)
        out = []
        for i in range(len(xyxy)):
            cid = int(cls[i])
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
            out.append(Detection(
                class_id=cid, class_name=_name(cid), confidence=float(conf[i]),
                bbox=(x1, y1, x2, y2),
                mask=(masks[i].astype(bool) if masks is not None else None),
            ))
        return out

    # ── (B) 검출 + 윗면 픽 자세 ──
    def process(self, color_bgr: np.ndarray, depth_m: np.ndarray,
                intrinsics, up_dir=None, conf_threshold: float | None = None,
                with_pointcloud: bool = False) -> SceneResult:
        """RGB-D 한 프레임 → 검출 + 가림 필터 + 윗면 픽 자세. SceneResult 반환."""
        h, w = color_bgr.shape[:2]
        up = up_dir if up_dir is not None else config.UP_DIR_CAMERA_FRAME
        thr = conf_threshold if conf_threshold is not None else self.conf_threshold
        state = run_detection(self.model, color_bgr, depth_m, tuple(intrinsics),
                              up, h, w, with_pointcloud=with_pointcloud,
                              conf_threshold=thr)

        picks = []
        for det in state["active"]:
            r = det["res"]
            picks.append(PickPose(
                class_id=det["cid"], class_name=det["name"],
                confidence=det["confidence"],
                center_3d=_vec3(r["center_3d"]),
                center_2d=(int(r["center_2d"][0]), int(r["center_2d"][1])),
                long_axis_3d=_vec3(r["long_axis_3d"]),
                short_axis_3d=_vec3(r["short_axis_3d"]),
                long_side_mm=float(r["long_side_mm"]),
                short_side_mm=float(r["short_side_mm"]),
                top_normal_3d=(_vec3(r["top_normal_3d"]) if r.get("top_normal_3d") is not None else None),
                mask=det["mask"],
            ))
        pallets = [Pallet(confidence=p["confidence"], center_2d=p["center_2d"],
                          mask=p["mask"]) for p in state["pallets"]]
        return SceneResult(
            pick_targets=picks, pallets=pallets,
            item_count=state["item_count"], done=state["done"],
            counts=state["counts"], inference_time=state["inference_time"],
        )
