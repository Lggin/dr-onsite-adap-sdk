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

import cv2
import numpy as np

from . import config
from .runtime.clustering import GROUP_PALETTE as _GROUP_PALETTE, cluster_by_appearance
from .runtime.detect import run_detection
from .runtime.inference import SegModel
from .runtime.viz import (
    draw_orientation_axes,
    draw_styled_text,
    draw_top_center,
    overlay_mask,
)


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
    group_id: int = -1                       # 외형 클러스터(cluster=True 일 때). -1=미할당
    group_label: str = ""                    # '<클래스명>#<서브그룹>' 예 'Carton#0'


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
    group_id: int = -1                       # 외형 클러스터(cluster=True 일 때). -1=미할당
    group_label: str = ""                    # '<클래스명>#<서브그룹>' 예 'Carton#0'


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
    intrinsics: tuple | None = None    # (fx,fy,cx,cy) — draw() 의 축 투영에 사용


def _vec3(a) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float).ravel()
    return (float(a[0]), float(a[1]), float(a[2]))


def _scene_from_state(state, intrinsics) -> "SceneResult":
    """run_detection() 의 state(dict) → SceneResult. OnsiteDetector.process 와
    OnsiteSession(스무딩 거친 state 포함)이 공유하는 패킹."""
    picks = []
    for det in state["active"]:
        r = det["res"]
        picks.append(PickPose(
            class_id=det["cid"], class_name=det["name"], confidence=det["confidence"],
            center_3d=_vec3(r["center_3d"]),
            center_2d=(int(r["center_2d"][0]), int(r["center_2d"][1])),
            long_axis_3d=_vec3(r["long_axis_3d"]), short_axis_3d=_vec3(r["short_axis_3d"]),
            long_side_mm=float(r["long_side_mm"]), short_side_mm=float(r["short_side_mm"]),
            top_normal_3d=(_vec3(r["top_normal_3d"]) if r.get("top_normal_3d") is not None else None),
            mask=det["mask"]))
    pallets = [Pallet(confidence=p["confidence"], center_2d=p["center_2d"], mask=p["mask"])
               for p in state["pallets"]]
    return SceneResult(pick_targets=picks, pallets=pallets, item_count=state["item_count"],
                       done=state["done"], counts=state["counts"],
                       inference_time=state["inference_time"], intrinsics=tuple(intrinsics))


def draw_scene(color_bgr, result):
    """검출(Detection 리스트) 또는 SceneResult 를 BGR 이미지에 그려 반환. UI 비종속.

    group_id 가 있으면(cluster=True) 클래스 색 대신 **그룹 색**으로 칠하고 라벨에 그룹명을 붙인다.
    """
    vis = color_bgr.copy()

    def _color(cid):
        entry = config.CLASSES.get(cid)
        return entry[1] if entry else (0, 255, 0)

    def _obj_color(cid, gid):
        return _GROUP_PALETTE[gid % len(_GROUP_PALETTE)] if gid is not None and gid >= 0 else _color(cid)

    if isinstance(result, SceneResult):
        for i, p in enumerate(result.pick_targets, 1):
            col = _obj_color(p.class_id, p.group_id)
            if p.mask is not None:
                overlay_mask(vis, p.mask, col)
            draw_top_center(vis, p.center_2d)
            if result.intrinsics is not None:
                draw_orientation_axes(
                    vis, np.asarray(p.center_3d, dtype=float),
                    np.asarray(p.long_axis_3d, dtype=float),
                    np.asarray(p.short_axis_3d, dtype=float),
                    p.long_side_mm, p.short_side_mm, result.intrinsics)
            u, v = p.center_2d
            tag = f" {p.group_label}" if p.group_label else ""
            draw_styled_text(
                vis, f"[{i}] {p.class_name}{tag} {p.long_side_mm:.0f}x{p.short_side_mm:.0f}mm",
                (u + 10, v - 10), col, scale=0.5, thickness=1)
        for pal in result.pallets:
            if pal.center_2d is not None:
                draw_styled_text(vis, "Pallet", pal.center_2d, (130, 100, 50),
                                 scale=0.6, thickness=2)
        if result.done:
            draw_styled_text(vis, "PALLET EMPTY - DONE", (20, 40), (0, 220, 0),
                             scale=1.0, thickness=2)
    else:  # list[Detection]
        for d in result:
            col = _obj_color(d.class_id, d.group_id)
            if d.mask is not None:
                overlay_mask(vis, d.mask, col)
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            tag = f" {d.group_label}" if d.group_label else ""
            draw_styled_text(vis, f"{d.class_name}{tag} {d.confidence:.2f}",
                             (x1, max(15, y1 - 6)), col, scale=0.5, thickness=1)
    return vis


class OnsiteDetector:
    """디팔레타이징 검출 + 윗면 픽 자세 facade.

    Parameters
    ----------
    weights : str | None
        가중치 경로. None 이면 config.WEIGHTS_PATH(env DEPAL_WEIGHTS).
    num_classes : int | None
        클래스 수. None 이면 config.NUM_CLASSES(기본 5). **가중치와 일치해야 함.**
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
               conf_threshold: float | None = None,
               cluster: bool = False) -> list[Detection]:
        """depth 없이 2D 검출만. Detection 리스트 반환(팔레트 포함, 가림필터 없음).

        cluster=True 면 검출 후 **같은 클래스 안에서** 외형(색·질감·크기)으로 서브그룹핑해
        각 Detection 의 group_id/group_label 을 채운다(분류 후 클러스터링).
        """
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
        if cluster and out:
            gids, labels = cluster_by_appearance(
                [d.mask for d in out], [d.class_id for d in out], color_bgr)
            for d, g, l in zip(out, gids, labels):
                d.group_id, d.group_label = g, l
        return out

    # ── (B) 검출 + 윗면 픽 자세 ──
    def process(self, color_bgr: np.ndarray, depth_m: np.ndarray,
                intrinsics, up_dir=None, conf_threshold: float | None = None,
                with_pointcloud: bool = False, cluster: bool = False) -> SceneResult:
        """RGB-D 한 프레임 → 검출 + 가림 필터 + 윗면 픽 자세. SceneResult 반환.

        cluster=True 면 픽 대상을 **같은 클래스 안에서** 외형(색·질감·크기)으로 서브그룹핑해
        각 PickPose 의 group_id/group_label 을 채운다(분류 후 클러스터링).
        """
        h, w = color_bgr.shape[:2]
        up = up_dir if up_dir is not None else config.UP_DIR_CAMERA_FRAME
        thr = conf_threshold if conf_threshold is not None else self.conf_threshold
        state = run_detection(self.model, color_bgr, depth_m, tuple(intrinsics),
                              up, h, w, with_pointcloud=with_pointcloud,
                              conf_threshold=thr)

        scene = _scene_from_state(state, tuple(intrinsics))
        if cluster and scene.pick_targets:
            pts = scene.pick_targets
            gids, labels = cluster_by_appearance(
                [p.mask for p in pts], [p.class_id for p in pts], color_bgr,
                sizes_mm=[(p.long_side_mm, p.short_side_mm) for p in pts],
                geoms=[(p.center_3d, p.long_axis_3d, p.short_axis_3d,
                        p.long_side_mm, p.short_side_mm) for p in pts],
                intrinsics=tuple(intrinsics))
            for p, g, l in zip(pts, gids, labels):
                p.group_id, p.group_label = g, l
        return scene

    # ── 시각화 ──
    def draw(self, color_bgr: np.ndarray, result) -> np.ndarray:
        """검출/픽자세를 그린 BGR 이미지를 반환.

        result : detect() 의 list[Detection] 또는 process() 의 SceneResult.
          * Detection 리스트 → 박스 + 마스크 + 클래스/신뢰도
          * SceneResult → 마스크 + 픽 중심점 + 장단축 + 치수(+팔레트/완료 배너)
        """
        return draw_scene(color_bgr, result)
