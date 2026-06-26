"""OnsiteSession — 고수준 SDK 진입점 (UI 비종속 오케스트레이터).

UI 를 올리는 팀을 위한 high-level API. SDK 가 **카메라 + 추론 루프를 소유**하고
백그라운드 스레드에서 돌리며, 결과를 **콜백(push)** 과 **폴링(pull)** 양쪽으로 제공한다.
UI 팀은 검출/픽자세 파이프라인을 다시 배선하지 않고 이벤트·제어 메서드에 바인딩만 하면 된다.
(Qt/웹/ROS 무엇과도 무관 — cv2.imshow/PySide 의존 없음.)

    from dr_onsite_adap import OnsiteSession

    sess = OnsiteSession(weights="continual_master_v10.pth", geom_algo="v2")
    sess.on_result(lambda scene: ...)   # push: 프레임마다 SceneResult
    sess.start()                         # 카메라+추론 스레드 시작
    ...
    scene = sess.latest()                # pull: 스레드세이프 최신 결과
    sess.set_roi(x, y, w, h); sess.measure_gravity(); sess.load_weights(path)
    sess.set_live(True); sess.set_threshold(0.5); sess.set_geometry("v2")
    sess.stop()                          # 또는 with OnsiteSession(...) as sess:

범위: 런타임 검출/픽자세 + 제어(ROI/중력/핫스왑/임계/기하/실시간). 학습 오케스트레이션은 추후.
"""
import json
import threading
import time

import cv2
import numpy as np

from .. import config
from ..api import _scene_from_state, draw_scene
from ..exceptions import CameraError
from .cameras import load_camera_class, resolve_backend
from .detect import run_detection
from .geometry import estimate_up_dir
from .inference import SegModel
from .smoothing import DetectionSmoother


# ── 작은 헬퍼 (ROI/up_dir) — app.py(=SAM 의존) 임포트 피하려 인라인 ──
def _load_roi(path):
    try:
        with open(path) as f:
            r = json.load(f)
        return (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) if r else None
    except Exception:
        return None


def _save_roi(path, roi):
    try:
        with open(path, "w") as f:
            json.dump({} if not roi else
                      {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]}, f)
    except Exception:
        pass


def _load_up(path):
    try:
        with open(path) as f:
            u = json.load(f)
        v = (float(u["x"]), float(u["y"]), float(u["z"]))
        return v if abs(v[0]) + abs(v[1]) + abs(v[2]) > 1e-6 else None
    except Exception:
        return None


def _save_up(path, vec):
    try:
        with open(path, "w") as f:
            json.dump({"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])},
                      f, indent=2)
    except Exception:
        pass


def _apply_roi(color, depth, intr, roi):
    if not roi:
        return color, depth, intr
    x, y, w, h = roi
    fx, fy, cx, cy = intr
    return (color[y:y + h, x:x + w], depth[y:y + h, x:x + w], (fx, fy, cx - x, cy - y))


class ReplaySource:
    """테스트/오프라인용 프레임 소스 — 'g' 캡처 npz 를 반복 재생(카메라 대체).
    카메라와 동일한 read() 계약: (color_bgr, depth_m, intrinsics, up_dir|None) 반환."""

    def __init__(self, npz_paths, loop=True, fps=15):
        self.paths = list(npz_paths)
        self.loop = loop
        self.i = 0
        self.dt = 1.0 / max(fps, 1)
        self.imu_enabled = False
        self.accel_fps = None

    def read(self):
        if not self.paths:
            return None
        if self.i >= len(self.paths):
            if not self.loop:
                return None
            self.i = 0
        d = np.load(self.paths[self.i])
        self.i += 1
        time.sleep(self.dt)
        bgr = cv2.cvtColor(d["rgb"], cv2.COLOR_RGB2BGR)
        depth = d["depth"].astype(float)
        intr = tuple(float(v) for v in d["intrinsics"])
        up = tuple(float(v) for v in d["up_dir"]) if "up_dir" in d else None
        return bgr, depth, intr, up

    def stop(self):
        pass


class OnsiteSession:
    """카메라+추론을 소유하는 고수준 세션. 자세한 사용법은 모듈 docstring 참고."""

    def __init__(self, weights=None, num_classes=None, device=None, geom_algo=None,
                 conf_threshold=None, camera_backend=None, source=None, live=True,
                 roi_path=None, up_path=None):
        if geom_algo:
            config.GEOM_ALGO = str(geom_algo).lower()
        self._device = device or config.DEVICE
        self._num_classes = num_classes if num_classes is not None else config.NUM_CLASSES
        self._model = SegModel(weights or config.WEIGHTS_PATH, self._num_classes, self._device)
        self._source = source
        self._owns_source = source is None
        self._camera_backend = camera_backend or config.CAMERA_BACKEND
        self._roi_path = roi_path or config.ROI_PATH
        self._up_path = up_path or config.UP_PATH
        self._live = bool(live)
        self._thr = conf_threshold if conf_threshold is not None else config.CONF_THRES
        self._roi = _load_roi(self._roi_path)
        self._measured_up = _load_up(self._up_path)
        self._smoother = DetectionSmoother()
        self._latest = None
        self._last_color = None
        self._last_depth = None
        self._last_intr = None
        self._fps = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._trigger = threading.Event()
        self._thread = None
        self._cb = {"result": [], "status": [], "error": []}

    # ── 이벤트 (push) ──
    def on_result(self, fn):
        """프레임마다 SceneResult 를 받는 콜백 등록. 데코레이터로도 사용 가능."""
        self._cb["result"].append(fn); return fn

    def on_status(self, fn):
        """{'fps': float} 상태 콜백."""
        self._cb["status"].append(fn); return fn

    def on_error(self, fn):
        """루프 예외(Exception) 콜백 — 한 프레임 실패해도 루프는 계속."""
        self._cb["error"].append(fn); return fn

    def _emit(self, kind, *a):
        for fn in self._cb[kind]:
            try:
                fn(*a)
            except Exception:
                pass

    # ── 폴링 (pull) ──
    def latest(self):
        """가장 최근 SceneResult(없으면 None) — 스레드세이프."""
        with self._lock:
            return self._latest

    @property
    def fps(self):
        with self._lock:
            return self._fps

    def draw_latest(self):
        """최신 결과를 최신 프레임에 그린 BGR 이미지(없으면 None)."""
        with self._lock:
            c, s = self._last_color, self._latest
        return None if (c is None or s is None) else draw_scene(c, s)

    # ── 생명주기 ──
    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        if self._source is None:
            try:
                Cam = load_camera_class(resolve_backend(self._camera_backend))
                self._source = Cam(config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS,
                                   use_imu=config.USE_IMU)
            except Exception as e:
                raise CameraError(f"카메라 시작 실패: {e}", error_code="E1001") from e
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="OnsiteSession", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._owns_source and self._source is not None:
            try:
                self._source.stop()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    # ── 제어 ──
    def set_live(self, on):
        """실시간 모드 on/off. off 면 trigger() 호출 시에만 1회 검출."""
        self._live = bool(on)
        self._smoother.reset()

    def trigger(self):
        """비실시간 모드에서 단발 검출 1회 요청."""
        self._trigger.set()

    def set_threshold(self, t):
        self._thr = float(t)

    def set_geometry(self, algo):
        """윗면 기하 알고리즘 'v1'/'v2'."""
        config.GEOM_ALGO = str(algo).lower()

    def set_roi(self, x, y, w, h):
        self._roi = (int(x), int(y), int(w), int(h))
        _save_roi(self._roi_path, self._roi)
        self._smoother.reset()

    def clear_roi(self):
        self._roi = None
        _save_roi(self._roi_path, None)
        self._smoother.reset()

    def measure_gravity(self):
        """현재 프레임의 depth 로 중력(up) 재측정·저장. 성공 시 up 튜플, 실패 시 None."""
        with self._lock:
            d, intr = self._last_depth, self._last_intr
        if d is None:
            return None
        up = estimate_up_dir(d, intr, config)
        if up is not None:
            self._measured_up = up
            _save_up(self._up_path, up)
            self._smoother.reset()
        return up

    def load_weights(self, path, num_classes=None):
        """재시작 없이 검출 가중치 핫스왑."""
        m = SegModel(path, num_classes if num_classes is not None else self._num_classes,
                     self._device)
        with self._lock:
            self._model = m

    # ── 내부 루프 ──
    def _loop(self):
        last = time.time()
        while not self._stop.is_set():
            try:
                pack = self._source.read()
            except Exception as e:
                self._emit("error", e)
                time.sleep(0.05)
                continue
            if pack is None:
                time.sleep(0.005)
                continue
            color_full, depth_full, intr_full, imu_up = pack
            color, depth_m, intr = _apply_roi(color_full, depth_full, intr_full, self._roi)
            up = imu_up if imu_up is not None else (self._measured_up or config.UP_DIR_CAMERA_FRAME)
            with self._lock:
                self._last_depth, self._last_intr = depth_m, intr

            if not (self._live or self._trigger.is_set()):
                time.sleep(0.005)
                continue
            self._trigger.clear()
            with self._lock:
                model = self._model
            try:
                h, w = color.shape[:2]
                state = run_detection(model, color, depth_m, intr, up, h, w,
                                      with_pointcloud=False, conf_threshold=self._thr)
                if self._live:
                    state = self._smoother.update(state)
                scene = _scene_from_state(state, intr)
            except Exception as e:
                self._emit("error", e)
                continue
            now = time.time()
            dt = max(now - last, 1e-3)
            last = now
            with self._lock:
                self._latest = scene
                self._last_color = color
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._emit("result", scene)
            self._emit("status", {"fps": self.fps})
