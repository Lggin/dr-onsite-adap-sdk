"""[실행·GUI] 데스크톱 GUI 앱 (PySide6).

cv2 키보드 앱(dr-onsite-run)의 상위 버전. 사이드바 버튼·카메라 드롭다운·검출 목록
표·상태바를 갖춘 창에서 클릭으로 조작한다(처음 보는 사람용). 모델 로딩·카메라·검출은
백그라운드 스레드라 창이 멈추지 않는다.

  실행:  dr-onsite-gui     (PySide6 필요:  pip install "dr-onsite-adap[gui]")

설계:
  _Loader      모델/SAM 로딩(느림) → 끝나면 검출 버튼 활성화
  _CameraThread 카메라 read() 루프 → 프레임 시그널
  _DetectThread 'd'(검출) 시 run_detection 을 UI 밖에서 실행 → 결과 시그널
영상 위 오버레이는 detect.draw_state 를 그대로 재사용한다(cv2 앱과 동일 결과).
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

from .. import config
from .._logging import setup_cli_logging
from . import cameras
from .detect import build_detections_3d, draw_state, run_detection
from .smoothing import DetectionSmoother
from .geometry import estimate_up_dir
from .viz3d import show_snapshot
from .app import _apply_roi, _load_roi, _save_roi, _load_up, _save_up

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, QThread, Signal
except Exception as e:  # noqa: BLE001
    sys.exit(
        "PySide6 가 필요합니다. 설치:  pip install \"dr-onsite-adap[gui]\"\n"
        f"(원래 오류: {e})"
    )


# ───────────────────────── 백그라운드 스레드 ─────────────────────────
class _Loader(QThread):
    """검출 모델 + SAM 라벨러 로딩(수십 초). 끝나면 ready 로 넘긴다."""
    ready = Signal(object, object)   # (SegModel, SAMLabeler|None)
    failed = Signal(str)

    def run(self):
        try:
            from .inference import SegModel
            model = SegModel(config.WEIGHTS_PATH, config.NUM_CLASSES, config.DEVICE)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"모델 로드 실패: {e}")
            return
        labeler = None
        try:
            from .labeler import SAMLabeler
            labeler = SAMLabeler(
                config.SAM_CHECKPOINT, config.ACTIVE_DATA_DIR, config.CLASSES,
                sam_type=config.SAM_TYPE, device=config.SAM_DEVICE,
            )
        except Exception as e:  # noqa: BLE001
            print(f"SAM 라벨링 비활성화: {e}")
        self.ready.emit(model, labeler)


class _ReloadThread(QThread):
    """학습·병합으로 만든 새 가중치를 백그라운드로 로드(재시작 없이 교체)."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path):
        super().__init__()
        self._path = path

    def run(self):
        try:
            from .inference import SegModel
            m = SegModel(self._path, config.NUM_CLASSES, config.DEVICE)
            self.done.emit(m)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"가중치 로드 실패: {e}")


class _CameraThread(QThread):
    """카메라 read() 루프. 최신 프레임을 frame 시그널로 보낸다."""
    frame = Signal(object, object, object, object)   # color, depth, intr, up_dir
    error = Signal(str)

    def __init__(self, serial=""):
        super().__init__()
        self._serial = serial
        self._run = True

    def run(self):
        try:
            backend = cameras.resolve_backend(config.CAMERA_BACKEND)
            Cam = cameras.load_camera_class(backend)
            if self._serial and backend == "orbbec":
                cam = Cam(config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS,
                          use_imu=config.USE_IMU, device_serial=self._serial)
            else:
                cam = Cam(config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS,
                          use_imu=config.USE_IMU)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"카메라 열기 실패: {e}")
            return
        self._cam = cam
        try:
            while self._run:
                pack = cam.read()
                if pack is None:
                    self.msleep(5)
                    continue
                color, depth, intr, up = pack
                self.frame.emit(color, depth, intr, up)
        finally:
            try:
                cam.stop()
            except Exception:
                pass

    def stop(self):
        self._run = False
        self.wait(3000)


class _DetectThread(QThread):
    """run_detection 을 UI 스레드 밖에서 1회 실행."""
    done = Signal(object, object)   # (state, intr)
    failed = Signal(str)

    def __init__(self, model, color, depth, intr, up, with_pointcloud=True,
                 cluster=False):
        super().__init__()
        self._args = (model, color, depth, intr, up)
        self._wp = with_pointcloud
        self._cluster = cluster

    def run(self):
        model, color, depth, intr, up = self._args
        try:
            h, w = color.shape[:2]
            state = run_detection(model, color, depth, intr, up, h, w,
                                  with_pointcloud=self._wp, cluster=self._cluster)
            self.done.emit(state, intr)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"검출 실패: {e}")


# ───────────────────────── 영상 라벨(ROI 드래그) ─────────────────────────
class VideoLabel(QtWidgets.QLabel):
    """프레임을 표시하고, ROI 모드에서 드래그로 사각형을 고른다(이미지 좌표로 변환)."""
    roi_selected = Signal(tuple)   # (x, y, w, h) in image coords

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;")
        self._roi_mode = False
        self._origin = None
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._img_w = self._img_h = 0      # 현재 표시 중인 원본 프레임 크기
        self._draw_rect = None             # 라벨 안에서 픽스맵이 그려진 영역

    def set_image_size(self, w, h):
        self._img_w, self._img_h = w, h

    def set_roi_mode(self, on):
        self._roi_mode = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    # 픽스맵이 라벨 안에서 차지하는 실제 사각형(레터박스 고려)
    def _pixmap_rect(self):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        x = (lw - pw) // 2
        y = (lh - ph) // 2
        return QtCore.QRect(x, y, pw, ph)

    def mousePressEvent(self, ev):
        if not self._roi_mode:
            return
        self._origin = ev.position().toPoint()
        self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
        self._rubber.show()

    def mouseMoveEvent(self, ev):
        if self._roi_mode and self._origin is not None:
            self._rubber.setGeometry(
                QtCore.QRect(self._origin, ev.position().toPoint()).normalized())

    def mouseReleaseEvent(self, ev):
        if not self._roi_mode or self._origin is None:
            return
        self._rubber.hide()
        sel = QtCore.QRect(self._origin, ev.position().toPoint()).normalized()
        self._origin = None
        self.set_roi_mode(False)
        pr = self._pixmap_rect()
        if pr is None or self._img_w == 0 or sel.width() < 8 or sel.height() < 8:
            self.roi_selected.emit(())     # 취소/전체
            return
        # 라벨 좌표 → 픽스맵 내부 → 원본 이미지 좌표
        sx = self._img_w / pr.width()
        sy = self._img_h / pr.height()
        x = int(max(0, (sel.x() - pr.x()) * sx))
        y = int(max(0, (sel.y() - pr.y()) * sy))
        w = int(sel.width() * sx)
        h = int(sel.height() * sy)
        w = max(1, min(w, self._img_w - x))
        h = max(1, min(h, self._img_h - y))
        self.roi_selected.emit((x, y, w, h))


# ───────────────────────────── 메인 윈도우 ─────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dr-onsite-adap — 디팔레타이징 비전")
        self.resize(1180, 760)

        self.model = None
        self.labeler = None
        self.cam_thread = None
        self.detect_thread = None
        self.loader = None

        self.last_pack = None          # (color, depth, intr, up)  ROI 적용 후
        self.last_color_raw = None     # ROI 적용 후 원본(라벨링용)
        self.state = None
        self.state_intr = None
        self.state_up = None
        self.roi = _load_roi(config.ROI_PATH)
        self.measured_up = _load_up(config.UP_PATH)   # 측정된 중력벡터(있으면 우선)
        self._fps = 0.0
        self._last_t = time.time()
        self._busy = False
        self.live_mode = False               # 실시간 연속 검출 토글
        self.group_mode = False              # 클래스별 외형 그룹화 토글
        self.smoother = DetectionSmoother()   # 실시간 깜빡임 방지(시간축 안정화)

        self._build_ui()
        self._refresh_camera_list()

    # ── UI 구성 ──
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 사이드바
        side = QtWidgets.QVBoxLayout()
        side.setSpacing(8)
        root.addLayout(side, 0)

        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.setToolTip("사용할 카메라 선택")
        self.cam_combo.activated.connect(self._on_camera_changed)
        side.addWidget(QtWidgets.QLabel("카메라"))
        side.addWidget(self.cam_combo)
        side.addSpacing(10)

        self.btn_detect = QtWidgets.QPushButton("▶  검출")
        self.btn_detect.setShortcut("d")
        self.btn_detect.clicked.connect(self.do_detect)
        self.btn_live = QtWidgets.QPushButton("▶▶  실시간 검출")
        self.btn_live.setCheckable(True)
        self.btn_live.setShortcut("l")
        self.btn_live.setToolTip("매 프레임 연속 검출(깜빡임 방지 적용). 다시 누르면 끔.")
        self.btn_live.toggled.connect(self._toggle_live)
        self.btn_group = QtWidgets.QPushButton("🎨  클래스별 그룹화")
        self.btn_group.setCheckable(True)
        self.btn_group.setToolTip("검출(분류) 후 같은 클래스 안에서 색·질감·크기가 비슷한 것끼리 "
                                  "그룹색으로 묶어 표시(예: Carton#0, Carton#1). 다시 누르면 끔.")
        self.btn_group.toggled.connect(self._toggle_group)
        self.btn_3d = QtWidgets.QPushButton("🧊  3D 보기")
        self.btn_3d.clicked.connect(self.do_3d)
        self.btn_roi = QtWidgets.QPushButton("▭  ROI 지정")
        self.btn_roi.setCheckable(True)
        self.btn_roi.clicked.connect(self._on_roi_button)
        self.btn_roi_clear = QtWidgets.QPushButton("▢  ROI 해제")
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        self.btn_label = QtWidgets.QPushButton("✎  라벨링(SAM)")
        self.btn_label.clicked.connect(self.do_label)
        self.btn_clear = QtWidgets.QPushButton("🧹  오버레이 지우기")
        self.btn_clear.clicked.connect(self._clear_overlay)
        self.btn_up = QtWidgets.QPushButton("🧭  중력 재측정")
        self.btn_up.setToolTip("바닥/팔레트 평면으로 up_dir(중력 벡터) 재측정 — 카메라 각도 바뀌었을 때")
        self.btn_up.clicked.connect(self.do_measure_up)
        self.btn_reload = QtWidgets.QPushButton("♻  새 가중치 로드")
        self.btn_reload.setToolTip("학습·병합으로 만든 새 가중치(DEPAL_FINAL)를 재시작 없이 적용 — 학습 전/후 비교용")
        self.btn_reload.clicked.connect(self.do_reload)
        for b in (self.btn_detect, self.btn_live, self.btn_group, self.btn_3d, self.btn_roi,
                  self.btn_roi_clear, self.btn_label, self.btn_up, self.btn_reload,
                  self.btn_clear):
            b.setMinimumHeight(40)
            b.setEnabled(False)
            side.addWidget(b)
        side.addStretch(1)

        # 오른쪽: 탭 (실시간 인식 / 라벨 데이터 / 학습)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        # 탭1: 실시간 인식 (영상 + 검출 목록)
        live = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(live)
        self.video = VideoLabel()
        self.video.roi_selected.connect(self._on_roi_selected)
        lv.addWidget(self.video, 1)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "클래스", "그룹", "크기(mm)", "중심(cm)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(160)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lv.addWidget(self.table, 0)
        self.tabs.addTab(live, "실시간 인식")

        # 탭2: 라벨 데이터 (저장된 라벨 썸네일)
        self.tabs.addTab(self._build_label_tab(), "라벨 데이터")
        # 탭3: 학습 (파이프라인 실행 + 로그)
        self.tabs.addTab(self._build_train_tab(), "학습")

        self.status = self.statusBar()
        self._set_status("모델 로딩 중… 잠시만요")

    # ── 시작: 모델 로딩 + 카메라 ──
    def start(self):
        self.loader = _Loader()
        self.loader.ready.connect(self._on_model_ready)
        self.loader.failed.connect(lambda m: self._set_status("⚠ " + m))
        self.loader.start()
        self._start_camera(self._current_serial())

    def _on_model_ready(self, model, labeler):
        self.model = model
        self.labeler = labeler
        for b in (self.btn_detect, self.btn_live, self.btn_group, self.btn_3d, self.btn_roi,
                  self.btn_roi_clear, self.btn_up, self.btn_reload, self.btn_clear):
            b.setEnabled(True)
        self.btn_label.setEnabled(labeler is not None)
        self._set_status("준비 완료 — '검출' 또는 '실시간 검출'을 누르세요")

    # ── 카메라 ──
    def _refresh_camera_list(self):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        self.cam_combo.addItem("자동(첫 장치)", "")
        for sn in cameras.list_orbbec_serials():
            self.cam_combo.addItem(sn, sn)
        # 환경변수로 지정돼 있으면 그걸 선택
        want = config.CAMERA_SERIAL
        if want:
            i = self.cam_combo.findData(want)
            if i >= 0:
                self.cam_combo.setCurrentIndex(i)
        self.cam_combo.blockSignals(False)

    def _current_serial(self):
        return self.cam_combo.currentData() or ""

    def _start_camera(self, serial):
        if self.cam_thread is not None:
            self.cam_thread.frame.disconnect()
            self.cam_thread.stop()
            self.cam_thread = None
        self.cam_thread = _CameraThread(serial)
        self.cam_thread.frame.connect(self._on_frame)
        self.cam_thread.error.connect(lambda m: self._set_status("⚠ " + m))
        self.cam_thread.start()

    def _on_camera_changed(self, _idx):
        self.state = None
        self.smoother.reset()
        self._start_camera(self._current_serial())
        self._set_status(f"카메라 전환: {self._current_serial() or '자동'}")

    # ── 프레임 수신/표시 ──
    def _on_frame(self, color, depth, intr, up):
        color_r, depth_r, intr_r = _apply_roi(color, depth, intr, self.roi)
        up_dir = up if up is not None else (self.measured_up or config.UP_DIR_CAMERA_FRAME)
        self.last_pack = (color_r, depth_r, intr_r, up_dir)
        self.last_color_raw = color_r.copy()

        # 실시간 모드: 직전 검출이 끝났으면 이번 프레임으로 다음 검출을 건다(1개씩 연속).
        if self.live_mode and not self._busy and self.model is not None:
            self._kick_detect()

        now = time.time()
        dt = max(now - self._last_t, 1e-3)
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._last_t = now

        disp = color_r.copy()
        if self.state is not None:
            # 영상 위 텍스트 패널은 끔(정보는 아래 '검출 목록' 표에 있어 가림 방지).
            draw_state(disp, self.state, self.state_intr, show_panel=False)
        self._show(disp)
        self._update_statusline()

    def _show(self, bgr):
        h, w = bgr.shape[:2]
        self.video.set_image_size(w, h)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._qbuf = np.ascontiguousarray(rgb)   # GC 방지로 참조 보관
        qimg = QtGui.QImage(self._qbuf.data, w, h, 3 * w,
                            QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pm)

    # ── 검출 ──
    def _kick_detect(self):
        """현재 프레임으로 검출 1회를 백그라운드로 시작. 진행 중(busy)이면 무시."""
        if self.model is None or self.last_pack is None or self._busy:
            return
        self._busy = True
        color, depth, intr, up = self.last_pack
        # 라이브(연속)에선 3D 점구름 생략해 가속, 단발 검출에선 계산('v' 3D 보기용).
        self.detect_thread = _DetectThread(self.model, color.copy(), depth.copy(),
                                           intr, up, with_pointcloud=not self.live_mode,
                                           cluster=self.group_mode)
        self.detect_thread.done.connect(self._on_detect_done)
        self.detect_thread.failed.connect(self._on_detect_fail)
        self.detect_thread.start()

    def do_detect(self):
        if self.model is None or self.last_pack is None or self._busy:
            return
        self.btn_detect.setEnabled(False)
        self._set_status("검출 중…")
        self._kick_detect()

    def _toggle_live(self, checked):
        self.live_mode = checked
        self.smoother.reset()
        self.btn_live.setText("⏸  실시간 끄기" if checked else "▶▶  실시간 검출")
        if checked:
            self._set_status("● 실시간 검출 ON")
            self._kick_detect()          # 즉시 시작(이후 프레임마다 자동 연속)
        else:
            self._set_status("실시간 검출 OFF")

    def _toggle_group(self, checked):
        """클래스별 외형 그룹화 토글. 켜면 다음 검출부터 그룹색/라벨로 표시."""
        self.group_mode = checked
        self.smoother.reset()            # 색/라벨이 바뀌므로 트랙 초기화
        self.btn_group.setText("🎨  그룹화 끄기" if checked else "🎨  클래스별 그룹화")
        self._set_status("● 클래스별 그룹화 ON" if checked else "클래스별 그룹화 OFF")
        # 정지 상태에 직전 검출이 있으면 즉시 재검출해 반영(라이브면 다음 프레임에 자동 적용).
        if not self.live_mode and self.model is not None and self.last_pack is not None:
            self._kick_detect()

    def _on_detect_done(self, state, intr):
        # 실시간 모드면 시간축 안정화(깜빡임 방지) 적용.
        self.state = self.smoother.update(state) if self.live_mode else state
        self.state_intr = intr
        self.state_up = self.last_pack[3] if self.last_pack else None
        self._busy = False
        self.btn_detect.setEnabled(True)
        self._fill_table(self.state)
        if not self.live_mode:
            self._set_status("검출 완료")

    def _on_detect_fail(self, msg):
        self._busy = False
        self.btn_detect.setEnabled(True)
        self._set_status("⚠ " + msg)

    def _fill_table(self, state):
        rows = state.get("active", [])
        self.table.setRowCount(len(rows))
        for i, det in enumerate(rows):
            res = det["res"]
            x, y, z = res["center_3d"]
            vals = [str(i + 1), det["name"], det.get("group_label", "") or "-",
                    f"{res['long_side_mm']:.0f} x {res['short_side_mm']:.0f}",
                    f"({x*100:+.1f}, {y*100:+.1f}, {z*100:+.1f})"]
            for c, v in enumerate(vals):
                self.table.setItem(i, c, QtWidgets.QTableWidgetItem(v))

    # ── 3D ──
    def do_3d(self):
        if self.state is None:
            self._set_status("먼저 '검출'을 하세요")
            return
        try:
            show_snapshot(build_detections_3d(self.state), up_dir=self.state_up)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"⚠ 3D 보기 실패: {e}")

    # ── 중력 벡터(up_dir) 재측정 ──
    def do_measure_up(self):
        if self.last_pack is None:
            self._set_status("프레임이 아직 없습니다")
            return
        _, depth, intr, _ = self.last_pack
        up = estimate_up_dir(depth, intr, config)
        if up is None:
            self._set_status("⚠ 중력 재측정 실패 — 바닥/팔레트 평면이 충분히 보이게 한 뒤 다시")
            return
        import math
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -up[2]))))
        self.measured_up = up
        _save_up(config.UP_PATH, up)
        self.state = None     # up 이 바뀌었으니 재검출 필요
        self.smoother.reset()
        self._set_status(
            f"중력 재측정 완료: ({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f}) · "
            f"기울기 {tilt:.1f}° · 저장됨(다음 실행에도 적용)")

    # ── 새 가중치 로드 (학습 후 재시작 없이 교체) ──
    def do_reload(self):
        path = config.FINAL_WEIGHTS
        if not (path and os.path.isfile(path)):
            self._set_status(f"⚠ 새 가중치가 없습니다: {path} (학습·병합 먼저)")
            return
        # VRAM 스파이크 방지: 옛 모델을 먼저 비우면 실시간 검출은 로딩 동안 잠시 멈춤(자동 재개).
        self.model = None
        self.state = None
        self.smoother.reset()
        self.btn_reload.setEnabled(False)
        self._set_status(f"새 가중치 로딩 중… {os.path.basename(path)}")
        self._reload = _ReloadThread(path)
        self._reload.done.connect(self._on_reloaded)
        self._reload.failed.connect(self._on_reload_fail)
        self._reload.start()

    def _on_reloaded(self, model):
        self.model = model
        self.smoother.reset()
        self.btn_reload.setEnabled(True)
        self._set_status(f"✅ 새 가중치 적용됨 — {os.path.basename(config.FINAL_WEIGHTS)}")

    def _on_reload_fail(self, msg):
        self.btn_reload.setEnabled(True)
        self._set_status("⚠ " + msg)

    # ── ROI ──
    def _on_roi_button(self, checked):
        if checked:
            self.video.set_roi_mode(True)
            self._set_status("영상 위에서 드래그해 ROI 지정 (작게 그리면 취소)")
        else:
            self.video.set_roi_mode(False)

    def _on_roi_selected(self, roi):
        self.btn_roi.setChecked(False)
        self.roi = roi if roi else None
        _save_roi(config.ROI_PATH, self.roi)
        self.state = None
        self.smoother.reset()
        self._set_status(f"ROI: {self.roi if self.roi else '전체 화면'}")

    def _clear_roi(self):
        self.roi = None
        _save_roi(config.ROI_PATH, None)
        self.state = None
        self.smoother.reset()
        self._set_status("ROI 해제 (전체 화면)")

    # ── 라벨링 (cv2 라벨러 재사용; 카메라 잠시 멈춤) ──
    def do_label(self):
        if self.labeler is None or self.last_color_raw is None:
            return
        self._set_status("SAM 라벨링 창에서 작업하세요 (Enter 저장 / ESC 취소)")
        QtWidgets.QApplication.processEvents()
        try:
            self.labeler.label_frame(self.last_color_raw.copy())
            self._set_status("라벨링 종료")
        except Exception as e:  # noqa: BLE001
            self._set_status(f"⚠ 라벨링 실패: {e}")
        self._refresh_labels()
        self.tabs.setCurrentIndex(1)   # '라벨 데이터' 탭으로 전환해 결과 확인

    def _clear_overlay(self):
        self.state = None
        self.smoother.reset()
        self.table.setRowCount(0)
        self._set_status("오버레이 지움")

    # ── 상태바 ──
    def _set_status(self, msg):
        self._status_msg = msg
        self._update_statusline()

    def _update_statusline(self):
        cnt = ""
        if self.state is not None:
            c = self.state.get("counts", {})
            cnt = " | " + "  ".join(f"{config.CLASSES[k][0]}:{v}"
                                    for k, v in sorted(c.items()) if k in config.CLASSES)
            if self.state.get("done"):
                cnt += "  ▣ 빈 팔레트(DONE)"
        cam = self._current_serial() or "자동"
        msg = getattr(self, "_status_msg", "")
        self.status.showMessage(
            f"{msg}{cnt}   ·   {self._fps:.1f} FPS · cam {cam} · "
            f"{config.NUM_CLASSES}cls")

    # ── 라벨 데이터 탭 ──
    def _build_label_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.lbl_count = QtWidgets.QLabel("라벨 0개")
        btn_ref = QtWidgets.QPushButton("새로고침")
        btn_ref.clicked.connect(self._refresh_labels)
        top.addWidget(self.lbl_count)
        top.addStretch(1)
        top.addWidget(btn_ref)
        v.addLayout(top)
        self.lbl_scroll = QtWidgets.QScrollArea()
        self.lbl_scroll.setWidgetResizable(True)
        self.lbl_grid_host = QtWidgets.QWidget()
        self.lbl_grid = QtWidgets.QGridLayout(self.lbl_grid_host)
        self.lbl_scroll.setWidget(self.lbl_grid_host)
        v.addWidget(self.lbl_scroll, 1)
        self._refresh_labels()
        return w

    def _refresh_labels(self):
        while self.lbl_grid.count():
            it = self.lbl_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        d = config.ACTIVE_DATA_DIR
        files = sorted(glob.glob(os.path.join(d, "sam_hard_*.jpg")), reverse=True)
        self.lbl_count.setText(f"라벨 이미지 {len(files)}개   ·   {d}")
        cols = 4
        for i, fp in enumerate(files[:60]):    # 최근 60장
            pm = QtGui.QPixmap(fp)
            if pm.isNull():
                continue
            thumb = QtWidgets.QLabel()
            thumb.setPixmap(pm.scaled(190, 135, Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))
            thumb.setToolTip(os.path.basename(fp))
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("border:1px solid #444; padding:2px;")
            self.lbl_grid.addWidget(thumb, i // cols, i % cols)

    # ── 학습 탭 (파이프라인 서브프로세스 + 로그) ──
    def _build_train_tab(self):
        self.proc = None
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        warn = QtWidgets.QLabel(
            "⚠ 학습은 GPU를 점유합니다 — 진행 중엔 실시간 검출이 느려집니다.\n"
            "   경로는 환경변수(DEPAL_BASE_WEIGHTS/TRAIN_DATA/FINAL 등)로 지정됩니다.")
        warn.setStyleSheet("color:#e09000; font-weight:bold;")
        v.addWidget(warn)
        row = QtWidgets.QHBoxLayout()
        self.train_btns = {}
        steps = [("① 취합", "ingest", []),
                 ("② 리플레이", "replay", []),
                 ("③ 학습", "train_lora", []),
                 ("④ 병합", "merge", []),
                 ("▶ 전체 라운드", "round", ["--replay"])]
        for label, mod, args in steps:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(
                lambda _=False, m=mod, a=args: self._run_pipeline(m, a))
            self.train_btns[mod] = b
            row.addWidget(b)
        self.btn_stop = QtWidgets.QPushButton("■ 중지")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        row.addWidget(self.btn_stop)
        v.addLayout(row)
        self.train_log = QtWidgets.QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet(
            "background:#111; color:#ddd; font-family:monospace; font-size:12px;")
        v.addWidget(self.train_log, 1)
        return w

    def _run_pipeline(self, module, args):
        if self.proc is not None and self.proc.state() != QtCore.QProcess.NotRunning:
            self._append_log("⚠ 이미 다른 작업이 실행 중입니다.\n")
            return
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_ready)
        self.proc.finished.connect(self._on_proc_finished)
        cargs = ["-u", "-m", f"dr_onsite_adap.pipeline.{module}"] + list(args)
        self._append_log(f"\n$ {sys.executable} {' '.join(cargs)}\n")
        for b in self.train_btns.values():
            b.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.proc.start(sys.executable, cargs)

    def _append_log(self, text):
        self.train_log.moveCursor(QtGui.QTextCursor.End)
        self.train_log.insertPlainText(text)
        self.train_log.moveCursor(QtGui.QTextCursor.End)

    def _on_proc_ready(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._append_log(data)

    def _on_proc_finished(self, code, _status):
        self._append_log(f"\n[종료] exit={code}\n")
        for b in self.train_btns.values():
            b.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_labels()

    def _stop_training(self):
        if self.proc is not None and self.proc.state() != QtCore.QProcess.NotRunning:
            self.proc.kill()
            self._append_log("\n[사용자 중지]\n")

    def closeEvent(self, ev):
        if self.cam_thread is not None:
            self.cam_thread.stop()
        if getattr(self, "proc", None) is not None and \
                self.proc.state() != QtCore.QProcess.NotRunning:
            self.proc.kill()
        ev.accept()


def main():
    setup_cli_logging()
    if not (config.WEIGHTS_PATH and os.path.isfile(config.WEIGHTS_PATH)):
        print(f"⚠ 가중치 없음: {config.WEIGHTS_PATH} — 환경변수 DEPAL_WEIGHTS 확인")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
