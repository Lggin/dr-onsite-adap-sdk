"""현장 프레임 조용히 모으기 — 평가셋(held-out) 촬영용.

현장 테스트 중에는 검출을 멈추거나 창을 띄울 수 없다. 그래서 이 모듈은 **이미 흘러가는
프레임을 저장만** 한다: 모델을 돌리지 않고, 창도 안 띄우고, 카메라도 새로 열지 않는다
(GUI/라이브 앱이 이미 들고 있는 프레임을 넘겨받는다).

두 가지를 자동으로 챙긴다.
  · **중복 제거** — 연속 프레임은 거의 같아서 평가셋 표본 수만 부풀린다("한 장면 1프레임").
    직전 저장분과 화면이 충분히 달라졌을 때만 저장한다(`min_diff`, 0 이면 끔).
  · **분리 보관** — 학습 폴더가 아니라 `DEPAL_FIELD_CAPTURE_DIR` 에 쌓는다. 라벨은 나중에
    `dr-onsite-label` 로 몰아서 하고, `dr-onsite-ingest --holdout` 으로 평가셋에 넣는다.

저장물: `cap_YYYYmmdd_HHMMSS_fff.jpg` (+ `DEPAL_CAPTURE_DEPTH=1` 이면 같은 이름 `.npz` 에
depth/intrinsics/up_dir). depth 는 기하 분석용이며 라벨링에는 필요 없다.
"""
import logging
import os
import time

import cv2
import numpy as np

from .. import config

logger = logging.getLogger(__name__)


class FieldCapture:
    """프레임을 받아 조건이 맞을 때만 파일로 떨어뜨린다(카메라 비소유)."""

    def __init__(self, out_dir=None, min_diff=None, save_depth=None):
        self.out_dir = os.path.expanduser(out_dir or config.FIELD_CAPTURE_DIR)
        self.min_diff = config.CAPTURE_MIN_DIFF if min_diff is None else float(min_diff)
        self.save_depth = config.CAPTURE_SAVE_DEPTH if save_depth is None else bool(save_depth)
        os.makedirs(self.out_dir, exist_ok=True)
        self._prev_small = None
        self.n_saved = 0
        self.n_skipped = 0

    @staticmethod
    def _small(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (80, 60), interpolation=cv2.INTER_AREA).astype(np.float32)

    def is_new_scene(self, color):
        """직전 저장분과 충분히 다른 장면인가(min_diff=0 이면 항상 True)."""
        if self.min_diff <= 0:
            return True
        small = self._small(color)
        if self._prev_small is None:
            return True
        return float(np.abs(small - self._prev_small).mean()) >= self.min_diff

    def save(self, color, depth=None, intr=None, up_dir=None, force=False):
        """조건이 맞으면 저장하고 경로를 돌려준다(건너뛰면 None)."""
        if color is None:
            return None
        if not force and not self.is_new_scene(color):
            self.n_skipped += 1
            return None
        # 같은 밀리초에 두 장이 들어와도 덮어쓰지 않도록 파일명 충돌을 피한다
        # (연사에서 조용히 프레임이 사라지는 게 제일 나쁘다).
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        path = os.path.join(self.out_dir, f"cap_{ts}.jpg")
        dup = 1
        while os.path.exists(path):
            path = os.path.join(self.out_dir, f"cap_{ts}-{dup}.jpg")
            dup += 1
        cv2.imwrite(path, color)
        if self.save_depth and depth is not None:
            np.savez_compressed(
                os.path.splitext(path)[0] + ".npz",
                depth=np.asarray(depth, dtype=np.float32),
                intrinsics=np.asarray(intr if intr is not None else [], dtype=np.float64),
                up_dir=np.asarray(up_dir if up_dir is not None else [], dtype=np.float64),
            )
        self._prev_small = self._small(color)
        self.n_saved += 1
        logger.info("촬영 %d장째 → %s", self.n_saved, path)
        return path

    def summary(self):
        return (f"촬영 {self.n_saved}장 (중복 건너뜀 {self.n_skipped}장) → {self.out_dir}\n"
                f"라벨링: dr-onsite-label   →   평가셋 적재: dr-onsite-ingest --holdout "
                f"--data-dir {self.out_dir}")


def main():
    """[독립 실행] 카메라를 직접 열어 일정 간격으로 촬영한다.

    ⚠️ 카메라는 한 프로세스만 잡을 수 있다. **GUI/라이브 앱이 떠 있으면 이걸 쓰지 말고**
    그쪽의 '📷 자동 촬영' 버튼(GUI) 이나 `p`/`P` 키(라이브 앱)를 쓰라 — 같은 폴더에 쌓인다.

    사용법:
      dr-onsite-capture                  # 3초 간격, Ctrl-C 로 종료
      dr-onsite-capture --interval 1.5 --max 200
      dr-onsite-capture --min-diff 0     # 중복 제거 끄기(전부 저장)
    """
    import argparse

    from .._logging import ensure_utf8_output, setup_cli_logging
    from .cameras import load_camera_class, resolve_backend

    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help=f"저장 폴더(기본 {config.FIELD_CAPTURE_DIR})")
    ap.add_argument("--interval", type=float, default=config.CAPTURE_INTERVAL_S,
                    help="촬영 간격(초)")
    ap.add_argument("--max", type=int, default=0, help="이 장수만큼 찍고 종료(0=무제한)")
    ap.add_argument("--min-diff", type=float, default=None,
                    help="직전 저장분과의 최소 화면 차이(0=중복 제거 끔)")
    ap.add_argument("--depth", action="store_true", help="depth 도 npz 로 함께 저장")
    ap.add_argument("--serial", default=None,
                    help="열 카메라 시리얼(여러 대일 때). --list-cameras 로 확인")
    ap.add_argument("--list-cameras", action="store_true", help="연결된 카메라 시리얼만 출력")
    ap.add_argument("--view", action="store_true",
                    help="창을 띄워 보면서 원할 때만 촬영(스페이스/p=1장, a=자동 토글, q=종료)")
    ap.add_argument("--roi", action="store_true",
                    help="ROI 사용(--view 전용). roi.json 을 읽어 적용하고 'r' 로 재설정, "
                         "'c' 로 해제. color/depth 크롭 + intrinsics 주점 보정까지 한다")
    args = ap.parse_args()
    ensure_utf8_output()
    setup_cli_logging()

    from .cameras import list_orbbec_serials

    if args.list_cameras:
        serials = list_orbbec_serials()
        print("연결된 Orbbec 카메라:", ", ".join(serials) if serials else "(없음)")
        return 0

    cap = FieldCapture(args.out, args.min_diff, args.depth or None)
    backend = resolve_backend(config.CAMERA_BACKEND)
    cls = load_camera_class(backend)
    # 생성 규약은 GUI/라이브 앱과 동일: (width, height, fps, use_imu=...).
    # 시리얼 지정은 orbbec 만 지원(여러 대 연결 상황이 orbbec 뿐).
    kw = {"use_imu": config.USE_IMU}
    if args.serial and backend == "orbbec":
        kw["device_serial"] = args.serial
    camera = cls(config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS, **kw)
    who = f"[{args.serial}] " if args.serial else ""
    print(f"📷 {who}{backend} 카메라 → {cap.out_dir}")

    if args.view:
        return _view_loop(camera, cap, args, who)

    print(f"   {args.interval}초 간격 자동 촬영. 중단: Ctrl-C")
    try:
        while True:
            pack = camera.read()
            if pack is not None:
                color, depth, intr, up = (list(pack) + [None] * 4)[:4]
                if cap.save(color, depth, intr, up) and args.max and cap.n_saved >= args.max:
                    break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    finally:
        _stop(camera)
    print(cap.summary())
    return 0


def _stop(camera):
    try:
        camera.stop()
    except Exception:  # noqa: BLE001
        pass


def _view_loop(camera, cap, args, who=""):
    """창을 띄워 보면서 원할 때만 찍는다 — GUI 가 다른 카메라를 쓰는 동안 두 번째 화면.

    검출은 하지 않는다(모델 미로드). 목적은 "다른 각도를 눈으로 보면서 원하는 순간에
    고른 카메라로 촬영"이다.

    ROI('r')는 라이브 앱과 **같은 규약**을 쓴다(app._apply_roi/_load_roi/_save_roi 재사용):
    color/depth 를 크롭하고 intrinsics 주점(cx,cy)을 크롭 원점만큼 이동시킨다. 이 보정이
    없으면 저장된 depth 로 물리 크기(mm)를 계산할 때 값이 틀어진다.
    """
    from .app import _apply_roi, _load_roi, _save_roi

    win = f"현장 촬영 {who}(space/p=1장, a=자동, r=ROI, c=ROI해제, q=종료)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    roi = _load_roi(config.ROI_PATH) if args.roi else None
    if roi:
        print(f"ROI: {roi} (roi.json 에서 로드) — 'r' 재설정 / 'c' 해제")
    elif args.roi:
        print("ROI: 없음(전체 프레임) — 'r' 로 지정")
    auto = False
    next_t = 0.0
    try:
        while True:
            pack = camera.read()
            if pack is None:
                continue
            color_f, depth_f, intr_f, up = (list(pack) + [None] * 4)[:4]
            # ROI 는 **저장 직전**에 적용한다 — 저장물이 곧 학습/평가 입력이므로.
            color, depth, intr = _apply_roi(color_f, depth_f, intr_f, roi)
            now = time.time()
            if auto and now >= next_t:
                next_t = now + args.interval
                cap.save(color, depth, intr, up)
            view = color.copy()
            txt = (f"{who}촬영 {cap.n_saved}장  (중복 건너뜀 {cap.n_skipped})"
                   f"{'  [자동 ON]' if auto else ''}"
                   f"{'  ROI ' + str(roi) if roi else ''}")
            cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(view, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if auto else (255, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord(" "), ord("p")):
                path = cap.save(color, depth, intr, up, force=True)
                print(f"📷 {cap.n_saved}장 저장 → {path}", flush=True)
            elif key == ord("a"):
                auto = not auto
                next_t = 0.0
                print(f"📷 자동 촬영 {'ON' if auto else 'OFF'} ({args.interval}초 간격)", flush=True)
            elif key == ord("r"):
                # 드래그로 영역 선택. 항상 **원본 프레임** 위에서 고른다(이미 ROI 가 걸린
                # 화면에서 고르면 좌표가 중첩돼 원본 기준이 아니게 된다).
                sel = cv2.selectROI(win, color_f, showCrosshair=False, fromCenter=False)
                if sel and sel[2] > 0 and sel[3] > 0:
                    roi = tuple(int(v) for v in sel)
                    _save_roi(config.ROI_PATH, roi)
                    print(f"ROI 지정됨: {roi} → {config.ROI_PATH}")
                else:
                    print("ROI 취소(선택 없음)")
            elif key == ord("c"):
                roi = None
                _save_roi(config.ROI_PATH, None)
                print("ROI 해제 — 전체 프레임으로 촬영합니다")
            elif key in (ord("q"), 27):
                break
            if args.max and cap.n_saved >= args.max:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(win)
        _stop(camera)
    print(cap.summary())
    return 0
