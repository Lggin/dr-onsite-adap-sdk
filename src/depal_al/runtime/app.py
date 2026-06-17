"""Live RGB-D app: detect pick targets, estimate each top-face pick pose,
filter occlusions, and signal an empty pallet.

Keys:
  d  (re)run detection on the current frame
  v  open a 3D snapshot of the last detection in the browser
  c  clear the overlay
  r  set/clear an ROI crop (drag = set, ESC = full frame)
  t  SAM active-learning labeling on the current frame
  q  quit
"""
import json
import os
import sys
import time

import cv2

from .. import config
from .cameras import load_camera_class, resolve_backend
from .detect import build_detections_3d, draw_state, run_detection
from .inference import SegModel
from .labeler import SAMLabeler
from .viz import draw_hud, draw_styled_text
from .viz3d import show_snapshot


def _apply_roi(color, depth_m, intr, roi):
    """ROI=(x,y,w,h) 로 color/depth 를 크롭하고 intrinsics 주점을 보정.
    fx,fy 는 불변, cx,cy 만 크롭 원점만큼 이동 → 3D 역투영이 ROI 안에서도 정확.
    roi 가 None 이면 원본 그대로."""
    if roi is None:
        return color, depth_m, intr
    x, y, w, h = roi
    H, W = color.shape[:2]
    x = max(0, min(int(x), W - 1)); y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x)); h = max(1, min(int(h), H - y))
    fx, fy, cx, cy = intr
    return (color[y:y + h, x:x + w], depth_m[y:y + h, x:x + w],
            (fx, fy, cx - x, cy - y))


def _load_roi(path):
    """roi.json -> (x,y,w,h) 또는 None."""
    try:
        with open(path) as f:
            r = json.load(f)
        roi = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
        return roi if roi[2] > 0 and roi[3] > 0 else None
    except Exception:
        return None


def _save_roi(path, roi):
    with open(path, "w") as f:
        if roi is None:
            json.dump({}, f)
        else:
            x, y, w, h = roi
            json.dump({"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                      f, indent=2)


def main():
    if not os.path.isfile(config.WEIGHTS_PATH):
        sys.exit(
            f"Weights not found: {config.WEIGHTS_PATH}\n"
            f"Set DEPAL_WEIGHTS env var to the correct path on this machine."
        )
    backend = resolve_backend(config.CAMERA_BACKEND)
    CameraStream = load_camera_class(backend)
    model = SegModel(config.WEIGHTS_PATH, config.NUM_CLASSES, config.DEVICE)
    try:
        labeler = SAMLabeler(
            config.SAM_CHECKPOINT, config.ACTIVE_DATA_DIR, config.CLASSES,
            sam_type=config.SAM_TYPE, device=config.SAM_DEVICE,
        )
    except FileNotFoundError as e:
        print(f"SAM disabled: {e}")
        labeler = None

    camera = CameraStream(
        config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS,
        use_imu=config.USE_IMU,
    )

    print("Press 'd' to (re)detect, 'v' for 3D snapshot, "
          "'c' to clear overlay, 'q' to quit, 't' for SAM labeling, "
          "'r' to set/clear ROI (crop).")

    # ROI(팔레트 영역): 검출/표시/라벨저장을 이 영역 크롭으로 한정. roi.json 에 저장돼
    # 다음 실행에도 유지. 'r' 로 다시 잡거나(드래그) ESC 로 해제(전체 프레임).
    roi = _load_roi(config.ROI_PATH)
    print(f"ROI: {roi if roi else '없음(전체 프레임)'}  ('r' 로 설정/해제)")
    if camera.imu_enabled:
        print(f"IMU: enabled (accel @ {camera.accel_fps} Hz)")
    else:
        print("IMU: disabled (using config fallback)")

    # 창을 미리 만들어 좌상단·앞으로 띄운다. WINDOW_NORMAL 이라 화면보다 크면
    # 드래그로 줄일 수 있고, TOPMOST 로 편집기 등 다른 창 뒤에 깔리지 않게 한다.
    WIN = "depal_al — Top Face Center"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, roi[2] if roi else config.CAM_WIDTH,
                     roi[3] if roi else config.CAM_HEIGHT)
    cv2.moveWindow(WIN, 0, 0)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1.0)
    except Exception:
        pass

    # Manual detection trigger: detection only runs on first frame and on
    # 'd' key press. Camera display stays at full fps in between. After
    # the first auto-trigger fires, `auto_triggered` flips so 'c' (clear)
    # can wipe the overlay without it springing back next frame.
    detection_state = None
    state_intr = None        # intrinsics in effect when state was captured
    state_up_dir = None
    auto_triggered = False
    fps_ema = 0.0
    last_t = time.time()
    try:
        while True:
            frame_pack = camera.read()
            if frame_pack is None:
                continue
            color_full, depth_full, intr_full, imu_up = frame_pack
            # ROI 크롭 적용 — 이후 검출/표시/라벨저장 모두 이 크롭 기준으로 통일.
            color, depth_m, intr = _apply_roi(color_full, depth_full, intr_full, roi)
            up_dir = imu_up if imu_up is not None else config.UP_DIR_CAMERA_FRAME
            h_img, w_img = color.shape[:2]
            color_raw = color.copy()  # untouched (ROI) frame for SAM labeling

            t_now = time.time()
            dt = max(t_now - last_t, 1e-3)
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
            last_t = t_now

            # First-frame auto trigger so the user sees a result without
            # having to press anything. Only fires once — after that the
            # user owns the overlay state via 'd' / 'c'.
            if not auto_triggered:
                detection_state = run_detection(
                    model, color_raw, depth_m, intr, up_dir, h_img, w_img,
                )
                state_intr = intr
                state_up_dir = up_dir
                auto_triggered = True

            if detection_state is not None:
                draw_state(color, detection_state, state_intr)
                counts_for_hud = detection_state["counts"]
            else:
                counts_for_hud = {}

            draw_hud(color, fps_ema, counts_for_hud, config.CLASSES)
            src = "IMU" if imu_up is not None else "cfg"
            draw_styled_text(
                color,
                f"up[{src}]=({up_dir[0]:+.2f},{up_dir[1]:+.2f},{up_dir[2]:+.2f})",
                (10, h_img - 12), (200, 200, 255), scale=0.55, thickness=1,
            )
            if detection_state is not None:
                age = t_now - detection_state["detect_time"]
                inf_ms = detection_state["inference_time"] * 1000.0
                status_text = (
                    f"last detect: {age:4.1f}s ago (took {inf_ms:.0f}ms) "
                    f"— 'd' refresh  'v' 3D  'c' clear"
                )
            else:
                status_text = "no detection — 'd' to detect"
            draw_styled_text(
                color, status_text,
                (10, h_img - 32), (255, 220, 120), scale=0.55, thickness=1,
            )
            cv2.imshow(WIN, color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                detection_state = run_detection(
                    model, color_raw, depth_m, intr, up_dir, h_img, w_img,
                )
                state_intr = intr
                state_up_dir = up_dir
            if key == ord("v") and detection_state is not None:
                dets = build_detections_3d(detection_state)
                show_snapshot(dets, up_dir=state_up_dir)
            if key == ord("c"):
                detection_state = None
                state_intr = None
                state_up_dir = None
            if key == ord("r"):
                # 전체 프레임에서 ROI 드래그 선택. ENTER=확정, ESC/빈선택=해제(전체).
                sel_win = "ROI select  (drag, ENTER=ok, ESC=clear)"
                sel = cv2.selectROI(sel_win, color_full,
                                    showCrosshair=False, fromCenter=False)
                try:
                    cv2.destroyWindow(sel_win)
                except Exception:
                    pass
                if sel[2] > 0 and sel[3] > 0:
                    roi = tuple(int(v) for v in sel)
                    cv2.resizeWindow(WIN, roi[2], roi[3])
                    print(f"ROI set: {roi}")
                else:
                    roi = None
                    cv2.resizeWindow(WIN, config.CAM_WIDTH, config.CAM_HEIGHT)
                    print("ROI cleared (full frame)")
                _save_roi(config.ROI_PATH, roi)
                # 좌표계가 바뀌었으니 기존 오버레이 무효화 + 자동 재검출.
                detection_state = None
                state_intr = None
                state_up_dir = None
                auto_triggered = False
            if key == ord("t") and labeler is not None:
                labeler.label_frame(color_raw)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Shutdown.")


if __name__ == "__main__":
    main()
