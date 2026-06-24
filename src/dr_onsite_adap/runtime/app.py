"""[실행] 라이브 RGB-D 앱: 픽 대상을 검출하고 각 물체 윗면의 픽 좌표/자세를
추정하며, 가려짐을 거르고, 빈 팔레트를 알려줍니다.

키 안내:
  d  현재 화면에서 검출(다시) 실행
  l  실시간 검출 모드 켜기/끄기 (매 프레임 자동 검출)
  v  마지막 검출 결과를 브라우저 3D로 보기
  c  화면 오버레이 지우기
  r  관심영역(ROI) 크롭 지정/해제 (드래그=지정, ESC=전체 화면)
  t  현재 화면을 SAM으로 라벨링(액티브러닝 데이터 수집)
  g  현재 RGB-D 프레임 덤프(.npz: rgb/depth/intrinsics/up_dir/masks) — 기하 알고리즘 오프라인 개발용
  q  종료
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from .. import config
from .._logging import setup_cli_logging
from .cameras import load_camera_class, resolve_backend
from .detect import build_detections_3d, draw_state, run_detection
from .inference import SegModel
from .labeler import SAMLabeler
from .smoothing import DetectionSmoother
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


def _load_up(path):
    """up.json -> (x,y,z) 단위벡터 또는 None. GUI '중력 재측정'이 쓴 값."""
    try:
        with open(path) as f:
            u = json.load(f)
        v = (float(u["x"]), float(u["y"]), float(u["z"]))
        return v if abs(v[0]) + abs(v[1]) + abs(v[2]) > 1e-6 else None
    except Exception:
        return None


def _save_up(path, vec):
    with open(path, "w") as f:
        if vec is None:
            json.dump({}, f)
        else:
            json.dump({"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])},
                      f, indent=2)


def main():
    setup_cli_logging()
    if not os.path.isfile(config.WEIGHTS_PATH):
        sys.exit(
            f"가중치 파일을 찾을 수 없습니다: {config.WEIGHTS_PATH}\n"
            f"환경변수 DEPAL_WEIGHTS 에 이 PC의 실제 가중치 경로를 지정하세요."
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
        print(f"SAM 라벨링 비활성화: {e}")
        labeler = None

    camera = CameraStream(
        config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS,
        use_imu=config.USE_IMU,
    )

    print("키: 'd' 검출(다시), 'l' 실시간 검출 ON/OFF, 'v' 3D 보기, "
          "'c' 오버레이 지우기, 't' SAM 라벨링, 'r' 관심영역 지정/해제, 'q' 종료")

    # ROI(팔레트 영역): 검출/표시/라벨저장을 이 영역 크롭으로 한정. roi.json 에 저장돼
    # 다음 실행에도 유지. 'r' 로 다시 잡거나(드래그) ESC 로 해제(전체 프레임).
    roi = _load_roi(config.ROI_PATH)
    print(f"ROI: {roi if roi else '없음(전체 프레임)'}  ('r' 로 설정/해제)")
    # 측정된 up_dir(중력 벡터) 있으면 우선 사용(GUI '중력 재측정'이 저장). IMU>측정값>config.
    measured_up = _load_up(config.UP_PATH)
    if measured_up is not None:
        print(f"up_dir(측정값): {tuple(round(v, 3) for v in measured_up)}")
    if camera.imu_enabled:
        print(f"IMU: 사용 (가속도계 @ {camera.accel_fps} Hz)")
    else:
        print("IMU: 미사용 (config 의 고정 up_dir 사용)")

    # 창을 미리 만들어 좌상단·앞으로 띄운다. WINDOW_NORMAL 이라 화면보다 크면
    # 드래그로 줄일 수 있고, TOPMOST 로 편집기 등 다른 창 뒤에 깔리지 않게 한다.
    WIN = "dr_onsite_adap — Top Face Center"
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
    live_mode = False        # 'l' 토글: 매 프레임 자동 검출(실시간)
    smoother = DetectionSmoother()   # 실시간 모드 깜빡임 방지(시간축 안정화)
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
            up_dir = (imu_up if imu_up is not None
                      else (measured_up or config.UP_DIR_CAMERA_FRAME))
            h_img, w_img = color.shape[:2]
            color_raw = color.copy()  # untouched (ROI) frame for SAM labeling

            t_now = time.time()
            dt = max(t_now - last_t, 1e-3)
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
            last_t = t_now

            # First-frame auto trigger so the user sees a result without
            # having to press anything. Only fires once — after that the
            # user owns the overlay state via 'd' / 'c'.
            # 실시간 모드면 매 프레임 검출 + 시간축 안정화(임계 경계 깜빡임 방지).
            if live_mode:
                raw_state = run_detection(
                    model, color_raw, depth_m, intr, up_dir, h_img, w_img,
                    with_pointcloud=False,   # 점구름은 3D 보기용 — 라이브선 생략(가속)
                )
                detection_state = smoother.update(raw_state)
                state_intr = intr
                state_up_dir = up_dir
            elif not auto_triggered:
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
            if live_mode and detection_state is not None:
                inf_ms = detection_state["inference_time"] * 1000.0
                status_text = (
                    f"● 실시간 검출 ON  ({inf_ms:.0f}ms, {fps_ema:.0f}fps) "
                    f"— 'l' 끄기  'v' 3D"
                )
            elif detection_state is not None:
                age = t_now - detection_state["detect_time"]
                inf_ms = detection_state["inference_time"] * 1000.0
                status_text = (
                    f"마지막 검출: {age:4.1f}초 전 ({inf_ms:.0f}ms 소요) "
                    f"— 'd' 다시  'l' 실시간  'v' 3D  'c' 지우기"
                )
            else:
                status_text = "검출 없음 — 'd' 1회 검출  'l' 실시간 모드"
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
            if key == ord("l"):
                live_mode = not live_mode
                smoother.reset()        # 트랙 초기화(켤/끌 때마다 새로 시작)
                auto_triggered = True   # 첫프레임 자동검출 로직과 겹치지 않게
                print("실시간 검출: " + ("ON" if live_mode else "OFF"))
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
                    print(f"관심영역 지정됨: {roi}")
                else:
                    roi = None
                    cv2.resizeWindow(WIN, config.CAM_WIDTH, config.CAM_HEIGHT)
                    print("관심영역 해제 (전체 화면)")
                _save_roi(config.ROI_PATH, roi)
                # 좌표계가 바뀌었으니 기존 오버레이 무효화 + 자동 재검출.
                detection_state = None
                state_intr = None
                state_up_dir = None
                auto_triggered = False
                smoother.reset()   # ROI 변경=좌표계 변경 → 트랙 무효
            if key == ord("g"):
                # 기하(윗면 중심/축/길이) 알고리즘 오프라인 개발용 RGB-D 덤프.
                # 모델을 한 번 돌려 원본 마스크/클래스/신뢰도를 함께 저장(수동 키라 비용 무방).
                cap_dir = os.path.expanduser(
                    os.environ.get("DEPAL_CAPTURE_DIR", "~/geom_captures"))
                os.makedirs(cap_dir, exist_ok=True)
                g_preds = model.predict(color_raw, conf_threshold=config.CONF_THRES)
                g_masks = getattr(g_preds, "mask", None)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_npz = os.path.join(cap_dir, f"rgbd_{ts}.npz")
                np.savez_compressed(
                    out_npz,
                    rgb=color_raw,
                    depth=depth_m.astype(np.float32),
                    intrinsics=np.asarray(intr, dtype=np.float64),
                    up_dir=np.asarray(up_dir, dtype=np.float64),
                    masks=(np.zeros((0,) + depth_m.shape[:2], np.uint8)
                           if g_masks is None else g_masks.astype(np.uint8)),
                    class_ids=np.asarray(g_preds.class_id, dtype=np.int32),
                    confidences=np.asarray(g_preds.confidence, dtype=np.float32),
                )
                n_g = 0 if g_masks is None else len(g_masks)
                print(f"[g] RGB-D 캡처 저장: {out_npz}  (검출 {n_g}개)")
            if key == ord("t") and labeler is not None:
                labeler.label_frame(color_raw)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("종료되었습니다.")


if __name__ == "__main__":
    main()
