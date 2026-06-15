import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

import config
from geometry import estimate_top_face


def _count_realsense():
    try:
        import pyrealsense2 as rs
        return len(rs.context().query_devices())
    except Exception:
        return -1  # SDK missing or broken


def _count_orbbec():
    try:
        import pyorbbecsdk as obs
        return len(obs.Context().query_devices())
    except Exception:
        return -1


def _resolve_backend(name):
    """Resolve config.CAMERA_BACKEND ('auto'|'realsense'|'orbbec') to a concrete
    backend with a known-present device. Exits if nothing works."""
    if name == "auto":
        n_rs = _count_realsense()
        n_ob = _count_orbbec()
        if n_rs > 0:
            print(f"Auto-detected camera: realsense ({n_rs} device)")
            return "realsense"
        if n_ob > 0:
            print(f"Auto-detected camera: orbbec ({n_ob} device)")
            return "orbbec"
        sys.exit("No connected RealSense or Orbbec camera found.")
    elif name in ("realsense", "orbbec"):
        n = _count_realsense() if name == "realsense" else _count_orbbec()
        if n < 0:
            sys.exit(f"CAMERA_BACKEND='{name}' but the SDK failed to load.")
        if n == 0:
            sys.exit(
                f"CAMERA_BACKEND='{name}' but no matching camera detected. "
                f"Set CAMERA_BACKEND='auto' to pick whichever is connected."
            )
        return name
    else:
        sys.exit(f"Unknown CAMERA_BACKEND: {name!r}")


def _load_camera_class(backend):
    if backend == "realsense":
        from camera import RealSenseStream
        return RealSenseStream
    if backend == "orbbec":
        from camera_orbbec import FemtoBoltStream
        return FemtoBoltStream
    raise ValueError(backend)


from inference import SegModel
from labeler import SAMLabeler
from viz import (
    draw_fail_marker,
    draw_hud,
    draw_info_panel,
    draw_numbered_badge,
    draw_orientation_axes,
    draw_styled_text,
    draw_top_center,
    draw_verify_flag,
    overlay_mask,
)
from viz3d import show_snapshot


def compute_occlusion_by_mask(collected, cfg):
    """Flag detections whose mask is too small relative to its own
    convex hull — i.e. enough of the silhouette is missing that the
    visible region almost certainly doesn't span the full top face.

    Single-mask test: we don't look at other detections to identify the
    occluder. Adjacent boxes with jittery mask boundaries used to trigger
    false positives in the cross-mask version; comparing each mask only
    to its own hull removes that failure mode entirely.

    Returns the set of indices (into `collected`) that should be hidden.
    """
    occluded = set()
    for j, det in enumerate(collected):
        mask = det["mask"]
        mask_area = int(mask.sum())
        if mask_area == 0:
            continue
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < 1.0:
            continue
        hidden_frac = 1.0 - (mask_area / hull_area)
        if hidden_frac > cfg.OCCLUSION_HIDDEN_FRACTION:
            occluded.add(j)
    return occluded


def run_detection(model, color, depth_m, intr, up_dir, h_img, w_img,
                  verifier=None):
    """Run the model + top-face estimation + mask-based occlusion once.

    Returns a state dict that the main loop overlays onto every frame
    until the next detection trigger. Heavy work (segmentation, 3D
    plane projection) lives entirely here; the loop just draws.

    If `verifier` is given, each kept detection also gets a DINOv2
    second-opinion verdict (flag-only) stored under det["verify"].
    """
    t0 = time.time()
    preds = model.predict(color, conf_threshold=config.CONF_THRES)
    inference_time = time.time() - t0

    masks = getattr(preds, "mask", None)
    class_ids = preds.class_id
    confidences = preds.confidence
    n = 0 if masks is None else len(masks)

    masks_with_color = []   # (mask, bgr) — drawn every frame as overlay
    collected = []          # detections that passed estimate_top_face
    held = []               # 검증기 'hold' override 로 픽에서 제외된 검출(불확실)
    fail_markers = []       # (image_pos, reason)
    pallets = []            # 팔레트 검출 (픽 대상 아님 — 존재/위치만)
    item_count = 0          # 픽 대상(박스/토트/랩) 검출 수 (빈 팔레트 판정용)
    counts = defaultdict(int)

    for i in range(n):
        cid = int(class_ids[i])
        if cid not in config.CLASSES:
            continue
        name, color_bgr = config.CLASSES[cid]
        mask = masks[i]
        masks_with_color.append((mask, color_bgr))

        # 팔레트: 픽 대상이 아니므로 top-face 추정/occlusion 필터를 건너뛴다.
        # (팔레트는 늘 물품에 가려져 occlusion 필터에 항상 걸려 버려질 것.)
        # 존재 여부 + 중심만 기록해 '빈 팔레트=완료' 신호와 위치 표시에 쓴다.
        if cid == config.PALLET_CID:
            M = cv2.moments(mask.astype(np.uint8))
            cen = ((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                   if M["m00"] > 0 else None)
            pallets.append({"mask": mask, "confidence": float(confidences[i]),
                            "center_2d": cen})
            counts[cid] += 1   # HUD 표시용
            continue

        item_count += 1   # 픽 대상 검출 (픽 성공 여부와 무관 — 남은 물품 카운트)

        res, reason = estimate_top_face(
            mask, depth_m, intr, config, up_dir=up_dir,
        )
        if res is None:
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] > 0:
                fail_markers.append((
                    (int(M["m10"] / M["m00"]),
                     int(M["m01"] / M["m00"])),
                    reason,
                ))
            continue
        u, v = res["center_2d"]
        if not (0 <= u < w_img and 0 <= v < h_img):
            fail_markers.append(((u, v), "off-screen"))
            continue

        # DINOv2 독립 검증. RFDETR 예측 마스크로 만든 크롭을 임베딩 뱅크와 비교.
        verdict = None
        if verifier is not None:
            try:
                verdict = verifier.verify(color, mask, cid)
            except Exception as e:
                print(f"[verify] skip det {i}: {e}")

        # 검증기 행동(config.VERIFY_ACTION): "flag"=표시만(검출 유지),
        # "reclassify"=확신 불일치면 DINO 라벨로 교체, "hold"=불확실하니 픽 제외.
        # flag 외 모드는 충분히 확신할 때(score>=VERIFY_OVERRIDE_MIN_SCORE)만 작동.
        ov_cid, ov_name, ov_color, action = cid, name, color_bgr, None
        act = getattr(config, "VERIFY_ACTION", "flag")
        if (verdict is not None and not verdict["agree"] and act != "flag"
                and verdict["score"] >= getattr(config, "VERIFY_OVERRIDE_MIN_SCORE", 0.7)):
            if act == "reclassify" and verdict["pred_cid"] in config.PICKABLE_CIDS:
                ov_cid = verdict["pred_cid"]
                ov_name, ov_color = config.CLASSES[ov_cid]
                action = "reclassify"
            elif act == "hold":
                action = "hold"

        counts[ov_cid] += 1
        rec = {
            "orig_idx": i,
            "cid": ov_cid,
            "name": ov_name,
            "color_bgr": ov_color,
            "mask": mask,
            "confidence": float(confidences[i]),
            "res": res,
            "verify": verdict,
            "override": action,   # None | "reclassify" | "hold"
            "rf_name": name,      # 원래 RFDETR 라벨(override 표시용)
        }
        if action == "hold":
            held.append(rec)      # 픽 대상에서 제외(여전히 item_count 엔 포함=물품은 존재)
        else:
            collected.append(rec)

    # Mask-based occlusion (replaces the old depth-based pair-wise check)
    occluded_idx = compute_occlusion_by_mask(collected, config)
    for oi in occluded_idx:
        u_oc, v_oc = collected[oi]["res"]["center_2d"]
        fail_markers.append(((u_oc, v_oc), "occluded"))
    active = [c for k, c in enumerate(collected) if k not in occluded_idx]

    # 빈 팔레트 = 완료 신호: 팔레트는 보이는데 픽 대상이 하나도 없음.
    done = bool(pallets) and item_count == 0

    return {
        "masks_with_color": masks_with_color,
        "active": active,
        "held": held,
        "fail_markers": fail_markers,
        "pallets": pallets,
        "item_count": item_count,
        "done": done,
        "counts": dict(counts),
        "detect_time": time.time(),
        "inference_time": inference_time,
        "frame": color.copy(),   # 's' 저장 키가 나중에 크롭을 다시 뜨기 위해 보관
    }


def draw_state(img, state, intr):
    """Overlay a stored detection state onto the current camera frame."""
    h_img, w_img = img.shape[:2]

    for mask, color_bgr in state["masks_with_color"]:
        overlay_mask(img, mask, color_bgr)

    panel_entries = []
    for badge_n, det in enumerate(state["active"], start=1):
        res = det["res"]
        u, v = res["center_2d"]
        x3, y3, z3 = res["center_3d"]

        draw_top_center(img, (u, v))
        draw_orientation_axes(
            img, res["center_3d"],
            res["long_axis_3d"], res["short_axis_3d"],
            res["long_side_mm"], res["short_side_mm"], intr,
        )

        badge_pos = (u + 18, v - 18)
        badge_pos = (max(15, min(badge_pos[0], w_img - 15)),
                     max(15, min(badge_pos[1], h_img - 15)))
        draw_numbered_badge(img, badge_pos, badge_n, det["color_bgr"])

        label = (
            f"[{badge_n}] {det['name']} "
            f"({x3*100:+5.1f},{y3*100:+5.1f},{z3*100:+5.1f})cm "
            f"{res['long_side_mm']:.0f}x{res['short_side_mm']:.0f}mm"
        )
        entry_color = det["color_bgr"]
        ver = det.get("verify")
        # 마진 게이트: 충분히 확신하는 불일치(score ≥ 임계)만 표시 → 약한 false-hold 억제
        if (ver is not None and not ver["agree"]
                and ver["score"] >= config.VERIFY_FLAG_MIN_SCORE):
            label += f"  !DINO={ver['pred_name']}({ver['score']:.0%})"
            entry_color = (0, 0, 255)
            draw_verify_flag(img, badge_pos, ver["pred_name"])
        if det.get("override") == "reclassify":
            label += f"  [DINO재분류 <- rf:{det['rf_name']}]"
            entry_color = (0, 200, 255)
        panel_entries.append((label, entry_color))

    # 검증기 'hold' override 로 픽에서 제외된 검출 — 노란색으로 별도 표시(불확실).
    for det in state.get("held", []):
        u, v = det["res"]["center_2d"]
        ver = det.get("verify")
        dino = ver["pred_name"] if ver else "?"
        cv2.circle(img, (u, v), 7, (0, 215, 255), 2)
        draw_styled_text(img, f"HELD {det['rf_name']}->{dino}",
                         (max(5, u - 40), max(20, v - 10)),
                         (0, 215, 255), scale=0.55, thickness=2)
        panel_entries.append(
            (f"[HELD] rf={det['rf_name']} dino={dino} (불확실-픽제외)", (0, 215, 255)))

    for pos, reason in state["fail_markers"]:
        draw_fail_marker(img, pos, reason)

    draw_info_panel(img, panel_entries)

    # 팔레트 위치 표시 (픽 대상 아님 — 마스크는 위에서 이미 오버레이됨)
    for p in state.get("pallets", []):
        if p["center_2d"] is not None:
            px, py = p["center_2d"]
            draw_styled_text(img, "Pallet", (max(5, px - 25), max(20, py)),
                             (130, 100, 50), scale=0.6, thickness=2)

    # 빈 팔레트 = 완료 신호 배너 (팔레트 보이는데 픽 대상 0)
    if state.get("done"):
        banner = "PALLET EMPTY - DONE"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cx = max(10, (w_img - tw) // 2)
        cv2.rectangle(img, (cx - 15, 18), (cx + tw + 15, 18 + th + 22),
                      (0, 130, 0), -1)
        cv2.putText(img, banner, (cx, 18 + th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3)


def build_detections_3d(state):
    """Pack the active detections into the format show_snapshot expects."""
    out = []
    for det in state["active"]:
        res = det["res"]
        bgr = det["color_bgr"]
        rgb01 = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32) / 255.0
        out.append({
            "label": det["name"],
            "near_pts": res["near_pts"],
            "inlier_pts": res["inlier_pts"],
            "outside_pts": res["outside_pts"],
            "center_3d": res["center_3d"],
            "long_axis_3d": res["long_axis_3d"],
            "short_axis_3d": res["short_axis_3d"],
            "long_len_mm": res["long_side_mm"],
            "short_len_mm": res["short_side_mm"],
            "color_rgb": rgb01,
        })
    return out


def save_verify_cases(state, out_dir):
    """현재 검출 상태의 active 검출들을 크롭 + verdict 로 덤프.
    파일명에 RFDETR/DINO 의견과 점수를 박아, false-hold(맞는데 막힘) / catch
    (오분류 잡음) 케이스를 폴더에서 바로 분류·검토할 수 있게 한다.
    마진 분포로 VERIFY_FLAG_MIN_SCORE 보정 + 현장 뱅크 보강 데이터 수집용."""
    import json
    from verifier import masked_crop_from_mask
    frame = state.get("frame")
    if frame is None:
        print("[save] no frame in state")
        return
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    saved = 0
    with open(os.path.join(out_dir, "cases.jsonl"), "a") as mf:
        for k, det in enumerate(state["active"]):
            crop = masked_crop_from_mask(frame, det["mask"])
            if crop is None:
                continue
            ver = det.get("verify")
            dino = ver["pred_name"] if ver else "NA"
            score = ver["score"] if ver else 0.0
            tag = "MISMATCH" if (ver and not ver["agree"]) else "agree"
            fn = f"{ts}_{k}_{tag}_rf-{det['name']}_dino-{dino}_{score:.2f}.jpg"
            cv2.imwrite(os.path.join(out_dir, fn),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            mf.write(json.dumps({
                "ts": ts, "idx": k, "crop": fn,
                "rfdetr": det["name"], "dino": dino,
                "agree": (ver["agree"] if ver else None),
                "score": (ver["score"] if ver else None),
                "margin": (ver["margin"] if ver else None),
                "rf_conf": det["confidence"],
            }, ensure_ascii=False) + "\n")
            saved += 1
    print(f"[save] {saved} cases -> {out_dir}")


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
        import json
        with open(path) as f:
            r = json.load(f)
        roi = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
        return roi if roi[2] > 0 and roi[3] > 0 else None
    except Exception:
        return None


def _save_roi(path, roi):
    import json
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
            f"Set LEE4_WEIGHTS env var to the correct path on this machine."
        )
    backend = _resolve_backend(config.CAMERA_BACKEND)
    CameraStream = _load_camera_class(backend)
    model = SegModel(config.WEIGHTS_PATH, config.NUM_CLASSES, config.DEVICE)
    try:
        labeler = SAMLabeler(
            config.SAM_CHECKPOINT, config.ACTIVE_DATA_DIR, config.CLASSES,
            device=config.SAM_DEVICE,
        )
    except FileNotFoundError as e:
        print(f"SAM disabled: {e}")
        labeler = None

    verifier = None
    if config.VERIFY_ENABLED:
        try:
            from verifier import Verifier
            verifier = Verifier(
                config.VERIFY_BANK_PATH, config.VERIFY_MODEL,
                config.VERIFY_IMG_SIZE, device=config.VERIFY_DEVICE,
                k=config.VERIFY_K, dino_weights=config.VERIFY_DINO_WEIGHTS,
            )
        except Exception as e:
            print(f"Verifier disabled: {e}")
            verifier = None

    camera = CameraStream(
        config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS,
        use_imu=config.USE_IMU,
    )

    print("Press 'd' to (re)detect, 'v' for 3D snapshot, "
          "'c' to clear overlay, 'q' to quit, 't' for SAM labeling, "
          "'s' to save verify cases, 'r' to set/clear ROI (crop).")

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
    # (메인 창은 키 입력만 받고 마우스 콜백이 없어 스케일링해도 안전.)
    WIN = "lee4 — Top Face Center"
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
            # color_full 은 'r' ROI 선택용으로 보관.
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
                    verifier=verifier,
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
                    verifier=verifier,
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
                # 좌표계가 바뀌었으니 기존 오버레이 무효화 + 자동 재검출(auto_triggered
                # 를 풀면 다음 프레임에서 새 ROI 로 검출이 한 번 다시 돈다).
                detection_state = None
                state_intr = None
                state_up_dir = None
                auto_triggered = False
            if key == ord("t") and labeler is not None:
                labeler.label_frame(color_raw)
            if key == ord("s") and detection_state is not None:
                save_verify_cases(detection_state, config.VERIFY_LOG_DIR)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Shutdown.")


if __name__ == "__main__":
    main()
