"""Detection state: run the model once, estimate each pick target's top-face
pick pose, filter occlusions, and overlay the result.

The model runs only on a trigger ('d' key / first frame); the live loop just
redraws the stored state. Heavy work (segmentation, 3D plane projection) lives
in run_detection(); draw_state() is cheap per-frame overlay.
"""
import time
from collections import defaultdict

import cv2
import numpy as np

from .. import config
from .geometry import estimate_top_face
from .occlusion import compute_occlusion_by_mask
from .viz import (
    draw_fail_marker,
    draw_info_panel,
    draw_numbered_badge,
    draw_orientation_axes,
    draw_styled_text,
    draw_top_center,
    overlay_mask,
)


def _mask_centroid(mask):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] > 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None


def _median_depth(depth_m, mask, cfg):
    """마스크 아래 유효 depth(미터) 중앙값 — 픽 순서(최상단=가장 가까움) 정렬용. 싼 연산."""
    d = depth_m[mask]
    d = d[(d > cfg.Z_MIN) & (d < cfg.Z_MAX)]
    return float(np.median(d)) if d.size else float("inf")


def run_detection(model, color, depth_m, intr, up_dir, h_img, w_img):
    """Run the model + top-face estimation + mask-based occlusion once.

    Returns a state dict that the main loop overlays onto every frame
    until the next detection trigger.
    """
    t0 = time.time()
    preds = model.predict(color, conf_threshold=config.CONF_THRES)
    inference_time = time.time() - t0

    masks = getattr(preds, "mask", None)
    class_ids = preds.class_id
    confidences = preds.confidence
    n = 0 if masks is None else len(masks)

    masks_with_color = []   # (mask, bgr) — 픽 대상 마스크(매 프레임 오버레이)
    fail_markers = []       # (image_pos, reason)
    pallets = []            # 팔레트 검출 (픽 대상 아님 — 존재/위치만)
    pickables = []          # 픽 대상 검출(기하 전): {cid,name,color_bgr,mask,confidence}
    counts = defaultdict(int)

    for i in range(n):
        cid = int(class_ids[i])
        if cid not in config.CLASSES:
            continue
        name, color_bgr = config.CLASSES[cid]
        mask = masks[i]

        # 팔레트: 픽 대상이 아니므로 top-face/occlusion 건너뜀. 존재+중심만 기록
        # ('빈 팔레트=완료' 신호 + 위치 표시). 마스크는 칠하지 않음(박스와 겹쳐 얼룩).
        if cid == config.PALLET_CID:
            pallets.append({"mask": mask, "confidence": float(confidences[i]),
                            "center_2d": _mask_centroid(mask)})
            counts[cid] += 1
            continue

        masks_with_color.append((mask, color_bgr))
        counts[cid] += 1   # 검출 수(HUD) — 기하 성공 여부와 무관(남은 물품 카운트)
        pickables.append({"cid": cid, "name": name, "color_bgr": color_bgr,
                          "mask": mask, "confidence": float(confidences[i])})

    item_count = len(pickables)

    # 가려짐 필터: 마스크만 사용 → 기하 전에 후보 선별(self-hull 테스트).
    occluded_idx = compute_occlusion_by_mask(pickables, config)
    for oi in occluded_idx:
        c = _mask_centroid(pickables[oi]["mask"])
        if c is not None:
            fail_markers.append((c, "occluded"))
    candidates = [p for k, p in enumerate(pickables) if k not in occluded_idx]

    # 픽 순서로 정렬: 최상단(카메라에 가장 가까움 = median depth 최소) 먼저.
    candidates.sort(key=lambda p: _median_depth(depth_m, p["mask"], config))

    # 기하(중점/장단축/길이/자세)는 '픽 대상' 상위 K개에만 — 라이브 FPS 핵심 최적화.
    # (모든 객체에 매 프레임 돌리던 게 병목이었음.) GEOM_TARGETS<=0 이면 전부.
    k_targets = config.GEOM_TARGETS if config.GEOM_TARGETS > 0 else len(candidates)
    active = []
    for p in candidates:
        if len(active) >= k_targets:
            break
        res, reason = estimate_top_face(p["mask"], depth_m, intr, config, up_dir=up_dir)
        if res is None:
            c = _mask_centroid(p["mask"])
            if c is not None:
                fail_markers.append((c, reason))
            continue
        u, v = res["center_2d"]
        if not (0 <= u < w_img and 0 <= v < h_img):
            fail_markers.append(((u, v), "off-screen"))
            continue
        active.append({**p, "res": res})

    # 빈 팔레트 = 완료 신호: 팔레트는 보이는데 픽 대상이 하나도 없음.
    done = bool(pallets) and item_count == 0

    return {
        "masks_with_color": masks_with_color,
        "active": active,
        "pickables": pickables,
        "fail_markers": fail_markers,
        "pallets": pallets,
        "item_count": item_count,
        "done": done,
        "counts": dict(counts),
        "detect_time": time.time(),
        "inference_time": inference_time,
        "frame": color.copy(),
    }


def draw_state(img, state, intr, show_panel=True):
    """Overlay a stored detection state onto the current camera frame.

    show_panel: draw the top-right text info panel. The cv2 app keeps it (no
    other place for the numbers); the GUI sets it False because it shows the
    same info in its detection table, so the panel would just cover the view.
    """
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
        panel_entries.append((label, det["color_bgr"]))

    for pos, reason in state["fail_markers"]:
        draw_fail_marker(img, pos, reason)

    if show_panel:
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
