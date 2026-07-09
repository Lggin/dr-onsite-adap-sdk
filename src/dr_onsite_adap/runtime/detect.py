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
from .clustering import cluster_by_appearance, group_color
from .geometry import estimate_top_face, estimate_top_face_v2
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


def run_detection(model, color, depth_m, intr, up_dir, h_img, w_img,
                  with_pointcloud=True, conf_threshold=None, cluster=False):
    """Run the model + top-face estimation + mask-based occlusion once.

    Returns a state dict that the main loop overlays onto every frame
    until the next detection trigger.

    with_pointcloud: 3D 스냅샷용 점구름까지 계산할지. 라이브(매 프레임)에선 False 로
    꺼서 비용을 줄인다(픽 자세에는 영향 없음). 단발 'd' / 3D 보기 흐름에선 True.

    cluster: True 면 픽 대상(active)을 **같은 클래스 안에서** 외형(색·질감·크기)으로
    서브그룹핑해 각 det 에 group_id/group_label 을 달고 마스크/뱃지를 그룹색으로 칠한다.
    """
    # 윗면 추정 알고리즘 선택(config.GEOM_ALGO): "v2"=평면 fit 방식, 그 외=기존 v1.
    estimate = (estimate_top_face_v2 if config.GEOM_ALGO == "v2"
                else estimate_top_face)

    if conf_threshold is None:
        conf_threshold = config.CONF_THRES
    t0 = time.time()
    preds = model.predict(color, conf_threshold=conf_threshold)
    inference_time = time.time() - t0

    masks = getattr(preds, "mask", None)
    class_ids = preds.class_id
    confidences = preds.confidence
    n = 0 if masks is None else len(masks)

    masks_with_color = []   # (mask, bgr) — drawn every frame as overlay
    collected = []          # detections that passed estimate_top_face
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

        # 픽 대상 마스크만 오버레이(팔레트 마스크는 칠하지 않음: 박스와 겹쳐 얼룩이 됨)
        masks_with_color.append((mask, color_bgr))
        item_count += 1   # 픽 대상 검출 (픽 성공 여부와 무관 — 남은 물품 카운트)

        res, reason = estimate(
            mask, depth_m, intr, config, up_dir=up_dir,
            with_pointcloud=with_pointcloud,
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

        counts[cid] += 1
        collected.append({
            "orig_idx": i,
            "cid": cid,
            "name": name,
            "color_bgr": color_bgr,
            "mask": mask,
            "confidence": float(confidences[i]),
            "res": res,
        })

    # Mask-based occlusion (single-mask self-hull test).
    occluded_idx = compute_occlusion_by_mask(collected, config)
    for oi in occluded_idx:
        u_oc, v_oc = collected[oi]["res"]["center_2d"]
        fail_markers.append(((u_oc, v_oc), "occluded"))
    active = [c for k, c in enumerate(collected) if k not in occluded_idx]

    # 외형 클러스터링(옵트인): 같은 클래스 안에서 색·질감·크기로 서브그룹 → 그룹색/라벨.
    if cluster and active:
        gids, glabels = cluster_by_appearance(
            [d["mask"] for d in active], [d["cid"] for d in active], color)
        active_mask_ids = set()
        for d, g, l in zip(active, gids, glabels):
            d["group_id"] = g
            d["group_label"] = l
            gc = group_color(g)
            if gc is not None:
                d["color_bgr"] = gc          # 뱃지/패널/마스크 색을 그룹색으로
            active_mask_ids.add(id(d["mask"]))
        # 마스크 오버레이 재구성: active 는 그룹색, 그 외(가림/실패)는 원 클래스색 유지.
        kept = [(m, c) for (m, c) in masks_with_color if id(m) not in active_mask_ids]
        masks_with_color = kept + [(d["mask"], d["color_bgr"]) for d in active]

    # 빈 팔레트 = 완료 신호: 팔레트는 보이는데 픽 대상이 하나도 없음.
    done = bool(pallets) and item_count == 0

    return {
        "masks_with_color": masks_with_color,
        "active": active,
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

        gl = det.get("group_label")
        gtag = f" [{gl}]" if gl else ""
        label = (
            f"[{badge_n}] {det['name']}{gtag} "
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
