import cv2


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_styled_text(img, text, pos, color, scale=0.5, thickness=1):
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - h - 5), (x + w, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness)


def overlay_mask(img, mask, color, alpha=0.4):
    overlay = img.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_top_center(img, center_2d, radius=6):
    cv2.circle(img, center_2d, radius, (0, 0, 255), -1)
    cv2.circle(img, center_2d, radius + 2, (255, 255, 255), 1)


def project_rect_corners(center_3d, long_axis_3d, short_axis_3d,
                         long_mm, short_mm, intrinsics):
    """Return the 4 image-space corners (u, v) of the top-face minAreaRect,
    or None if any corner falls behind the camera."""
    fx, fy, cx, cy = intrinsics
    half_L = long_axis_3d * (long_mm * 0.5 / 1000.0)
    half_S = short_axis_3d * (short_mm * 0.5 / 1000.0)
    out = []
    for sa in (-1, 1):
        for sb in (-1, 1):
            p = center_3d + sa * half_L + sb * half_S
            if p[2] <= 1e-6:
                return None
            out.append((int(round(p[0] * fx / p[2] + cx)),
                       int(round(p[1] * fy / p[2] + cy))))
    return out


def draw_orientation_axes(img, center_3d, long_axis_3d, short_axis_3d,
                          long_len_mm, short_len_mm, intrinsics,
                          long_color=(0, 255, 255), short_color=(255, 255, 0),
                          thickness=2, tip_length=0.18):
    """Project the box principal axes to pixels and draw two arrows from the
    centroid: one along +long_axis_3d, one along +short_axis_3d."""
    fx, fy, cx, cy = intrinsics

    def proj(p):
        if p[2] <= 1e-6:
            return None
        return (int(round(p[0] * fx / p[2] + cx)),
                int(round(p[1] * fy / p[2] + cy)))

    cen = proj(center_3d)
    if cen is None:
        return
    half_long = long_axis_3d * (long_len_mm * 0.5 / 1000.0)
    half_short = short_axis_3d * (short_len_mm * 0.5 / 1000.0)
    for tip3d, col in (
        (center_3d + half_long, long_color),
        (center_3d + half_short, short_color),
    ):
        tip = proj(tip3d)
        if tip is not None:
            cv2.arrowedLine(img, cen, tip, col, thickness, tipLength=tip_length)


def draw_fail_marker(img, pos, reason, color=(0, 165, 255)):
    """Small '?' marker + short reason label. Used on detections where the
    segmentation mask is present but the top-face centre couldn't be derived
    (e.g. depth dropout, plane not horizontal, obscured by a higher box)."""
    x, y = int(pos[0]), int(pos[1])
    cv2.circle(img, (x, y), 11, (0, 0, 0), -1)
    cv2.circle(img, (x, y), 11, color, 2)
    (tw, th), _ = cv2.getTextSize("?", _FONT, 0.55, 2)
    cv2.putText(img, "?", (x - tw // 2, y + th // 2),
                _FONT, 0.55, color, 2)
    # Short reason text immediately to the right, with a black halo for
    # contrast against any background.
    label = str(reason)
    (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
    lx = x + 16
    ly = y + lh // 2
    cv2.putText(img, label, (lx, ly), _FONT, 0.45, (0, 0, 0), 3)
    cv2.putText(img, label, (lx, ly), _FONT, 0.45, color, 1)


def draw_verify_flag(img, badge_pos, dino_name, color=(0, 0, 255)):
    """검증 불일치 표시. 번호 배지 위쪽에 빨간 '!' 원 + DINOv2 예측 클래스명.
    flag-only 모드라 검출을 막지는 않고, RFDETR 와 의견이 갈린 검출만 눈에 띄게 한다."""
    x, y = int(badge_pos[0]), int(badge_pos[1]) - 26
    cv2.circle(img, (x, y), 10, (0, 0, 0), -1)
    cv2.circle(img, (x, y), 10, color, 2)
    (tw, th), _ = cv2.getTextSize("!", _FONT, 0.55, 2)
    cv2.putText(img, "!", (x - tw // 2, y + th // 2), _FONT, 0.55, color, 2)
    label = f"DINO:{dino_name}"
    cv2.putText(img, label, (x + 14, y + 4), _FONT, 0.45, (0, 0, 0), 3)
    cv2.putText(img, label, (x + 14, y + 4), _FONT, 0.45, color, 1)


def draw_numbered_badge(img, pos, n, fill_color,
                         text_color=(255, 255, 255), radius=11):
    """Small filled circle in class color with the detection's index inside.
    Use this to label each kept detection so the right-hand info panel can
    cross-reference. Halo + outline make it readable on any background."""
    x, y = int(pos[0]), int(pos[1])
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), -1)
    cv2.circle(img, (x, y), radius, fill_color, -1)
    cv2.circle(img, (x, y), radius, (255, 255, 255), 1)
    text = str(n)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 2)
    cv2.putText(img, text, (x - tw // 2, y + th // 2),
                _FONT, 0.5, text_color, 2)


def draw_info_panel(img, entries, scale=0.5, line_height=18, margin=8):
    """Top-right semi-transparent panel: one row per (text, color) entry.
    All per-detection text moves here so multiple boxes don't fight for
    on-image real estate."""
    if not entries:
        return
    h_img, w_img = img.shape[:2]
    max_w = max(cv2.getTextSize(t, _FONT, scale, 1)[0][0] for t, _ in entries)
    panel_w = max_w + 16
    panel_h = len(entries) * line_height + 10
    x0 = w_img - panel_w - margin
    y0 = margin

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (100, 100, 100), 1)

    y = y0 + line_height - 2
    for text, color in entries:
        cv2.putText(img, text, (x0 + 8, y), _FONT, scale, color, 1)
        y += line_height


def draw_hud(img, fps, class_counts, classes):
    draw_styled_text(img, f"AI: {fps:5.1f} FPS", (10, 25),
                     (0, 255, 255), scale=0.6, thickness=2)
    y = 50
    for cid, (name, color) in classes.items():
        n = class_counts.get(cid, 0)
        draw_styled_text(img, f"{name}: {n}", (10, y),
                         color, scale=0.6, thickness=2)
        y += 25
