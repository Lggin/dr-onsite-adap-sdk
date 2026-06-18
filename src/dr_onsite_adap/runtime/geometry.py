import cv2
import numpy as np


def estimate_top_face(mask, depth_m, intrinsics, cfg, up_dir=None,
                      top_layer_band=None):
    """Return (result_dict, None) on success or (None, reason_str) on failure.

    Algorithm (plane-space minAreaRect, height-along-up):
      1. Largest 2D mask contour.
      2. For every valid mask pixel: back-project to a 3D point, then
         compute its height along -up_dir. Using height (instead of raw
         camera depth) keeps the percentile selection invariant to
         camera tilt — points on the same horizontal plane share the
         same height regardless of where they sit in the image.
      3. Trimmed median over the [TOP_DEPTH_PERCENTILE_LO, _HI] slice
         of heights → location of the horizontal plane along up_dir.
      4. Build the horizontal plane (normal = up_dir) through the
         anchor point whose height is closest to the trimmed median.
      5. Ray-project every contour pixel onto the plane, express in 2D
         plane coords (orthonormal basis e1, e2 perpendicular to up).
      6. minAreaRect IN PLANE COORDS — perspective foreshortening
         already corrected.
      7. Lift the rect centre + corners back to 3D.

    up_dir: optional (3,) world-up direction in camera frame (e.g. from IMU).
            If None, falls back to cfg.UP_DIR_CAMERA_FRAME.
    top_layer_band: kept for API compatibility but no longer used.

    Failure reasons (short labels for HUD):
      "no mask"        — empty mask
      "degenerate"     — mask collapsed to a line / zero-area rect
      "few depth"      — too few valid depth pixels under the mask
      "ray parallel"   — pixel ray parallel to the fitted plane
      "behind cam"     — degenerate corner / centroid (depth ≤ 0)
    """
    fx, fy, cx, cy = intrinsics

    # ─── 1. Largest mask contour ───
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None, "no mask"
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < 1.0:
        return None, "degenerate"

    # ─── 2. Valid mask pixels → 3D points (camera frame) ───
    ys, xs = np.where(mask)
    depths = depth_m[ys, xs]
    valid = (depths > cfg.Z_MIN) & (depths < cfg.Z_MAX)
    if int(valid.sum()) < cfg.MIN_PLANE_INLIERS:
        return None, "few depth"
    valid_depths = depths[valid]
    valid_xs = xs[valid]
    valid_ys = ys[valid]
    valid_X = (valid_xs - cx) * valid_depths / fx
    valid_Y = (valid_ys - cy) * valid_depths / fy
    valid_pts = np.stack([valid_X, valid_Y, valid_depths],
                         axis=1).astype(np.float64)

    # ─── 3. Per-point height along up_dir ───
    if up_dir is None:
        up_dir = cfg.UP_DIR_CAMERA_FRAME
    up_arr = np.asarray(up_dir, dtype=np.float64)
    up_arr = up_arr / np.linalg.norm(up_arr)
    # Sign convention: smaller value = higher up. (Equivalent to camera
    # depth when up_dir = (0,0,-1), so the same LO/HI percentile labels
    # still mean "topmost".)
    heights = -(valid_pts @ up_arr)

    # ─── 4. Trimmed-median height → plane location ───
    lo_cutoff = float(np.percentile(heights, cfg.TOP_DEPTH_PERCENTILE_LO))
    hi_cutoff = float(np.percentile(heights, cfg.TOP_DEPTH_PERCENTILE_HI))
    near_top = (heights >= lo_cutoff - 1e-9) & (heights <= hi_cutoff + 1e-9)
    if not near_top.any():
        return None, "few depth"
    h_top = float(np.median(heights[near_top]))

    # ─── 4b. 윗면 영역만 추출 (옆면 제외) ───
    # 높은 박스는 옆면이 마스크 실루엣에 포함된다. 전체 외곽선을 윗면 평면에 투영해
    # minAreaRect 를 잡으면 옆면까지 들어가 사각형이 부풀고 중점이 치우친다.
    # → 윗면 평면 높이 ±TOP_FACE_BAND_M(기본 3cm) 안의 픽셀(=실제 윗면)만의 외곽선을 쓴다.
    # 윗면이 충분히 안 잡히면(드물게) 전체 외곽선으로 폴백.
    band = getattr(cfg, "TOP_FACE_BAND_M", 0.03)
    on_top = np.abs(heights - h_top) < band
    src_contour = contour
    if int(on_top.sum()) >= cfg.MIN_PLANE_INLIERS:
        top_img = np.zeros(mask.shape[:2], dtype=np.uint8)
        top_img[valid_ys[on_top], valid_xs[on_top]] = 255
        top_img = cv2.morphologyEx(
            top_img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        tcnts, _ = cv2.findContours(
            top_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if tcnts:
            tc = max(tcnts, key=cv2.contourArea)
            if float(cv2.contourArea(tc)) >= 1.0:
                src_contour = tc

    # Anchor = the actual 3D point closest in height to h_top
    anchor_idx = int(np.argmin(np.abs(heights - h_top)))
    P_top = valid_pts[anchor_idx]
    d_plane = -float(up_arr @ P_top)   # plane: up_arr · P + d_plane = 0

    # In-plane 2D basis (e1, e2) — orthonormal, both perpendicular to up.
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(up_arr @ ref)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = ref - (ref @ up_arr) * up_arr
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up_arr, e1)

    # ─── 5. Ray-project every contour pixel onto the plane ───
    # src_contour = 윗면 영역 외곽선(옆면 제외; 폴백 시 전체 외곽선)
    pts_img = src_contour.reshape(-1, 2).astype(np.float64)
    dx = (pts_img[:, 0] - cx) / fx
    dy = (pts_img[:, 1] - cy) / fy
    denom = up_arr[0] * dx + up_arr[1] * dy + up_arr[2]
    if np.any(np.abs(denom) < 1e-9):
        return None, "ray parallel"
    t = -d_plane / denom
    if np.any(t <= 0):
        return None, "behind cam"
    P_3D = np.stack([t * dx, t * dy, t], axis=1)
    rel = P_3D - P_top
    plane_pts = np.stack([rel @ e1, rel @ e2], axis=1).astype(np.float32)

    # ─── 6. 사각형: 방향은 minAreaRect, 크기·중심은 윗면 점들의 백분위(robust) ───
    # minAreaRect 만 쓰면 모든 점을 감싸므로(가장자리 잡음/마스크 spill 포함) 변 길이가
    # 체계적으로 과대 추정된다. 방향(각도)만 minAreaRect 에서 가져오고, 변 길이/중심은
    # 윗면 점들을 그 축에 투영해 [trim, 100-trim] 백분위 구간으로 구해 가장자리 이상치를
    # 잘라낸다(TOP_FACE_TRIM_PCT). 중심 = 백분위 구간 중점.
    rect_plane = cv2.minAreaRect(plane_pts)
    if rect_plane[1][0] < 1e-4 or rect_plane[1][1] < 1e-4:
        return None, "degenerate"
    corners_plane = cv2.boxPoints(rect_plane)
    ax_u = corners_plane[1] - corners_plane[0]
    ax_v = corners_plane[2] - corners_plane[1]
    nu = float(np.linalg.norm(ax_u))
    nv = float(np.linalg.norm(ax_v))
    if nu < 1e-6 or nv < 1e-6:
        return None, "degenerate"
    ax_u = ax_u / nu
    ax_v = ax_v / nv

    trim = getattr(cfg, "TOP_FACE_TRIM_PCT", 1.0)
    if int(on_top.sum()) >= cfg.MIN_PLANE_INLIERS:
        # 실제 윗면 점들(채워진 영역)을 평면 좌표로
        top_rel = valid_pts[on_top] - P_top
        pts2 = np.stack([top_rel @ e1, top_rel @ e2], axis=1)
    else:
        pts2 = plane_pts.astype(np.float64)   # 폴백: ray-projected contour
    pu = pts2 @ ax_u
    pv = pts2 @ ax_v
    u_lo, u_hi = np.percentile(pu, [trim, 100.0 - trim])
    v_lo, v_hi = np.percentile(pv, [trim, 100.0 - trim])
    len_u = float(u_hi - u_lo)
    len_v = float(v_hi - v_lo)
    center_plane = ((u_lo + u_hi) * 0.5) * ax_u + ((v_lo + v_hi) * 0.5) * ax_v

    # ─── 7. Lift centre + axes back to 3D ───
    centroid = (P_top + center_plane[0] * e1 + center_plane[1] * e2).astype(np.float64)
    if centroid[2] <= 0:
        return None, "behind cam"
    u_center = int(round(centroid[0] * fx / centroid[2] + cx))
    v_center = int(round(centroid[1] * fy / centroid[2] + cy))

    axis_u_3d = ax_u[0] * e1 + ax_u[1] * e2
    axis_v_3d = ax_v[0] * e1 + ax_v[1] * e2
    if len_u >= len_v:
        long_axis = axis_u_3d.astype(np.float32)
        short_axis = axis_v_3d.astype(np.float32)
        long_mm = len_u * 1000.0
        short_mm = len_v * 1000.0
    else:
        long_axis = axis_v_3d.astype(np.float32)
        short_axis = axis_u_3d.astype(np.float32)
        long_mm = len_v * 1000.0
        short_mm = len_u * 1000.0
    if long_mm < 1e-3 or short_mm < 1e-3:
        return None, "degenerate"

    # ─── 8. Snapshot point clouds (height-based 3-way split) ───
    all_pts_f32 = valid_pts.astype(np.float32)
    too_near = heights < lo_cutoff - 1e-9     # above the LO band  → red
    too_far = heights > hi_cutoff + 1e-9      # below the HI band  → grey
    near_pts = all_pts_f32[too_near]
    in_pts = all_pts_f32[near_top]
    out_pts = all_pts_f32[too_far]

    return {
        "center_3d": centroid.astype(np.float32),
        "center_2d": (u_center, v_center),
        "short_side_mm": float(short_mm),
        "long_side_mm": float(long_mm),
        "long_axis_3d": long_axis,
        "short_axis_3d": short_axis,
        "near_pts": near_pts,
        "inlier_pts": in_pts,
        "outside_pts": out_pts,
        "n_total": int(valid.sum()),
        "n_top": int(near_top.sum()),
        "n_inliers": int(near_top.sum()),
    }, None


def estimate_up_dir(depth_m, intrinsics, cfg, iters=500, thresh=0.012,
                    subsample=15000, seed=0, refine_iters=8):
    """현재 depth 프레임에서 **지배 수평면(바닥/팔레트 등)** 의 법선을 RANSAC 으로 찾아
    카메라 프레임 기준 '월드 위(up)' 단위벡터를 추정한다.

    고정 마운트에서 카메라 기울기가 바뀌면 up_dir 도 바뀌는데, 이 함수로 한 번
    다시 측정해 상수로 쓰면 IMU 없이도 정확하다. 반환 벡터는 카메라 쪽(=위)을 향하도록
    z 성분이 음수가 되게 정렬한다(오버헤드 카메라 관례, 예: (-0.005,-0.428,-0.904)).

    실패 시 None (유효 depth 부족 등).
    """
    fx, fy, cx, cy = intrinsics
    ys, xs = np.where((depth_m > cfg.Z_MIN) & (depth_m < cfg.Z_MAX))
    if len(xs) < 500:
        return None
    rng = np.random.default_rng(seed)
    if len(xs) > subsample:
        idx = rng.choice(len(xs), subsample, replace=False)
        xs, ys = xs[idx], ys[idx]
    z = depth_m[ys, xs].astype(np.float64)
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    P = np.stack([X, Y, z], axis=1)

    best_inl = 0
    best_mask = None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        dist = np.abs((P - s[0]) @ n)
        inl = int((dist < thresh).sum())
        if inl > best_inl:
            best_inl, best_mask = inl, dist < thresh
    if best_mask is None or best_inl < 200:
        return None

    # inlier 들로 법선 반복 정밀화: PCA(최소분산 방향=법선)로 적합 → 그 평면 기준으로
    # inlier 재선정 → 다시 적합. 몇 번 돌리면 total-least-squares 평면으로 수렴해
    # RANSAC 3점 추정보다 훨씬 정밀한 법선을 얻는다(up_dir 미세오차↓).
    Q = P[best_mask]
    n = None
    for _ in range(refine_iters):
        c = Q.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        n = Vt[-1]
        n = n / np.linalg.norm(n)
        d = -float(n @ c)
        dist = np.abs(P @ n + d)
        newQ = P[dist < thresh]
        if len(newQ) < 200 or len(newQ) == len(Q):
            Q = newQ if len(newQ) >= 200 else Q
            break
        Q = newQ
    if n[2] > 0:                       # 위(카메라 쪽)를 향하도록 z<0 로
        n = -n
    n = n / np.linalg.norm(n)
    return tuple(float(v) for v in n)
