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
    pts_img = contour.reshape(-1, 2).astype(np.float64)
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

    # ─── 6. minAreaRect in plane coordinates ───
    rect_plane = cv2.minAreaRect(plane_pts)
    if rect_plane[1][0] < 1e-4 or rect_plane[1][1] < 1e-4:
        return None, "degenerate"
    corners_plane = cv2.boxPoints(rect_plane)

    # ─── 7. Lift back to 3D ───
    corners_3d = (
        P_top[None, :]
        + corners_plane[:, 0:1] * e1[None, :]
        + corners_plane[:, 1:2] * e2[None, :]
    ).astype(np.float64)
    centroid = corners_3d.mean(axis=0)
    if centroid[2] <= 0:
        return None, "behind cam"

    u_center = int(round(centroid[0] * fx / centroid[2] + cx))
    v_center = int(round(centroid[1] * fy / centroid[2] + cy))

    edge_a = corners_3d[1] - corners_3d[0]
    edge_b = corners_3d[2] - corners_3d[1]
    len_a = float(np.linalg.norm(edge_a))
    len_b = float(np.linalg.norm(edge_b))
    if len_a < 1e-6 or len_b < 1e-6:
        return None, "degenerate"
    if len_a >= len_b:
        long_axis = (edge_a / len_a).astype(np.float32)
        short_axis = (edge_b / len_b).astype(np.float32)
        long_mm = len_a * 1000.0
        short_mm = len_b * 1000.0
    else:
        long_axis = (edge_b / len_b).astype(np.float32)
        short_axis = (edge_a / len_a).astype(np.float32)
        long_mm = len_b * 1000.0
        short_mm = len_a * 1000.0

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
