import numpy as np
import plotly.graph_objects as go


LONG_RGB = "rgb(255,255,0)"        # yellow — matches 2D long-axis arrow
SHORT_RGB = "rgb(0,255,255)"       # cyan  — matches 2D short-axis arrow
CENTER_RGB = "rgb(255,0,0)"
OUTSIDE_RGB = "rgb(115,115,115)"   # mask pts deeper than the rim band
NEAR_RGB = "rgb(220,40,40)"        # mask pts closer than LO percentile —
                                    # the "trimmed out" group (noise / rim
                                    # curvature). Red to signal "rejected".


def _subsample(arr, k):
    if arr.shape[0] <= k:
        return arr
    idx = np.random.default_rng(0).choice(arr.shape[0], k, replace=False)
    return arr[idx]


def _rgb01_to_str(c):
    return f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"


def _world_rotation(up_dir):
    """Rotation R that maps `up_dir` (camera frame) onto world Z+ (0,0,1).

    All camera-frame points X are rendered as R @ X. After rotation the
    horizontal top-face plane is perpendicular to the rendered Z axis,
    so a flat box top appears axis-aligned in the 3D viewer instead of
    tilted by the camera mounting angle.
    """
    a = np.asarray(up_dir, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        # a is parallel (R = I) or anti-parallel (180° flip about X) to b.
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _add_arrow(fig, origin, vec, color, name, line_width=8, head_size=0.04):
    """Line + cone tip = real 3D arrow visible above the point cloud."""
    end = origin + vec
    fig.add_trace(go.Scatter3d(
        x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
        mode='lines', line=dict(color=color, width=line_width),
        name=name, showlegend=True,
    ))
    fig.add_trace(go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[vec[0]], v=[vec[1]], w=[vec[2]],
        sizemode='absolute', sizeref=head_size, anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False, showlegend=False, hoverinfo='skip',
    ))


def show_snapshot(detections, up_dir=None, max_pts_per_layer=20000):
    """Open a browser-based 3D plot in a virtual coordinate space.
    WebGL-backed so navigation is smooth: left-drag rotates, shift/right-drag
    pans, scroll zooms; hovering any point shows its (x, y, z) in meters.

    Per detection three point groups are drawn so the trimmed-median band
    selection is visible:
      • near_pts    (red)    — depth < LO%ile, rejected as noise/curvature
      • inlier_pts  (class)  — LO–HI%ile band, set z_top
      • outside_pts (grey)   — depth > HI%ile, deeper layer (e.g. tote bottom)
    Centre dot (red), long-axis (yellow), short-axis (cyan) arrows on top.

    If `up_dir` is given (camera-frame world-up), every point is rotated
    so up_dir becomes world Z+. The horizontal top-face then renders as
    a level surface regardless of camera mounting tilt.

    Non-blocking — the live 2D loop keeps running while the tab is open.
    """
    print("\n--- 3D snapshot (opening in browser; meters; hover for values) ---")
    if not detections:
        print("  (no detections in this frame)")
    for i, det in enumerate(detections, 1):
        cx, cy, cz = det["center_3d"]
        print(f"  [{i}] {det['label']:5s}  "
              f"center=({cx*100:+6.1f}, {cy*100:+6.1f}, {cz*100:+6.1f}) cm  "
              f"long={det['long_len_mm']:.0f} mm  "
              f"short={det['short_len_mm']:.0f} mm")

    R = _world_rotation(up_dir) if up_dir is not None else np.eye(3)

    def rot_pts(pts):
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64) @ R.T

    def rot_vec(v):
        return R @ np.asarray(v, dtype=np.float64)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=4, color='black', symbol='cross'),
        name='camera origin',
    ))

    if up_dir is not None:
        ud_rot = np.array([0.0, 0.0, 1.0]) * 0.30   # 30 cm visual along Z+
        fig.add_trace(go.Scatter3d(
            x=[0, ud_rot[0]], y=[0, ud_rot[1]], z=[0, ud_rot[2]],
            mode='lines+markers',
            line=dict(color='magenta', width=5),
            marker=dict(size=[0, 6], color='magenta'),
            name='world-up',
        ))

    for i, det in enumerate(detections, 1):
        color = _rgb01_to_str(det["color_rgb"])

        # outside (deeper than HI percentile) — grey
        out = rot_pts(det.get("outside_pts", None))
        if out.size:
            o = _subsample(out, max_pts_per_layer)
            fig.add_trace(go.Scatter3d(
                x=o[:, 0], y=o[:, 1], z=o[:, 2], mode='markers',
                marker=dict(size=1, color=OUTSIDE_RGB),
                opacity=0.4,
                name=f"#{i} off-plane", showlegend=False,
            ))

        # too-near (closer than LO percentile) — red, "trimmed out"
        near = rot_pts(det.get("near_pts", None))
        if near.size:
            n = _subsample(near, max_pts_per_layer)
            fig.add_trace(go.Scatter3d(
                x=n[:, 0], y=n[:, 1], z=n[:, 2], mode='markers',
                marker=dict(size=1, color=NEAR_RGB),
                opacity=0.55,
                name=f"#{i} trimmed (near)", showlegend=True,
            ))

        # inliers (LO–HI band) — class colour
        ip = rot_pts(det["inlier_pts"])
        if ip.size:
            p = _subsample(ip, max_pts_per_layer)
            fig.add_trace(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                marker=dict(size=2, color=color),
                name=f"#{i} {det['label']} inliers",
            ))

        c = rot_vec(det["center_3d"])
        fig.add_trace(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[c[2]], mode='markers',
            marker=dict(size=6, color=CENTER_RGB,
                        line=dict(color='black', width=1)),
            name=f"#{i} center",
        ))

        L = rot_vec(det["long_axis_3d"].astype(np.float64)
                    * (det["long_len_mm"] * 0.5 / 1000.0))
        S = rot_vec(det["short_axis_3d"].astype(np.float64)
                    * (det["short_len_mm"] * 0.5 / 1000.0))
        _add_arrow(fig, c, L, LONG_RGB,
                   name=f"#{i} long {det['long_len_mm']:.0f}mm")
        _add_arrow(fig, c, S, SHORT_RGB,
                   name=f"#{i} short {det['short_len_mm']:.0f}mm")

    # After rotation, world-up is always +Z in the scene. Place the eye
    # high (Z>0) so we look "down" at the rotated scene — for an overhead
    # camera the objects sit at Z<0 (camera is at the origin, looking up
    # in rotated-frame terms because we mapped its optical axis through R).
    scene_up = dict(x=0, y=0, z=1)
    scene_eye = dict(x=1.4, y=-1.4, z=1.4)

    fig.update_layout(
        title="dr_onsite_adap — 3D snapshot (world frame; up_dir → +Z)",
        legend=dict(orientation="h", yanchor="top", y=-0.02, x=0),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m, world-up)",
            aspectmode='data',
            camera=dict(up=scene_up, eye=scene_eye),
        ),
        margin=dict(l=0, r=0, t=40, b=40),
    )
    fig.show()
