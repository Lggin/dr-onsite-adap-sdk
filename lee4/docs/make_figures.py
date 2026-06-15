"""Generate the figures referenced from DEVLOG.md.

Run with `python3 docs/make_figures.py` from the lee4 root. Re-running
overwrites the PNGs in docs/imgs/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import cv2


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "imgs")
os.makedirs(OUT, exist_ok=True)

# ── Shared camera intrinsics (matches the synthetic test) ──
H_IMG, W_IMG = 720, 1280
fx = fy = 900.0
cx, cy = W_IMG / 2.0, H_IMG / 2.0


def project(P):
    """3D camera-frame point -> (u, v) pixel."""
    u = P[..., 0] * fx / P[..., 2] + cx
    v = P[..., 1] * fy / P[..., 2] + cy
    return np.stack([u, v], axis=-1)


# ════════════════════════════════════════════════════════════════════
# Figure 1 — image-space vs plane-space minAreaRect (tilted camera)
# ════════════════════════════════════════════════════════════════════
def fig_image_vs_plane():
    """Camera tilted 25° looking at a horizontal rect that sits to one
    side of the optical axis. Image-space minAreaRect bulges outward and
    its centre is offset; plane-space minAreaRect hugs the silhouette
    and recovers the true centre.
    """
    tilt = np.deg2rad(35)
    up_cam = np.array([0.0, np.sin(tilt), -np.cos(tilt)])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(up_cam, e1)

    # Ground-truth rect, in 3D camera frame. Put it well off-axis and
    # close to the camera so the perspective foreshortening (and the
    # resulting image-space minAreaRect bulge) is obvious.
    center_gt = np.array([0.30, -0.25, 0.9])
    long_m, short_m = 0.55, 0.40
    aa = np.linspace(-1, 1, 80) * (long_m / 2)
    bb = np.linspace(-1, 1, 80) * (short_m / 2)
    AA, BB = np.meshgrid(aa, bb)
    rect_3d = center_gt[None, None, :] + AA[..., None]*e1 + BB[..., None]*e2

    # The four GT corners (clockwise)
    s = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    gt_corners_3d = center_gt + s[:, 0:1] * (long_m / 2) * e1 \
                              + s[:, 1:2] * (short_m / 2) * e2
    gt_corners_uv = project(gt_corners_3d)

    # Silhouette in image space (just the projected GT corners; gives a
    # quadrilateral due to perspective)
    silh_uv = gt_corners_uv

    # ── image-space minAreaRect ──
    rect_img = cv2.minAreaRect(silh_uv.astype(np.float32))
    img_corners_uv = cv2.boxPoints(rect_img)

    # Ray-project image-space corners onto the horizontal plane to get 3D
    # corners (the OLD algorithm), then back-project the centroid to uv.
    d_plane = -float(up_cam @ center_gt)  # plane passes through GT center
    def ray_to_plane(uv):
        dx = (uv[:, 0] - cx) / fx
        dy = (uv[:, 1] - cy) / fy
        ray = np.stack([dx, dy, np.ones_like(dx)], axis=1)
        denom = ray @ up_cam
        t = -d_plane / denom
        return ray * t[:, None]

    img_corners_3d = ray_to_plane(img_corners_uv)
    img_center_3d = img_corners_3d.mean(axis=0)
    img_center_uv = project(img_center_3d)

    # ── plane-space minAreaRect ──
    # Ray-project the silhouette to the plane, drop into plane coords (e1, e2)
    silh_3d = ray_to_plane(silh_uv)
    rel = silh_3d - center_gt
    silh_plane = np.stack([rel @ e1, rel @ e2], axis=1).astype(np.float32)
    rect_plane = cv2.minAreaRect(silh_plane)
    plane_corners_2d = cv2.boxPoints(rect_plane)
    plane_corners_3d = (center_gt[None, :]
                        + plane_corners_2d[:, 0:1] * e1
                        + plane_corners_2d[:, 1:2] * e2)
    plane_center_3d = plane_corners_3d.mean(axis=0)
    plane_center_uv = project(plane_center_3d)
    plane_corners_uv = project(plane_corners_3d)

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel — image view
    ax = axes[0]
    ax.set_title("Image view (camera tilted 35°)")
    ax.add_patch(Polygon(silh_uv, closed=True, facecolor="#cfd8e6",
                         edgecolor="#5a6776", linewidth=2,
                         label="True top face (silhouette)"))
    ax.add_patch(Polygon(img_corners_uv, closed=True, fill=False,
                         edgecolor="#d04848", linewidth=2.2, linestyle="--",
                         label="image-space minAreaRect"))
    ax.add_patch(Polygon(plane_corners_uv, closed=True, fill=False,
                         edgecolor="#2f8f3f", linewidth=2.2,
                         label="plane-space minAreaRect"))
    ax.plot(*project(center_gt), "ko", markersize=8, label="True centre")
    ax.plot(*img_center_uv, marker="x", color="#d04848", markersize=11,
            mew=2.5, linestyle="None", label="Image-space recovered")
    ax.plot(*plane_center_uv, marker="+", color="#2f8f3f", markersize=14,
            mew=2.5, linestyle="None", label="Plane-space recovered")
    ax.set_xlim(0, W_IMG); ax.set_ylim(H_IMG, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.legend(loc="lower left", fontsize=8)

    # Right panel — top-down view in plane coordinates (in cm for clarity)
    ax = axes[1]
    ax.set_title("Top-down (horizontal plane coords)")
    gt_corners_plane = s * np.array([long_m / 2, short_m / 2])
    ax.add_patch(Polygon(gt_corners_plane * 100, closed=True,
                         facecolor="#cfd8e6", edgecolor="#5a6776",
                         linewidth=2, label="True top face"))

    # Lift the image-space algorithm's corners into plane coords (relative
    # to GT centre, in cm)
    img_rel = img_corners_3d - center_gt
    img_plane = np.stack([img_rel @ e1, img_rel @ e2], axis=1) * 100
    ax.add_patch(Polygon(img_plane, closed=True, fill=False,
                         edgecolor="#d04848", linewidth=2.2, linestyle="--",
                         label="image-space → projected"))
    ax.add_patch(Polygon(plane_corners_2d * 100, closed=True, fill=False,
                         edgecolor="#2f8f3f", linewidth=2.2,
                         label="plane-space"))
    ax.plot(0, 0, "ko", markersize=8, label="True centre")
    img_c_plane = img_plane.mean(axis=0)
    ax.plot(*img_c_plane, marker="x", color="#d04848", markersize=11,
            mew=2.5, linestyle="None", label="Image-space centre")
    ax.plot(*(rect_plane[0][0] * 100, rect_plane[0][1] * 100),
            marker="+", color="#2f8f3f", markersize=14, mew=2.5,
            linestyle="None", label="Plane-space centre")
    ax.axhline(0, color="#bbb", lw=0.5); ax.axvline(0, color="#bbb", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("e1 axis (cm)"); ax.set_ylabel("e2 axis (cm)")
    err_img_mm = np.linalg.norm(img_c_plane) * 10  # cm → mm
    err_plane_mm = np.linalg.norm(rect_plane[0]) * 1000
    ax.text(0.02, 0.98,
            f"image-space error: {err_img_mm:.1f} mm\n"
            f"plane-space error: {err_plane_mm:.2f} mm",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(facecolor="#fff8d6", edgecolor="#bba84a", lw=1))
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT, "image_vs_plane.png")
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 2 — trimmed-median depth selection on an empty tote
# ════════════════════════════════════════════════════════════════════
def fig_trimmed_median():
    """Bimodal depth distribution for an empty tote (rim + inner bottom)
    with a sprinkle of close-side noise. Show how the 5-15% trimmed
    median lands on the rim while plain 5%-percentile-median is pulled
    upward by the noise."""
    rng = np.random.default_rng(0)
    # Rim depth ~ 1.50 m, inner bottom ~ 1.62 m, sensor noise σ ~ 5 mm
    rim = rng.normal(1.500, 0.005, size=900)
    bottom = rng.normal(1.620, 0.006, size=1500)
    noise = rng.normal(1.480, 0.004, size=80)   # close-side spikes
    depths = np.concatenate([rim, bottom, noise])
    depths.sort()

    p5 = np.percentile(depths, 5)
    p15 = np.percentile(depths, 15)
    z_old = np.median(depths[depths <= p5 + 1e-6])
    mask_band = (depths >= p5) & (depths <= p15)
    z_new = np.median(depths[mask_band])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(depths, bins=60, color="#cfd8e6", edgecolor="#5a6776")

    # Mark the bands
    ymax = ax.get_ylim()[1]
    ax.axvspan(depths.min(), p5, color="#d04848", alpha=0.18,
               label=f"closest 5% (noise + rim curvature)")
    ax.axvspan(p5, p15, color="#2f8f3f", alpha=0.20,
               label=f"5–15% slice (rim body)")
    ax.axvline(z_old, color="#d04848", linewidth=2,
               label=f"old z_top = {z_old*100:.2f} cm")
    ax.axvline(z_new, color="#2f8f3f", linewidth=2,
               label=f"new z_top = {z_new*100:.2f} cm")
    ax.axvline(1.500, color="#444", linestyle=":", linewidth=1.5,
               label="true rim 1.500 m")

    ax.set_xlabel("depth (m)")
    ax.set_ylabel("# pixels")
    ax.set_title("Empty tote depth histogram — trimmed median locks onto the rim")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT, "trimmed_median.png")
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 3 — occlusion criterion (hidden fraction vs hull)
# ════════════════════════════════════════════════════════════════════
def fig_occlusion():
    """Three masks side by side: (a) clean rect, low hidden fraction;
    (b) adjacent boxes with jittery border, low hidden fraction (no
    false positive); (c) heavily occluded mask, high hidden fraction
    (flagged)."""
    def hidden_frac(mask):
        m_u8 = mask.astype(np.uint8)
        ctrs, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        c = max(ctrs, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        hull_mask = np.zeros_like(m_u8)
        cv2.fillPoly(hull_mask, [hull], 1)
        return 1 - mask.sum() / max(hull_mask.sum(), 1), hull_mask.astype(bool)

    H, W = 200, 240
    # (a) clean rect
    a = np.zeros((H, W), bool); a[40:160, 50:190] = True
    # (b) adjacent with jitter — only one mask, the left box; its right
    # edge is jagged (jitter)
    b = np.zeros((H, W), bool); b[40:160, 30:120] = True
    for y in range(50, 150, 8):
        b[y:y+3, 115:120] = False
    # (c) heavily occluded — only an L-shaped corner of the original box
    # is visible. The convex hull restores the full rect, so the missing
    # region is large.
    c = np.zeros((H, W), bool); c[40:160, 30:200] = True
    c[40:140, 60:200] = False    # upper-right chunk removed → narrow L

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, mask, label in zip(
        axes,
        [a, b, c],
        ["(a) Clean rect", "(b) Jittery edge (adjacent box)",
         "(c) Heavily occluded"],
    ):
        frac, hull = hidden_frac(mask)
        rgb = np.full((H, W, 3), 255, np.uint8)
        rgb[hull & ~mask] = (216, 110, 110)   # missing region (red-ish)
        rgb[mask] = (90, 140, 200)            # visible mask (blue)
        ax.imshow(rgb)
        flag = " → FLAGGED" if frac > 0.5 else ""
        ax.set_title(f"{label}\nhidden = {frac*100:.1f}%{flag}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Single-mask occlusion test: (hull − mask) / hull > 50%",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, "occlusion.png")
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_image_vs_plane()
    fig_trimmed_median()
    fig_occlusion()
