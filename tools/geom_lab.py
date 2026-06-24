"""[dev] 윗면 기하(중심/축/길이) 알고리즘 오프라인 개발 하네스.

라이브 앱의 'g' 키로 저장한 RGB-D 덤프(rgbd_*.npz)를 읽어, 현재(및 새) 윗면
추정 알고리즘을 돌리고 결과를 시각화한다. 카메라 없이 4090에서 빠르게 반복.

덤프 키: rgb, depth(m), intrinsics(fx,fy,cx,cy), up_dir, masks(N,H,W), class_ids, confidences

출력 (--out 디렉토리, 기본 = npz 옆):
  <stem>_overlay.png   2D: 마스크 경계 + 중심점 + 장/단축 + 추정 사각형
  <stem>_det<k>.html   3D: 마스크 점구름(높이색) + inlier/near/outside + 중심 + 축
                        → 브라우저로 열어 노이즈/평면 상태를 직접 확인

사용:
  python tools/geom_lab.py ~/geom_captures/rgbd_20260624_153000.npz
  python tools/geom_lab.py ~/geom_captures/*.npz --algo both
"""
import argparse
import glob
import os

import cv2
import numpy as np

from dr_onsite_adap import config
from dr_onsite_adap.runtime.geometry import estimate_top_face
from dr_onsite_adap.runtime.viz import (
    draw_orientation_axes,
    draw_top_center,
)

try:
    import plotly.graph_objects as go
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False

# 새 알고리즘이 생기면 여기에 매핑(없으면 old 만).
try:
    from dr_onsite_adap.runtime.geometry import estimate_top_face_v2
    _HAVE_V2 = True
except Exception:
    estimate_top_face_v2 = None
    _HAVE_V2 = False

ALGOS = {"old": estimate_top_face}
if _HAVE_V2:
    ALGOS["new"] = estimate_top_face_v2


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    rgb = d["rgb"]
    depth = d["depth"].astype(np.float64)
    intr = tuple(float(v) for v in d["intrinsics"])
    up_dir = tuple(float(v) for v in d["up_dir"])
    masks = d["masks"].astype(bool)
    cids = d["class_ids"].astype(int)
    confs = d["confidences"].astype(float) if "confidences" in d else np.ones(len(cids))
    return rgb, depth, intr, up_dir, masks, cids, confs


def _overlay(rgb, depth, intr, up_dir, masks, cids, results, out_png):
    # rgb 는 모델 입력(RGB로 가정) — cv2 저장 위해 BGR 변환.
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb.copy()
    for k, (cid, mask, res) in enumerate(zip(cids, masks, results), start=1):
        cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 1)
        if res is None:
            continue
        u, v = res["center_2d"]
        draw_top_center(img, (u, v))
        draw_orientation_axes(
            img, res["center_3d"], res["long_axis_3d"], res["short_axis_3d"],
            res["long_side_mm"], res["short_side_mm"], intr,
        )
        cv2.putText(img, f"{k}:{res['long_side_mm']:.0f}x{res['short_side_mm']:.0f}",
                    (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(out_png, img)
    return out_png


def _scatter3d(res, label, out_html):
    if not _HAVE_PLOTLY or res is None:
        return None
    fig = go.Figure()

    def _add(pts, color, name, size=2):
        if pts is None or len(pts) == 0:
            return
        p = pts[:: max(1, len(pts) // 20000)]
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=size, color=color), name=name))

    _add(res.get("outside_pts"), "lightgrey", "outside (HI 아래)")
    _add(res.get("near_pts"), "crimson", "near (LO 위)")
    _add(res.get("inlier_pts"), "limegreen", "윗면 inlier")
    c = np.asarray(res["center_3d"], float)
    fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers",
                  marker=dict(size=6, color="blue"), name="center"))
    for axis_key, col, mm in (("long_axis_3d", "orange", res["long_side_mm"]),
                              ("short_axis_3d", "purple", res["short_side_mm"])):
        a = np.asarray(res[axis_key], float)
        half = (mm / 1000.0) / 2.0
        seg = np.stack([c - a * half, c + a * half])
        fig.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                      mode="lines", line=dict(color=col, width=6),
                      name=f"{axis_key} {mm:.0f}mm"))
    fig.update_layout(title=label, scene=dict(aspectmode="data"))
    fig.write_html(out_html)
    return out_html


def process(npz_path, out_dir, algos):
    rgb, depth, intr, up_dir, masks, cids, confs = _load(npz_path)
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    H, W = depth.shape[:2]
    print(f"\n=== {stem}  ({W}x{H}, up={tuple(round(x,3) for x in up_dir)}, "
          f"검출 {len(masks)}개) ===")
    pick = [i for i, c in enumerate(cids) if c != config.PALLET_CID]
    for algo_name in algos:
        fn = ALGOS[algo_name]
        results = []
        for i in pick:
            res, reason = fn(masks[i], depth, intr, config, up_dir=up_dir,
                             with_pointcloud=True)
            results.append(res)
            tag = "OK " if res else f"FAIL({reason})"
            extra = (f" c3d=({res['center_3d'][0]*100:+.1f},"
                     f"{res['center_3d'][1]*100:+.1f},{res['center_3d'][2]*100:+.1f})cm "
                     f"{res['long_side_mm']:.0f}x{res['short_side_mm']:.0f}mm "
                     f"top={res['n_top']}/{res['n_total']}") if res else ""
            print(f"  [{algo_name}] det#{i} cid={cids[i]} conf={confs[i]:.2f} {tag}{extra}")
        op = os.path.join(out_dir, f"{stem}_{algo_name}_overlay.png")
        _overlay(rgb, depth, intr, up_dir, [masks[i] for i in pick], [cids[i] for i in pick],
                 results, op)
        print(f"  → {op}")
        for k, (i, res) in enumerate(zip(pick, results)):
            if res is not None:
                hp = os.path.join(out_dir, f"{stem}_{algo_name}_det{k}.html")
                _scatter3d(res, f"{stem} {algo_name} det{k}", hp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="rgbd_*.npz (glob 가능)")
    ap.add_argument("--out", default=None, help="출력 디렉토리(기본=npz 옆)")
    ap.add_argument("--algo", default="old", choices=["old", "new", "both"])
    args = ap.parse_args()
    algos = ["old", "new"] if args.algo == "both" else [args.algo]
    algos = [a for a in algos if a in ALGOS]
    if not algos:
        algos = ["old"]
        print("⚠️ 요청 알고리즘 없음 → old 만 실행 (new 미구현)")

    paths = []
    for p in args.npz:
        paths.extend(sorted(glob.glob(os.path.expanduser(p))))
    if not paths:
        print("npz 없음")
        return
    for p in paths:
        out_dir = args.out or os.path.dirname(os.path.abspath(p))
        os.makedirs(out_dir, exist_ok=True)
        process(p, out_dir, algos)


if __name__ == "__main__":
    main()
