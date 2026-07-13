"""[dev] 그룹화 특징 검증 하네스 — "각 특징 블록이 해당 속성 차이를 잡아내는가" 증명용.

한 속성만 다른 박스 2개를 찍은 RGB-D 덤프(npz)를 케이스별로 넣으면, 두 박스의 특징
벡터를 **색/질감/크기 블록으로 분해한 거리**로 보여준다. 기대: 크기case→크기블록↑,
색case→색블록↑, 질감case→질감블록↑, 같은종류(기준선)→모두↓.

사용:
  python tools/prove_grouping.py \
      --size    a.npz b.npz \
      --color   c.npz d.npz \
      --texture e.npz \
      --baseline f.npz \
      [--lw 0.35] [--out ~/prove_grouping.png]

각 npz 는 그 케이스의 두 박스가 담긴 한 프레임. 프레임에 픽대상(박스류)이 정확히 2개면
그 둘을, 더 많으면 신뢰도 상위 2개를 쓴다. 여러 npz 를 주면 케이스 내 평균.
"""
import argparse
import os

import numpy as np

from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL
from dr_onsite_adap.runtime.detect import run_detection
from dr_onsite_adap.runtime.inference import SegModel

BLOCKS = [("색", slice(0, 3)), ("질감", slice(3, 5)), ("크기", slice(5, 7))]


def _pick_two(npz, model):
    d = np.load(npz, allow_pickle=True)
    bgr = d["rgb"]; depth = d["depth"]
    intr = tuple(float(x) for x in d["intrinsics"]); up = d["up_dir"].astype(float)
    h, w = bgr.shape[:2]
    st = run_detection(model, bgr, depth, intr, up, h, w, with_pointcloud=False, cluster=False)
    act = st["active"]
    if len(act) < 2:
        return None
    act = sorted(act, key=lambda a: -a["confidence"])[:2]
    feats, sizes = [], []
    for a in act:
        r = a["res"]; s = (r["long_side_mm"], r["short_side_mm"])
        geom = (r["center_3d"], r["long_axis_3d"], r["short_axis_3d"], s[0], s[1])
        feats.append(CL.prototype_feature(a["mask"].astype(bool), bgr, s,
                                          geom=geom, intrinsics=intr))
        sizes.append(s)
    return feats, sizes


def _block_dists(fa, fb, scale, bw):
    diff = (fa - fb) / scale * bw
    out = {name: float(np.linalg.norm(diff[sl])) for name, sl in BLOCKS}
    out["총"] = float(np.linalg.norm(diff))
    return out


def main():
    ap = argparse.ArgumentParser()
    for c in ("size", "color", "texture", "baseline"):
        ap.add_argument(f"--{c}", nargs="*", default=[])
    ap.add_argument("--lw", type=float, default=None, help="L_WEIGHT override(색case 밝기차 보려면 1.0)")
    ap.add_argument("--weights", default="/home/dr/lee3/thirdtask/lora_weights_b1/continual_master_v11.pth")
    ap.add_argument("--out", default=os.path.expanduser("~/prove_grouping.png"))
    args = ap.parse_args()
    if args.lw is not None:
        config.CLUSTER_L_WEIGHT = args.lw

    model = SegModel(args.weights, config.NUM_CLASSES, "cuda")
    scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, config)
    tau = config.CLUSTER_TAU

    cases = {"크기 다름": args.size, "색 다름": args.color,
             "질감 다름": args.texture, "같은 종류(기준)": args.baseline}
    rows = {}
    for case, files in cases.items():
        vals = []
        for f in files:
            r = _pick_two(f, model)
            if r is None:
                print(f"  ⚠ {f}: 픽대상 2개 미만 — 건너뜀"); continue
            (fa, fb), _sz = r
            vals.append(_block_dists(fa, fb, scale, bw))
        if vals:
            rows[case] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}

    print(f"\n=== 블록별 거리 (L_WEIGHT={config.CLUSTER_L_WEIGHT}, 구별 임계 TAU={tau}) ===")
    print(f"{'케이스':>14} | {'색':>6} {'질감':>6} {'크기':>6} | {'총거리':>6} | 구별")
    for case, r in rows.items():
        mark = "✅" if r["총"] > tau else "⚠"
        print(f"{case:>14} | {r['색']:>6.2f} {r['질감']:>6.2f} {r['크기']:>6.2f} | {r['총']:>6.2f} | {mark}")

    # 막대그래프 (라벨 영문 — 한글 폰트 의존 회피)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        EN = {"색": "Color", "질감": "Texture", "크기": "Size",
              "크기 다름": "Size diff", "색 다름": "Color diff",
              "질감 다름": "Texture(+size) diff", "같은 종류(기준)": "Baseline\n(same box)"}
        cs = list(rows.keys()); x = np.arange(len(cs)); wid = 0.25
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for k, (nm, _sl) in enumerate(BLOCKS):
            ax.bar(x + (k - 1) * wid, [rows[c][nm] for c in cs], wid, label=EN[nm])
        ax.axhline(tau, ls="--", color="red", lw=1, label=f"discrimination threshold {tau}")
        ax.set_xticks(x); ax.set_xticklabels([EN.get(c, c) for c in cs])
        ax.set_ylabel("normalized block distance")
        ax.set_title(f"Feature block distance per case (L_WEIGHT={config.CLUSTER_L_WEIGHT})")
        ax.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=130)
        print("그래프 저장:", args.out)
    except Exception as e:
        print("그래프 생략:", e)


if __name__ == "__main__":
    main()
