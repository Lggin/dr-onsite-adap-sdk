#!/usr/bin/env python3
"""[검증] 검출 모델 가중치들을 비교합니다 — 클래스별 AP@0.5와 recall.

주 용도: 증분 학습(LoRA 라운드) 뒤에 특정 클래스가 망가지지 않았는지 확인하는
것입니다(예: 실측에 팔레트 라벨이 없어 팔레트 AP가 0으로 붕괴 — 리플레이가 막아
주려는 그 현상). held-out(학습에 안 쓴) 데이터로 평가합니다.

사용법:
  dr-onsite-eval --models 이전=/경로/a.pth 새것=/경로/b.pth
  dr-onsite-eval --models v8=/a.pth v8_3=/b.pth --data DIR --n 250
"""
import argparse
import json
import os
import random

import numpy as np

from .. import config
from .._logging import ensure_utf8_output
from ..runtime.inference import SegModel

CLASSES = {cid: name for cid, (name, _) in config.CLASSES.items()}   # config 순서 따름
CONF_EVAL = 0.05   # AP 곡선용 낮은 임계값


def iou(a, B):
    if len(B) == 0:
        return np.zeros((0,))
    ix1 = np.maximum(a[0], B[:, 0]); iy1 = np.maximum(a[1], B[:, 1])
    ix2 = np.minimum(a[2], B[:, 2]); iy2 = np.minimum(a[3], B[:, 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None); inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + ab - inter + 1e-9)


def ap_from(tp_scores, n_gt):
    # tp_scores: list[(score, is_tp)]; all-point interpolation AP
    if n_gt == 0:
        return None
    if not tp_scores:
        return 0.0
    tp_scores.sort(key=lambda x: -x[0])
    tp = np.array([t for _, t in tp_scores]); fp = 1 - tp
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp / n_gt; prec = ctp / (ctp + cfp + 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="'이름=경로' 쌍. 예: 이전=/a.pth 새것=/b.pth")
    ap.add_argument("--data", default=os.path.join(str(config.PALLET_DATA_DIR), "valid"),
                    help="평가용 COCO valid 폴더(held-out)")
    ap.add_argument("--n", type=int, default=250, help="평가할 이미지 수")
    ap.add_argument("--device", default="cuda", help="cpu 또는 cuda")
    args = ap.parse_args()
    ensure_utf8_output()

    models = {}
    for spec in args.models:
        if "=" not in spec:
            ap.error(f"--models 항목은 '이름=경로' 형식이어야 합니다: {spec!r}")
        name, path = spec.split("=", 1)
        models[name] = path

    import cv2
    coco = json.load(open(os.path.join(args.data, "_annotations.coco.json")))
    imgs = {im["id"]: im for im in coco["images"]}
    gt = {}  # img_id -> list[(cls, [x1y1x2y2])]
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt.setdefault(a["image_id"], []).append((a["category_id"], [x, y, x + w, y + h]))
    # 팔레트 포함 이미지를 우선 선택한 뒤 나머지로 채움
    pal = [i for i, g in gt.items() if any(c == config.PALLET_CID for c, _ in g)]
    rest = [i for i in imgs if i not in pal]
    random.seed(0); random.shuffle(rest)
    sel = pal[:200] + rest[:max(0, args.n - min(len(pal), 200))]
    sel = sel[:args.n]
    ng = {c: 0 for c in CLASSES}
    for i in sel:
        for c, _ in gt.get(i, []):
            ng[c] += 1
    print(f"평가 이미지 {len(sel)}장 (팔레트 포함 {sum(i in pal for i in sel)}장) | 정답 라벨: "
          + ", ".join(f"{CLASSES[c]}:{ng[c]}" for c in CLASSES))

    results = {}
    for name, wp in models.items():
        model = SegModel(wp, config.NUM_CLASSES, args.device)
        tps = {c: [] for c in CLASSES}; rec_tp = {c: 0 for c in CLASSES}
        for i in sel:
            im = imgs[i]; img = cv2.imread(os.path.join(args.data, im["file_name"]))
            if img is None:
                continue
            preds = model.predict(img, conf_threshold=CONF_EVAL)
            pb = getattr(preds, "xyxy", None); pc = preds.class_id; ps = preds.confidence
            gl = gt.get(i, [])
            for c in CLASSES:
                gboxes = np.array([b for cc, b in gl if cc == c], float).reshape(-1, 4)
                used = np.zeros(len(gboxes), bool)
                order = np.argsort(-ps)
                for j in order:
                    if int(pc[j]) != c:
                        continue
                    sc = float(ps[j]); bb = pb[j]
                    if len(gboxes) == 0:
                        tps[c].append((sc, 0)); continue
                    iou_v = iou(bb, gboxes); iou_v[used] = -1
                    k = int(np.argmax(iou_v))
                    if iou_v[k] >= 0.5:
                        used[k] = True; tps[c].append((sc, 1)); rec_tp[c] += 1
                    else:
                        tps[c].append((sc, 0))
        aps = {c: ap_from(tps[c], ng[c]) for c in CLASSES}
        results[name] = (aps, rec_tp)
        del model
        import torch
        torch.cuda.empty_cache()

    names = list(results.keys())
    print("\n=== 클래스별 AP@0.5  (recall) ===")
    print(f"{'클래스':8} " + " ".join(f"{nm:>16}" for nm in names))

    def fmt(ap_v, tp, c):
        return "  GT0" if ap_v is None else f"{ap_v:.3f} (r={tp/ng[c]:.2f})"

    for c in CLASSES:
        cells = []
        for nm in names:
            aps, rec = results[nm]
            cells.append(f"{fmt(aps[c], rec[c], c):>16}")
        print(f"{CLASSES[c]:8} " + " ".join(cells))
    print("\n핵심: 이전 버전보다 Pallet AP/recall 이 떨어졌다면 팔레트 헤드 열화입니다.")


if __name__ == "__main__":
    main()
