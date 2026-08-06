#!/usr/bin/env python3
"""[게이트] 라운드 결과를 현장 held-out 으로 채점해 회귀면 배포를 막습니다 (Phase 2).

`dr-onsite-eval` 은 사람이 표를 보고 판단하는 도구입니다. 이건 **기계가 판정**합니다:
클래스별 AP@0.5·recall 을 base 와 비교해 임계 이상 떨어지면 **0 이 아닌 종료코드**를
돌려주므로, 라운드 스크립트가 그 자리에서 멈출 수 있습니다.

판정 규칙
  · 붕괴(collapse)  — base recall ≥ 0.3 인 클래스가 candidate 에서 절반 이하로 떨어짐 → FAIL
  · 하락(drop)      — AP 또는 recall 이 `--max-*-drop` 초과로 떨어짐 → FAIL
  · 판정보류(SKIP)  — GT 인스턴스가 `--min-instances` 미만이면 표본이 얇아 판정하지 않음
  · 전이(transfer)  — GT 가 다른 클래스로 예측된 건수를 함께 출력(예: 팔레트→슬립시트).
                      mAP 한 숫자는 이 전이를 숨긴다. 2026-08-04 실험에서 부분 라벨
                      한 라운드로 팔레트가 120/120 → 0/120 이 됐는데 회귀 mAP 는 0.96 이었다.

누수 검사: 평가셋 이미지가 학습셋에 있으면 그 채점은 무의미하므로 **먼저 실패**시킵니다
(파일 내용 해시 비교. `--allow-leak` 로만 무시 가능).

사용법:
  dr-onsite-gate --base 이전.pth --cand 새것.pth
  dr-onsite-gate --base a.pth --cand b.pth --data /경로/holdout/valid --n 200
  dr-onsite-gate ... --max-ap-drop 0.03 --min-instances 15 --out gate.md
종료코드: 0=통과, 1=회귀(배포 금지), 2=실행 불가(누수·데이터 없음 등)
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

from .. import config
from .._logging import ensure_utf8_output
from ..runtime.inference import SegModel
from .eval import CLASSES, CONF_EVAL, ap_from, iou

CONF_MATCH = 0.3      # 혼동행렬·recall 판정용(운영 임계와 같은 눈높이)
IOU_MATCH = 0.5


def _sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def leak_check(eval_dir, train_dirs):
    """평가셋 이미지가 학습셋에 있으면 그 파일명들을 돌려준다(내용 해시 기준)."""
    ev = {}
    for p in glob.glob(os.path.join(eval_dir, "*.jpg")) + glob.glob(os.path.join(eval_dir, "*.png")):
        ev[_sha1(p)] = os.path.basename(p)
    if not ev:
        return []
    hit = []
    for d in train_dirs:
        for split in ("train", "valid", "test"):
            for p in glob.glob(os.path.join(d, split, "*.jpg")) + glob.glob(os.path.join(d, split, "*.png")):
                h = _sha1(p)
                if h in ev:
                    hit.append(f"{ev[h]}  ←  {os.path.join(os.path.basename(d), split)}")
    return sorted(set(hit))


def load_eval_set(data_dir, n):
    """COCO 폴더 → (imgs, gt, sel, ng). 팔레트 포함 프레임을 우선 담는다."""
    jp = os.path.join(data_dir, "_annotations.coco.json")
    if not os.path.isfile(jp):
        return None, None, None, None
    coco = json.load(open(jp))
    imgs = {im["id"]: im for im in coco["images"]}
    gt = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt.setdefault(a["image_id"], []).append((a["category_id"], [x, y, x + w, y + h]))
    pal = [i for i, g in gt.items() if any(c == config.PALLET_CID for c, _ in g)]
    rest = [i for i in sorted(imgs) if i not in set(pal)]
    sel = (sorted(pal) + rest)[:n]
    ng = {c: 0 for c in CLASSES}
    for i in sel:
        for c, _ in gt.get(i, []):
            ng[c] += 1
    return imgs, gt, sel, ng


def score(weights, data_dir, imgs, gt, sel, ng, device):
    """가중치 하나를 채점 → {'ap','recall','confusion'}.

    AP 는 낮은 임계(CONF_EVAL)에서 곡선으로, recall·혼동행렬은 운영 눈높이(CONF_MATCH)에서
    구한다. 혼동행렬은 클래스 무관 최대 IoU 매칭이라 "어디로 갔는지"가 보인다.
    """
    import cv2

    model = SegModel(weights, config.NUM_CLASSES, device)
    tps = {c: [] for c in CLASSES}
    rec_tp = {c: 0 for c in CLASSES}
    conf = {c: {"미검출": 0, **{k: 0 for k in CLASSES}} for c in CLASSES}
    try:
        for i in sel:
            im = imgs[i]
            img = cv2.imread(os.path.join(data_dir, im["file_name"]))
            if img is None:
                continue
            preds = model.predict(img, conf_threshold=CONF_EVAL)
            pb, pc, ps = preds.xyxy, preds.class_id, preds.confidence
            gl = gt.get(i, [])

            # (1) 클래스별 AP·recall
            for c in CLASSES:
                gboxes = np.array([b for cc, b in gl if cc == c], float).reshape(-1, 4)
                used = np.zeros(len(gboxes), bool)
                for j in np.argsort(-ps):
                    if int(pc[j]) != c:
                        continue
                    sc = float(ps[j])
                    if len(gboxes) == 0:
                        tps[c].append((sc, 0))
                        continue
                    v = iou(pb[j], gboxes)
                    v[used] = -1
                    k = int(np.argmax(v))
                    if v[k] >= IOU_MATCH:
                        used[k] = True
                        tps[c].append((sc, 1))
                        if sc >= CONF_MATCH:
                            rec_tp[c] += 1
                    else:
                        tps[c].append((sc, 0))

            # (2) 혼동행렬 — GT 마다 클래스 무관 최고 IoU 예측 하나
            order = [j for j in np.argsort(-ps) if float(ps[j]) >= CONF_MATCH]
            taken = set()
            for cc, box in gl:
                best, best_v = None, IOU_MATCH
                for j in order:
                    if j in taken:
                        continue
                    v = float(iou(np.array(box, float), np.array([pb[j]], float))[0])
                    if v >= best_v:
                        best, best_v = j, v
                if best is None:
                    conf[cc]["미검출"] += 1
                else:
                    taken.add(best)
                    conf[cc][int(pc[best])] += 1
    finally:
        del model
        import torch
        torch.cuda.empty_cache()

    return {
        "ap": {c: ap_from(tps[c], ng[c]) for c in CLASSES},
        "recall": {c: (rec_tp[c] / ng[c] if ng[c] else None) for c in CLASSES},
        "confusion": conf,
    }


def judge(base, cand, ng, args):
    """클래스별 판정 → (rows, verdict). rows = [(cls, 상태, 사유)]."""
    rows, failed = [], False
    for c in CLASSES:
        name = CLASSES[c]
        n = ng[c]
        b_ap, c_ap = base["ap"][c], cand["ap"][c]
        b_rc, c_rc = base["recall"][c], cand["recall"][c]
        if n < args.min_instances or b_ap is None or c_ap is None:
            rows.append((name, n, b_ap, c_ap, b_rc, c_rc, "SKIP",
                         f"GT {n}개 < {args.min_instances} — 표본 부족, 판정보류"))
            continue
        d_ap, d_rc = c_ap - b_ap, c_rc - b_rc
        why = []
        if b_rc >= 0.3 and c_rc <= b_rc * 0.5:
            why.append(f"붕괴: recall {b_rc:.2f}→{c_rc:.2f} (절반 이하)")
        if d_ap < -args.max_ap_drop:
            why.append(f"AP {d_ap:+.3f} (허용 -{args.max_ap_drop})")
        if d_rc < -args.max_recall_drop:
            why.append(f"recall {d_rc:+.3f} (허용 -{args.max_recall_drop})")
        # 어디로 갔는지: 이 클래스 GT 가 다른 클래스로 예측된 건수(candidate 기준)
        moved = {CLASSES[k]: v for k, v in cand["confusion"][c].items()
                 if isinstance(k, int) and k != c and v > 0}
        if why and moved:
            why.append("전이 " + ", ".join(f"→{k} {v}건" for k, v in sorted(moved.items(), key=lambda x: -x[1])))
        state = "FAIL" if why else "PASS"
        failed |= state == "FAIL"
        rows.append((name, n, b_ap, c_ap, b_rc, c_rc, state, " · ".join(why) or "이상 없음"))
    return rows, ("FAIL" if failed else "PASS")


def render(rows, verdict, meta):
    L = [f"# 회귀 게이트 — {verdict}", "",
         f"- base: `{meta['base']}`", f"- candidate: `{meta['cand']}`",
         f"- 평가셋: `{meta['data']}` (이미지 {meta['n_img']}장)",
         f"- 임계: AP -{meta['max_ap_drop']} / recall -{meta['max_recall_drop']} / 최소 GT {meta['min_instances']}개",
         "",
         "| 클래스 | GT | base AP | cand AP | base recall | cand recall | 판정 | 사유 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, n, b_ap, c_ap, b_rc, c_rc, state, why in rows:
        f = lambda v: "—" if v is None else f"{v:.3f}"   # noqa: E731
        mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚪"}[state]
        L.append(f"| {name} | {n} | {f(b_ap)} | {f(c_ap)} | {f(b_rc)} | {f(c_rc)} | {mark} {state} | {why} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=config.BASE_WEIGHTS, help="비교 기준 가중치(라운드 시작점)")
    ap.add_argument("--cand", default=config.FINAL_WEIGHTS, help="판정할 가중치(라운드 결과)")
    ap.add_argument("--data", default=os.path.join(config.HOLDOUT_DATA_DIR, "valid"),
                    help="현장 held-out COCO 폴더(학습에 안 쓴 것)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-ap-drop", type=float, default=0.05)
    ap.add_argument("--max-recall-drop", type=float, default=0.05)
    ap.add_argument("--min-instances", type=int, default=10)
    ap.add_argument("--out", default=None, help="판정표 저장(.md). 같은 이름 .json 도 함께 저장")
    ap.add_argument("--allow-leak", action="store_true", help="누수가 있어도 진행(권장하지 않음)")
    args = ap.parse_args()
    ensure_utf8_output()

    imgs, gt, sel, ng = load_eval_set(args.data, args.n)
    if imgs is None:
        print(f"⛔ 평가셋이 없습니다: {args.data}\n"
              "   현장 프레임을 찍어 `dr-onsite-ingest --holdout` 으로 쌓으세요 "
              "(학습셋과 분리되어 이 게이트 전용으로만 쓰입니다).", file=sys.stderr)
        return 2
    if not sel:
        print(f"⛔ 평가셋에 이미지가 없습니다: {args.data}", file=sys.stderr)
        return 2

    leaks = leak_check(args.data, [config.TRAIN_DATA_DIR, config.ACTIVE_DATA_DIR])
    if leaks:
        head = "\n".join("     " + s for s in leaks[:8])
        msg = (f"⛔ 누수 {len(leaks)}건 — 평가셋 이미지가 학습셋에 있습니다:\n{head}\n"
               "   학습에 쓴 프레임으로 채점하면 회귀가 가려집니다.")
        if not args.allow_leak:
            print(msg + "\n   (정말 무시하려면 --allow-leak)", file=sys.stderr)
            return 2
        print(msg + "\n   --allow-leak 로 계속합니다(판정 신뢰도 낮음).")

    print(f"평가셋 {len(sel)}장 | GT: " + ", ".join(f"{CLASSES[c]}:{ng[c]}" for c in CLASSES), flush=True)
    print(f"1/2 base 채점 … {args.base}", flush=True)
    base = score(args.base, args.data, imgs, gt, sel, ng, args.device)
    print(f"2/2 candidate 채점 … {args.cand}", flush=True)
    cand = score(args.cand, args.data, imgs, gt, sel, ng, args.device)

    rows, verdict = judge(base, cand, ng, args)
    meta = {"base": args.base, "cand": args.cand, "data": args.data, "n_img": len(sel),
            "max_ap_drop": args.max_ap_drop, "max_recall_drop": args.max_recall_drop,
            "min_instances": args.min_instances}
    md = render(rows, verdict, meta)
    print("\n" + md)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(md)
        js = {"verdict": verdict, "meta": meta, "gt_counts": {CLASSES[c]: ng[c] for c in CLASSES},
              "base": {"ap": {CLASSES[c]: base["ap"][c] for c in CLASSES},
                       "recall": {CLASSES[c]: base["recall"][c] for c in CLASSES}},
              "cand": {"ap": {CLASSES[c]: cand["ap"][c] for c in CLASSES},
                       "recall": {CLASSES[c]: cand["recall"][c] for c in CLASSES},
                       "confusion": {CLASSES[c]: {(CLASSES[k] if isinstance(k, int) else k): v
                                                  for k, v in cand["confusion"][c].items()}
                                     for c in CLASSES}},
              "rows": [{"class": r[0], "gt": r[1], "state": r[6], "why": r[7]} for r in rows]}
        json.dump(js, open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"저장: {args.out} / {os.path.splitext(args.out)[0]}.json")

    if verdict == "FAIL":
        print("\n❌ 회귀 감지 — 이 가중치는 배포하지 마세요.", file=sys.stderr)
        return 1
    print("\n✅ 통과 — 회귀 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
