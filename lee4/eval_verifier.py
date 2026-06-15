"""
③ 검증기 평가 — DINOv2 검증 레이어가 RFDETR 오분류를 얼마나 잡고(catch),
맞은 검출을 얼마나 잘못 막는지(false-hold)를 거리 기준선 τ 에 따라 스윕한다.

파이프라인(평가 프레임 1장당):
  RFDETR(v7_7) 예측  ──IoU>0.5 매칭──▶  GT 클래스
        │                                   │
        └── pred 마스크로 크롭 → DINOv2 ──▶ kNN 투표(뱅크) → dino_pred, margin
  hold 규칙: dino_pred ≠ rfdetr  AND  margin ≥ τ   (margin = 득표율차)
  catch_rate  = RFDETR 틀린 것 중 hold 된 비율   (↑ 좋음)
  false_hold  = RFDETR 맞은 것 중 hold 된 비율   (↓ 좋음)

⚠️ 의미 있는 결과를 보려면 **RFDETR 이 학습에 쓰지 않은 신규 프레임**을 줘야 한다.
   (현재 lee3 valid 102장은 학습데이터라 오분류가 거의 없음 → 배관 검증용 데모만 됨.)
   신규 프레임은 뱅크와 겹치지 않으므로 --exclude_same_image 는 자동 무영향.

사용:
  HF_HUB_OFFLINE=0 python3 eval_verifier.py --eval_dir /path/to/frames
    (eval_dir 안에 _annotations.coco.json + jpg)
"""
import os, json, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import verifier as V
from inference import SegModel

OUT = os.path.join(os.path.dirname(__file__), "eval_out")
os.makedirs(OUT, exist_ok=True)
NAMES = V.CLASS_NAMES


def poly_to_mask(seg, H, W):
    m = np.zeros((H, W), np.uint8)
    if isinstance(seg, list):
        for poly in seg:
            if len(poly) >= 6:
                pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return inter / float(np.logical_or(a, b).sum())


def knn_vote(bank_emb, bank_lab, emb, k, drop=None):
    """drop: 제외할 뱅크 인덱스 boolean mask (leave-one-image-out). votes(정규화) 반환."""
    sims = bank_emb @ emb
    if drop is not None:
        sims = sims.copy(); sims[drop] = -2.0
    idx = np.argsort(-sims)[:k]
    votes = {}
    for j in idx:
        c = int(bank_lab[j])
        votes[c] = votes.get(c, 0.0) + float(max(sims[j], 1e-6))
    tot = sum(votes.values()) or 1.0
    return {c: v / tot for c, v in votes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir",
                    default="/home/dr/lee3/active_learning_data 1/active_learning_data/valid",
                    help="_annotations.coco.json + jpg 가 든 폴더")
    ap.add_argument("--max_images", type=int, default=0, help="0=전체")
    ap.add_argument("--conf", type=float, default=config.CONF_THRES)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=config.VERIFY_K)
    ap.add_argument("--false_hold_target", type=float, default=0.02)
    ap.add_argument("--exclude_same_image", action="store_true", default=True,
                    help="평가 이미지와 같은 출처의 뱅크 크롭을 kNN에서 제외(뱅크-소스 겹칠 때만 작동)")
    args = ap.parse_args()

    # ---- 뱅크 + DINOv2 ----
    bank = np.load(config.VERIFY_BANK_PATH, allow_pickle=True)
    bank_emb = bank["emb"].astype(np.float32)
    bank_lab = bank["labels"].astype(int)
    bank_src = bank["sources"] if "sources" in bank else np.array([None] * len(bank_lab))
    model, tf = V.load_dinov2(config.VERIFY_MODEL, config.VERIFY_IMG_SIZE,
                              config.VERIFY_DEVICE, weights_file=config.VERIFY_DINO_WEIGHTS)
    print(f"bank: {len(bank_lab)} crops | DINOv2 {config.VERIFY_MODEL}@{config.VERIFY_IMG_SIZE}")

    # ---- RFDETR ----
    seg = SegModel(config.WEIGHTS_PATH, config.NUM_CLASSES, config.DEVICE)

    coco = json.load(open(os.path.join(args.eval_dir, "_annotations.coco.json")))
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    img_ids = list(imgs.keys())
    if args.max_images:
        img_ids = img_ids[:args.max_images]

    # ---- 프레임별 추론 + 매칭 + 검증 ----
    dets = []   # 매칭된 검출 레코드
    n_pred = n_matched = n_spurious = n_missed = 0
    for n, iid in enumerate(img_ids):
        im = imgs[iid]
        fn = im["file_name"]
        img = cv2.imread(os.path.join(args.eval_dir, fn))
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = [(poly_to_mask(a["segmentation"], H, W), int(a["category_id"]))
               for a in anns_by_img.get(iid, [])]
        gt_used = [False] * len(gts)

        preds = seg.predict(img, conf_threshold=args.conf)
        pmasks = getattr(preds, "mask", None)
        if pmasks is None:
            n_missed += len(gts)
            continue
        drop = (bank_src == fn) if args.exclude_same_image else None

        for i in range(len(pmasks)):
            cid = int(preds.class_id[i])
            if cid not in config.CLASSES:
                continue
            n_pred += 1
            pmask = pmasks[i].astype(bool)
            # GT 매칭 (최대 IoU)
            best_j, best_iou = -1, 0.0
            for j, (gm, gc) in enumerate(gts):
                iou = mask_iou(pmask, gm)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou < args.iou:
                n_spurious += 1
                continue
            gt_cid = gts[best_j][1]
            gt_used[best_j] = True
            n_matched += 1
            # 검증기
            crop = V.masked_crop_from_mask(img, pmask)
            if crop is None:
                continue
            emb = V.embed_crops_rgb(model, tf, [crop], config.VERIFY_DEVICE)[0]
            votes = knn_vote(bank_emb, bank_lab, emb, args.k, drop=drop)
            dino_pred = max(votes, key=votes.get)
            margin = votes[dino_pred] - votes.get(cid, 0.0)
            dets.append({
                "file": fn, "rfdetr": cid, "gt": gt_cid, "dino": dino_pred,
                "margin": float(margin), "conf": float(preds.confidence[i]),
                "disagree": dino_pred != cid,
                "rf_correct": cid == gt_cid,
            })
        n_missed += sum(1 for u in gt_used if not u)
        if (n + 1) % 20 == 0:
            print(f"  ...{n+1}/{len(img_ids)} 프레임")

    # ---- 요약 ----
    rf_err = [d for d in dets if not d["rf_correct"]]
    rf_ok = [d for d in dets if d["rf_correct"]]
    print(f"\n프레임 {len(img_ids)} | pred {n_pred} | 매칭 {n_matched} | "
          f"spurious(FP) {n_spurious} | missed(FN) {n_missed}")
    print(f"매칭된 검출 중 RFDETR 오분류 {len(rf_err)} / 정분류 {len(rf_ok)}")
    if rf_err:
        from collections import Counter
        conf = Counter((NAMES[d['rfdetr']], NAMES[d['gt']]) for d in rf_err)
        print("오분류 유형 (RFDETR→GT):", {f"{a}→{b}": c for (a, b), c in conf.items()})

    # ---- τ 스윕 ----
    taus = np.linspace(0.0, 1.0, 51)
    rows = []
    for tau in taus:
        held_err = sum(1 for d in rf_err if d["disagree"] and d["margin"] >= tau)
        held_ok = sum(1 for d in rf_ok if d["disagree"] and d["margin"] >= tau)
        catch = held_err / len(rf_err) if rf_err else float("nan")
        fhold = held_ok / len(rf_ok) if rf_ok else 0.0
        rows.append((tau, catch, fhold))

    print(f"\n{'tau':>5} {'catch(오분류잡음)':>16} {'false_hold(오보류)':>18}")
    for tau, catch, fhold in rows[::5]:
        cs = "  n/a" if catch != catch else f"{catch:6.1%}"
        print(f"{tau:>5.2f} {cs:>16} {fhold:>17.1%}")

    # 권장 τ: false_hold ≤ target 중 catch 최대
    rec = None
    if rf_err:
        cand = [(t, c, f) for t, c, f in rows if f <= args.false_hold_target]
        if cand:
            rec = max(cand, key=lambda r: r[1])
            print(f"\n권장 τ={rec[0]:.2f}: catch={rec[1]:.1%}, "
                  f"false_hold={rec[2]:.1%} (목표 ≤{args.false_hold_target:.0%})")

    # ---- ROC ----
    catches = [c for _, c, _ in rows]
    fholds = [f for _, _, f in rows]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fholds, catches, "-o", ms=3)
    if rec:
        ax.scatter([rec[2]], [rec[1]], c="red", zorder=5,
                   label=f"recommended tau={rec[0]:.2f}")
        ax.axvline(args.false_hold_target, ls="--", c="gray", lw=1)
        ax.legend()
    ax.set_xlabel("false-hold rate (correct detections wrongly held)")
    ax.set_ylabel("catch rate (misclassifications caught)")
    ax.set_title("Verifier operating point (tau sweep)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/eval_roc.png", dpi=130); plt.close(fig)

    json.dump({
        "n_images": len(img_ids), "n_pred": n_pred, "n_matched": n_matched,
        "n_spurious": n_spurious, "n_missed": n_missed,
        "n_rf_err": len(rf_err), "n_rf_ok": len(rf_ok),
        "recommend_tau": (rec[0] if rec else None),
        "dets": dets,
        "sweep": [{"tau": t, "catch": c, "false_hold": f} for t, c, f in rows],
    }, open(f"{OUT}/eval_summary.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[done] -> {OUT}/eval_roc.png, eval_summary.json")


if __name__ == "__main__":
    main()
