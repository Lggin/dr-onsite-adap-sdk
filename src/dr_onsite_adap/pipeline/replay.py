#!/usr/bin/env python3
"""[3단계] 학습용 데이터셋을 만듭니다(리플레이 혼합).

만들어지는 학습셋:
  train = 실측(현장에서 라벨한) 이미지 전부
          + 팔레트가 들어간 합성 이미지 일부(리플레이)
  valid = 실측 valid + 팔레트 들어간 합성 valid 일부

"리플레이"가 핵심입니다. 현장에서 모은 적은 양의 새 이미지만으로 학습하면 모델이
기존에 잘하던 클래스(특히 팔레트)를 잊어버립니다(catastrophic forgetting). 그래서
팔레트가 든 합성 이미지를 섞어 기존 능력을 붙잡아 둡니다 — 이 덕분에 적은 현장
데이터로도 안전하게 배포 가능한 모델이 나옵니다.

이미지는 심볼릭 링크로 연결합니다(원본 보존·디스크 절약). 파일명 충돌은 al_/rp_
접두어로 피합니다.

사용법:
  dr-onsite-replay                       # 기본 경로로 생성
  dr-onsite-replay --k-train 400 --k-valid 100   # 합성 리플레이 샘플 수 조절
  dr-onsite-replay --al DIR --pallet DIR --out DIR   # 경로 직접 지정
"""
import argparse
import json
import os
import random
from collections import Counter

from .. import config

CATS = [{"id": 0, "name": "Box", "supercategory": "object"},
        {"id": 1, "name": "Tote Box", "supercategory": "object"},
        {"id": 2, "name": "Shrink Wrap", "supercategory": "object"},
        {"id": 3, "name": "Pallet", "supercategory": "object"}]
NAMES = {0: "Box", 1: "Tote", 2: "Wrap", 3: "Pallet"}


def load(split, base):
    return json.load(open(os.path.join(base, split, "_annotations.coco.json")))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--al", default=config.DATA_DIR,
                    help="실측 액티브러닝 데이터셋 폴더")
    ap.add_argument("--pallet", default=config.PALLET_DATA_DIR,
                    help="팔레트 라벨이 있는 합성 데이터셋(리플레이 출처)")
    ap.add_argument("--out", default=config.REPLAY_OUT_DIR,
                    help="결과 데이터셋을 저장할 폴더")
    ap.add_argument("--k-train", type=int, default=400,
                    help="train 에 섞을 팔레트 포함 합성 이미지 수")
    ap.add_argument("--k-valid", type=int, default=100,
                    help="valid 에 섞을 팔레트 포함 합성 이미지 수")
    ap.add_argument("--exclude-prefix", default="lee5synth",
                    help="이 접두어로 시작하는 실측 이미지는 제외(실측에 섞여 있던 "
                         "옛 합성 데이터 누수 방지). 빈 문자열이면 전부 포함")
    args = ap.parse_args()
    AL, PAL, OUT = str(args.al), str(args.pallet), str(args.out)
    random.seed(0)

    for sp in ["train", "valid"]:
        os.makedirs(os.path.join(OUT, sp), exist_ok=True)
    for sp, K in [("train", args.k_train), ("valid", args.k_valid)]:
        merged = {"images": [], "annotations": [], "categories": CATS}
        iid = aid = 0
        cc = Counter()

        # 1) 실측 액티브러닝 (옵션으로 특정 접두어 제외)
        al = load(sp, AL)
        anns_by = {}
        for a in al["annotations"]:
            anns_by.setdefault(a["image_id"], []).append(a)
        for im in al["images"]:
            if args.exclude_prefix and im["file_name"].startswith(args.exclude_prefix):
                continue
            src = os.path.join(AL, sp, im["file_name"])
            if not os.path.isfile(src):
                continue
            newname = "al_" + im["file_name"]
            link = os.path.join(OUT, sp, newname)
            if not os.path.islink(link) and not os.path.exists(link):
                os.symlink(src, link)
            iid += 1
            merged["images"].append({"id": iid, "file_name": newname,
                                     "width": im.get("width", 0), "height": im.get("height", 0)})
            for a in anns_by.get(im["id"], []):
                aid += 1
                na = dict(a); na["id"] = aid; na["image_id"] = iid
                merged["annotations"].append(na); cc[a["category_id"]] += 1

        n_real = iid

        # 2) 팔레트가 든 합성 이미지(리플레이)
        pal = load(sp, PAL)
        panns = {}
        for a in pal["annotations"]:
            panns.setdefault(a["image_id"], []).append(a)
        pal_imgs = {im["id"]: im for im in pal["images"]}
        with_pallet = [i for i, al_ in panns.items() if any(a["category_id"] == 3 for a in al_)]
        random.shuffle(with_pallet)
        pick = with_pallet[:K]
        for pid in pick:
            im = pal_imgs[pid]
            src = os.path.join(PAL, sp, im["file_name"])
            if not os.path.isfile(src):
                continue
            newname = "rp_" + im["file_name"]
            link = os.path.join(OUT, sp, newname)
            if not os.path.islink(link) and not os.path.exists(link):
                os.symlink(src, link)
            iid += 1
            merged["images"].append({"id": iid, "file_name": newname,
                                     "width": im.get("width", 0), "height": im.get("height", 0)})
            for a in panns.get(pid, []):
                aid += 1
                na = dict(a); na["id"] = aid; na["image_id"] = iid
                merged["annotations"].append(na); cc[a["category_id"]] += 1

        with open(os.path.join(OUT, sp, "_annotations.coco.json"), "w") as f:
            json.dump(merged, f)
        print(f"[{sp}] {len(merged['images'])}장 (실측 {n_real} + 리플레이 {iid-n_real}) | "
              + ", ".join(f"{NAMES[k]}:{cc[k]}" for k in sorted(cc)))


if __name__ == "__main__":
    main()
