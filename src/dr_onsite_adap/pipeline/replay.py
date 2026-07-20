#!/usr/bin/env python3
"""[3단계] 학습용 데이터셋을 만듭니다(리플레이 혼합, 클래스 층화추출).

만들어지는 학습셋:
  train = 실측(현장에서 라벨한) 이미지 전부
          + 리플레이 소스에서 클래스 층화추출한 이미지 일부
  valid = 실측 valid + 리플레이 소스 valid 층화추출 일부

"리플레이"가 핵심입니다. 현장에서 모은 적은 양의 새 이미지만으로 학습하면 모델이
기존에 잘하던 클래스를 잊어버립니다(catastrophic forgetting). 그래서 기존 데이터를
섞어 능력을 붙잡아 둡니다.

⚠️ 무작위 추출의 함정: 소스에 희귀 클래스(예 슬립시트 ~4.5%)가 있으면 무작위로 K장
뽑을 때 그 클래스가 통째로 빠질 수 있어 보존이 안 됩니다. 그래서 **클래스 층화추출**을
씁니다 — 희귀 클래스부터 각 클래스에 최소 per_class_min 장을 보장한 뒤 나머지를
무작위로 채웁니다. 이 덕분에 팔레트뿐 아니라 5클래스 전부가 리플레이에 확실히 포함됩니다.

리플레이 소스는 DEPAL_REPLAY_SRC(기본: 팔레트 합성셋, 하위호환)로 지정합니다.
전 클래스 보존을 원하면 final_dataset(5클래스) 로 지정하세요.

이미지는 심볼릭 링크로 연결합니다(원본 보존·디스크 절약). 파일명 충돌은 al_/rp_
접두어로 피합니다.

사용법:
  dr-onsite-replay                       # config 기본 경로/설정으로 생성
  dr-onsite-replay --k-train 400 --k-valid 100   # 리플레이 샘플 수 조절
  dr-onsite-replay --per-class-min 60            # 클래스별 최소 보장 수
  dr-onsite-replay --al DIR --replay-src DIR --out DIR   # 경로 직접 지정
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

from .. import config

# config.CLASSES 단일 소스에서 파생 → 클래스 순서 변경 시 자동 정합(v11: 0:Pallet..4:Shrink Wrap).
CATS = [{"id": cid, "name": name, "supercategory": "object"}
        for cid, (name, _c) in sorted(config.CLASSES.items())]
NAMES = {cid: name for cid, (name, _c) in config.CLASSES.items()}


def load(split, base):
    return json.load(open(os.path.join(base, split, "_annotations.coco.json")))


def stratified_pick(imgs_by_cat, all_ids, K, per_class_min, rng):
    """리플레이용 이미지 id 를 클래스 층화추출한다.

    imgs_by_cat: {cid: [해당 클래스를 포함한 image_id, ...]}
    all_ids:     소스의 전체 image_id (2단계 무작위 채움용)
    반환: 최대 K 개의 image_id 리스트.

    희귀 클래스(가용 이미지 적은 순)부터 각 클래스에 per_class_min 장(가용 범위 내)을
    보장한 뒤, 남는 슬롯을 전체에서 무작위로 채운다. per_class_min*클래스수 > K 여도
    희귀 클래스가 먼저 채워지므로 희귀 클래스 보존이 우선된다.
    """
    picked, seen = [], set()
    # 1) 클래스별 최소 보장 — 희귀 클래스 우선(가용 수 오름차순).
    order = sorted((c for c in imgs_by_cat if imgs_by_cat[c]),
                   key=lambda c: len(imgs_by_cat[c]))
    for c in order:
        if len(picked) >= K:
            break
        pool = [i for i in imgs_by_cat[c] if i not in seen]
        rng.shuffle(pool)
        for i in pool[:per_class_min]:
            if len(picked) >= K:
                break
            picked.append(i); seen.add(i)
    # 2) 나머지를 전체에서 무작위로 채움.
    rest = [i for i in all_ids if i not in seen]
    rng.shuffle(rest)
    for i in rest:
        if len(picked) >= K:
            break
        picked.append(i); seen.add(i)
    return picked


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--al", default=config.DATA_DIR,
                    help="실측 액티브러닝 데이터셋 폴더")
    ap.add_argument("--replay-src", dest="replay_src", default=config.REPLAY_SRC_DIR,
                    help="리플레이 소스 데이터셋(층화추출 출처). 전 클래스 보존은 "
                         "final_dataset 지정")
    ap.add_argument("--pallet", dest="replay_src_alias", default=None,
                    help="(구) 리플레이 소스 별칭. --replay-src 와 동일, 하위호환용")
    ap.add_argument("--out", default=config.REPLAY_OUT_DIR,
                    help="결과 데이터셋을 저장할 폴더")
    ap.add_argument("--k-train", type=int, default=400,
                    help="train 에 섞을 리플레이 이미지 수")
    ap.add_argument("--k-valid", type=int, default=100,
                    help="valid 에 섞을 리플레이 이미지 수")
    ap.add_argument("--per-class-min", type=int, default=config.REPLAY_PER_CLASS_MIN,
                    help="리플레이에서 각 클래스별 최소 보장 이미지 수(가용 범위 내)")
    ap.add_argument("--exclude-prefix", default="lee5synth",
                    help="이 접두어로 시작하는 실측 이미지는 제외(실측에 섞여 있던 "
                         "옛 합성 데이터 누수 방지). 빈 문자열이면 전부 포함")
    args = ap.parse_args()
    AL = str(args.al)
    SRC = str(args.replay_src_alias or args.replay_src)
    OUT = str(args.out)
    PCM = max(0, int(args.per_class_min))
    rng = random.Random(0)   # 결정론적 추출(라운드 재현성)

    print(f"🔀 리플레이 소스: {SRC}  (클래스별 최소 {PCM}장 보장 + 무작위 채움)")
    for sp in ["train", "valid"]:
        os.makedirs(os.path.join(OUT, sp), exist_ok=True)
    for sp, K in [("train", args.k_train), ("valid", args.k_valid)]:
        merged = {"images": [], "annotations": [], "categories": CATS}
        iid = aid = 0
        cc = Counter()

        # 1) 실측 액티브러닝 (옵션으로 특정 접두어 제외) — 전량 포함.
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

        # 2) 리플레이 소스에서 클래스 층화추출.
        rp = load(sp, SRC)
        ranns = defaultdict(list)
        for a in rp["annotations"]:
            ranns[a["image_id"]].append(a)
        rp_imgs = {im["id"]: im for im in rp["images"]}
        imgs_by_cat = defaultdict(list)
        for i, alist in ranns.items():
            for c in {a["category_id"] for a in alist}:
                imgs_by_cat[c].append(i)
        pick = stratified_pick(imgs_by_cat, list(rp_imgs.keys()), K, PCM, rng)

        # 리플레이 선택 결과의 클래스별 커버리지 요약(보존 확인용).
        cov = Counter()
        for pid in pick:
            for c in {a["category_id"] for a in ranns.get(pid, [])}:
                cov[c] += 1

        for pid in pick:
            im = rp_imgs[pid]
            src = os.path.join(SRC, sp, im["file_name"])
            if not os.path.isfile(src):
                continue
            newname = "rp_" + im["file_name"]
            link = os.path.join(OUT, sp, newname)
            if not os.path.islink(link) and not os.path.exists(link):
                os.symlink(src, link)
            iid += 1
            merged["images"].append({"id": iid, "file_name": newname,
                                     "width": im.get("width", 0), "height": im.get("height", 0)})
            for a in ranns.get(pid, []):
                aid += 1
                na = dict(a); na["id"] = aid; na["image_id"] = iid
                merged["annotations"].append(na); cc[a["category_id"]] += 1

        with open(os.path.join(OUT, sp, "_annotations.coco.json"), "w") as f:
            json.dump(merged, f)
        cov_s = ", ".join(f"{NAMES.get(k, k)}:{cov[k]}" for k in sorted(cov))
        print(f"[{sp}] {len(merged['images'])}장 (실측 {n_real} + 리플레이 {iid-n_real}) | "
              + ", ".join(f"{NAMES.get(k, k)}:{cc[k]}" for k in sorted(cc)))
        print(f"        리플레이 클래스 커버리지(이미지 수): {cov_s}")


if __name__ == "__main__":
    main()
