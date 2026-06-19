#!/usr/bin/env python3
"""[2단계] 라벨을 학습 데이터셋에 추가합니다.

라이브 앱('t' 키)의 SAM 라벨러가 만든 sam_hard 파일 묶음(jpg + mask.png + json)을
액티브러닝 COCO 데이터셋(train·valid)에 **기존 데이터를 지우지 않고 추가만** 합니다.

- 데이터셋 폴더 안의 sam_hard_*.json 중 아직 train 에 없는 것만 대상으로 합니다.
- train 과 valid **양쪽에 똑같이** 추가합니다(이 파이프라인의 valid 는 별도 검증셋이
  아니라 학습이 새 크롭을 보게 하려는 용도입니다).
- category_id 는 라벨러의 class_id 그대로(0=Box, 1=Tote, 2=Wrap, 3=Pallet).
- 쓰기 전에 train/valid COCO 를 타임스탬프 백업합니다.

사용법:
  dr-onsite-ingest                 # 기본 데이터셋 폴더에 추가
  dr-onsite-ingest --dry-run       # 무엇이 추가될지 미리보기(변경 없음)
  dr-onsite-ingest --data-dir DIR  # 데이터셋 폴더 직접 지정
"""
import argparse
import datetime
import glob
import json
import os
import shutil
from collections import Counter

import cv2

from .. import config


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=config.ACTIVE_DATA_DIR,
                    help="액티브러닝 데이터셋 폴더(train/valid COCO + sam_hard 파일). "
                         "기본값은 환경변수 DEPAL_ACTIVE_DATA_DIR / DEPAL_DATA_DIR")
    ap.add_argument("--dry-run", action="store_true",
                    help="실제로 추가하지 않고 대상만 출력")
    args = ap.parse_args()
    data_dir = args.data_dir

    train_p = os.path.join(data_dir, "train", "_annotations.coco.json")
    valid_p = os.path.join(data_dir, "valid", "_annotations.coco.json")
    test_p = os.path.join(data_dir, "test", "_annotations.coco.json")

    # 처음(데이터셋 없음)이면 빈 COCO(4클래스)로 생성 → 맨바닥에서 라벨→학습 가능.
    if not os.path.exists(train_p):
        cats = [{"id": 0, "name": "Box", "supercategory": "object"},
                {"id": 1, "name": "Tote Box", "supercategory": "object"},
                {"id": 2, "name": "Shrink Wrap", "supercategory": "object"},
                {"id": 3, "name": "Pallet", "supercategory": "object"}]
        for sp_p in (train_p, valid_p, test_p):
            os.makedirs(os.path.dirname(sp_p), exist_ok=True)
            with open(sp_p, "w") as f:
                json.dump({"images": [], "annotations": [], "categories": cats}, f)
        print(f"🆕 빈 데이터셋 생성(train/valid/test, 4클래스) → {data_dir}")

    train = load(train_p)
    valid = load(valid_p)
    existing = {im["file_name"] for im in train["images"]}

    # 폴더 안 sam_hard json 중 아직 COCO 에 없는 것
    todo = []
    for jp in sorted(glob.glob(os.path.join(data_dir, "sam_hard_*.json"))):
        if jp.endswith("_mask.png"):
            continue
        d = load(jp)
        jpg = d.get("image_file") or os.path.basename(jp).replace(".json", ".jpg")
        if jpg in existing:
            continue
        src = os.path.join(data_dir, jpg)
        if not os.path.isfile(src):
            print(f"  ⚠️ jpg 없음, 건너뜀: {jpg}")
            continue
        todo.append((jp, d, jpg, src))

    if not todo:
        print("추가할 새 sam_hard 라벨이 없습니다(모두 이미 들어있음).")
        return

    print(f"추가 대상: {len(todo)}장")
    cc = Counter()
    for jp, d, jpg, src in todo:
        c = Counter(a["class_id"] for a in d["annotations"])
        cc += c
        print(f"  {jpg}  라벨 {len(d['annotations'])}개 {dict(c)}")
    print(f"클래스 합계: {dict(cc)}")
    if args.dry_run:
        print("(--dry-run: 변경 없음)")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(train_p, train_p + f".bak_ingest_{ts}")
    shutil.copy(valid_p, valid_p + f".bak_ingest_{ts}")
    print(f"백업 생성: *.bak_ingest_{ts}")

    def append_to(coco, split):
        nimg = max([im["id"] for im in coco["images"]] + [0])
        nann = max([a["id"] for a in coco["annotations"]] + [0])
        added_i = added_a = 0
        for jp, d, jpg, src in todo:
            dst = os.path.join(data_dir, split, jpg)
            shutil.copy(src, dst)
            img = cv2.imread(dst)
            H, W = img.shape[:2]
            nimg += 1
            img_id = nimg
            coco["images"].append({"id": img_id, "file_name": jpg,
                                   "width": int(W), "height": int(H)})
            added_i += 1
            for a in d["annotations"]:
                if "bbox" not in a or len(a["bbox"]) != 4:
                    continue
                x1, y1, x2, y2 = a["bbox"]
                w, h = x2 - x1, y2 - y1
                nann += 1
                coco["annotations"].append({
                    "id": nann, "image_id": img_id,
                    "category_id": int(a["class_id"]),
                    "bbox": [x1, y1, w, h],
                    "segmentation": [a.get("segmentation", [])],
                    "area": int(w * h), "iscrowd": 0,
                })
                added_a += 1
        return added_i, added_a

    ti, ta = append_to(train, "train")
    vi, va = append_to(valid, "valid")
    with open(train_p, "w") as f:
        json.dump(train, f)
    with open(valid_p, "w") as f:
        json.dump(valid, f)
    # test split 이 있으면(처음 생성 시 포함) 동일하게 채워 학습 로더가 안전하게 동작.
    if os.path.exists(test_p):
        test = load(test_p)
        append_to(test, "test")
        with open(test_p, "w") as f:
            json.dump(test, f)
    print(f"✅ train +{ti}장/{ta}라벨 (총 {len(train['images'])}장), "
          f"valid +{vi}장/{va}라벨 (총 {len(valid['images'])}장)")


if __name__ == "__main__":
    main()
