#!/usr/bin/env python3
"""[준비] 데이터셋 준비 도구 (처음 한 번 / 가끔만 사용).

하위 명령:
  build-pallet    4클래스(+Pallet) 학습 데이터셋을 조립합니다. 팔레트 없는 실측
                  베이스(심볼릭 링크) + 팔레트가 라벨된 합성 이미지(복사)를 합치고
                  Pallet 카테고리를 선언합니다. dr-onsite-replay 의 리플레이 출처가
                  되는 데이터셋입니다.
  declare-pallet  기존 COCO 데이터셋에 Pallet(id 3) 카테고리만 선언합니다(라벨은
                  건드리지 않음 — 나중에 들어올 실측 팔레트 라벨 자리 확보).
                  여러 번 실행해도 안전하며, 먼저 백업합니다.

사용법:
  dr-onsite-dataset build-pallet --src-base DIR --synth DIR [DIR ...] --out DIR
  dr-onsite-dataset declare-pallet [--data-dir DIR]
"""
import argparse
import collections
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from .. import config

PALLET_CAT = {"id": config.PALLET_CID,
              "name": config.CLASSES[config.PALLET_CID][0],
              "supercategory": "object"}
NAMES = {0: "Box", 1: "Tote", 2: "Shrink Wrap", 3: "Pallet"}


# ---------------------------------------------------------------------------
# build-pallet
# ---------------------------------------------------------------------------
def _add_images(coco, out, src_dir, split, images, anns, prefix, link,
                img_id, ann_id):
    smap = {}
    for im in images:
        img_id += 1
        smap[im["id"]] = img_id
        newname = f"{prefix}{im['file_name']}" if prefix else im["file_name"]
        dst = out / split / newname
        srcf = src_dir / split / im["file_name"]
        if not dst.exists():
            if link:
                os.symlink(srcf, dst)
            else:
                shutil.copy(srcf, dst)
        ni = dict(im); ni["id"] = img_id; ni["file_name"] = newname
        coco["images"].append(ni)
    for an in anns:
        if an["image_id"] in smap:
            ann_id += 1
            na = dict(an); na["id"] = ann_id; na["image_id"] = smap[an["image_id"]]
            coco["annotations"].append(na)
    return img_id, ann_id


def _build_split(split, src_base, synth_dirs, out, exclude_prefix):
    (out / split).mkdir(parents=True, exist_ok=True)
    src = json.loads((src_base / split / "_annotations.coco.json").read_text())
    cats = src["categories"][:]
    if not any(c["id"] == config.PALLET_CID for c in cats):
        cats.append(PALLET_CAT)
    coco = {"info": src.get("info", {}), "licenses": src.get("licenses", []),
            "categories": cats, "images": [], "annotations": []}
    img_id = ann_id = 0

    # 베이스: 팔레트 없는 실측 이미지만(옛 합성 누수 제외), 심볼릭 링크
    base_imgs = [im for im in src["images"]
                 if not (exclude_prefix and im["file_name"].startswith(exclude_prefix))]
    base_ids = {im["id"] for im in base_imgs}
    base_anns = [an for an in src["annotations"] if an["image_id"] in base_ids]
    img_id, ann_id = _add_images(coco, out, src_base, split, base_imgs, base_anns,
                                 prefix="", link=True, img_id=img_id, ann_id=ann_id)
    n_base = img_id

    # 주입: 팔레트가 라벨된 합성 이미지(복사)
    for sdir in synth_dirs:
        sj = sdir / split / "_annotations.coco.json"
        if not sj.exists():
            continue
        sd = json.loads(sj.read_text())
        img_id, ann_id = _add_images(coco, out, sdir, split, sd["images"],
                                     sd["annotations"], prefix=f"{sdir.name}_",
                                     link=False, img_id=img_id, ann_id=ann_id)

    (out / split / "_annotations.coco.json").write_text(json.dumps(coco))
    cc = collections.Counter(a["category_id"] for a in coco["annotations"])
    print(f"[{split}] 이미지 {len(coco['images'])}장 (베이스 {n_base} + 합성 "
          f"{len(coco['images'])-n_base}) | 라벨 {len(coco['annotations'])}개")
    print(f"   클래스별: { {NAMES.get(k,k): v for k,v in sorted(cc.items())} }")


def cmd_build_pallet(args):
    src_base = Path(args.src_base)
    synth_dirs = [Path(s) for s in args.synth]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ["train", "valid"]:
        _build_split(split, src_base, synth_dirs, out, args.exclude_prefix)
    print(f"✅ 조립 완료 → {out}")


# ---------------------------------------------------------------------------
# declare-pallet
# ---------------------------------------------------------------------------
def cmd_declare_pallet(args):
    root = Path(args.data_dir)
    for split in ["train", "valid"]:
        p = root / split / "_annotations.coco.json"
        d = json.loads(p.read_text())
        ids = {c["id"] for c in d.get("categories", [])}
        if 3 in ids:
            print(f"[{split}] Pallet 이미 선언됨 — 건너뜀")
            continue
        shutil.copy(p, p.parent / f"{p.name}.bak_{datetime.now():%Y%m%d_%H%M%S}")
        d["categories"].append(PALLET_CAT)
        p.write_text(json.dumps(d))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        print(f"[{split}] Pallet 선언 추가 → categories={cats}")
    print("완료 (라벨 변경 없음).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-pallet", help="4클래스 팔레트 데이터셋 조립")
    bp.add_argument("--src-base", required=True,
                    help="심볼릭 링크할 베이스 COCO 데이터셋(실측, 팔레트 없음)")
    bp.add_argument("--synth", nargs="+", required=True,
                    help="주입할 팔레트 라벨 합성 COCO 폴더들")
    bp.add_argument("--out", default=str(config.PALLET_DATA_DIR),
                    help="결과 데이터셋 폴더")
    bp.add_argument("--exclude-prefix", default="lee5synth",
                    help="이 접두어로 시작하는 베이스 이미지는 제외")
    bp.set_defaults(func=cmd_build_pallet)

    dp = sub.add_parser("declare-pallet", help="데이터셋에 Pallet 카테고리 선언")
    dp.add_argument("--data-dir", default=str(config.DATA_DIR),
                    help="COCO 데이터셋 폴더(train/valid)")
    dp.set_defaults(func=cmd_declare_pallet)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
