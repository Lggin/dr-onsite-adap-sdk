#!/usr/bin/env python3
"""Append raw sam_hard triples (jpg + mask.png + json) produced by the live
SAM labeler into the active-learning COCO dataset (train + valid), without
destroying existing data — append only.

- Targets sam_hard_*.json at the dataset root that aren't already in train.
- Adds each to BOTH train and valid (the existing convention: valid is not a
  held-out split here, so LoRA train + any downstream consumer both see the
  new crops).
- category_id = the labeler's class_id (0=Box, 1=Tote, 2=Wrap, 3=Pallet).
- Timestamped backup of train/valid COCO before writing.

Usage:  depal-ingest [--data-dir DIR] [--dry-run]
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=config.ACTIVE_DATA_DIR,
                    help="active-learning dataset root (train/valid COCO + sam_hard triples)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data_dir = args.data_dir

    train_p = os.path.join(data_dir, "train", "_annotations.coco.json")
    valid_p = os.path.join(data_dir, "valid", "_annotations.coco.json")
    train = load(train_p)
    valid = load(valid_p)
    existing = {im["file_name"] for im in train["images"]}

    # sam_hard json at the root not yet in the COCO
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
            print(f"  ⚠️ jpg missing, skip: {jpg}")
            continue
        todo.append((jp, d, jpg, src))

    if not todo:
        print("No new sam_hard triples (all already in COCO).")
        return

    print(f"To add: {len(todo)} images")
    cc = Counter()
    for jp, d, jpg, src in todo:
        c = Counter(a["class_id"] for a in d["annotations"])
        cc += c
        print(f"  {jpg}  ann={len(d['annotations'])} {dict(c)}")
    print(f"class totals: {dict(cc)}")
    if args.dry_run:
        print("(--dry-run: no changes)")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(train_p, train_p + f".bak_ingest_{ts}")
    shutil.copy(valid_p, valid_p + f".bak_ingest_{ts}")
    print(f"backup: *.bak_ingest_{ts}")

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
    print(f"✅ train +{ti}img/{ta}ann (total {len(train['images'])}img), "
          f"valid +{vi}img/{va}ann (total {len(valid['images'])}img)")


if __name__ == "__main__":
    main()
