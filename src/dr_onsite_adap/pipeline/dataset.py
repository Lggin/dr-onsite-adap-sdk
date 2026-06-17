#!/usr/bin/env python3
"""Dataset setup helpers (one-off / occasional).

Subcommands:
  build-pallet    Assemble a 4-class (+Pallet) training dataset: a real
                  pallet-free base (symlinked) + pallet-labeled synthetic
                  images (copied), with a Pallet category declared. This is
                  the replay source consumed by `depal-replay`.
  declare-pallet  Add the Pallet (id 3) category to an existing COCO dataset
                  without touching annotations (reserves the slot for future
                  real pallet labels). Idempotent; backs up first.

Usage:
  depal-dataset build-pallet --src-base DIR --synth DIR [DIR ...] --out DIR
  depal-dataset declare-pallet [--data-dir DIR]
"""
import argparse
import collections
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from .. import config

PALLET_CAT = {"id": 3, "name": "Pallet", "supercategory": "object"}
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
    if not any(c["id"] == 3 for c in cats):
        cats.append(PALLET_CAT)
    coco = {"info": src.get("info", {}), "licenses": src.get("licenses", []),
            "categories": cats, "images": [], "annotations": []}
    img_id = ann_id = 0

    # base: pallet-free real images only (exclude legacy leaky synth), symlinked
    base_imgs = [im for im in src["images"]
                 if not (exclude_prefix and im["file_name"].startswith(exclude_prefix))]
    base_ids = {im["id"] for im in base_imgs}
    base_anns = [an for an in src["annotations"] if an["image_id"] in base_ids]
    img_id, ann_id = _add_images(coco, out, src_base, split, base_imgs, base_anns,
                                 prefix="", link=True, img_id=img_id, ann_id=ann_id)
    n_base = img_id

    # inject pallet-labeled synthetic images (copied)
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
    print(f"[{split}] images {len(coco['images'])} (base {n_base} + synth "
          f"{len(coco['images'])-n_base}) | anns {len(coco['annotations'])}")
    print(f"   per-class: { {NAMES.get(k,k): v for k,v in sorted(cc.items())} }")


def cmd_build_pallet(args):
    src_base = Path(args.src_base)
    synth_dirs = [Path(s) for s in args.synth]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ["train", "valid"]:
        _build_split(split, src_base, synth_dirs, out, args.exclude_prefix)
    print(f"✅ assembled → {out}")


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
            print(f"[{split}] Pallet already declared — skip")
            continue
        shutil.copy(p, p.parent / f"{p.name}.bak_{datetime.now():%Y%m%d_%H%M%S}")
        d["categories"].append(PALLET_CAT)
        p.write_text(json.dumps(d))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        print(f"[{split}] Pallet declared → categories={cats}")
    print("done (annotations unchanged).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-pallet", help="assemble 4-class pallet dataset")
    bp.add_argument("--src-base", required=True,
                    help="base COCO dataset (real, pallet-free) to symlink")
    bp.add_argument("--synth", nargs="+", required=True,
                    help="pallet-labeled synthetic COCO dirs to inject")
    bp.add_argument("--out", default=str(config.PALLET_DATA_DIR),
                    help="output dataset dir")
    bp.add_argument("--exclude-prefix", default="lee5synth",
                    help="skip base images with this filename prefix")
    bp.set_defaults(func=cmd_build_pallet)

    dp = sub.add_parser("declare-pallet", help="declare Pallet category in a dataset")
    dp.add_argument("--data-dir", default=str(config.DATA_DIR),
                    help="COCO dataset root (train/valid)")
    dp.set_defaults(func=cmd_declare_pallet)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
