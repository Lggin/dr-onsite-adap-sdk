#!/usr/bin/env python3
"""Build a replay-mixed training set for a LoRA round.

  train = all real active-learning images
          + a sample of pallet-containing synthetic images (replay: prevents
            the detector forgetting pallet/box/tote while learning new
            field hard-cases from a small real batch)
  valid = real active-learning valid + pallet-containing synthetic valid sample

Images are symlinked (originals preserved, disk saved). Filename collisions
avoided with al_/rp_ prefixes.

The replay strategy is what makes a small field batch deployable: a handful of
new real frames teach the new appearance, while the synthetic replay holds the
existing classes in place (avoids the catastrophic forgetting seen without it).

Usage:  depal-replay [--al DIR] [--pallet DIR] [--out DIR]
                      [--k-train N] [--k-valid N] [--exclude-prefix STR]
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--al", default=config.DATA_DIR,
                    help="real active-learning dataset root")
    ap.add_argument("--pallet", default=config.PALLET_DATA_DIR,
                    help="synthetic dataset with pallet labels (replay source)")
    ap.add_argument("--out", default=config.REPLAY_OUT_DIR,
                    help="output dataset dir")
    ap.add_argument("--k-train", type=int, default=400,
                    help="pallet-containing replay samples for train")
    ap.add_argument("--k-valid", type=int, default=100,
                    help="pallet-containing replay samples for valid")
    ap.add_argument("--exclude-prefix", default="lee5synth",
                    help="skip real images whose filename starts with this "
                         "(legacy leaky synth mixed into the real set); "
                         "empty string keeps everything")
    args = ap.parse_args()
    AL, PAL, OUT = str(args.al), str(args.pallet), str(args.out)
    random.seed(0)

    for sp in ["train", "valid"]:
        os.makedirs(os.path.join(OUT, sp), exist_ok=True)
    for sp, K in [("train", args.k_train), ("valid", args.k_valid)]:
        merged = {"images": [], "annotations": [], "categories": CATS}
        iid = aid = 0
        cc = Counter()

        # 1) real active-learning (optionally excluding a legacy prefix)
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

        # 2) pallet-containing synthetic images (replay)
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
        print(f"[{sp}] {len(merged['images'])}img (real {n_real} + replay {iid-n_real}) | "
              + ", ".join(f"{NAMES[k]}:{cc[k]}" for k in sorted(cc)))


if __name__ == "__main__":
    main()
