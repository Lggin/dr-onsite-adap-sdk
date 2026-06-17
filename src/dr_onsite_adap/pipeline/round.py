#!/usr/bin/env python3
"""Active-learning round orchestrator: refresh the detector from the current
corpus.

  (optional) depal-replay   build the replay-mixed training set
  1) depal-train            LoRA fine-tune from BASE_WEIGHTS
  2) depal-merge            fuse the adapter → FINAL_WEIGHTS

Each step runs as its own subprocess so GPU/Lightning state stays isolated.

⚠️ Guardrail: only labeled/curated data is used. Do NOT auto-absorb live
   un-curated detections — that corrupts the model (unmanned, very-low-error
   constraint), so updates happen as a deliberate "round".
   + If the training set has 0 pallet annotations the round stops: training
     the detector on a pallet-free batch can weaken the pallet head (the
     forgetting that replay is designed to prevent). Add real pallet labels,
     or pass --force, or mix in a pallet-bearing replay set (--replay).

Recommended recipe (per round):
  field collect (live app 't') → depal-ingest → depal-replay
    → depal-round → depal-eval → deploy (point DEPAL_WEIGHTS at FINAL_WEIGHTS)

Usage:
  depal-round                 # train + merge
  depal-round --replay        # build replay set first, then train + merge
  depal-round --force         # proceed even with 0 pallet annotations
"""
import argparse
import json
import os
import subprocess
import sys
import time

from .. import config

PY = sys.executable


def pallet_ann_count(data_dir):
    n = 0
    for split in ["train", "valid"]:
        p = os.path.join(data_dir, split, "_annotations.coco.json")
        if os.path.exists(p):
            d = json.loads(open(p).read())
            n += sum(1 for a in d["annotations"] if a["category_id"] == config.PALLET_CID)
    return n


def run(title, module):
    cmd = [PY, "-m", f"dr_onsite_adap.pipeline.{module}"]
    print(f"\n{'='*60}\n▶ {title}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"❌ '{title}' failed (exit {r.returncode}). Round aborted.")
    print(f"✅ {title} done ({(time.time()-t)/60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true",
                    help="build the replay-mixed training set before training")
    ap.add_argument("--force", action="store_true",
                    help="proceed even with 0 pallet annotations")
    args = ap.parse_args()

    if args.replay:
        run("0/2 build replay set", "replay")

    npal = pallet_ann_count(config.TRAIN_DATA_DIR)
    print(f"📊 pallet (cid={config.PALLET_CID}) annotations in {config.TRAIN_DATA_DIR}: {npal}")
    if npal == 0 and not args.force:
        sys.exit(
            "⛔ 0 pallet annotations. Training on a pallet-free batch can weaken "
            "the pallet head.\n   → add real pallet labels, use --replay to mix in "
            "pallet-bearing synthetic frames, or --force to proceed anyway."
        )

    run("1/2 LoRA train (detector)", "train_lora")
    run("2/2 LoRA merge", "merge")

    print("\n🎉 Round complete.")
    print(f"   • new merged weights: {config.FINAL_WEIGHTS}")
    print("   • evaluate: depal-eval --models prev=<old.pth> new="
          f"{config.FINAL_WEIGHTS}")
    print("   • deploy: point DEPAL_WEIGHTS at the new file")
    print("   • next round: bump DEPAL_BASE_WEIGHTS to this result")


if __name__ == "__main__":
    main()
