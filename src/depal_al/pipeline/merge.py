#!/usr/bin/env python3
"""Fuse the LoRA adapter from a training round into the base weights,
producing a single deployable .pth.

⚠️ BASE_WEIGHTS must match the one train_lora.py used (the adapter is only
valid on the feature distribution it was trained against).

Per round: bump BASE_WEIGHTS to the previous round result and FINAL_WEIGHTS to
the next version. Deploy by pointing DEPAL_WEIGHTS at the new file.

Paths from config / env: DEPAL_BASE_WEIGHTS, DEPAL_LORA_SAFE, DEPAL_FINAL.

Usage:  depal-merge
"""
import torch
from rfdetr import RFDETRSegMedium
from peft import PeftModel

from .. import config


def main():
    print(f"1️⃣ load base ({config.NUM_CLASSES} classes): {config.BASE_WEIGHTS}")
    wrapper = RFDETRSegMedium(num_classes=config.NUM_CLASSES, device="cuda",
                              pretrain_weights=config.BASE_WEIGHTS)

    if hasattr(wrapper.model, "model"):
        raw_model = wrapper.model.model
    else:
        raw_model = wrapper.model.net

    print(f"2️⃣ attach LoRA adapter: {config.LORA_SAFE_DIR}")
    lora_model = PeftModel.from_pretrained(raw_model, config.LORA_SAFE_DIR)

    print("3️⃣ merge LoRA into base...")
    merged_model = lora_model.merge_and_unload()

    print(f"4️⃣ save single .pth: {config.FINAL_WEIGHTS}")
    torch.save({"model": merged_model.state_dict()}, config.FINAL_WEIGHTS)

    print(f"✅ done: {config.FINAL_WEIGHTS}")


if __name__ == "__main__":
    main()
