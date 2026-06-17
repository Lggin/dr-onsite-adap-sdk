#!/usr/bin/env python3
"""LoRA fine-tune of the segmentation detector for one active-learning round.

stack+cumulative strategy: start from the previous production weights
(BASE_WEIGHTS) and train LoRA adapters on the current corpus (the replay-mixed
set is the recommended TRAIN_DATA_DIR). The class head is unchanged, so this
is incremental — no architecture change.

It intercepts PyTorch-Lightning's fit() to (a) wrap the inner model in a PEFT
LoRA model and (b) train only LoRA params, then loads the best checkpoint back
into the PEFT model and saves the adapter.

Per round: bump BASE_WEIGHTS to the previous merged result and FINAL_WEIGHTS
(in merge.py) to the next version. All paths come from config / env:
  DEPAL_BASE_WEIGHTS, DEPAL_TRAIN_DATA, DEPAL_LORA_SAFE, DEPAL_LORA_RUN

Usage:  depal-train
"""
import types
from pathlib import Path

import torch
import pytorch_lightning as pl
from rfdetr import RFDETRSegMedium
from peft import LoraConfig, get_peft_model

from .. import config

AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "Rotate": {"limit": 15, "p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5,
    },
    "HueSaturationValue": {
        "hue_shift_limit": 8,
        "sat_shift_limit": 15,
        "val_shift_limit": 10,
        "p": 0.3,
    },
}


def _load_best_into_peft(peft_model, best_path):
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    raw = getattr(peft_model, "_orig_mod", peft_model)
    if not isinstance(raw, torch.nn.Module):
        raw = peft_model
    missing, unexpected = raw.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"   (load_state_dict missing keys: {len(missing)})")
    if unexpected:
        print(f"   (load_state_dict unexpected keys: {len(unexpected)})")


def _make_custom_fit(original_fit):
    def custom_fit(trainer, lightning_module, datamodule=None, **kwargs):
        print("\n" + "=" * 50)
        print("🚨 [Interceptor] wrapping Lightning module in LoRA")
        if hasattr(lightning_module, "model"):
            inner_model = lightning_module.model
        elif hasattr(lightning_module, "net"):
            inner_model = lightning_module.net
        else:
            raise RuntimeError("inner model not found on the Lightning module")

        lora_cfg = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules="all-linear",
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
        )
        peft_model = get_peft_model(inner_model, lora_cfg)
        if hasattr(lightning_module, "model"):
            lightning_module.model = peft_model
        elif hasattr(lightning_module, "net"):
            lightning_module.net = peft_model

        def custom_configure_optimizers(self):
            print("🛠️ training LoRA params only")
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable_params, lr=2e-4)

        lightning_module.configure_optimizers = types.MethodType(
            custom_configure_optimizers, lightning_module,
        )
        peft_model.print_trainable_parameters()
        print("=" * 50 + "\n")

        result = original_fit(trainer, lightning_module, datamodule, **kwargs)

        best_path = Path(config.LORA_RUN_DIR) / "checkpoint_best_total.pth"
        if best_path.exists():
            print(f"\n📥 loading best checkpoint: {best_path}")
            _load_best_into_peft(peft_model, best_path)
        else:
            print("\n⚠️ no best checkpoint — saving last-epoch state")

        print("\n💾 extracting LoRA weights...")
        peft_model.save_pretrained(config.LORA_SAFE_DIR)
        print(f"🎉 LoRA saved: {config.LORA_SAFE_DIR}")
        return result

    return custom_fit


def main():
    print(f"🧠 1. base model ({config.NUM_CLASSES} classes) ← {config.BASE_WEIGHTS}")
    wrapper = RFDETRSegMedium(num_classes=config.NUM_CLASSES, device="cuda",
                              pretrain_weights=config.BASE_WEIGHTS)

    pl.Trainer.fit = _make_custom_fit(pl.Trainer.fit)

    print(f"⚡ 2. training on {config.TRAIN_DATA_DIR}")
    wrapper.train(
        dataset_dir=config.TRAIN_DATA_DIR,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR,
        num_workers=4,
        aug_config=AUG_CONFIG,
        output_dir=config.LORA_RUN_DIR,
        early_stopping=True,
        early_stopping_patience=config.EARLY_STOP_PATIENCE,
        early_stopping_min_delta=0.001,
        wandb=False,
        project="Depal_Active_Learning",
        run="lora_round",
    )
    print("🚀 done. Next: depal-merge to fuse the adapter into the base.")


if __name__ == "__main__":
    main()
