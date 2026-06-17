#!/usr/bin/env python3
"""[4단계 내부] LoRA 어댑터를 베이스 가중치에 합쳐 배포용 단일 .pth 를 만듭니다.

⚠️ BASE_WEIGHTS 는 train_lora.py 가 쓴 것과 반드시 같아야 합니다(어댑터는 학습 당시
feature 분포에서만 유효).

라운드마다: BASE_WEIGHTS 를 직전 결과로, FINAL_WEIGHTS 를 다음 버전으로 올리세요.
배포는 환경변수 DEPAL_WEIGHTS 를 새 파일로 지정하면 됩니다.

경로는 환경변수로: DEPAL_BASE_WEIGHTS, DEPAL_LORA_SAFE, DEPAL_FINAL.

사용법:  dr-onsite-merge   (보통은 dr-onsite-round 가 알아서 부릅니다)
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
