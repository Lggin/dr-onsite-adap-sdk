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
    merged_sd = merged_model.state_dict()
    torch.save({"model": merged_sd}, config.FINAL_WEIGHTS)

    # 병합이 실제로 모델을 바꿨는지 검증(LoRA_B=0 이면 base 와 동일 → 라운드 무의미)
    try:
        bck = torch.load(config.BASE_WEIGHTS, map_location="cpu", weights_only=False)
        base_sd = bck["model"] if isinstance(bck, dict) and "model" in bck else bck
        diff = maxd = 0
        for k, v in merged_sd.items():
            if k in base_sd and base_sd[k].shape == v.shape:
                d = float((base_sd[k].float() - v.float()).abs().max())
                if d > 1e-6:
                    diff += 1; maxd = max(maxd, d)
        if diff == 0:
            print("⚠️ 병합 결과가 base 와 동일합니다(LoRA 델타 0) — 이 라운드는 모델을 "
                  "바꾸지 않았습니다. train 로그의 'LoRA_B 절대합' 을 확인하세요.")
        else:
            print(f"🔎 병합 확인: base 대비 {diff}개 텐서 변경, 최대차 {maxd:.5f} (정상)")
    except Exception as e:
        print(f"   (병합 검증 생략: {e})")

    print(f"✅ done: {config.FINAL_WEIGHTS}")


if __name__ == "__main__":
    main()
