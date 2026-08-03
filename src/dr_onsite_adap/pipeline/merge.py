#!/usr/bin/env python3
"""[4단계 내부] LoRA 어댑터를 베이스 가중치에 합쳐 배포용 단일 .pth 를 만듭니다.

⚠️ BASE_WEIGHTS 는 train_lora.py 가 쓴 것과 반드시 같아야 합니다(어댑터는 학습 당시
feature 분포에서만 유효).

라운드마다: BASE_WEIGHTS 를 직전 결과로, FINAL_WEIGHTS 를 다음 버전으로 올리세요.
배포는 환경변수 DEPAL_WEIGHTS 를 새 파일로 지정하면 됩니다.

경로는 환경변수로: DEPAL_BASE_WEIGHTS, DEPAL_LORA_SAFE, DEPAL_FINAL.

사용법:  dr-onsite-merge   (보통은 dr-onsite-round 가 알아서 부릅니다)
"""
import os

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
    merged_sd = merged_model.state_dict()

    # 4️⃣ 저장 전에 병합이 실제로 모델을 바꿨는지 검증 → 델타 0(무효 라운드)이면
    #    배포 가중치(FINAL_WEIGHTS)를 덮어쓰지 않고 중단(무결성 게이트).
    print("4️⃣ 병합 검증 (base 대비 실제 변화 확인)...")
    try:
        bck = torch.load(config.BASE_WEIGHTS, map_location="cpu", weights_only=False)
        base_sd = bck["model"] if isinstance(bck, dict) and "model" in bck else bck
        diff = maxd = 0
        for k, v in merged_sd.items():
            if k in base_sd and base_sd[k].shape == v.shape:
                # ⚠️ base 는 map_location="cpu" 로 읽지만 merged_sd 는 **GPU 텐서**다.
                # 그냥 빼면 "Expected all tensors to be on the same device" 예외가 나고,
                # 아래 except 가 이를 삼켜 **무결성 게이트가 통째로 생략**된다
                # (2026-08-03 실측: "병합 검증 생략 … 검증 없이 저장합니다").
                # LoRA 병합 무효화를 잡으려고 만든 게이트가 정작 안 걸리던 원인이라
                # 비교 전에 CPU 로 내린다.
                bt = base_sd[k].detach().to("cpu").float()
                vt = v.detach().to("cpu").float()
                d = float((bt - vt).abs().max())
                if d > 1e-6:
                    diff += 1; maxd = max(maxd, d)
    except Exception as e:
        # 검증 자체가 불가능(base 로드 실패 등)하면 게이트를 우회하되 명확히 알림.
        print(f"   ⚠️ 병합 검증 생략(base 비교 불가: {e}) — 검증 없이 저장합니다.")
        diff = None

    if diff == 0:
        msg = ("병합 결과가 base 와 동일합니다(LoRA 델타 0) — 이 라운드는 모델을 "
               "바꾸지 않았습니다. train 로그의 'LoRA_B 절대합' 을 확인하세요.")
        if os.environ.get("DEPAL_ALLOW_EMPTY_MERGE") == "1":
            print(f"   ⚠️ {msg}\n   (DEPAL_ALLOW_EMPTY_MERGE=1 → 강제 저장)")
        else:
            raise SystemExit(
                f"❌ {msg}\n   배포 가중치를 덮어쓰지 않고 중단합니다. "
                "우회: DEPAL_ALLOW_EMPTY_MERGE=1")
    elif diff:
        print(f"   🔎 병합 확인: base 대비 {diff}개 텐서 변경, 최대차 {maxd:.5f} (정상)")

    print(f"5️⃣ save single .pth: {config.FINAL_WEIGHTS}")
    torch.save({"model": merged_sd}, config.FINAL_WEIGHTS)
    print(f"✅ done: {config.FINAL_WEIGHTS}")


if __name__ == "__main__":
    main()
