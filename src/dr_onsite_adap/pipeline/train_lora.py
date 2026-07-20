#!/usr/bin/env python3
"""[4단계 내부] 검출 모델을 LoRA로 미세조정합니다(보통 dr-onsite-round 가 자동 호출).

전략(stack+cumulative): 직전 운영 가중치(BASE_WEIGHTS)에서 시작해 현재 데이터셋
(권장: 리플레이 혼합셋, TRAIN_DATA_DIR)으로 LoRA 어댑터를 학습합니다. 클래스 구조는
그대로라 증분 학습입니다(아키텍처 변경 없음).

PyTorch-Lightning 의 fit() 을 가로채 (a) 내부 모델을 PEFT LoRA 로 감싸고 (b) LoRA
파라미터만 학습한 뒤, best 체크포인트를 다시 불러와 어댑터를 저장합니다.

라운드마다: BASE_WEIGHTS 를 직전 결과로, (merge.py 의) FINAL_WEIGHTS 를 다음 버전으로
올리세요. 경로는 모두 환경변수로 지정합니다:
  DEPAL_BASE_WEIGHTS, DEPAL_TRAIN_DATA, DEPAL_LORA_SAFE, DEPAL_LORA_RUN

사용법:  dr-onsite-train   (보통은 dr-onsite-round 가 알아서 부릅니다)
"""
import json
import os
import time
import types
from pathlib import Path

import torch
import pytorch_lightning as pl
from rfdetr import RFDETRSegMedium
from peft import LoraConfig, get_peft_model

from .. import config
from ..exceptions import PipelineError


class RoundProgressCallback(pl.Callback):
    """라운드 학습의 전체 진행 상황을 매 검증 에폭마다 한 줄로 요약한다.

    클래스별 지표 테이블(AP/AR/F1/…)은 rfdetr 의 COCOEvalCallback 이 이미
    출력하므로 여기선 재발명하지 않는다. 대신 그 지표가 callback_metrics 에
    채워진 뒤(=이 콜백을 COCOEvalCallback 뒤에 append) 발화해:
      - 진행률 %, 남은 에폭 수, 경과/ETA(에폭당 평균 시간 기반)
      - mAP50 / mAP50:95 / F1 추세 + best 에폭
    을 한 줄로 콘솔·파일에 남긴다(rich 테이블은 스크롤로 사라지므로 사후 검토용).
    """

    def __init__(self, fallback_epochs, log_path=None):
        self.total = int(fallback_epochs)
        self.log_path = log_path
        self.t0 = None
        self.best_map = None
        self.best_epoch = None

    def on_fit_start(self, trainer, pl_module):
        self.t0 = time.time()
        # 실제 최대 에폭(early stopping 과 무관한 상한)을 우선 사용.
        me = getattr(trainer, "max_epochs", None)
        if me and me > 0:
            self.total = int(me)
        if self.log_path:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            except OSError:
                self.log_path = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return   # fit 시작 전 예열 검증은 건너뜀
        m = trainer.callback_metrics

        def g(*keys):
            for k in keys:
                v = m.get(k)
                if v is not None:
                    return float(v)
            return None

        ep = int(trainer.current_epoch) + 1
        pct = 100.0 * ep / max(self.total, 1)
        elapsed = (time.time() - self.t0) if self.t0 else 0.0
        remaining = max(self.total - ep, 0)
        eta = (elapsed / max(ep, 1)) * remaining   # 에폭당 평균 × 남은 에폭

        map50 = g("val/mAP_50", "val/ema_mAP_50")
        map5095 = g("val/mAP_50_95", "val/ema_mAP_50_95")
        f1 = g("val/F1")

        cur = map50 if map50 is not None else map5095
        star = ""
        if cur is not None and (self.best_map is None or cur > self.best_map):
            self.best_map, self.best_epoch = cur, ep
            star = " ⭐best"

        def mm(x):
            return f"{x:.4f}" if x is not None else "—"

        best_s = (f"{mm(self.best_map)}@ep{self.best_epoch}"
                  if self.best_map is not None else "—")
        line = (f"[진행] Epoch {ep}/{self.total} ({pct:.1f}%) | 남은 ~{remaining}ep "
                f"| 경과 {elapsed/60:.1f}m ETA ~{eta/60:.1f}m "
                f"| mAP50={mm(map50)} mAP50:95={mm(map5095)} F1={mm(f1)} "
                f"| best {best_s}{star}")
        print(line, flush=True)
        if self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass

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
    sd = ckpt["model"]
    # ⚠️ 버그수정(2026-07-15): best 체크포인트의 LoRA 파라미터는 RF-DETR 저장/EMA
    # 경로에서 초기값(B=0)으로 저장된다. 그대로 로드하면 방금 학습한 in-memory LoRA
    # 델타를 0 으로 덮어써(merge 가 base 를 그대로 뱉음 = 온사이트 라운드가 모델을
    # 전혀 안 바꾸는 버그) LoRA 키는 제외하고 동결 base 만 로드해 학습된 LoRA 를 보존한다.
    n_before = len(sd)
    sd = {k: v for k, v in sd.items() if "lora_A" not in k and "lora_B" not in k}
    print(f"   (best 로드: LoRA 키 {n_before - len(sd)}개 제외 → 학습된 LoRA 보존, base만 로드)")
    missing, unexpected = raw.load_state_dict(sd, strict=False)
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
            raise PipelineError(
                "Lightning 모듈에서 inner model(model/net 속성)을 찾지 못했습니다. "
                "rfdetr 버전이 예상과 다를 수 있습니다.",
                error_code="E2001")

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

        # 진행률/ETA/지표추세 요약 콜백을 rfdetr 콜백들 뒤에 붙인다(지표가
        # callback_metrics 에 채워진 뒤 발화하도록 순서상 마지막에 append).
        try:
            prog = RoundProgressCallback(
                config.EPOCHS,
                log_path=os.path.join(config.LORA_RUN_DIR, "progress.log"),
            )
            if isinstance(getattr(trainer, "callbacks", None), list):
                trainer.callbacks.append(prog)
            else:
                trainer.callbacks = [prog]
        except Exception as e:   # 진행 로그는 부가기능 — 실패해도 학습은 계속
            print(f"⚠️ 진행 콜백 등록 실패(학습은 계속): {e}")

        result = original_fit(trainer, lightning_module, datamodule, **kwargs)

        best_path = Path(config.LORA_RUN_DIR) / "checkpoint_best_total.pth"
        if best_path.exists():
            print(f"\n📥 loading best checkpoint: {best_path}")
            _load_best_into_peft(peft_model, best_path)
        else:
            print("\n⚠️ no best checkpoint — saving last-epoch state")

        print("\n💾 extracting LoRA weights...")
        # 학습된 LoRA 델타가 실제로 존재하는지 확인(0 이면 merge 가 무효 → 모델 안 바뀜).
        # 무효 어댑터를 저장하면 다음 merge 가 배포 가중치를 무변화로 덮어써 라운드가
        # 조용히 실패하므로, 여기서 저장 전에 막는다(무결성 게이트).
        bsum = sum(float(p.detach().abs().sum())
                   for n, p in peft_model.named_parameters() if "lora_B" in n)
        print(f"   LoRA_B 절대합 = {bsum:.6f}  (0 이면 학습이 델타를 안 만든 것)")
        if bsum == 0.0:
            msg = ("LoRA_B 가 0 입니다 — 학습이 델타를 만들지 못했습니다. "
                   "이 어댑터를 merge 해도 모델이 안 바뀝니다(무효 라운드).")
            if os.environ.get("DEPAL_ALLOW_EMPTY_LORA") == "1":
                print(f"   ⚠️ {msg}\n   (DEPAL_ALLOW_EMPTY_LORA=1 → 무시하고 저장)")
            else:
                raise SystemExit(
                    f"❌ {msg}\n   무효 어댑터 저장을 막습니다. "
                    "우회: DEPAL_ALLOW_EMPTY_LORA=1")
        peft_model.save_pretrained(config.LORA_SAFE_DIR)
        print(f"🎉 LoRA saved: {config.LORA_SAFE_DIR}")
        return result

    return custom_fit


def main():
    print(f"🧠 1. base model ({config.NUM_CLASSES} classes) ← {config.BASE_WEIGHTS}")
    wrapper = RFDETRSegMedium(num_classes=config.NUM_CLASSES, device="cuda",
                              pretrain_weights=config.BASE_WEIGHTS)

    pl.Trainer.fit = _make_custom_fit(pl.Trainer.fit)

    # 학습 이미지 해상도가 섞여 있으면(예: ROI 크롭을 프레임마다 다르게 잡은 경우)
    # rfdetr 가 배치로 마스크를 못 합쳐 터진다(matcher: tensors must match). 그럴 땐
    # batch_size=1 로 자동 강제해 안전하게 학습한다(소량 AL 데이터엔 충분).
    bs = config.BATCH_SIZE
    tp = os.path.join(config.TRAIN_DATA_DIR, "train", "_annotations.coco.json")
    try:
        sizes = {(im.get("width"), im.get("height"))
                 for im in json.load(open(tp))["images"]}
    except Exception:
        sizes = set()
    if len(sizes) > 1 and bs > 1:
        print(f"⚠️ 학습 이미지 해상도가 여러 종류({len(sizes)}종) → batch_size=1 로 자동 전환"
              f" (배치 마스크 불일치 회피). 균일 ROI 로 라벨하면 bs>1 가능.")
        bs = 1

    print(f"⚡ 2. training on {config.TRAIN_DATA_DIR} (batch_size={bs})")
    wrapper.train(
        dataset_dir=config.TRAIN_DATA_DIR,
        epochs=config.EPOCHS,
        batch_size=bs,
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
    print("🚀 완료. 다음: dr-onsite-merge 로 어댑터를 베이스에 합치세요.")


if __name__ == "__main__":
    main()
