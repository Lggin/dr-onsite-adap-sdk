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

# main() 이 실제 적용 batch_size(해상도 혼재 시 1 로 자동 다운그레이드된 값)를
# 여기 담고, fit 시점에 실행되는 custom_fit 클로저가 읽어 run_meta 스냅샷에 남긴다.
_effective_bs = [None]


def _resolve_lora_targets(inner_model, spec):
    """`config.LORA_TARGETS` 를 peft 가 받는 형태로 변환한다.

    "all-linear" 는 peft 가 그대로 이해하므로 문자열을 넘긴다.
    "no-backbone" 은 backbone(DINOv2) 을 뺀 Linear 모듈 **전체 이름 리스트**를 만든다.
    (peft 는 리스트 항목을 `key == t or key.endswith("."+t)` 로 매칭하므로 전체 이름이 안전하다.)

    왜 backbone 을 뺄 수 있게 했나: 실측상 Linear 186개 중 backbone 이 72개·21.2M 로
    선형 파라미터의 66% 다. 라운드는 현장 신규 라벨 수십 장 규모라 백본까지 적응시키면
    과적합 위험이 크고, 일반 시각 특징을 망가뜨릴 수 있다.
    """
    if spec in (None, "", "all-linear"):
        return "all-linear"
    if spec == "no-backbone":
        names = [n for n, m in inner_model.named_modules()
                 if isinstance(m, torch.nn.Linear) and not n.startswith("backbone")]
        if not names:
            raise PipelineError(
                "LORA_TARGETS='no-backbone' 인데 backbone 이외의 Linear 를 찾지 못했습니다. "
                "모델 구조가 예상과 다릅니다.", error_code="E2002")
        return names
    # 콤마 구분 접미어 목록도 허용(예: "q_proj,v_proj")
    return [s.strip() for s in spec.split(",") if s.strip()]


class RoundProgressCallback(pl.Callback):
    """라운드 학습의 진행 상황·지표·파라미터를 관측 가능하게 남긴다(#3 로그 구체화).

    클래스별 지표 테이블(AP/AR/F1/…)은 rfdetr 의 COCOEvalCallback 이 이미
    출력하므로 재발명하지 않는다. 대신 그 지표가 callback_metrics 에 채워진 뒤
    (=이 콜백을 COCOEvalCallback 뒤에 append) 발화해 아래 산출물을 남긴다.
    모두 run_dir(=LORA_RUN_DIR) 아래에 모인다:

      - progress.log : 매 검증 에폭마다 진행률/ETA/mAP·F1 추세 한 줄(스크롤로
        사라지는 rich 테이블의 사후 검토용).
      - run_meta.json: 라운드 시작 시 파라미터 스냅샷(BASE_WEIGHTS·LR·LoRA r/alpha/
        dropout·epochs·batch·데이터셋 장수). 재현성/사후 추적용.
      - round_metrics.csv : 에폭별 실제 적용 LR + train/val loss + mAP50/mAP50:95/F1.
        (⚠️`metrics.csv` 는 Lightning CSVLogger 가 같은 폴더에 쓰므로 이름을 피했다)
        (LR 은 옵티마이저에서 직접 읽어 스케줄러 변화까지 반영)
      - summary.json : 종료 시 first→best mAP 변화·best 에폭·총 소요 요약.
    """

    def __init__(self, fallback_epochs, run_dir=None, batch_size=None):
        self.total = int(fallback_epochs)
        self.run_dir = run_dir
        self.batch_size = batch_size
        self.log_path = os.path.join(run_dir, "progress.log") if run_dir else None
        self.meta_path = os.path.join(run_dir, "run_meta.json") if run_dir else None
        # ⚠️ 파일명 주의: Lightning 의 CSVLogger 가 같은 디렉터리에 **metrics.csv** 를 쓴다.
        # 이름이 겹치면 두 쪽이 서로의 헤더를 덮어써 CSVLogger 가
        # `ValueError: dict contains fields not in fieldnames: None` 로 **학습을 죽인다**
        # (2026-08-03 스윕 스모크에서 실측: 첫 검증 끝에서 크래시 → 어댑터 미저장).
        self.csv_path = os.path.join(run_dir, "round_metrics.csv") if run_dir else None
        self.summary_path = os.path.join(run_dir, "summary.json") if run_dir else None
        self.t0 = None
        self.best_map = None
        self.best_epoch = None
        self.first_map = None     # 첫 전체 검증 에폭 mAP(before/after 요약의 'before')
        self.first_epoch = None

    def _append(self, path, text):
        if not path:
            return
        # ⚠️ encoding 을 명시하지 않으면 로케일 기본값을 따르는데, 로케일이 안 잡힌 환경
        # (cron·서브프로세스·서비스)에서는 ASCII 가 되어 한글/이모지 진행로그가
        # UnicodeEncodeError 로 **학습 자체를 죽인다**(2026-08-03 스윕 스모크에서 실측:
        # 첫 검증 epoch 끝에서 크래시 → 어댑터 미저장). 항상 utf-8 로 쓴다.
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass

    def on_fit_start(self, trainer, pl_module):
        self.t0 = time.time()
        # 실제 최대 에폭(early stopping 과 무관한 상한)을 우선 사용.
        me = getattr(trainer, "max_epochs", None)
        if me and me > 0:
            self.total = int(me)
        if self.run_dir:
            try:
                os.makedirs(self.run_dir, exist_ok=True)
            except OSError:
                self.run_dir = self.log_path = self.meta_path = None
                self.csv_path = self.summary_path = None

        # ── 파라미터 스냅샷(run_meta.json) ─────────────────────────────
        # 데이터셋 장수는 학습 COCO 에서 최선껏 읽는다(실패해도 무시).
        n_train_imgs = None
        try:
            tp = os.path.join(config.TRAIN_DATA_DIR, "train", "_annotations.coco.json")
            n_train_imgs = len(json.load(open(tp))["images"])
        except Exception:
            pass
        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_weights": config.BASE_WEIGHTS,
            "num_classes": config.NUM_CLASSES,
            "train_data_dir": config.TRAIN_DATA_DIR,
            "n_train_images": n_train_imgs,
            "epochs_max": self.total,
            "batch_size": self.batch_size,
            "lr": config.LR,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "lora_dropout": config.LORA_DROPOUT,
            "lora_targets": config.LORA_TARGETS,
            "weight_decay": config.WEIGHT_DECAY,
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
        }
        if self.meta_path:
            try:
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except OSError:
                pass
        print(f"📄 라운드 파라미터 스냅샷 → {self.meta_path or '(생략)'}")
        print(f"   base={os.path.basename(config.BASE_WEIGHTS)} lr={config.LR:g} "
              f"r={config.LORA_R} alpha={config.LORA_ALPHA} "
              f"epochs≤{self.total} bs={self.batch_size} "
              f"train_imgs={n_train_imgs}", flush=True)

        # metrics.csv 헤더(덮어쓰기 시작).
        if self.csv_path:
            try:
                with open(self.csv_path, "w", encoding="utf-8") as f:
                    f.write("epoch,lr,train_loss,val_loss,mAP50,mAP50_95,F1\n")
            except OSError:
                self.csv_path = None

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
        train_loss = g("train_loss", "train/loss", "loss", "train_loss_epoch")
        val_loss = g("val_loss", "val/loss", "val_loss_epoch", "val/loss_total")

        # 실제 적용 LR 은 옵티마이저에서 직접 읽는다(스케줄러 변화·config 배선 확인).
        lr = None
        try:
            opts = getattr(trainer, "optimizers", None) or []
            if opts and opts[0].param_groups:
                lr = float(opts[0].param_groups[0]["lr"])
        except Exception:
            pass

        cur = map50 if map50 is not None else map5095
        if cur is not None and self.first_map is None:
            self.first_map, self.first_epoch = cur, ep
        star = ""
        if cur is not None and (self.best_map is None or cur > self.best_map):
            self.best_map, self.best_epoch = cur, ep
            star = " ⭐best"

        def mm(x):
            return f"{x:.4f}" if x is not None else "—"

        # ── metrics.csv 한 줄 append ──────────────────────────────────
        def cs(x):
            return f"{x:.6g}" if x is not None else ""
        self._append(self.csv_path,
                     f"{ep},{cs(lr)},{cs(train_loss)},{cs(val_loss)},"
                     f"{cs(map50)},{cs(map5095)},{cs(f1)}\n")

        best_s = (f"{mm(self.best_map)}@ep{self.best_epoch}"
                  if self.best_map is not None else "—")
        lr_s = f"{lr:g}" if lr is not None else "—"
        line = (f"[진행] Epoch {ep}/{self.total} ({pct:.1f}%) | 남은 ~{remaining}ep "
                f"| 경과 {elapsed/60:.1f}m ETA ~{eta/60:.1f}m "
                f"| lr={lr_s} loss(t/v)={mm(train_loss)}/{mm(val_loss)} "
                f"| mAP50={mm(map50)} mAP50:95={mm(map5095)} F1={mm(f1)} "
                f"| best {best_s}{star}")
        print(line, flush=True)
        self._append(self.log_path, line + "\n")

    def on_fit_end(self, trainer, pl_module):
        # ── before/after 요약 ────────────────────────────────────────
        # 'before'=첫 전체 검증 에폭 mAP, 'after'=best mAP. (BASE_WEIGHTS 자체를
        # valid 로 재평가한 진짜 base 기준은 eval 게이트 Phase2 의 몫으로 남긴다.)
        elapsed = (time.time() - self.t0) if self.t0 else 0.0
        delta = (self.best_map - self.first_map
                 if self.best_map is not None and self.first_map is not None else None)

        def mm(x):
            return f"{x:.4f}" if x is not None else "—"

        arrow = "→"
        dstr = (f"{delta:+.4f}" if delta is not None else "—")
        line = (f"[요약] first(ep{self.first_epoch})={mm(self.first_map)} {arrow} "
                f"best(ep{self.best_epoch})={mm(self.best_map)}  Δ={dstr}  "
                f"| 총 {elapsed/60:.1f}m")
        print("\n" + line, flush=True)
        self._append(self.log_path, line + "\n")
        summary = {
            "first_epoch": self.first_epoch, "first_map": self.first_map,
            "best_epoch": self.best_epoch, "best_map": self.best_map,
            "delta_map": delta, "elapsed_min": round(elapsed / 60, 2),
            "note": "before=first full-val epoch, after=best; true-base eval는 eval 게이트 Phase2",
        }
        if self.summary_path:
            try:
                with open(self.summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
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

        targets = _resolve_lora_targets(inner_model, config.LORA_TARGETS)
        lora_cfg = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=targets,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
        )
        peft_model = get_peft_model(inner_model, lora_cfg)
        _n_tr = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"🔧 LoRA 대상={config.LORA_TARGETS} r={config.LORA_R} "
              f"alpha={config.LORA_ALPHA} → 학습 파라미터 {_n_tr:,}", flush=True)
        if hasattr(lightning_module, "model"):
            lightning_module.model = peft_model
        elif hasattr(lightning_module, "net"):
            lightning_module.net = peft_model

        def custom_configure_optimizers(self):
            # ⚠️ 버그수정(2026-07-27): 이전엔 lr=2e-4 를 하드코딩해 config.LR(튜닝값)이
            # 이 monkeypatch 에 덮여 무시됐다(wrapper.train(lr=config.LR) 도 무효화).
            # LR 을 실제 튜닝 노브(config.LR)로 배선하고, 적용 LR·학습 파라미터 수를
            # 로그에 남겨(#3) 이후 파라미터 최적화가 관측 가능하도록 한다.
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            n_train = sum(p.numel() for p in trainable_params)
            print(f"🛠️ training LoRA params only (lr={config.LR:g}, "
                  f"trainable={n_train:,})")
            return torch.optim.AdamW(trainable_params, lr=config.LR)

        lightning_module.configure_optimizers = types.MethodType(
            custom_configure_optimizers, lightning_module,
        )
        peft_model.print_trainable_parameters()
        print("=" * 50 + "\n")

        # 진행률/지표/파라미터 로그 콜백을 rfdetr 콜백들 뒤에 붙인다(지표가
        # callback_metrics 에 채워진 뒤 발화하도록 순서상 마지막에 append).
        try:
            prog = RoundProgressCallback(
                config.EPOCHS,
                run_dir=config.LORA_RUN_DIR,
                batch_size=_effective_bs[0],
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
    _effective_bs[0] = bs   # run_meta 스냅샷용(콜백이 fit 시점에 읽음)

    print(f"⚡ 2. training on {config.TRAIN_DATA_DIR} (batch_size={bs})")
    wrapper.train(
        dataset_dir=config.TRAIN_DATA_DIR,
        epochs=config.EPOCHS,
        batch_size=bs,
        lr=config.LR,
        # ⚠️ 이전에는 이 인자가 없어 rfdetr 기본값(1e-4)이 조용히 쓰였다. 이제 탐색 가능.
        weight_decay=config.WEIGHT_DECAY,
        num_workers=4,
        aug_config=AUG_CONFIG,
        output_dir=config.LORA_RUN_DIR,
        early_stopping=True,
        early_stopping_patience=config.EARLY_STOP_PATIENCE,
        early_stopping_min_delta=0.001,
        wandb=False,
        # ⚠️ 반드시 False. rfdetr 기본값 True 면 TensorBoard writer 가 TensorFlow(2.18)
        #    1600여 모듈을 끌어오고, TF 가 torch 위에 cuFFT/cuDNN/cuBLAS 팩토리를 이중
        #    등록한다("Unable to register ... already been registered"). 그 결과 프로세스
        #    종료 시점에 glibc 힙이 깨져 SIGABRT(rc=-6)로 죽는다
        #    (`double free or corruption (out)` / `corrupted size vs. prev_size`).
        #    2026-08-03 스윕에서 학습 5건 전부 이렇게 아보트했고, 비정상 종료라 CUDA
        #    정리 경로가 커널에서 폭주해 PC 가 18시간 프리즈했다. 지표/진행률은
        #    PTL CSVLogger + RoundProgressCallback 로 충분하다.
        tensorboard=False,
        project="Depal_Active_Learning",
        run="lora_round",
    )
    print("🚀 완료. 다음: dr-onsite-merge 로 어댑터를 베이스에 합치세요.")


if __name__ == "__main__":
    main()
