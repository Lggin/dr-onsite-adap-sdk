#!/usr/bin/env python3
"""[4단계] 라운드 자동 실행 — 현장 데이터로 검출 모델을 갱신합니다.

순서:
  (선택) dr-onsite-replay   학습용 리플레이 데이터셋 생성
  1) dr-onsite-train        직전 가중치(BASE_WEIGHTS)에서 LoRA 미세조정
  2) dr-onsite-merge        LoRA 어댑터를 합쳐 단일 가중치(FINAL_WEIGHTS) 생성

각 단계는 별도 프로세스로 실행돼 GPU/학습 상태가 서로 섞이지 않습니다.

⚠️ 안전장치: 사람이 검수한 라벨만 사용합니다. 라이브 검출을 자동으로 흡수하지
   마세요(무인·극저에러 운영 원칙에 어긋나며 모델이 오염됩니다). 그래서 갱신은
   의도적인 "라운드" 단위로만 합니다.
   + 학습셋에 라벨이 0개인 클래스가 있으면 라운드를 멈춥니다: 프레임에 그 물체가
     보이는데 라벨만 없으면 배경(negative)으로 학습돼 그 클래스가 무너집니다.
     실측(2026-08-04)으로 20프레임에서 팔레트 라벨 20개를 빼자 팔레트 검출이
     120/120 → 0/120 으로 전멸했고, 같은 프레임을 전 클래스 라벨하면 무손실,
     리플레이 400장을 섞으면 118/120 로 복구됐다.
     → 보이는 클래스를 전부 라벨하거나, --replay 로 층화 리플레이를 섞거나,
     --force 로 강행하세요.

권장 흐름(한 라운드):
  현장 수집(라이브 앱 't') → dr-onsite-ingest → dr-onsite-replay
    → dr-onsite-round → dr-onsite-eval → 배포(DEPAL_WEIGHTS 를 새 가중치로)

사용법:
  dr-onsite-round            # 학습 + 합치기
  dr-onsite-round --replay   # 리플레이 데이터셋부터 만들고 학습 + 합치기
  dr-onsite-round --force    # 라벨 0개인 클래스가 있어도 강행
"""
import argparse
import json
import os
import subprocess
import sys
import time

from .. import config

PY = sys.executable


def _cls_name(cid):
    """config.CLASSES 값이 (이름, 색) 튜플이므로 이름만 꺼낸다."""
    spec = config.CLASSES.get(cid, cid)
    return spec[0] if isinstance(spec, (tuple, list)) else str(spec)


def class_ann_counts(data_dir):
    """클래스별 라벨 수를 센다({cid: 개수}, 0개인 클래스도 포함)."""
    counts = {cid: 0 for cid in config.CLASSES}
    for split in ["train", "valid"]:
        p = os.path.join(data_dir, split, "_annotations.coco.json")
        if os.path.exists(p):
            d = json.loads(open(p).read())
            for a in d["annotations"]:
                cid = a["category_id"]
                counts[cid] = counts.get(cid, 0) + 1
    return counts


def run(title, module):
    cmd = [PY, "-m", f"dr_onsite_adap.pipeline.{module}"]
    print(f"\n{'='*60}\n▶ {title}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"❌ '{title}' 실패 (종료코드 {r.returncode}). 라운드 중단.")
    print(f"✅ {title} 완료 ({(time.time()-t)/60:.1f}분)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true",
                    help="학습 전에 리플레이 혼합 데이터셋을 먼저 만든다")
    ap.add_argument("--force", action="store_true",
                    help="라벨이 0개인 클래스가 있어도 진행한다")
    args = ap.parse_args()

    if args.replay:
        run("0/2 리플레이 데이터셋 생성", "replay")

    counts = class_ann_counts(config.TRAIN_DATA_DIR)
    print(f"📊 클래스별 라벨 수 @ {config.TRAIN_DATA_DIR}")
    for cid, spec in config.CLASSES.items():
        name = spec[0] if isinstance(spec, (tuple, list)) else str(spec)
        mark = "  ⛔0개" if counts.get(cid, 0) == 0 else ""
        print(f"   {cid}:{name:<12s} {counts.get(cid, 0):6d}{mark}")
    missing = [f"{cid}:{_cls_name(cid)}" for cid in config.CLASSES if counts.get(cid, 0) == 0]
    if missing and not args.force:
        sys.exit(
            f"⛔ 라벨이 0개인 클래스: {', '.join(missing)}\n"
            "   이 배치로 학습하면 해당 클래스 인식이 무너집니다. 프레임에 그 물체가 "
            "보이는데 라벨만 없으면 배경(negative)으로 학습되기 때문입니다.\n"
            "   실측(2026-08-04, 같은 20프레임·같은 베이스·all-linear r=8):\n"
            "     · 슬립시트만 라벨(팔레트 20개 라벨 누락) → 팔레트 검출 120/120 → 0/120 (전멸)\n"
            "     · 같은 프레임 전 클래스 라벨            → 120/120 (무손실)\n"
            "     · 슬립시트만 라벨 + 리플레이 400장      → 118/120 (리플레이가 상쇄)\n"
            "   → 그 프레임에 보이는 다른 클래스도 라벨하거나, --replay 로 전 클래스 "
            "층화 리플레이를 섞거나, --force 로 강행하세요."
        )

    run("1/2 LoRA 학습 (검출기)", "train_lora")
    run("2/2 LoRA 합치기 (merge)", "merge")

    print("\n🎉 라운드 완료.")
    print(f"   • 새 가중치: {config.FINAL_WEIGHTS}")
    print("   • 평가: dr-onsite-eval --models prev=<이전.pth> new="
          f"{config.FINAL_WEIGHTS}")
    print("   • 배포: 환경변수 DEPAL_WEIGHTS 를 새 가중치로 지정")
    print("   • 다음 라운드: DEPAL_BASE_WEIGHTS 를 이 결과로 올리세요")


if __name__ == "__main__":
    main()
