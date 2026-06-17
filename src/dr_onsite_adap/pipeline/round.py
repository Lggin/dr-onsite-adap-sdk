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
   + 학습셋에 팔레트 라벨이 0개면 라운드를 멈춥니다: 팔레트 없는 배치로 학습하면
     팔레트 인식 능력이 약해질 수 있습니다(리플레이가 막아주려는 그 망각).
     팔레트 실측 라벨을 추가하거나, --replay 로 합성 팔레트를 섞거나, --force 로
     강행하세요.

권장 흐름(한 라운드):
  현장 수집(라이브 앱 't') → dr-onsite-ingest → dr-onsite-replay
    → dr-onsite-round → dr-onsite-eval → 배포(DEPAL_WEIGHTS 를 새 가중치로)

사용법:
  dr-onsite-round            # 학습 + 합치기
  dr-onsite-round --replay   # 리플레이 데이터셋부터 만들고 학습 + 합치기
  dr-onsite-round --force    # 팔레트 라벨이 0개여도 강행
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
        sys.exit(f"❌ '{title}' 실패 (종료코드 {r.returncode}). 라운드 중단.")
    print(f"✅ {title} 완료 ({(time.time()-t)/60:.1f}분)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true",
                    help="학습 전에 리플레이 혼합 데이터셋을 먼저 만든다")
    ap.add_argument("--force", action="store_true",
                    help="팔레트 라벨이 0개여도 진행한다")
    args = ap.parse_args()

    if args.replay:
        run("0/2 리플레이 데이터셋 생성", "replay")

    npal = pallet_ann_count(config.TRAIN_DATA_DIR)
    print(f"📊 팔레트(cid={config.PALLET_CID}) 라벨 수 @ {config.TRAIN_DATA_DIR}: {npal}개")
    if npal == 0 and not args.force:
        sys.exit(
            "⛔ 팔레트 라벨이 0개입니다. 팔레트 없는 배치로 학습하면 팔레트 인식이 "
            "약해질 수 있습니다.\n   → 팔레트 실측 라벨을 추가하거나, --replay 로 "
            "팔레트 포함 합성을 섞거나, --force 로 강행하세요."
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
