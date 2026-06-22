#!/usr/bin/env python3
"""RF-DETR seg 추론 FPS 측정 — Thor(엣지)에서 실행.

실제 파이프라인과 똑같은 SegModel 경로(= optimize_for_inference 포함)로 잰다.
모델 추론 지연만 본다(카메라/기하/표시 제외).

필요 환경변수:
  DEPAL_WEIGHTS       검출 가중치 .pth (예: 현장 운영 v8_3)
  DEPAL_DEVICE        cuda (기본)
  DEPAL_NUM_CLASSES   가중치 클래스 수 (기본 4)

사용 예:
  DEPAL_WEIGHTS=/경로/continual_master_v8_3_pallet.pth DEPAL_DEVICE=cuda \
    python3 bench_infer.py --iters 100 --warmup 20 --image /경로/대표프레임.png

  # 대표 프레임이 없으면 합성 프레임으로 (검출 0개라 후처리 비용이 약간 과소평가됨):
  DEPAL_WEIGHTS=... python3 bench_infer.py --iters 100
"""
import os
import time
import argparse
import statistics

import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100, help="측정 반복 수")
    ap.add_argument("--warmup", type=int, default=20, help="워밍업 반복 수(측정 제외)")
    ap.add_argument("--conf", type=float, default=0.5, help="검출 conf 임계값")
    ap.add_argument("--image", default=None, help="대표 프레임 경로(권장). 없으면 합성")
    ap.add_argument("--size", type=int, default=720, help="합성 프레임 한 변(px)")
    args = ap.parse_args()

    weights = os.environ.get("DEPAL_WEIGHTS")
    if not weights:
        raise SystemExit("DEPAL_WEIGHTS 환경변수에 .pth 경로를 지정하세요.")
    device = os.environ.get("DEPAL_DEVICE", "cuda")
    num_classes = int(os.environ.get("DEPAL_NUM_CLASSES", "4"))

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"이미지를 못 읽음: {args.image}")
        src = f"image {args.image}"
    else:
        frame = (np.random.rand(args.size, args.size, 3) * 255).astype(np.uint8)
        src = f"synthetic {args.size}x{args.size} (검출 0 → 후처리 과소평가 가능)"
    h, w = frame.shape[:2]

    from dr_onsite_adap.runtime.inference import SegModel

    t_load = time.perf_counter()
    model = SegModel(weights, num_classes, device)
    load_ms = (time.perf_counter() - t_load) * 1000.0

    # 워밍업: CUDA init / autotune / 첫 추론 캐싱은 측정에서 뺀다
    for _ in range(args.warmup):
        model.predict(frame, args.conf)

    # 측정 (predict 는 CPU 결과를 반환 → GPU sync 강제됨)
    times = []
    n_det_last = None
    for _ in range(args.iters):
        t0 = time.perf_counter()
        det = model.predict(frame, args.conf)
        times.append((time.perf_counter() - t0) * 1000.0)
        try:
            n_det_last = len(det.xyxy)
        except Exception:
            pass

    times.sort()

    def pct(p):
        return times[min(len(times) - 1, int(len(times) * p))]

    med = statistics.median(times)
    mean = statistics.fmean(times)

    print(f"\n=== RF-DETR Seg 추론 지연 ===")
    print(f"  weights : {weights}")
    print(f"  device  : {device}   num_classes : {num_classes}")
    print(f"  frame   : {w}x{h} ({src})")
    print(f"  검출 수 : {n_det_last}")
    print(f"  iters={args.iters}  warmup={args.warmup}  model_load={load_ms:.0f} ms")
    print(f"  ---------------------------------------------")
    print(f"  median  : {med:6.1f} ms   ({1000.0 / med:5.1f} FPS)")
    print(f"  mean    : {mean:6.1f} ms   ({1000.0 / mean:5.1f} FPS)")
    print(f"  p95     : {pct(0.95):6.1f} ms")
    print(f"  min/max : {times[0]:.1f} / {times[-1]:.1f} ms")


if __name__ == "__main__":
    main()
