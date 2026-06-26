"""[학습] UI 팀이 *저수준(low-level)* API 로 개발하는 모습 — OnsiteDetector.

저수준 facade = SDK 가 **"한 프레임 처리(process)"** 만 해준다.
카메라 열기 / 루프 돌리기 / 프레임 꺼내기 / 언제 호출할지 / 스레드 / 종료
== 전부 'UI 팀(당신)' 이 직접 짠다.   ← 통제권 큼, 그러나 배선 코드도 당신 몫.

실행(카메라 없이, npz 재생):
  python lowlevel_uiapp.py
"""
import glob
import time

import cv2
import numpy as np

from dr_onsite_adap import OnsiteDetector

WEIGHTS = "/home/dr/lee3/thirdtask/lora_weights_b1/continual_master_v9_full.pth"
DEVICE = "cpu"        # 데모는 GPU가 학습중이라 cpu. 실전은 None(자동 cuda).
NPZ = sorted(glob.glob("/home/dr/geom_captures/rgbd_*.npz"))

# ── (A) 당신이 직접: 검출기 준비 ──
det = OnsiteDetector(weights=WEIGHTS, geom_algo="v2", device=DEVICE)

# ── (B) 당신이 직접: 프레임 소스. 여기선 카메라 대신 npz. 실전이면 카메라 SDK 를
#        당신이 열고 read 해서 (color, depth, intrinsics, up_dir) 를 만들어야 한다. ──
def read_frame(i):
    d = np.load(NPZ[i % len(NPZ)])
    color = cv2.cvtColor(d["rgb"], cv2.COLOR_RGB2BGR)
    depth = d["depth"].astype(float)
    intr = tuple(float(v) for v in d["intrinsics"])
    up = tuple(float(v) for v in d["up_dir"])
    return color, depth, intr, up

# ── (C) 당신이 직접: 루프 + 매 프레임 process 호출 + 결과 처리 + 그리기 ──
#        (실시간 UI 라면 이 루프를 별도 스레드로 돌리고, 종료/예외/FPS 도 직접 관리해야 한다.
#         단일 스레드로 UI 에서 돌리면 검출하는 동안 화면이 멈춘다.)
last_vis = None
t0 = time.time()
N = 8
for i in range(N):
    color, depth, intr, up = read_frame(i)
    scene = det.process(color, depth, intrinsics=intr, up_dir=up)   # ← SDK 몫: 한 프레임
    picks = [f"{p.class_name} {p.long_side_mm:.0f}x{p.short_side_mm:.0f}mm" for p in scene.pick_targets]
    print(f"[low] frame {i}: picks={len(scene.pick_targets)} done={scene.done} {picks}")
    last_vis = det.draw(color, scene)                                # 그리기 호출도 당신이
fps = N / (time.time() - t0)
print(f"[low] 평균 {fps:.1f} fps (이 루프·스레드·카메라 관리는 전부 당신이 짰다)")
if last_vis is not None:
    cv2.imwrite("/tmp/lowlevel_out.jpg", last_vis)
    print("[low] 시각화 저장: /tmp/lowlevel_out.jpg")
