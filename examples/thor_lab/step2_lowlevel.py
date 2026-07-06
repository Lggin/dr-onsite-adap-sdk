"""STEP 2 — 저수준(low-level) facade: OnsiteDetector

핵심 감각: **SDK 는 '한 프레임 처리(process)' 만 해준다.**
프레임을 어떻게 구할지(카메라 열기 등)·언제 부를지·루프는 == '당신(개발자)' 이 짠다.
여기선 카메라 대신 저장된 캡처 1장을 '당신이' 넣어 본다.

실행:  bash run.sh step2_lowlevel.py
"""
import glob

import cv2
import numpy as np

from dr_onsite_adap import OnsiteDetector

det = OnsiteDetector()   # env 가중치 사용

# ── (당신 몫) 프레임 준비 — 실전이면 카메라에서 color/depth/intrinsics/up 를 뽑아야 한다.
#     여기선 예전에 'g' 로 저장한 캡처 1장을 읽는다. ──
files = sorted(glob.glob("/home/dr/geom_captures/rgbd_*.npz"))
if not files:
    raise SystemExit("~/geom_captures 에 npz 가 없습니다.")
d = np.load(files[0])
color = cv2.cvtColor(d["rgb"], cv2.COLOR_RGB2BGR)
depth = d["depth"].astype(float)
intr = tuple(float(v) for v in d["intrinsics"])
up = tuple(float(v) for v in d["up_dir"])
print(f"프레임 준비(당신 몫): {files[0].split('/')[-1]}  {color.shape}")

# ── (SDK 몫) 딱 이 한 줄 — 한 프레임 → 검출 + 윗면 픽자세 ──
scene = det.process(color, depth, intrinsics=intr, up_dir=up)

print(f"\n검출 {len(scene.pick_targets)}개:")
for p in scene.pick_targets:
    x, y, z = p.center_3d
    print(f"  {p.class_name:12s} 중심({x:+.2f},{y:+.2f},{z:+.2f})m  {p.long_side_mm:.0f}x{p.short_side_mm:.0f}mm")

cv2.imwrite("/tmp/step2_out.jpg", det.draw(color, scene))
print("\n시각화 저장: /tmp/step2_out.jpg")
print("→ 배운 것: SDK 는 process() 한 프레임만. 프레임 준비·언제 호출은 전부 당신이 했다.")
print("   실시간으로 하려면 이 위에 '카메라 열기 + 루프 + 스레드' 를 당신이 더 짜야 한다 (STEP 3 와 비교).")
