"""[학습] UI 팀이 *고수준(high-level)* API 로 개발하는 모습 — OnsiteSession.

고수준 facade = SDK 가 **카메라 + 루프 + 백그라운드 스레드까지 다** 한다.
당신(UI 팀)은 **콜백을 연결**하고 **제어 메서드**를 버튼에 바인딩만 하면 된다.
저수준 예제(lowlevel_uiapp.py)의 (B)프레임소스 (C)루프/스레드 가 통째로 사라진다.

실행(카메라 없이, npz 재생):
  python highlevel_uiapp.py
"""
import glob
import time

import cv2

from dr_onsite_adap import OnsiteSession
from dr_onsite_adap.runtime.session import ReplaySource

WEIGHTS = "/home/dr/lee3/thirdtask/lora_weights_b1/continual_master_v11.pth"
DEVICE = "cpu"        # 데모는 GPU가 학습중이라 cpu. 실전은 None(자동 cuda).

# 카메라 없으면 ReplaySource 주입. 카메라 있으면 source 인자를 아예 빼면 SDK 가 카메라를 연다.
src = ReplaySource(sorted(glob.glob("/home/dr/geom_captures/rgbd_*.npz")), loop=True)

sess = OnsiteSession(weights=WEIGHTS, geom_algo="v2", device=DEVICE, source=src, live=True)

# ── 당신이 짜는 건 사실상 이거 하나: 결과가 오면 UI 갱신(여기선 print) ──
@sess.on_result
def on_scene(scene):
    picks = [f"{p.class_name} {p.long_side_mm:.0f}x{p.short_side_mm:.0f}mm" for p in scene.pick_targets]
    print(f"[high] picks={len(scene.pick_targets)} done={scene.done} {picks}")

# 카메라 열기·루프·스레드는 SDK 가 한다. 당신은 start/stop 과 버튼만.
sess.start()
time.sleep(8)                       # UI 라면 그냥 앱이 떠 있는 시간
print(f"[high] 평균 {sess.fps:.1f} fps (카메라·루프·스레드는 SDK 가 했다)")
vis = sess.draw_latest()            # 최신 결과 시각화 (UI 에 표시)
if vis is not None:
    cv2.imwrite("/tmp/highlevel_out.jpg", vis)
    print("[high] 시각화 저장: /tmp/highlevel_out.jpg")
sess.stop()
