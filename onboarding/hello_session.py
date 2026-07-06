"""온보딩 검증 예제 — 고수준 OnsiteSession 이 도는지 확인.

x86 개발기엔 보통 카메라가 없으므로, 번들된 샘플 프레임을 ReplaySource 로 재생한다.
설치·설정(.env: 가중치·기하·장치)이 맞으면 검출 결과가 흘러나온다.

실행:  bash onboarding/run.sh
"""
import os
import time

from dr_onsite_adap import OnsiteSession
from dr_onsite_adap.runtime.session import ReplaySource

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample", "frame.npz")

# 카메라 없이 검증: 샘플 프레임 재생.
#   실제 카메라를 붙일 땐 아래 두 줄을 지우고 OnsiteSession(source=None) 으로 두면
#   SDK 가 .env 의 DEPAL_CAMERA_* 설정으로 카메라를 연다.
src = ReplaySource([SAMPLE], loop=True)
sess = OnsiteSession(source=src)          # weights/geom/device 는 .env(환경변수)에서

@sess.on_result
def on_scene(scene):
    names = [p.class_name for p in scene.pick_targets]
    print(f"[hello] picks={len(scene.pick_targets)} done={scene.done}  {names}")

@sess.on_error
def on_error(e):
    print("[error]", repr(e)[:160])

print("OnsiteSession 시작 (카메라 없이 샘플 재생)...")
sess.start()
time.sleep(6)
print(f"평균 {sess.fps:.1f} fps")
sess.stop()
print("✅ 설치·설정 OK — 고수준 SDK(OnsiteSession) 정상 동작. 이제 당신 앱을 짜면 됩니다.")
