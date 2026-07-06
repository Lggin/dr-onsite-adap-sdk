"""STEP 3 — 고수준(high-level) facade: OnsiteSession

핵심 감각: **SDK 가 카메라 + 루프 + 백그라운드 스레드까지 다 한다.**
당신은 콜백(on_result)을 연결하고 start() 만. STEP 2 의 '프레임 준비·루프' 가 사라진다.

실행(실제 Thor 카메라):   bash run.sh step3_highlevel.py
실행(카메라 없이, 재생):  bash run.sh step3_highlevel.py --replay '/home/dr/geom_captures/rgbd_*.npz'
"""
import argparse
import glob
import time

from dr_onsite_adap import OnsiteSession

ap = argparse.ArgumentParser()
ap.add_argument("--replay", default=None, help="npz glob — 주면 카메라 대신 재생(카메라 불필요)")
ap.add_argument("--seconds", type=float, default=15.0)
args = ap.parse_args()

# 카메라 없이 돌리려면 ReplaySource 를 주입. 안 주면 SDK 가 '실제 카메라' 를 연다.
source = None
if args.replay:
    from dr_onsite_adap.runtime.session import ReplaySource
    files = sorted(glob.glob(args.replay))
    if not files:
        raise SystemExit(f"npz 없음: {args.replay}")
    source = ReplaySource(files, loop=True)
    print(f"[카메라 없이] {len(files)}개 프레임 재생")
else:
    print("[실제 카메라] SDK 가 Thor 카메라를 엽니다... (당신은 카메라 코드를 한 줄도 안 씀)")

sess = OnsiteSession(source=source)   # weights/geom 은 env 에서

# ── 당신이 짜는 건 사실상 이것 하나: 결과가 오면 무엇을 할지 ──
@sess.on_result
def on_scene(scene):
    names = " ".join(p.class_name for p in scene.pick_targets)
    print(f"[live] picks={len(scene.pick_targets)} done={scene.done}  {names}")

@sess.on_error
def on_error(e):
    print("[error]", repr(e)[:140])

sess.start()               # ← SDK 가 카메라 열고 루프를 스레드로 돌린다
time.sleep(args.seconds)   # UI 라면 그냥 앱이 떠 있는 시간
print(f"\n평균 {sess.fps:.1f} fps")
vis = sess.draw_latest()
if vis is not None:
    import cv2
    cv2.imwrite("/tmp/step3_out.jpg", vis)
    print("시각화 저장: /tmp/step3_out.jpg")
sess.stop()
print("→ 배운 것: 카메라 열기·프레임 추출·루프·스레드 == SDK 가 다 했다. 당신은 콜백 하나 + start/stop.")
