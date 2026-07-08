"""OnsiteSession 빠른 시작 예제 — UI 팀이 SDK 를 붙이는 최소 패턴.

콜백(push) + 폴링(pull) + 제어 + (카메라 없이) 재생까지 한 파일로 보여준다.

실행:
  # 실제 카메라
  python session_quickstart.py --weights continual_master_v11.pth --seconds 10

  # 카메라 없이 — 저장된 RGB-D 캡처(.npz) 재생
  python session_quickstart.py --weights continual_master_v11.pth \
        --replay '/home/dr/geom_captures/rgbd_*.npz' --seconds 10 --save out.jpg
"""
import argparse
import glob
import time

from dr_onsite_adap import OnsiteSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="검출 가중치(.pth)")
    ap.add_argument("--replay", default=None,
                    help="npz glob — 주면 카메라 대신 재생(카메라 불필요)")
    ap.add_argument("--seconds", type=float, default=10.0, help="실행 시간(초)")
    ap.add_argument("--geom", default="v2", help="윗면 기하 v1/v2")
    ap.add_argument("--device", default=None, help="cuda/cpu (기본 자동)")
    ap.add_argument("--save", default=None, help="마지막 시각화 저장 경로(.jpg)")
    args = ap.parse_args()

    # ── 카메라 없이 돌리려면 ReplaySource(저장된 npz)를 source 로 주입 ──
    source = None
    if args.replay:
        from dr_onsite_adap.runtime.session import ReplaySource
        files = sorted(glob.glob(args.replay))
        if not files:
            raise SystemExit(f"npz 없음: {args.replay}")
        source = ReplaySource(files, loop=True)
        print(f"[replay] 카메라 없이 {len(files)}개 프레임 재생")

    sess = OnsiteSession(weights=args.weights, geom_algo=args.geom,
                         device=args.device, source=source, live=True)

    # (1) PUSH — 프레임마다 콜백으로 결과 받기
    @sess.on_result
    def on_scene(scene):
        items = [f"{p.class_name}@({p.center_3d[0]:+.2f},{p.center_3d[1]:+.2f},"
                 f"{p.center_3d[2]:+.2f})m {p.long_side_mm:.0f}x{p.short_side_mm:.0f}mm"
                 for p in scene.pick_targets]
        flag = "  [PALLET EMPTY-DONE]" if scene.done else ""
        print(f"[result] picks={len(scene.pick_targets)} pallets={len(scene.pallets)}{flag}  {items}")

    @sess.on_error
    def on_error(e):
        print("[error]", repr(e)[:160])

    sess.start()
    try:
        time.sleep(args.seconds)
    finally:
        # (2) PULL — 언제든 최신 결과를 직접 조회
        scene = sess.latest()
        if scene is not None:
            print(f"[latest] fps={sess.fps:.1f}  picks={len(scene.pick_targets)}  done={scene.done}")
        # 시각화 저장(원하면)
        if args.save:
            import cv2
            vis = sess.draw_latest()
            if vis is not None:
                cv2.imwrite(args.save, vis)
                print("시각화 저장:", args.save)
        sess.stop()
        print("세션 종료")


if __name__ == "__main__":
    main()
