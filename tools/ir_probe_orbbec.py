"""IR 프로브 (Orbbec Femto Mega) — "흑백 학습 + IR 추론" 안의 승패 요인을 실측한다.

⚠️ 카메라는 **Femto Mega(iToF, Azure Kinect DK 계열 센서)** 다. RealSense(활성 스테레오)가
   아니므로 다음이 달라진다 — 이 안에 유리하게:
     • ToF 는 구조광 점패턴을 뿌리지 않는다 → **IR 에 점패턴 없음**. emitter 교대 촬영도,
       그로 인한 fps 반감도 없다(그건 RealSense D4xx 특성이었다).
     • IR 은 depth 와 **같은 ToF 센서**가 만든다 → 픽셀 완전 정합. 정렬 불필요.
   대신 ToF 고유의 갭이 생긴다:
     • IR amplitude ∝ 반사율 × cos(입사각) / 거리²  → **멀거나 기울어진 면이 어둡다**.
       단순 흑백 변환 렌더에는 이 falloff 가 없다. 이게 남은 주된 도메인 갭이다.
     • 근거리/재귀반사(테이프·비닐)에서 포화, 원거리에서 노이즈.

목적: 학습을 걸기 **전에** 아래를 눈과 숫자로 확인한다.
  ① 어두운 광택 네이비 박스가 IR 에서 어떻게 보이나(NIR 에선 염료가 투명해 크라프트처럼
     보일 수 있다 = 우리 렌더가 재현 못 하던 축이 사라짐 = 이 안의 핵심 가설)
  ② 거리/각도에 따른 밝기 falloff 가 얼마나 센가(렌더로 근사해야 할 양)
  ③ 인쇄·테이프 경계가 IR 에서 사라지는가(마스크 품질 영향)
  ④ 슈링크랩/투명 토트가 IR 에서 어떻게 보이나(현재 depth 취약 대상)

이 스크립트는 **탐색 우선**이다: 장치가 실제 지원하는 센서/프로파일을 먼저 나열해 출력한
뒤 IR 을 연다. pyorbbecsdk 의 열거자 이름이 기종·버전마다 달라서 하드코딩하지 않는다.

⚠️ 카메라는 한 프로세스만 잡을 수 있다. **실행 전 GUI/라이브앱 종료**, 종료 후 바로
   재접속 금지(몇 초 대기, 'Timestamp anomaly' 방지).

사용(Thor 에서):
    python3 tools/ir_probe_orbbec.py --list          # 지원 스트림만 나열하고 종료
    python3 tools/ir_probe_orbbec.py --shots 3
출력: ~/ir_probe/<ts>/  color.png gray.png ir_vis.png depth_vis.png raw.npz montage.png
"""
import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pyorbbecsdk as obs


def sensor_types():
    """이 SDK 빌드가 아는 센서 타입 이름들(IR 후보 탐색용)."""
    return [n for n in dir(obs.OBSensorType) if not n.startswith("_")]


def describe_profiles(pipeline, type_name):
    """해당 센서 타입의 스트림 프로파일들을 (w,h,fmt,fps) 로 나열."""
    st = getattr(obs.OBSensorType, type_name, None)
    if st is None:
        return None
    try:
        lst = pipeline.get_stream_profile_list(st)
    except Exception as e:
        return f"조회 실패: {e}"
    out = []
    try:
        for i in range(lst.get_count()):
            p = lst.get_stream_profile_by_index(i)
            try:
                v = p.as_video_stream_profile()
                out.append((v.get_width(), v.get_height(), str(v.get_format()), v.get_fps()))
            except Exception:
                out.append((None, None, str(p), None))
    except Exception as e:
        return f"열거 실패: {e}"
    return out


def pick_ir(pipeline):
    """IR 센서 타입과 프로파일을 고른다. Femto Mega 는 IR_SENSOR 하나가 일반적이나
    기종에 따라 LEFT_IR_SENSOR 등으로 나온다 → 후보를 순서대로 시도."""
    cands = [n for n in ("IR_SENSOR", "LEFT_IR_SENSOR", "IR_LEFT_SENSOR", "RIGHT_IR_SENSOR")
             if hasattr(obs.OBSensorType, n)]
    for name in cands:
        profs = describe_profiles(pipeline, name)
        if isinstance(profs, list) and profs:
            return name, profs
    return None, None


def to_vis(img):
    """16bit IR/depth 를 보기용 8bit 로(1~99 백분위 스트레치)."""
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def stats(name, img):
    v = img.astype(np.float32)
    mx = float(v.max()) if v.size else 0
    sat = 100.0 * float((v >= mx * 0.999).mean()) if mx > 0 else 0.0
    return (f"{name:<10} min {v.min():8.1f} p50 {np.percentile(v,50):8.1f} "
            f"p99 {np.percentile(v,99):8.1f} max {mx:8.1f}  최대값비율 {sat:4.1f}%")


def falloff_report(ir, depth_m):
    """거리에 따른 IR 밝기 감쇠를 실측 — 렌더 근사에 쓸 계수를 얻는다.
    이론상 amplitude ∝ 1/r² 이므로 ir*r² 가 거리에 무관해야 한다(재질 동일 가정)."""
    m = (depth_m > 0.2) & (depth_m < 5.0) & np.isfinite(depth_m)
    if m.sum() < 1000:
        print("  falloff: 유효 depth 부족")
        return
    r = depth_m[m]
    v = ir[m].astype(np.float32)
    print("  거리대별 IR 밝기(원시 / r² 보정):")
    for lo, hi in ((0.3, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 4.0)):
        s = (r >= lo) & (r < hi)
        if s.sum() < 200:
            continue
        print(f"    {lo:.1f}~{hi:.1f}m  n={s.sum():>8}  raw {v[s].mean():8.1f}   "
              f"×r² {(v[s] * r[s] ** 2).mean():8.1f}")
    print("    → 'raw' 가 거리에 따라 크게 떨어지고 '×r²' 가 평평하면 1/r² 모델이 맞다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="지원 스트림만 나열하고 종료")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--gap", type=float, default=4.0, help="장면 간 대기 초")
    ap.add_argument("--out", default=os.path.expanduser("~/ir_probe"))
    ap.add_argument("--align-color", action="store_true",
                    help="color 를 depth/IR 격자로 정렬(C2D). 세 채널이 한 격자에 오므로 "
                         "gray vs IR 을 픽셀단위로 공정 비교할 수 있고, "
                         "'IR 로 검출 + color 로 그룹화' 구조와도 같은 배선이다. "
                         "(런타임 기본은 반대 방향인 D2C: depth→color)")
    a = ap.parse_args()

    print(f"pyorbbecsdk: {getattr(obs, '__version__', '버전 미상')}")
    pipeline = obs.Pipeline()
    try:
        info = pipeline.get_device().get_device_info()
        print(f"장치: {info.get_name()}  시리얼 {info.get_serial_number()}  "
              f"FW {info.get_firmware_version()}")
    except Exception as e:
        print(f"장치 정보 조회 실패: {e}")

    print(f"\nSDK 가 아는 센서 타입: {sensor_types()}")
    for t in ("COLOR_SENSOR", "DEPTH_SENSOR"):
        p = describe_profiles(pipeline, t)
        print(f"\n[{t}] {len(p) if isinstance(p, list) else p}"
              + (f"  예: {p[:4]}" if isinstance(p, list) else ""))
    ir_name, ir_profs = pick_ir(pipeline)
    print(f"\n[IR] 센서={ir_name}  프로파일 {len(ir_profs) if ir_profs else 0}개"
          + (f"  예: {ir_profs[:6]}" if ir_profs else "  ⚠️ IR 스트림을 찾지 못함"))
    if a.list or not ir_name:
        return

    cfg = obs.Config()
    n_en = 0
    for t in ("COLOR_SENSOR", "DEPTH_SENSOR", ir_name):
        st = getattr(obs.OBSensorType, t)
        try:
            lst = pipeline.get_stream_profile_list(st)
            prof = None
            if t == "COLOR_SENSOR" and a.align_color:
                # ⚠️ AlignFilter 는 **압축 포맷(MJPG/H264/H265)을 처리하지 못해 color 프레임을
                # 떨궈버린다**(실측: C2D 켜면 "color 프레임 없음"). 그래서 정렬 시엔 무압축
                # BGR 을 요청한다. 무압축은 대역폭을 많이 먹으므로 fps 를 낮춘다
                # (운영 스크립트가 USB 대역폭 때문에 depth 를 320x288 로 낮춰 쓰는 상황이다).
                for w, h, fps in ((1280, 720, 5), (1280, 720, 15), (1280, 960, 5)):
                    try:
                        prof = lst.get_video_stream_profile(w, h, obs.OBFormat.BGR, fps)
                        print(f"  color: BGR {w}x{h}@{fps} (무압축, 정렬용)")
                        break
                    except Exception:
                        continue
            if prof is None:
                prof = lst.get_default_video_stream_profile()
            cfg.enable_stream(prof)
            n_en += 1
            print(f"  활성화: {t}")
        except Exception as e:
            print(f"  ⚠️ {t} 활성화 실패: {e}")
    if n_en == 0:
        print("스트림을 하나도 열지 못했습니다."); return

    outdir = os.path.join(a.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    print(f"\n저장 위치: {outdir}")

    # ⚠️ 이 카메라는 **네트워크 Femto Mega** 라 'Timestamp anomaly' 로 프레임셋이 통째로
    # 버려져 0fps 가 되는 이슈가 있다. 런타임 백엔드(orbbec.py)와 똑같이 동기 요구를 완화하고
    # 스트림별 '마지막 프레임' 을 캐싱해 짝짓는다(고정 마운트라 시점 차는 무시 가능).
    for fn, arg in (("set_frame_aggregate_output_mode",
                     getattr(getattr(obs, "OBFrameAggregateOutputMode", None),
                             "ANY_SITUATION", None)),):
        try:
            if arg is not None:
                getattr(cfg, fn)(arg)
                print(f"  {fn}(ANY_SITUATION)")
        except Exception as e:
            print(f"  ⚠️ {fn} 미지원: {e}")
    try:
        pipeline.disable_frame_sync()
        print("  disable_frame_sync()")
    except Exception as e:
        print(f"  ⚠️ disable_frame_sync 미지원: {e}")

    pipeline.start(cfg)
    align = None
    if a.align_color:
        try:
            align = obs.AlignFilter(obs.OBStreamType.DEPTH_STREAM)   # C2D: color → depth 격자
            print("  color→depth 정렬(C2D) 활성화 — 세 채널이 IR/depth 격자를 공유합니다")
        except Exception as e:
            print(f"  ⚠️ C2D 정렬 생성 실패: {e}")
    try:
        for s in range(a.shots):
            if s:
                print(f"\n다음 장면까지 {a.gap:.0f}초 — 박스 종류/거리/각도를 바꿔주세요")
                print("  (특히: 어두운 광택 네이비 박스, 슈링크랩, 투명 토트, 원거리 팔레트)")
                time.sleep(a.gap)
            tag = f"shot{s:02d}"
            print(f"\n=== {tag} ===")

            # 동기 완화 상태에선 한 프레임셋이 세 스트림을 다 담지 않는다 → 스트림별 최신을 캐싱.
            # 정렬(C2D)은 color+depth 가 **같은** 프레임셋에 들어온 순간에만 걸 수 있으므로,
            # 그런 셋이 나타나면 그때 정렬해서 채운다.
            cf = df = irf = None
            got_aligned = False
            for _ in range(90):
                f = pipeline.wait_for_frames(2000)
                if f is None:
                    continue
                c, d_, i_ = f.get_color_frame(), f.get_depth_frame(), f.get_ir_frame()
                if i_ is not None:
                    irf = i_
                if align is not None and c is not None and d_ is not None and not got_aligned:
                    out = align.process(f)
                    if out is not None:
                        try:
                            aset = out.as_frame_set()
                        except Exception:
                            aset = out
                        ac, ad = aset.get_color_frame(), aset.get_depth_frame()
                        if ac is not None and ad is not None:
                            cf, df, got_aligned = ac, ad, True
                            print("  ✅ color+depth 동일 프레임셋 확보 → C2D 정렬 적용")
                    continue
                if align is None:
                    if c is not None:
                        cf = c
                    if d_ is not None:
                        df = d_
                if cf is not None and df is not None and irf is not None:
                    break
            if cf is None and df is None and irf is None:
                print("  프레임 수신 실패"); continue
            if align is not None and not got_aligned:
                print("  ⚠️ color+depth 가 같은 프레임셋에 안 들어와 정렬 실패 "
                      "(fps 불일치/타임스탬프 이상) → 정렬 없이 진행")
                # 정렬 못 했으면 최신 프레임이라도 쓴다
                for _ in range(30):
                    f = pipeline.wait_for_frames(2000)
                    if f is None:
                        continue
                    c, d_ = f.get_color_frame(), f.get_depth_frame()
                    if c is not None:
                        cf = c
                    if d_ is not None:
                        df = d_
                    if cf is not None and df is not None:
                        break

            def arr(fr, ch=1, dt=np.uint8):
                if fr is None:
                    return None
                d = np.frombuffer(fr.get_data(), dtype=dt)
                h, w = fr.get_height(), fr.get_width()
                return d.reshape((h, w, ch)) if ch > 1 else d.reshape((h, w))

            color = None
            if cf is not None:
                buf = np.frombuffer(cf.get_data(), dtype=np.uint8)
                try:                                 # 대개 MJPG 또는 BGR/RGB
                    color = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                except Exception:
                    color = None
                if color is None:
                    try:
                        color = buf.reshape((cf.get_height(), cf.get_width(), 3))
                    except Exception:
                        color = None

            depth = arr(df, dt=np.uint16)
            ir = arr(irf, dt=np.uint16)
            if ir is None and irf is not None:       # Y8 IR 인 기종
                ir = arr(irf, dt=np.uint8)

            depth_m = None
            if depth is not None:
                scale = 1.0
                try:
                    scale = df.get_depth_scale()     # mm 단위 스케일
                except Exception:
                    pass
                depth_m = depth.astype(np.float32) * (scale if scale else 1.0) / 1000.0

            base = os.path.join(outdir, tag)
            if color is not None:
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"{base}_color.png", color)
                cv2.imwrite(f"{base}_gray.png", gray)
                print("  " + stats("gray", gray))
            else:
                gray = None
                print("  ⚠️ color 프레임 없음")
            if ir is not None:
                cv2.imwrite(f"{base}_ir_vis.png", to_vis(ir))
                print("  " + stats("ir", ir))
                print(f"  IR 해상도 {ir.shape[::-1]}  dtype {ir.dtype}")
            if depth_m is not None:
                cv2.imwrite(f"{base}_depth_vis.png",
                            cv2.applyColorMap(to_vis(depth_m), cv2.COLORMAP_JET))
                print(f"  depth 유효픽셀 {100 * float((depth_m > 0).mean()):.1f}%")
                if ir is not None and ir.shape == depth_m.shape:
                    falloff_report(ir, depth_m)
                elif ir is not None:
                    print(f"  ⚠️ IR{ir.shape} 과 depth{depth_m.shape} 해상도 불일치 "
                          f"→ falloff 분석 생략(정합 확인 필요)")

            np.savez_compressed(f"{base}_raw.npz",
                                color=(color if color is not None else np.zeros((1,))),
                                ir=(ir if ir is not None else np.zeros((1,))),
                                depth=(depth if depth is not None else np.zeros((1,))))

            panels = []
            if color is not None:
                panels.append(("color", color))
                panels.append(("gray(RGB→L)", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
            if ir is not None:
                panels.append(("IR (ToF, 점패턴 없음)", cv2.cvtColor(to_vis(ir), cv2.COLOR_GRAY2BGR)))
            if depth_m is not None:
                panels.append(("depth", cv2.applyColorMap(to_vis(depth_m), cv2.COLORMAP_JET)))
            if panels:
                tiles = []
                for name, img in panels:
                    t = cv2.resize(img, (640, 360))
                    cv2.rectangle(t, (0, 0), (640, 26), (0, 0, 0), -1)
                    cv2.putText(t, name, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 1)
                    tiles.append(t)
                while len(tiles) % 2:
                    tiles.append(np.zeros_like(tiles[0]))
                rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
                cv2.imwrite(f"{base}_montage.png", np.vstack(rows))
                print(f"  저장: {base}_montage.png")
    finally:
        pipeline.stop()
        print("\n카메라 해제 완료. 라이브앱 재실행 전 몇 초 기다리세요.")

    print("\n확인 포인트:")
    print("  ① 어두운 광택 네이비 박스가 IR 에서 크라프트처럼 보이는가")
    print("     → 그렇다면 '렌더로 재현 못하는 색 축' 이 사라진다 = 이 안의 핵심 이득")
    print("  ② 거리별 falloff 표: raw 가 급감하면 렌더 근사에 1/r² 를 넣어야 한다")
    print("     (Isaac 은 depth·normals 를 이미 렌더하므로 gray×cosθ/r² 로 합성 가능)")
    print("  ③ 인쇄/테이프 경계가 사라지는가(마스크 품질), 테이프가 포화되는가")
    print("  ④ 슈링크랩/투명 토트가 IR 에서 보이는가(현재 depth 취약 대상)")


if __name__ == "__main__":
    main()
