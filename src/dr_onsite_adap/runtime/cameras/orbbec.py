"""Orbbec Femto Bolt / Femto Mega-class camera I/O via pyorbbecsdk.

Mirrors RealSenseStream's contract:
  • __init__(width, height, fps)
  • read() -> (color_bgr, depth_m_float32, (fx, fy, cx, cy), up_dir or None)
  • stop()
  • .imu_enabled : bool
  • .accel_fps  : int | None

ToF cameras like Femto Bolt have a fixed set of native depth resolutions
(640x576 / 512x512 / 320x288 / 1024x1024), so the requested width/height
applies to COLOR only. Depth is picked from the device's native modes and
the AlignFilter resamples it to color's resolution.
"""

import os

import cv2
import numpy as np
import pyorbbecsdk as obs


# Femto Bolt depth (K4A ToF) native modes, in preference order.
_DEPTH_PREFERRED = [(640, 576), (512, 512), (320, 288), (1024, 1024)]


# Femto Bolt's accelerometer reports in the IMU body frame, which is rotated
# relative to the color/depth camera frame. Empirically derived from two
# orientations (lens-up: imu=(-1,0,0), color frame upright: imu=(0,0,-1)):
#   IMU +X  →  color  -Z  (camera back)
#   IMU +Y  →  color  -X  (camera left)
#   IMU +Z  →  color  +Y  (image down)
# So v_color = _IMU_TO_COLOR @ v_imu.
_IMU_TO_COLOR = np.array([
    [ 0, -1,  0],
    [ 0,  0,  1],
    [-1,  0,  0],
], dtype=np.float64)


def _enable_color(pipeline, cfg, width, height, fps):
    """Enable a color stream at (width, height, fps). Returns (profile, decode_kind),
    where decode_kind is one of:
      None       — raw BGR, zero-cost.
      'bgra2bgr' — drop alpha channel (zero-copy view).
      'rgb2bgr'  — cv2.cvtColor.
      'mjpg'     — cv2.imdecode JPEG per frame (slow, ~5-15ms at 720p).
    Femto Bolt at 720p typically lacks raw BGR but supports BGRA, so we
    prefer BGRA over MJPG to avoid the per-frame JPEG decode."""
    color_list = pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
    candidates = [
        ("BGR",  None),
        ("BGRA", "bgra2bgr"),
        ("RGB",  "rgb2bgr"),
        ("MJPG", "mjpg"),
    ]
    for fmt_name, decode_kind in candidates:
        fmt = getattr(obs.OBFormat, fmt_name, None)
        if fmt is None:
            continue
        try:
            p = color_list.get_video_stream_profile(width, height, fmt, fps)
            cfg.enable_stream(p)
            print(f"  color: {width}x{height}@{fps} {fmt_name}")
            return p, decode_kind
        except Exception:
            continue
    raise RuntimeError(
        f"Femto Bolt color: {width}x{height}@{fps} not available in BGR/BGRA/RGB/MJPG."
    )


def _enable_depth(pipeline, cfg, fps):
    """Enable a depth stream using one of Femto Bolt's native ToF resolutions."""
    depth_list = pipeline.get_stream_profile_list(obs.OBSensorType.DEPTH_SENSOR)
    for sw, sh in _DEPTH_PREFERRED:
        try:
            p = depth_list.get_video_stream_profile(sw, sh, obs.OBFormat.Y16, fps)
            cfg.enable_stream(p)
            return p
        except Exception:
            continue
    # Last resort: SDK default (whatever combination the device exposes).
    p = depth_list.get_default_video_stream_profile()
    cfg.enable_stream(p)
    return p


def _try_enable_accel(pipeline, cfg):
    """Return (enabled, fps) after attempting to enable the accel stream."""
    try:
        prof_list = pipeline.get_stream_profile_list(obs.OBSensorType.ACCEL_SENSOR)
        if prof_list is None or len(prof_list) == 0:
            return False, None
        accel_profile = prof_list[0]
        cfg.enable_stream(accel_profile)
        fps = None
        for getter in ("fps", "get_fps"):
            try:
                fps = int(getattr(accel_profile, getter)())
                break
            except Exception:
                continue
        return True, fps
    except Exception:
        return False, None


def _open_pipeline(serial, index):
    """(pipeline, ctx) 를 돌려준다. serial(우선)/index 로 특정 Orbbec 장치를 열고,
    둘 다 비면 device[0] 을 연다(ctx=None).

    ⚠️ Context 를 살려둬야 한다: `obs.Context().query_devices()` 처럼 Context 를 바로
    버리면 DeviceList 의 내부 매니저 포인터가 NULL 이 돼 get_device_by_* 가
    'NULL pointer ... deviceMgr' 로 실패한다(시리얼 목록 읽기는 캐시라 됨). 그래서
    ctx 를 함께 반환해 호출측(FemtoBoltStream)이 장치 사용 동안 참조를 유지한다.
    """
    if not serial and index is None:
        return obs.Pipeline(), None
    ctx = obs.Context()
    dl = ctx.query_devices()
    if serial:
        try:
            dev = dl.get_device_by_serial_number(serial)
        except Exception:
            dev = None
        if dev is None:
            avail = [dl.get_device_serial_number_by_index(i)
                     for i in range(dl.get_count())]
            raise RuntimeError(
                f"DEPAL_CAMERA_SERIAL '{serial}' 카메라를 찾을 수 없습니다. "
                f"연결된 시리얼: {avail}")
        return obs.Pipeline(dev), ctx
    return obs.Pipeline(dl.get_device_by_index(int(index))), ctx


class FemtoBoltStream:
    """Color+depth (aligned to color) + accel-derived world-up direction.

    device_serial / device_index 로 특정 카메라를 고를 수 있다(여러 대 연결 시).
    둘 다 None 이면 환경변수 DEPAL_CAMERA_SERIAL / DEPAL_CAMERA_INDEX 를 보고,
    그것도 없으면 device[0] 을 연다.
    """

    def __init__(self, width, height, fps, use_imu=True,
                 ema_alpha=0.1, frame_timeout_ms=1000,
                 device_serial=None, device_index=None):
        if device_serial is None:
            device_serial = os.environ.get("DEPAL_CAMERA_SERIAL") or None
        if device_index is None:
            _idx = os.environ.get("DEPAL_CAMERA_INDEX")
            device_index = int(_idx) if _idx not in (None, "") else None
        # _ctx 를 인스턴스에 보관해 장치 사용 동안 Context 가 살아있게 한다(위 주석 참고).
        self.pipeline, self._ctx = _open_pipeline(device_serial, device_index)
        cfg = obs.Config()

        self._color_profile, self._color_decode = _enable_color(
            self.pipeline, cfg, width, height, fps
        )
        self._depth_profile = _enable_depth(self.pipeline, cfg, fps)
        if use_imu:
            self.imu_enabled, self.accel_fps = _try_enable_accel(self.pipeline, cfg)
        else:
            self.imu_enabled, self.accel_fps = False, None

        # Align depth onto the color stream's space (so depth[v,u] matches color[v,u]).
        self.align_filter = obs.AlignFilter(obs.OBStreamType.COLOR_STREAM)

        self.pipeline.start(cfg)

        params = self.pipeline.get_camera_param()
        intr = params.rgb_intrinsic
        self.intrinsics = (float(intr.fx), float(intr.fy),
                           float(intr.cx), float(intr.cy))

        self._up_ema = None
        self._ema_alpha = ema_alpha
        self._timeout = frame_timeout_ms

    def _accel_from_frames(self, frames):
        """Pull (x,y,z) m/s² out of whatever accel-frame API this sdk uses."""
        f = None
        try:
            f = frames.get_frame(obs.OBFrameType.ACCEL_FRAME)
        except Exception:
            pass
        if f is None:
            try:
                for fr in frames:
                    if fr.get_type() == obs.OBFrameType.ACCEL_FRAME:
                        f = fr
                        break
            except Exception:
                pass
        if f is None:
            return None
        try:
            af = f.as_accel_frame()
            v = af.get_value()
            return float(v.x), float(v.y), float(v.z)
        except Exception:
            return None

    def _update_up(self, frames):
        triple = self._accel_from_frames(frames)
        if triple is None:
            return
        v_imu = np.array(triple, dtype=np.float64)
        v_color = _IMU_TO_COLOR @ v_imu
        norm = np.linalg.norm(v_color)
        if norm < 1e-6:
            return
        new = v_color / norm  # proper acceleration at rest = -gravity = world-up
        if self._up_ema is None:
            self._up_ema = new
        else:
            self._up_ema = (1 - self._ema_alpha) * self._up_ema + self._ema_alpha * new
            self._up_ema = self._up_ema / np.linalg.norm(self._up_ema)

    def _decode_color(self, color_frame):
        w = color_frame.get_width()
        h = color_frame.get_height()
        buf = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        kind = self._color_decode
        if kind == "mjpg":
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if kind == "bgra2bgr":
            # BGRA stride = w * 4. Take the first 3 channels — alpha is opaque.
            return buf.reshape((h, w, 4))[:, :, :3]
        if kind == "rgb2bgr":
            return cv2.cvtColor(buf.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        return buf.reshape((h, w, 3))

    def read(self):
        frames = self.pipeline.wait_for_frames(self._timeout)
        if frames is None:
            return None

        # IMU lives on the unaligned frameset (AlignFilter only touches video).
        if self.imu_enabled:
            self._update_up(frames)

        aligned = self.align_filter.process(frames)
        if aligned is None:
            return None
        try:
            aligned_set = aligned.as_frame_set()
        except Exception:
            aligned_set = aligned

        color_frame = aligned_set.get_color_frame()
        depth_frame = aligned_set.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        color = self._decode_color(color_frame)
        if color is None:
            return None

        dw = depth_frame.get_width()
        dh = depth_frame.get_height()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((dh, dw))
        try:
            scale_mm = float(depth_frame.get_depth_scale())
        except Exception:
            scale_mm = 1.0
        depth_m = (depth_raw.astype(np.float32) * scale_mm) * 0.001

        up_dir = (None if self._up_ema is None
                  else tuple(float(x) for x in self._up_ema))
        return color, depth_m, self.intrinsics, up_dir

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
