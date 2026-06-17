import numpy as np
import pyrealsense2 as rs


def _accel_fps_or_none():
    """Probe connected RealSense devices for a supported accel profile fps.
    Returns the first available fps, or None if no IMU is present."""
    try:
        for dev in rs.context().query_devices():
            for sensor in dev.query_sensors():
                for prof in sensor.get_stream_profiles():
                    if prof.stream_type() == rs.stream.accel:
                        return prof.fps()
    except Exception:
        pass
    return None


class RealSenseStream:
    """Aligned color + depth (meters) + smoothed world-up vector from the IMU.

    The accelerometer's proper-acceleration reading at rest equals -gravity,
    which in this body's coordinate frame is the world-up direction. We
    expose that as `up_dir`. If the camera has no IMU or no reading has
    arrived yet, `up_dir` is None and the caller should fall back to a
    static config value. The IMU axes on D-series cameras are aligned with
    the depth/color frame to within a few degrees — we ignore the precise
    extrinsic rotation for now.
    """

    def __init__(self, width, height, fps, use_imu=True, ema_alpha=0.1):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.imu_enabled = False
        self.accel_fps = _accel_fps_or_none() if use_imu else None
        if self.accel_fps is not None:
            try:
                cfg.enable_stream(
                    rs.stream.accel, rs.format.motion_xyz32f, self.accel_fps
                )
                self.imu_enabled = True
            except RuntimeError:
                pass
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.intrinsics = None
        self._up_ema = None
        self._ema_alpha = ema_alpha

    def _update_up(self, frames):
        for f in frames:
            if f.profile.stream_type() != rs.stream.accel:
                continue
            a = f.as_motion_frame().get_motion_data()
            v = np.array([a.x, a.y, a.z], dtype=np.float64)
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                return
            new = v / norm  # proper acceleration at rest = -gravity = world-up
            if self._up_ema is None:
                self._up_ema = new
            else:
                self._up_ema = (1 - self._ema_alpha) * self._up_ema + self._ema_alpha * new
                self._up_ema = self._up_ema / np.linalg.norm(self._up_ema)
            return

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        if self.intrinsics is None:
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            self.intrinsics = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        if self.imu_enabled:
            # rs.align only touches video streams; read motion frames from
            # the original frameset, not the aligned one.
            self._update_up(frames)
        up_dir = (None if self._up_ema is None
                  else tuple(float(x) for x in self._up_ema))
        return color, depth_m, self.intrinsics, up_dir

    def stop(self):
        self.pipeline.stop()
