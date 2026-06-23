"""Camera backends + auto-detection.

resolve_backend("auto") probes for a connected RealSense or Orbbec device
and returns whichever is present. load_camera_class(backend) returns the
matching stream class (lazy import so the unused SDK never has to load).
"""
import sys

from ...exceptions import ValidationError


def _count_realsense():
    try:
        import pyrealsense2 as rs
        return len(rs.context().query_devices())
    except Exception:
        return -1  # SDK missing or broken


def _count_orbbec():
    try:
        import pyorbbecsdk as obs
        return len(obs.Context().query_devices())
    except Exception:
        return -1


def resolve_backend(name):
    """Resolve config.CAMERA_BACKEND ('auto'|'realsense'|'orbbec') to a concrete
    backend with a known-present device. Exits if nothing works."""
    if name == "auto":
        n_rs = _count_realsense()
        n_ob = _count_orbbec()
        if n_rs > 0:
            print(f"Auto-detected camera: realsense ({n_rs} device)")
            return "realsense"
        if n_ob > 0:
            print(f"Auto-detected camera: orbbec ({n_ob} device)")
            return "orbbec"
        sys.exit("No connected RealSense or Orbbec camera found.")
    elif name in ("realsense", "orbbec"):
        n = _count_realsense() if name == "realsense" else _count_orbbec()
        if n < 0:
            sys.exit(f"CAMERA_BACKEND='{name}' but the SDK failed to load.")
        if n == 0:
            sys.exit(
                f"CAMERA_BACKEND='{name}' but no matching camera detected. "
                f"Set CAMERA_BACKEND='auto' to pick whichever is connected."
            )
        return name
    else:
        sys.exit(f"Unknown CAMERA_BACKEND: {name!r}")


def load_camera_class(backend):
    if backend == "realsense":
        from .realsense import RealSenseStream
        return RealSenseStream
    if backend == "orbbec":
        from .orbbec import FemtoBoltStream
        return FemtoBoltStream
    raise ValidationError(
        f"알 수 없는 카메라 backend: {backend!r}. 'realsense' 또는 'orbbec' 여야 합니다.",
        error_code="E3001")


def list_orbbec_serials():
    """연결된 Orbbec 카메라 시리얼 목록(없거나 SDK 미설치면 빈 리스트).
    GUI 카메라 선택 드롭다운용. 시리얼만 읽으므로 장치를 점유하지 않는다."""
    try:
        import pyorbbecsdk as obs
        dl = obs.Context().query_devices()
        return [dl.get_device_serial_number_by_index(i)
                for i in range(dl.get_count())]
    except Exception:
        return []
