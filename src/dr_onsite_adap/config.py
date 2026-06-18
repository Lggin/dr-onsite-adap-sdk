"""Central configuration for dr_onsite_adap.

All paths are resolved from environment variables with sensible relative
defaults so nothing is hard-coded to a particular machine. Override per
deployment, e.g.:

    export DEPAL_HOME=/opt/depal            # base for default sub-paths
    export DEPAL_WEIGHTS=/opt/depal/weights/model.pth
    export DEPAL_DATA_DIR=/data/active_learning_data
    export DEPAL_DEVICE=cuda

Runtime (live camera app) reads the DETECTION + CAMERA + GEOMETRY sections.
The training pipeline reads the PIPELINE section. The two halves share
CLASSES / NUM_CLASSES so a single deployment stays consistent.
"""
import os
from pathlib import Path


def _env_path(name, default):
    v = os.environ.get(name)
    return Path(v) if v else Path(default)


# ---------------------------------------------------------------------------
# Base locations. DEPAL_HOME anchors the relative defaults; point the more
# specific vars at real files/dirs in production.
# ---------------------------------------------------------------------------
DEPAL_HOME = _env_path("DEPAL_HOME", os.getcwd())
WEIGHTS_DIR = _env_path("DEPAL_WEIGHTS_DIR", DEPAL_HOME / "weights")
DATA_DIR = _env_path("DEPAL_DATA_DIR", DEPAL_HOME / "data" / "active_learning_data")

# ---------------------------------------------------------------------------
# DETECTION (runtime)
# ---------------------------------------------------------------------------
# Detection weights must match NUM_CLASSES. 4 = Box/Tote/Wrap/Pallet.
WEIGHTS_PATH = str(_env_path("DEPAL_WEIGHTS", WEIGHTS_DIR / "model.pth"))
NUM_CLASSES = int(os.environ.get("DEPAL_NUM_CLASSES", "4"))
CONF_THRES = float(os.environ.get("DEPAL_CONF_THRES", "0.4"))
DEVICE = os.environ.get("DEPAL_DEVICE", "cpu")

# Pallet is not a pick target — it is excluded from top-face/occlusion and
# only used as the "empty pallet = done" signal + reference plane.
PALLET_CID = 3
PICKABLE_CIDS = (0, 1, 2)

# class_id -> (display name, BGR color)
CLASSES = {
    0: ("Box",    (0, 255, 0)),
    1: ("Tote",   (255, 165, 0)),
    2: ("Wrap",   (0, 0, 255)),
    3: ("Pallet", (50, 100, 160)),
}

# ---------------------------------------------------------------------------
# SAM active-learning labeler (press 't' in the live app)
# ---------------------------------------------------------------------------
SAM_CHECKPOINT = str(_env_path("DEPAL_SAM_WEIGHTS", WEIGHTS_DIR / "sam_vit_b_01ec64.pth"))
SAM_TYPE = os.environ.get("DEPAL_SAM_TYPE", "vit_b")
SAM_DEVICE = os.environ.get("DEPAL_SAM_DEVICE", DEVICE)
# Saved jpg + mask.png + json triples land here, ready for the ingest step.
ACTIVE_DATA_DIR = os.environ.get("DEPAL_ACTIVE_DATA_DIR", str(DATA_DIR))

# ROI crop coords saved here ('r' in the live app). Empty file = full frame.
ROI_PATH = os.environ.get("DEPAL_ROI_PATH", str(DEPAL_HOME / "roi.json"))
# 측정한 up_dir(중력 벡터)을 저장하는 파일. GUI '중력 재측정'이 여기 쓰고, 다음 실행에
# 자동 적용된다. IMU/UP_DIR_CAMERA_FRAME 보다 우선(있으면). 카메라 각도 바뀌면 재측정.
UP_PATH = os.environ.get("DEPAL_UP_PATH", str(DEPAL_HOME / "up.json"))

# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------
# "auto" probes for a connected RealSense or Orbbec device and picks one.
CAMERA_BACKEND = os.environ.get("DEPAL_CAMERA_BACKEND", "auto")
# 여러 대 연결됐을 때 특정 Orbbec 카메라 선택. 시리얼 권장(고정), index 는 연결순서로 바뀜.
# 비워두면 device[0] 을 연다. (orbbec 백엔드가 이 환경변수를 직접 읽는다.)
CAMERA_SERIAL = os.environ.get("DEPAL_CAMERA_SERIAL", "")
CAMERA_INDEX = os.environ.get("DEPAL_CAMERA_INDEX", "")
USE_IMU = os.environ.get("DEPAL_USE_IMU", "1") not in ("0", "false", "False")
CAM_WIDTH = int(os.environ.get("DEPAL_CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.environ.get("DEPAL_CAM_HEIGHT", "720"))
CAM_FPS = int(os.environ.get("DEPAL_CAM_FPS", "30"))

# ---------------------------------------------------------------------------
# GEOMETRY (top-face estimation)
# ---------------------------------------------------------------------------
# Physical depth range (meters); drops invalid/out-of-range pixels.
Z_MIN = float(os.environ.get("DEPAL_Z_MIN", "0.1"))
Z_MAX = float(os.environ.get("DEPAL_Z_MAX", "4.8"))

# Top-layer isolation: take the median of depth in percentile slice [LO, HI].
# LO skips the closest pixels (sensor noise / rounded rim on empty totes);
# HI stops short of the inner-bottom band. Flat box/wrap tops are ~uniform
# so the slice is a no-op for them.
TOP_DEPTH_PERCENTILE_LO = float(os.environ.get("DEPAL_TOP_PCT_LO", "5"))
TOP_DEPTH_PERCENTILE_HI = float(os.environ.get("DEPAL_TOP_PCT_HI", "15"))

# Minimum valid depth pixels under a mask before we attempt a plane.
MIN_PLANE_INLIERS = int(os.environ.get("DEPAL_MIN_PLANE_INLIERS", "80"))

# Top-face band (meters): pixels within ±this of the fitted top plane are taken
# as the top face, so the rectangle/center ignore visible side faces of tall
# boxes. Larger = more tolerant (may include sides); smaller = stricter.
TOP_FACE_BAND_M = float(os.environ.get("DEPAL_TOP_FACE_BAND_M", "0.03"))

# Edge length is the [trim, 100-trim] percentile span of top-face points along
# each rectangle axis (not min/max), trimming edge outliers / mask spill that
# would otherwise inflate the measured side lengths. 0 = raw min/max.
TOP_FACE_TRIM_PCT = float(os.environ.get("DEPAL_TOP_FACE_TRIM_PCT", "1.0"))

# World "up" in the camera frame. Used to pick the top slice and to express
# the horizontal plane. Overridden by IMU at runtime when available; this is
# the fallback for fixed mounts. Calibrate per mounting:
#   overhead straight-down: (0, 0, -1)
#   tilted forward ~25°:    (-0.005, -0.428, -0.904)
_up = os.environ.get("DEPAL_UP_DIR")
if _up:
    UP_DIR_CAMERA_FRAME = tuple(float(x) for x in _up.split(","))
else:
    UP_DIR_CAMERA_FRAME = (0.0, 0.0, -1.0)

# Reject a plane whose normal makes a larger angle than acos(this) with up.
TOP_PLANE_MIN_ALIGNMENT = float(os.environ.get("DEPAL_TOP_PLANE_MIN_ALIGN", "0.7"))

# Mask-based occlusion: drop a detection whose visible mask area is below
# (1 - this) of its own convex hull area. Single-mask test (no cross-mask).
OCCLUSION_HIDDEN_FRACTION = float(os.environ.get("DEPAL_OCCLUSION_FRAC", "0.5"))

# ---------------------------------------------------------------------------
# PIPELINE (active-learning training round). Paths default under DEPAL_HOME.
# ---------------------------------------------------------------------------
# Continual-learning base weights the next LoRA round starts from.
BASE_WEIGHTS = str(_env_path("DEPAL_BASE_WEIGHTS", WEIGHTS_DIR / "base.pth"))
# Training set dir (COCO train/valid). Defaults to the active-learning data.
TRAIN_DATA_DIR = os.environ.get("DEPAL_TRAIN_DATA", str(DATA_DIR))
LORA_SAFE_DIR = str(_env_path("DEPAL_LORA_SAFE", DEPAL_HOME / "runs" / "lora_safe"))
LORA_RUN_DIR = str(_env_path("DEPAL_LORA_RUN", DEPAL_HOME / "runs" / "lora_run"))
# Merged single-file output of a round.
FINAL_WEIGHTS = str(_env_path("DEPAL_FINAL", WEIGHTS_DIR / "merged.pth"))

# Synthetic dataset (with pallet labels) used as replay source + eval held-out.
PALLET_DATA_DIR = os.environ.get("DEPAL_PALLET_DATA", str(DEPAL_HOME / "data" / "pallet_synth"))
# Output dir for the replay-mixed training set builder.
REPLAY_OUT_DIR = os.environ.get("DEPAL_REPLAY_OUT", str(DEPAL_HOME / "data" / "replay_round"))

# LoRA hyperparameters (shared across rounds).
LORA_R = int(os.environ.get("DEPAL_LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("DEPAL_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("DEPAL_LORA_DROPOUT", "0.052412"))
LR = float(os.environ.get("DEPAL_LR", "0.00018334"))
EPOCHS = int(os.environ.get("DEPAL_EPOCHS", "60"))
BATCH_SIZE = int(os.environ.get("DEPAL_BATCH_SIZE", "4"))
EARLY_STOP_PATIENCE = int(os.environ.get("DEPAL_EARLY_STOP_PATIENCE", "15"))
