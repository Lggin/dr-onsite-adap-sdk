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
# Detection weights must match NUM_CLASSES. 5 = Pallet/Slip Sheet/Carton/Tote/Shrink Wrap (v11).
WEIGHTS_PATH = str(_env_path("DEPAL_WEIGHTS", WEIGHTS_DIR / "model.pth"))
NUM_CLASSES = int(os.environ.get("DEPAL_NUM_CLASSES", "5"))
# 기본 0.5: 과거 conf_threshold 인자명 버그로 RF-DETR predict 가 항상 default 0.5 를
# 적용해 왔다. 인자명 수정(threshold=) 시 실효 동작을 그대로 유지하려고 기본값을 0.5 로 둔다.
CONF_THRES = float(os.environ.get("DEPAL_CONF_THRES", "0.5"))
DEVICE = os.environ.get("DEPAL_DEVICE", "cpu")

# Pallet is not a pick target — it is excluded from top-face/occlusion and
# only used as the "empty pallet = done" signal + reference plane.
PALLET_CID = 0
PICKABLE_CIDS = (1, 2, 3, 4)   # Pallet(0) 외 전부 픽 대상 (Slip Sheet/Carton/Tote/Shrink Wrap)

# class_id -> (display name, BGR color)
# ★ v11 최종 순서 (다운스트림 UI/로봇팀 합의 완료, 2026-07-08 적용):
#   0:Pallet, 1:Slip Sheet, 2:Carton, 3:Tote, 4:Shrink Wrap.
#   v10(0:Pallet,1:Carton,2:Tote,3:Wrap) 대비 Slip Sheet 를 1 에 삽입, Carton 1→2 이동.
#   가중치 continual_master_v11.pth (5클래스) 와 순서 일치해야 함.
CLASSES = {
    0: ("Pallet",      (50, 100, 160)),
    1: ("Slip Sheet",  (255, 0, 255)),
    2: ("Carton",      (0, 255, 0)),
    3: ("Tote",        (255, 165, 0)),
    4: ("Shrink Wrap", (0, 0, 255)),
}

# ── 외형 클러스터링(분류 후, 같은 클래스 안에서 물리크기·색·질감으로 서브그룹) ──
# 검출(분류)이 먼저 → 클래스별로 나눈 뒤 각 클래스 안에서만 비슷한 것끼리 묶는다(타 클래스 불혼합).
# 결합 특징: 색(Lab 중앙값, 밝기 L 다운웨이트) + 질감(로그-하이패스·Canny, 조명강건) +
#           크기(물리 long/short mm — 상자 종류의 최강 단서). 고정 스케일 정규화 후 블록 가중 →
#           유클리드 거리 < CLUSTER_TAU 면 연결(union-find). **크기 우선** 기본값.
# 0.35 유지 (2026-08-12 조사 후 결론). 0.15 로 낮추는 안을 검토했다가 **보류**했다.
#   측정: 현장 held-out 40장(Carton 481개)에 조명 변형을 걸고 그룹 유지율을 잼.
#     · 조명 강건성은 L 가중이 낮을수록 단조 개선 — 명도 ×0.4 에서 유지율
#       L_W=1.0 → 6.2% / 0.35 → 65.1% / 0.15 → 78.4% / 0.0 → 84.6%
#     · 그러나 0.15 는 실제로 **다른 SKU 두 쌍을 합쳤다**: 파란 인쇄박스↔네이비 무지박스,
#       Pacific 브랜드박스↔무지 흰박스. 이 현장은 박스 종류를 가르는 신호가 밝기라서
#       (Lab 차이 중 L 이 68~82% 차지, a·b 는 거의 동일) L 을 죽이면 구분 근거도 죽는다.
#   → L_WEIGHT 로는 "전역 조명 변화"와 "같은 프레임 내 물체 간 밝기 차"를 구분할 수 없다.
#     올바른 방향은 **프레임 단위 L 정규화**(프레임 L 중앙값 기준으로 각 객체 L 을 상대화):
#     조명이 통째로 변해도 상대 순서는 보존되므로 강건성과 구분력을 동시에 얻는다.
#   ⚠️ 위 측정은 depth 없이 돌려 크기 2차원이 픽셀(실운영은 물리 mm), 질감이 저해상
#     폴백(실운영은 워프 패치)이었다. 물리 크기가 들어가면 구분을 크기가 떠받쳐 L 의존도가
#     달라질 수 있다 → RGB-D held-out(`dr-onsite-capture --depth`) 확보 후 재검증할 것.
CLUSTER_L_WEIGHT = float(os.environ.get("DEPAL_CLUSTER_L_WEIGHT", "0.35"))     # 밝기(L) 다운웨이트 → 조명차 완화
# 동일 가중치 기본(색=질감=크기) — 크기 확 다른 케이스도 유지되면서, 크기 비슷·색 미묘차
# 케이스가 색/질감으로 갈린다. 특정 축을 더 보려면 해당 env 를 올리면 됨.
CLUSTER_W_COLOR = float(os.environ.get("DEPAL_CLUSTER_W_COLOR", "1.0"))        # 색 가중
CLUSTER_W_TEX = float(os.environ.get("DEPAL_CLUSTER_W_TEX", "0.5"))            # 질감 가중(노이즈 바닥 높아 0.5로 낮춤)
CLUSTER_W_SIZE = float(os.environ.get("DEPAL_CLUSTER_W_SIZE", "1.0"))          # 크기 가중
CLUSTER_TAU = float(os.environ.get("DEPAL_CLUSTER_TAU", "1.3"))                # 연결 거리 임계
CLUSTER_SIZE_SCALE_MM = float(os.environ.get("DEPAL_CLUSTER_SIZE_SCALE_MM", "50"))  # 크기 구분 문턱(mm)
CLUSTER_MIN_PX = int(os.environ.get("DEPAL_CLUSTER_MIN_PX", "60"))             # 이보다 작은 마스크는 그룹 제외
CLUSTER_FEAT_SCALE = float(os.environ.get("DEPAL_CLUSTER_FEAT_SCALE", "0.5"))  # 특징 계산 해상도 배율(속도; 색/질감은 저해상 무해)
# 프로토타입(정답) 모드: 클래스당 최대 10개 레퍼런스에 최근접 배정. 뱅크는 JSON(유저 편집 가능).
# 정답 있는 클래스는 최근접 배정(거리>MATCH_TAU 면 '기타'), 정답 없는 클래스는 비지도 폴백.
PROTOTYPE_PATH = os.environ.get("DEPAL_PROTOTYPE_PATH", "~/cluster_prototypes.json")
CLUSTER_MATCH_TAU = float(os.environ.get("DEPAL_CLUSTER_MATCH_TAU", "3.0"))    # 이보다 멀면 '기타'
CLUSTER_MAX_PROTOTYPES = int(os.environ.get("DEPAL_CLUSTER_MAX_PROTOTYPES", "10"))

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
# Orbbec depth 해상도 모드 "WxH" (예: "320x288"). 비우면 기본(640x576) 우선.
# 낮추면 depth 대역폭이 크게 줄어 여러 클라이언트가 동시에 봐도 depth 가 안 굶음.
# 윗면 추정은 color 해상도로 정렬되므로 낮은 depth 해상도도 충분(2m에서 cm급).
DEPTH_MODE = os.environ.get("DEPAL_DEPTH_MODE", "")

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

# 윗면 추정 알고리즘: "v1"(기존: 수평가정 + 컨투어 minAreaRect) /
# "v2"(점들에 평면 직접 fit + n·up 게이트로 옆면 거부 + 최소면적 trim 사각형).
# v2 는 휜/노이즈 점구름에 강건하고 옆면을 윗면으로 오인하지 않으며 프레임 간 안정적.
GEOM_ALGO = os.environ.get("DEPAL_GEOM_ALGO", "v1").lower()
# v2 전용 파라미터(geometry 가 기본값 갖지만 여기 두면 env 로 튜닝 가능).
TOP_CAND_BAND_M = float(os.environ.get("DEPAL_TOP_CAND_BAND_M", "0.08"))         # 최상단 후보 밴드(m)
TOP_PLANE_THRESH_M = float(os.environ.get("DEPAL_TOP_PLANE_THRESH_M", "0.02"))   # 평면 inlier 거리(m)
TOP_ANGLE_STEP_DEG = int(os.environ.get("DEPAL_TOP_ANGLE_STEP_DEG", "2"))        # 사각형 각도 스윕 간격(deg)

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
# 현장 held-out 평가셋(학습 금지). `dr-onsite-ingest --holdout` 이 여기에만 쌓고,
# `dr-onsite-gate` 가 이걸로 회귀를 판정한다. 학습셋과 물리적으로 분리해 두는 이유는
# 2026-08-06 확인 — 지금까지 현장 라벨 42장이 전부 학습에 쓰여 held-out 이 0장이었다.
HOLDOUT_DATA_DIR = str(_env_path("DEPAL_HOLDOUT_DIR", DEPAL_HOME / "data" / "field_holdout"))

# 현장 평가셋 촬영(runtime/capture.py). 검출을 멈추지 않고 프레임만 쌓는 폴더.
FIELD_CAPTURE_DIR = str(_env_path("DEPAL_FIELD_CAPTURE_DIR", DEPAL_HOME / "field_captures"))
CAPTURE_INTERVAL_S = float(os.environ.get("DEPAL_CAPTURE_INTERVAL", "3.0"))
# 직전 저장분 대비 최소 화면 차이(0~255 평균). 연속 프레임 중복을 걸러 "한 장면 1프레임"을
# 유지한다 — 평가셋에서 거의 같은 프레임은 표본 수만 부풀리고 판정을 왜곡한다. 0 이면 끔.
CAPTURE_MIN_DIFF = float(os.environ.get("DEPAL_CAPTURE_MIN_DIFF", "6.0"))
CAPTURE_SAVE_DEPTH = os.environ.get("DEPAL_CAPTURE_DEPTH", "0") == "1"
LORA_SAFE_DIR = str(_env_path("DEPAL_LORA_SAFE", DEPAL_HOME / "runs" / "lora_safe"))
LORA_RUN_DIR = str(_env_path("DEPAL_LORA_RUN", DEPAL_HOME / "runs" / "lora_run"))
# Merged single-file output of a round.
FINAL_WEIGHTS = str(_env_path("DEPAL_FINAL", WEIGHTS_DIR / "merged.pth"))
# 운영 가중치 경로(재시작해도 로드되는 기본값이 가리키는 곳). GUI '운영 승격'이 현재 가중치를
# 여기로 복사 → 다음 실행부터 적용. demo 스크립트의 DEPAL_WEIGHTS/BASE 가 이 경로를 가리키면 됨.
OPERATIONAL_WEIGHTS = str(_env_path("DEPAL_OPERATIONAL_WEIGHTS", WEIGHTS_DIR / "operational.pth"))

# Synthetic dataset (with pallet labels) used as replay source + eval held-out.
PALLET_DATA_DIR = os.environ.get("DEPAL_PALLET_DATA", str(DEPAL_HOME / "data" / "pallet_synth"))
# 리플레이 소스 데이터셋. 기본 = **고정 리플레이 셋 `replay_fixed_v11`**
# (2026-08-12 생성, train 400 + valid 100, v11 5클래스).
#   · 출처는 final_dataset_ultimate — 현 운영 모델의 학습셋. 리플레이의 목적이 "베이스가
#     이미 배운 것을 유지"시키는 것이므로 그 모델이 학습한 데이터에서 뽑는 게 맞다.
#   · 만들 때 두 가지를 걸렀다: 현장 프레임의 ×50 오버샘플링 중복(같은 사진이 400장을
#     잠식), train·valid 교차 23장(ingest 가 현장 라벨을 양쪽에 넣기 때문 → valid 가
#     held-out 이 아니게 됨). 상세는 데이터셋 안 README.md.
#   · ⚠️ valid 의 Slip Sheet 는 3장뿐(소스 valid 자체에 3장) — 슬립시트 회귀 판정은
#     현장 held-out + `dr-onsite-gate` 로 볼 것.
# 예전 동작(합성 팔레트셋에서 매 라운드 추출)으로 돌리려면 DEPAL_REPLAY_SRC 지정.
REPLAY_SRC_DIR = os.environ.get("DEPAL_REPLAY_SRC",
                                str(DEPAL_HOME / "data" / "replay_fixed_v11"))
# 리플레이에서 각 클래스별로 최소 보장할 이미지 수(가용 범위 내). 희귀 클래스
# (슬립시트 등)가 무작위 추출에서 누락되는 것을 막는다. K 를 초과하면 희귀 우선.
REPLAY_PER_CLASS_MIN = int(os.environ.get("DEPAL_REPLAY_PER_CLASS_MIN", "40"))
# Output dir for the replay-mixed training set builder.
REPLAY_OUT_DIR = os.environ.get("DEPAL_REPLAY_OUT", str(DEPAL_HOME / "data" / "replay_round"))
# 리플레이 고정 명단. 한 번 뽑은 train/valid 파일명을 여기 기록해 **라운드마다 같은
# 리플레이**를 쓰게 한다(시드가 고정이어도 소스 내용이 바뀌면 추출 결과가 달라지므로
# 명단 없이는 고정되지 않는다). 다시 뽑으려면 `dr-onsite-replay --refresh-manifest`.
# 기본값은 고정 셋 안의 명단 — 데이터셋과 명단이 함께 움직여 짝이 어긋나지 않는다.
REPLAY_MANIFEST = os.environ.get(
    "DEPAL_REPLAY_MANIFEST",
    str(DEPAL_HOME / "data" / "replay_fixed_v11" / "replay_manifest.json"))

# LoRA hyperparameters (shared across rounds).
LORA_R = int(os.environ.get("DEPAL_LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("DEPAL_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("DEPAL_LORA_DROPOUT", "0.052412"))
# 어댑터를 붙일 범위. 모델의 Linear 186개 중 **backbone(DINOv2)이 72개·21.2M 파라미터로
# 전체 선형 파라미터의 66%** 를 차지한다(실측). 라운드는 현장 신규 라벨 수십 장 규모라
# 백본까지 흔들면 과적합·저효율이다.
#   "all-linear"  : 전부(기존 동작). r=32 기준 학습 파라미터 4,861,312 = 베이스의 13.74%
#   "no-backbone" : backbone 제외 = transformer+heads 114개(10.2M) — 일반 시각특징 보존
LORA_TARGETS = os.environ.get("DEPAL_LORA_TARGETS", "all-linear")
# ⚠️ 이전에는 wrapper.train() 에 weight_decay 를 **전달조차 하지 않아** rfdetr 기본값
# (TrainConfig.weight_decay=1e-4)이 그대로 쓰였다. 2026-05 W&B 스윕에도 WD 는 없었다.
# 기본값을 1e-4 로 둬 기존 동작을 유지하면서 탐색 가능하게만 만든다.
WEIGHT_DECAY = float(os.environ.get("DEPAL_WEIGHT_DECAY", "1e-4"))
LR = float(os.environ.get("DEPAL_LR", "0.00018334"))
EPOCHS = int(os.environ.get("DEPAL_EPOCHS", "60"))
BATCH_SIZE = int(os.environ.get("DEPAL_BATCH_SIZE", "4"))
EARLY_STOP_PATIENCE = int(os.environ.get("DEPAL_EARLY_STOP_PATIENCE", "15"))
