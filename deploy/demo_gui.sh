#!/usr/bin/env bash
# 데모용 GUI 실행 — 실시간 검출 + 현장 학습(라벨→취합→학습→병합→재로드) 한 번에.
# 학습 탭 버튼들이 아래 env 를 상속받아 동작합니다.
set -a
# 모델/디바이스
DEPAL_WEIGHTS=${1:-$HOME/depal_weights_backup/continual_master_v10.pth}   # 1번째 인자로 가중치 경로 주면 그걸 사용
DEPAL_DEVICE=cuda
DEPAL_NUM_CLASSES=4
DEPAL_GEOM_ALGO=v2   # 윗면 기하 v2(평면fit+옆면거부+최소면적)
# SAM 라벨러
DEPAL_SAM_WEIGHTS=$HOME/depal_weights_backup/sam_vit_b_01ec64.pth
DEPAL_SAM_DEVICE=cuda
# 데이터셋(라벨저장 = 취합 = 학습, 한 폴더)
DEPAL_DATA_DIR=$HOME/depal_data
# 학습 라운드
DEPAL_BASE_WEIGHTS=$HOME/depal_weights_backup/continual_master_v10.pth
DEPAL_LORA_SAFE=$HOME/demo_runs/lora_safe
DEPAL_LORA_RUN=$HOME/demo_runs/lora_run
DEPAL_FINAL=$HOME/demo_runs/merged_demo.pth
DEPAL_EPOCHS=10
# 카메라 (현재 잘 되는 설정과 다르면 여기만 수정)
DEPAL_CAMERA_BACKEND=orbbec
DEPAL_CAMERA_SERIAL=CL2Z16500ZW
DEPAL_DEPTH_MODE=320x288   # USB 대역폭 절감(카메라 다중연결 시 프레임 굶음 방지)
DEPAL_USE_IMU=0
DEPAL_UP_DIR=-0.584,-0.032,-0.811   # 현재 마운트(X틸트 35.8deg) — up.json 측정값과 일치
DEPAL_UP_PATH=$HOME/up.json          # 측정 중력벡터 항상 로드(실행 폴더 무관)
set +a
mkdir -p "$HOME/demo_runs"
echo "GUI 실행: 가중치=$DEPAL_WEIGHTS  데이터=$DEPAL_DATA_DIR  epochs=$DEPAL_EPOCHS"
exec ~/ganginlee/AL/lee4/ganginenv/bin/dr-onsite-gui
