#!/usr/bin/env bash
# 랩 실행 래퍼 (Thor 전용) — 운영과 동일한 env 를 세팅한 뒤 파이썬 스크립트를 실행한다.
# 사용:  bash run.sh step1_hello.py   /   bash run.sh step3_highlevel.py --replay '/home/dr/geom_captures/rgbd_*.npz'
set -a
DEPAL_WEIGHTS=$HOME/depal_weights_backup/continual_master_v10.pth   # 현 운영 가중치(새 클래스순서)
DEPAL_NUM_CLASSES=4
DEPAL_DEVICE=cuda
DEPAL_GEOM_ALGO=v2                      # 윗면 기하 v2
DEPAL_CAMERA_BACKEND=orbbec
DEPAL_CAMERA_SERIAL=CL2Z16500ZW        # 작업장면 보는 카메라
DEPAL_DEPTH_MODE=320x288
DEPAL_USE_IMU=1
DEPAL_UP_PATH=$HOME/up.json
set +a
exec ~/ganginlee/AL/lee4/ganginenv/bin/python "$@"
