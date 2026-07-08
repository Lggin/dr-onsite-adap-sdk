#!/usr/bin/env bash
# UI 팀 x86 개발기 온보딩 설치 — venv 생성 + SDK(+의존성) 설치 + .env 준비.
# 사용:  bash onboarding/install.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
echo "== dr_onsite_adap 온보딩 설치 =="
echo "레포: $REPO"

# 1) 가상환경
if [ ! -d "$REPO/.venv" ]; then
    python3 -m venv "$REPO/.venv"
    echo "[1/3] venv 생성: $REPO/.venv"
else
    echo "[1/3] venv 이미 있음: $REPO/.venv"
fi
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
python -m pip install -U pip wheel >/dev/null

# 2) SDK + 의존성 설치 (editable). torch/rfdetr/opencv/numpy 등 자동.
#    x86 GPU 개발기는 PyPI torch(CUDA) 기본 휠이면 됨. CPU 전용이면 그대로도 동작(느림).
echo "[2/3] SDK + 의존성 설치 중... (torch 등 커서 수 분 소요)"
pip install -e "$REPO"

# 3) 설정 파일 준비
if [ ! -f "$HERE/.env" ]; then
    cp "$HERE/.env.example" "$HERE/.env"
    echo "[3/3] 설정 생성: onboarding/.env  (가중치 경로 등 확인/수정하세요)"
else
    echo "[3/3] 설정 이미 있음: onboarding/.env"
fi

cat <<EOF

✅ 설치 완료.

다음 3가지만 하면 실행됩니다:
  1) 가중치 파일(continual_master_v11.pth)을 담당자에게 받아 아래 경로에 두기:
       $REPO/weights/continual_master_v11.pth
     (다른 경로면 onboarding/.env 의 DEPAL_WEIGHTS 를 수정)
  2) (GPU 없으면) onboarding/.env 에서 DEPAL_DEVICE=cpu 로
  3) 설치 검증 실행:
       bash onboarding/run.sh
     → 카메라 없이 샘플 프레임으로 OnsiteSession 이 도는지 확인합니다.
EOF
