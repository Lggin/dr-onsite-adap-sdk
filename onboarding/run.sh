#!/usr/bin/env bash
# 설정(.env)을 읽어 환경변수로 넣고, venv 파이썬으로 스크립트를 실행한다.
# 사용:
#   bash onboarding/run.sh                     # 기본 검증 예제(hello_session.py)
#   bash onboarding/run.sh path/to/your_app.py # 당신의 앱 실행
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
if [ ! -d "$REPO/.venv" ]; then
    echo "먼저 설치하세요: bash onboarding/install.sh"; exit 1
fi
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
set -a
[ -f "$HERE/.env" ] && source "$HERE/.env"
set +a
exec python "${1:-$HERE/hello_session.py}" "${@:2}"
