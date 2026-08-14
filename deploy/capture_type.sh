#!/usr/bin/env bash
# 박스 종류별 촬영 — 그룹화(서브클래스) 정답 데이터 수집용.
#
#   bash ~/capture_type.sh <종류이름> [옵션]
#
# 예:
#   bash ~/capture_type.sh blue_printed      # 파란 인쇄 소매박스
#   bash ~/capture_type.sh navy_plain        # 네이비 무지 "DO NOT STACK"
#   bash ~/capture_type.sh pacific           # Pacific 브랜드 tan
#   bash ~/capture_type.sh plain_white       # 무지 흰/tan
#   bash ~/capture_type.sh blue_printed single   # 낱개 1개만 놓고 촬영(등록용)
#
# ▶ 왜 종류별로 나눠 찍나
#   한 프레임에 **한 종류만** 있으면 그 프레임의 모든 Carton 이 같은 그룹이다.
#   즉 폴더 이름이 곧 정답 라벨이 되어 객체별 라벨링이 필요 없다.
#
# ▶ 조작 (창이 뜬다)
#   스페이스 또는 p = 1장 촬영 / a = 자동촬영 토글 / q = 종료
#
# ▶ 꼭 지켜줄 것
#   · 한 프레임에 **한 종류만** 나오게 (다른 종류가 보이면 그 프레임은 쓸 수 없다)
#   · 같은 박스를 **조명 바꿔가며** — 형광등 on/off, 밝은 쪽/그늘, 커튼 열고/닫고
#     (오늘까지의 학습은 합성 조명이었다. 실제 조명 변화가 이 데이터의 핵심 가치다)
#   · 거리·각도도 조금씩 바꿔서 (depth 가 있으면 물리 크기는 거리에 무관하다)
#   · 종류당 쌓은 상태 5~8장 + 낱개 1~2장
set -euo pipefail

TYPE="${1:-}"
MODE="${2:-stack}"

# 카메라 목록만 보고 싶을 때
if [ "$TYPE" = "cams" ] || [ "$TYPE" = "--list-cameras" ]; then
  exec ~/ganginlee/AL/lee4/ganginenv/bin/dr-onsite-capture --list-cameras
fi

if [ -z "$TYPE" ]; then
  echo "사용법: bash ~/capture_type.sh <종류이름> [single]"
  echo "        bash ~/capture_type.sh cams          # 연결된 카메라 시리얼 보기"
  echo "        CAM=<시리얼> bash ~/capture_type.sh <종류이름>   # 카메라 지정"
  echo "  이미 찍은 종류:"
  for d in "$HOME/box_types"/*/; do
    [ -d "$d" ] || continue
    n=$(ls "$d"*.jpg 2>/dev/null | wc -l)
    printf "    %-18s %s장\n" "$(basename "$d")" "$n"
  done
  exit 1
fi

OUT="$HOME/box_types/${TYPE}${MODE:+_$MODE}"
OUT="${OUT%_stack}"            # stack 은 접미어 없이(기본)
mkdir -p "$OUT"

set -a
DEPAL_DEVICE=cuda
DEPAL_NUM_CLASSES=5
# ★ depth 저장 — 물리 크기(mm)를 쓰기 위해 반드시 필요. 이게 없으면 크기 축이 픽셀이 되고,
#   픽셀 크기는 카메라 거리에 따라 변해 그룹화의 가장 믿을 만한 축을 잃는다.
DEPAL_CAPTURE_DEPTH=1
# 중복 제거는 끈다 — 같은 장면을 조명만 바꿔 찍는 것이 목적이므로, 화면이 비슷하다고
# 걸러버리면 정작 원하는 조명 변화 프레임이 버려진다.
DEPAL_CAPTURE_MIN_DIFF=0
# 카메라 — 기본은 demo_gui.sh 와 같은 카메라. 바꾸려면 CAM 환경변수에 시리얼을 준다.
#   bash ~/capture_type.sh cams            # 연결된 시리얼 확인
#   CAM=ABC123 bash ~/capture_type.sh pacific
DEPAL_CAMERA_BACKEND=orbbec
DEPAL_CAMERA_SERIAL="${CAM:-CL2Z1650107}"
DEPAL_DEPTH_MODE=320x288
DEPAL_USE_IMU=0
DEPAL_UP_PATH=$HOME/up.json
# ROI 는 종류별로 다를 수 있으니 이 촬영 전용 파일에 저장한다(라이브 앱의 roi.json 을
# 덮어써서 운영 설정이 바뀌는 사고를 막는다).
DEPAL_ROI_PATH=$HOME/box_types/roi_capture.json
set +a

echo "════════════════════════════════════════════════════════"
echo " 종류: $TYPE   모드: $MODE"
echo " 저장: $OUT"
echo " 현재까지: $(ls "$OUT"/*.jpg 2>/dev/null | wc -l)장"
echo "────────────────────────────────────────────────────────"
if [ "$MODE" = "single" ]; then
  echo " ▶ 낱개 모드 — 박스 1개만 놓고 1~2장"
else
  echo " ▶ 쌓기 모드 — 이 종류만 팔레트에 쌓고 5~8장"
fi
echo " ▶ 조명을 바꿔가며 찍어주세요 (이게 이 데이터의 핵심)"
echo " ▶ 한 프레임에 다른 종류가 보이면 안 됩니다"
echo "────────────────────────────────────────────────────────"
echo " 카메라: $DEPAL_CAMERA_SERIAL   (바꾸려면 CAM=<시리얼>, 목록은 'cams')"
echo " 스페이스/p = 1장   a = 자동촬영   r = ROI 지정   c = ROI 해제   q = 종료"
echo "════════════════════════════════════════════════════════"

exec ~/ganginlee/AL/lee4/ganginenv/bin/dr-onsite-capture \
     --view --depth --roi --out "$OUT" --serial "$DEPAL_CAMERA_SERIAL" "${@:3}"
