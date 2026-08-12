"""OpenCV 창 백엔드(Qt5) 예열 — torchvision 로드 후 창 생성이 멈추는 문제 회피.

pip 로 설치한 ``opencv-python`` 휠은 GUI 백엔드가 **Qt5** 이고, 번들된
``cv2/qt/plugins/platforms/libqxcb.so`` 의 xcb 초기화가 **torchvision 이 이미
로드된 프로세스에서는 반환하지 않는다.** 증상은 예외가 아니라 **무한 대기**라
디버깅이 까다롭다(2026-08-10 4090 개발 PC 실측):

  · ``cv2.namedWindow`` 에서 멈춤, 메인 스레드가 CPU 한 코어를 계속 소모
  · X 연결(unix 소켓)조차 열지 못한 상태 → 창이 아예 안 뜸
  · ``QT_DEBUG_PLUGINS=1`` 기준 libqxcb.so 를 **로드한 직후** 멈춤
  · 임포트 순서를 바꿔도(torchvision 먼저/cv2 먼저) 동일
  · ``QT_QPA_PLATFORM_PLUGIN_PATH`` 로 시스템 Qt5 플러그인을 써도 동일

torchvision 이 로드되기 **전에** 창을 하나 만들었다 닫아 Qt/xcb 초기화를 끝내두면,
그 뒤로는 (창 이름이 달라도) 정상 동작한다. 그래서 cv2 창을 쓰는 CLI 진입점은
**torch/torchvision 을 임포트하기 전에** :func:`warmup` 을 부른다.

SAM 라벨러처럼 torch 를 쓰는 경로에서만 문제가 되므로, 검출·학습 등 창을 안 쓰는
경로는 이 모듈을 부를 필요가 없다.
"""
import os
import sys

_WARMUP_WINDOW = "__cv2_qt_warmup__"
_done = False


def warmup() -> bool:
    """Qt/xcb 를 미리 초기화한다. 성공하면 True.

    화면이 없거나(headless) cv2 가 GUI 없이 빌드된 환경에서는 아무 것도 하지 않고
    False 를 돌려준다 — 예열은 어디까지나 사고 예방이므로, 실패해도 호출자를
    막지 않는다. 이미 torchvision 이 로드된 뒤라면 예열해도 늦으므로 경고만 남긴다.
    """
    global _done
    if _done:
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False          # headless — 창을 띄울 일 자체가 없다
    if "torchvision" in sys.modules:
        print("⚠️ torchvision 이 이미 로드된 뒤라 창 백엔드 예열이 늦었습니다 "
              "— 창 생성에서 멈추면 이 호출을 임포트보다 앞으로 옮기세요.")
        return False
    try:
        import cv2
        import numpy as np
        cv2.namedWindow(_WARMUP_WINDOW)
        cv2.imshow(_WARMUP_WINDOW, np.zeros((1, 1, 3), np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow(_WARMUP_WINDOW)
        cv2.waitKey(1)
    except Exception as e:      # GUI 없는 빌드(headless 휠) 등 — 조용히 넘어간다
        print(f"창 백엔드 예열 생략({type(e).__name__}: {e})")
        return False
    _done = True
    return True
