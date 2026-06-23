"""SDK 로깅 (팀 규칙 #6).

라이브러리 위생: 패키지 로거(``dr_onsite_adap``)에는 __init__ 에서 NullHandler 만 단다.
임포트만으로는 아무 것도 출력하지 않고, 임베딩하는 앱이 핸들러/레벨을 제어한다.

CLI/앱 진입점(main)은 ``setup_cli_logging()`` 으로 stdout StreamHandler 를 붙여
사람이 메시지를 보게 한다(기존 print 동작 보존, GUI 가 서브프로세스 stdout 을 캡처하던
경로도 유지). 모듈은 ``logging.getLogger(__name__)`` 로 자식 로거를 쓴다.
"""
import logging
import sys

_PKG = "dr_onsite_adap"


def setup_cli_logging(level: int = logging.INFO) -> None:
    """패키지 로거에 stdout 핸들러를 1회 부착(idempotent). CLI main 에서 호출."""
    logger = logging.getLogger(_PKG)
    logger.setLevel(level)
    if any(getattr(h, "_dr_onsite_cli", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler._dr_onsite_cli = True   # 중복 부착 방지 표식
    logger.addHandler(handler)
    logger.propagate = False
