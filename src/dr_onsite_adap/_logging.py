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


def ensure_utf8_output() -> None:
    """stdout/stderr 를 UTF-8 로 고정한다(한글 출력이 로케일 때문에 죽는 것 방지).

    cron·systemd·서브프로세스처럼 로케일이 안 잡힌 환경에서는 stdout 인코딩이 ascii 가 되고,
    한글이나 이모지를 print 하는 순간 UnicodeEncodeError 로 **명령이 실패**한다. 실제로
    2026-08-03 에는 이 원인으로 학습이 첫 검증 epoch 에서 죽었고(progress.log 파일 인코딩),
    2026-08-06 에는 회귀 게이트가 판정은 PASS 인데 종료코드 1 을 돌려줬다(마지막 print 가 터져서).
    판정 결과를 종료코드로 쓰는 도구에서 이건 곧 오판이므로 진입점에서 못 박는다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            if enc not in ("utf8", "utf8mb4") and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass   # 재설정 불가한 스트림(파이프 래핑 등)은 그대로 둔다


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
