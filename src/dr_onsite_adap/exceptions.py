"""예외 계층 — PAP 팀 SDK 에러 계약(message + error_code + retryable)을 따른다.

팀 표준의 공통 base(`PapSDKError`)는 별도 패키지라 직접 상속할 수 없어, 동일 계약을 갖는
자체 base `DrOnsiteError` 를 둔다. 모든 SDK 예외는 이를 상속하고, 내부 라이브러리 예외
(pyorbbec, torch 등)는 이 타입으로 wrapping 해서 올린다(`raise ... from e`).

에러 코드 규약: E1xxx 카메라 · E2xxx 파이프라인 · E3xxx 검증(입력/설정) · E5xxx 모델.
"""
from typing import Optional


class DrOnsiteError(Exception):
    """모든 dr_onsite_adap 예외의 base.

    Args:
        message: 무엇이 잘못됐고 어떻게 고치는지.
        error_code: 안정적 식별 코드 (예: ``"E3001"``). 생략 시 클래스 기본값.
        retryable: 같은 입력으로 재시도하면 성공할 수 있는지 여부.

    Example:
        >>> try:
        ...     raise CameraError("카메라 못 찾음", error_code="E1002")
        ... except DrOnsiteError as e:
        ...     print(e.error_code, e.retryable)
        E1002 False
    """

    default_code = "E0000"

    def __init__(self, message: str, *, error_code: Optional[str] = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.default_code
        self.retryable = retryable

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class ValidationError(DrOnsiteError):
    """잘못된 입력/설정 (형상·경로·값 등)."""
    default_code = "E3000"


class CameraError(DrOnsiteError):
    """카메라 연결/스트림 열기 실패."""
    default_code = "E1000"


class ModelError(DrOnsiteError):
    """검출 모델 로드/추론 실패."""
    default_code = "E5000"


class PipelineError(DrOnsiteError):
    """액티브러닝 파이프라인(ingest/train/merge 등) 실패."""
    default_code = "E2000"
