"""AI 공통: 오류 코드 · 호출 로그 · 개인정보 가림 (명세 REQ-AI-12)."""

from .ai_call import (
    ERROR_CODES,
    FEATURE_ERROR,
    TIMEOUT_SEC,
    AiCallLog,
    AiErrorResponse,
    call_ai,
    mask_phone,
)

__all__ = ["ERROR_CODES", "FEATURE_ERROR", "TIMEOUT_SEC", "AiCallLog", "AiErrorResponse", "call_ai", "mask_phone"]
