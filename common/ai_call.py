"""AI 공통 (명세 REQ-AI-12).

- 실패 응답은 정해진 오류 코드 5종, 형식 { code, message, detail }
- 호출마다 로그 필드 6개: feature, model_version, input_kind, result_summary, latency_ms, error
- 로그·프롬프트에 이름·전화번호를 남기지 않는다
- 서버 대체 경로 확인용으로 오류를 강제로 낼 수 있다 (환경변수 `AI_FORCE_ERROR`)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ErrorCode = Literal["STT-FAIL", "LLM-FAIL", "PART-FAIL", "BEHAVIOR-FAIL", "RAG-NONE"]
ERROR_CODES: tuple[str, ...] = ("STT-FAIL", "LLM-FAIL", "PART-FAIL", "BEHAVIOR-FAIL", "RAG-NONE")

Feature = Literal["stt", "route", "part_check", "behavior", "guide", "summary"]
# 기능 → 실패 시 오류 코드
FEATURE_ERROR: dict[str, str] = {
    "stt": "STT-FAIL",
    "route": "LLM-FAIL",
    "part_check": "PART-FAIL",
    "behavior": "BEHAVIOR-FAIL",
    "guide": "RAG-NONE",
    "summary": "LLM-FAIL",
}
# 모르는 예외(코드 버그 등)의 오류 코드. guide 는 근거 없음(RAG-NONE)이 아니라 장애(LLM-FAIL)로 본다
UNEXPECTED_ERROR: dict[str, str] = {**FEATURE_ERROR, "guide": "LLM-FAIL"}
# 명세 타임아웃: 일반 8초, 부위 검증 1.5초
TIMEOUT_SEC: dict[str, float] = {"default": 8.0, "part_check": 1.5}

logger = logging.getLogger("mnm_ai.calls")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AiErrorResponse(_Strict):
    code: ErrorCode
    message: str
    detail: Optional[dict] = None


class AiCallLog(_Strict):
    """A-S07 모니터링으로 보내는 한 줄. 원문(발화·사진·영상)은 넣지 않는다."""

    feature: Feature
    model_version: str
    input_kind: str = Field(description='예: "audio/webm", "text", "image/jpeg", "video/mp4"')
    result_summary: str = Field(description="결과 요약 (코드·개수만, 원문 없음)")
    latency_ms: int = Field(ge=0)
    error: Optional[ErrorCode] = None


# 010-1234-5678, 01012345678, 010 1234 5678, +82 10-1234-5678, 지역번호 02-123-4567 등
_PHONE = re.compile(
    r"(?<!\d)(?:\+82[-\s]?0?|0)1[016789][-\s.]?\d{3,4}[-\s.]?\d{4}(?!\d)"  # 휴대전화
    r"|(?<!\d)0(?:2|[3-6]\d)[-\s.]\d{3,4}[-\s.]\d{4}(?!\d)"  # 지역번호 02, 031~064
)


def mask_phone(text: str) -> str:
    """전화번호를 [전화번호] 로 바꾼다. 프롬프트·로그에 넣기 전에 쓴다."""
    return _PHONE.sub("[전화번호]", text)


def _forced(feature: str) -> Optional[str]:
    """AI_FORCE_ERROR=route,stt 처럼 기능 이름 또는 오류 코드를 쉼표로.

    기능 이름·all 은 그 기능의 기본 코드(guide 는 RAG-NONE). 오류 코드는 그 코드를 낼 수 있는 기능 전부에 적용
    (LLM-FAIL 이면 route·summary·guide).
    """
    raw = os.getenv("AI_FORCE_ERROR", "")
    want = {s.strip() for s in raw.split(",") if s.strip()}
    code = FEATURE_ERROR[feature]
    if feature in want or code in want or "all" in want:
        return code
    other = UNEXPECTED_ERROR[feature]
    return other if other in want else None


def _to_error(exc: Exception, feature: str) -> AiErrorResponse:
    """각 기능의 예외(to_response 보유)를 공통 형식으로. 모르는 예외는 기능의 오류 코드로 감싼다."""
    if hasattr(exc, "to_response"):
        r = exc.to_response()
        d = r.model_dump() if isinstance(r, BaseModel) else dict(r)
        return AiErrorResponse(code=d["code"], message=mask_phone(d["message"]), detail=d.get("detail"))
    return AiErrorResponse(code=UNEXPECTED_ERROR[feature], message="처리 실패", detail={"type": type(exc).__name__})


def call_ai(
    feature: str,
    model_version: str,
    input_kind: str,
    fn: Callable[[], Any],
    summarize: Callable[[Any], str] = lambda r: "ok",
    sink: Callable[[AiCallLog], None] | None = None,
) -> tuple[Any, AiCallLog]:
    """기능 호출을 감싸 (결과 또는 AiErrorResponse, 로그)를 돌려준다. 예외를 밖으로 내지 않는다."""
    t0 = time.perf_counter()
    forced = _forced(feature)
    if forced:
        result: Any = AiErrorResponse(code=forced, message="강제 오류 (AI_FORCE_ERROR)", detail={"forced": True})
    else:
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 — 모든 실패를 정해진 코드로
            result = _to_error(e, feature)
    err = result.code if isinstance(result, AiErrorResponse) else None
    if err:
        summary = f"error={err}"
    else:
        try:
            summary = mask_phone(summarize(result))
        except Exception:  # noqa: BLE001
            summary = "ok"
    log = AiCallLog(
        feature=feature,
        model_version=model_version,
        input_kind=input_kind,
        result_summary=summary,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        error=err,
    )
    (sink or _default_sink)(log)
    return result, log


def _default_sink(log: AiCallLog) -> None:
    logger.info(json.dumps(log.model_dump(), ensure_ascii=False))
