"""AI 오류(code·detail) → HTTP 상태 코드.

기준: mnm-dev `docs/api/ai-internal.yaml`
- 422 입력 때문에 실패 (다시 불러도 같음)
- 404 근거 조항 없음 (`RAG-NONE`)
- 503 외부 API·모델 호출 실패
- 504 시간 초과
재시도 여부는 호출하는 서버(Spring)가 엔드포인트별로 정한다 (mnm-dev#5).
"""

from __future__ import annotations

from typing import Optional

# 기능별 detail.reason → 상태 코드. 표에 없는 reason 은 503.
_REASON_STATUS: dict[str, dict[str, int]] = {
    "stt": {
        "format": 422, "empty": 422, "too_long": 422, "silence": 422, "unreadable": 422, "no_text": 422,
        "timeout": 504,
    },
    "part_check": {"open": 422, "bad_request": 422, "timeout": 504},
    "behavior": {"open": 422, "no_frames": 422, "timeout": 504},
    "guide": {"no_rule": 404, "no_evidence": 404, "timeout": 504},
    "summary": {"report_fields": 422},
    "route": {},
}

# 서빙 계층에서 붙이는 reason (모든 기능 공통)
SERVING_REASON_STATUS: dict[str, int] = {
    "bad_request": 422,     # 요청 형식 오류
    "fetch_rejected": 422,  # 받을 수 없는 주소 (허용 호스트 밖, 만료된 서명 URL 등)
    "too_large": 422,       # 파일이 상한보다 큼
    "fetch_failed": 503,    # 저장소 연결 실패·5xx
    "not_loaded": 503,      # 모델이 서버에 올라와 있지 않음
    "server_timeout": 504,  # 서버 시간 제한 초과
}


def status_for(feature: str, code: str, detail: Optional[dict]) -> int:
    if code == "RAG-NONE":
        return 404  # 근거 없음은 언제나 404 (AI_FORCE_ERROR=guide 강제 오류 포함)
    detail = detail or {}
    reason = detail.get("reason")
    if reason in SERVING_REASON_STATUS:
        return SERVING_REASON_STATUS[reason]
    if feature == "route" and "limit_sec" in detail:
        return 504  # router 시간 초과는 reason 없이 limit_sec 를 넣는다
    return _REASON_STATUS.get(feature, {}).get(reason, 503)


class ServingError(Exception):
    """서빙 계층 실패. `common.call_ai` 가 `to_response()` 로 공통 오류 형식으로 바꾼다."""

    def __init__(self, code: str, reason: str, message: str, **extra):
        super().__init__(message)
        self.code, self.reason, self.extra = code, reason, extra

    def to_response(self) -> dict:
        return {"code": self.code, "message": str(self), "detail": {"reason": self.reason, **self.extra}}
