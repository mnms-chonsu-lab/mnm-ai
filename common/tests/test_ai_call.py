import pytest
from pydantic import BaseModel

from common import ERROR_CODES, AiErrorResponse, call_ai, mask_phone


class FeatErr(Exception):
    def to_response(self):
        return {"code": "PART-FAIL", "message": "열 수 없음 010-1234-5678", "detail": {"reason": "open"}}


class ModelErr(Exception):
    class R(BaseModel):
        code: str = "LLM-FAIL"
        message: str = "시간 초과"
        detail: dict | None = None

    def to_response(self):
        return self.R()


def run(feature, fn, **kw):
    logs = []
    res, log = call_ai(feature, "m-v1", "text", fn, sink=logs.append, **kw)
    assert logs == [log]
    return res, log


def test_success_log_fields():
    res, log = run("route", lambda: {"intent": "diary"}, summarize=lambda r: f"intent={r['intent']}")
    assert res == {"intent": "diary"}
    assert set(log.model_dump()) == {"feature", "model_version", "input_kind", "result_summary", "latency_ms", "error"}
    assert log.error is None and log.result_summary == "intent=diary"


def test_feature_error_passthrough_and_masked():
    res, log = run("part_check", lambda: (_ for _ in ()).throw(FeatErr()))
    assert isinstance(res, AiErrorResponse) and res.code == "PART-FAIL" and "010" not in res.message
    assert log.error == "PART-FAIL" and log.result_summary == "error=PART-FAIL"


def test_pydantic_error_response():
    res, _ = run("route", lambda: (_ for _ in ()).throw(ModelErr()))
    assert res.code == "LLM-FAIL"


def test_unknown_exception_uses_feature_code():
    res, _ = run("behavior", lambda: 1 / 0)
    assert res.code == "BEHAVIOR-FAIL" and res.detail == {"type": "ZeroDivisionError"}


@pytest.mark.parametrize("feature,code", [("stt", "STT-FAIL"), ("route", "LLM-FAIL"), ("part_check", "PART-FAIL"),
                                          ("behavior", "BEHAVIOR-FAIL"), ("guide", "RAG-NONE")])
def test_forced_errors(monkeypatch, feature, code):
    monkeypatch.setenv("AI_FORCE_ERROR", feature)
    called = []
    res, log = run(feature, lambda: called.append(1))
    assert res.code == code and log.error == code and not called


def test_force_all_and_by_code(monkeypatch):
    monkeypatch.setenv("AI_FORCE_ERROR", "all")
    assert run("stt", lambda: 1)[0].code == "STT-FAIL"
    monkeypatch.setenv("AI_FORCE_ERROR", "RAG-NONE")
    assert run("guide", lambda: 1)[0].code == "RAG-NONE"
    assert run("stt", lambda: 1)[0] == 1


@pytest.mark.parametrize("s", ["010-1234-5678", "01012345678", "010 1234 5678", "+82 10-1234-5678", "02-123-4567", "031.123.4567"])
def test_mask_phone(s):
    assert mask_phone(f"연락처 {s} 입니다") == "연락처 [전화번호] 입니다"


def test_mask_phone_keeps_tags_and_dates():
    t = "1234번 소 2026-11-15 접종, 개체번호 002 1234 5678"
    assert mask_phone(t) == t


def test_codes():
    assert ERROR_CODES == ("STT-FAIL", "LLM-FAIL", "PART-FAIL", "BEHAVIOR-FAIL", "RAG-NONE")
