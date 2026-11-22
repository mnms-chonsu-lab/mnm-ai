"""서빙 계층 테스트. 모델·OpenAI 는 가짜로 바꾸고, 주소·토큰·상태 코드·응답 형식만 본다.

모델 자체의 정확도·속도는 각 모듈(inference/·agents/) 테스트와 스테이징 측정에서 본다.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import httpx
import jsonschema
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mlops.serving import app as app_mod
from mlops.serving.config import Settings
from mlops.serving.models import Models

ROOT = Path(__file__).resolve().parents[3]
TOKEN = "test-token"
H = {"X-Internal-Token": TOKEN}
STORAGE = "storage.test"

KB = [
    dict(rule_id="action.fmd.1", kind="action", disease="FMD", stage="after_report", order=1, text="조항 원문 A1",
         doc="문서 A", clause="제1조", revised_at="2026-01-02", status="게시"),
]

SUMMARY_REQ = dict(
    record_id="r-1",
    found_date="2026-11-15",
    cow={"tag4": "1234", "desc": "큰 암소", "barn": "A1"},
    symptom_answers={"drool": True, "blister": True, "lump": False, "feed": False},
    photos=[{"part": "mouth", "match": True}],
    diary=[],
    report_fields={"owner_and_location": "농장주, 경기 ○○군 ○○리", "species": "한우", "head_count": 23,
                   "symptoms": ["침 흘림"], "found_date": None, "reporter": "농장주, 경기 ○○군 ○○리"},
)


def schema(name: str, part: str) -> dict:
    return json.loads((ROOT / "schemas" / f"{name}.json").read_text())[part]


# ---------------------------------------------------------------------------- 가짜 모델
class FakeEmbedder:
    """항상 mouth 로 판정하는 임베더."""

    text_matrix = np.eye(4, dtype=np.float32)

    def encode_image(self, img):
        return np.array([1, 0, 0, 0], dtype=np.float32)


def fake_detector(frames):
    from behavior_yolo11s.behavior import Detection

    return [[Detection(label="stand", confidence=0.9, bbox=(1, 1, 10, 10))] for _ in frames]


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 120, 90)).save(buf, format="PNG")
    return buf.getvalue()


def mp4_bytes(tmp_path: Path) -> bytes:
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(15):
        w.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    w.release()
    return p.read_bytes()


def storage(files: dict[str, bytes]):
    def handler(req: httpx.Request) -> httpx.Response:
        body = files.get(req.url.path)
        return httpx.Response(200, content=body) if body is not None else httpx.Response(403)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def make_client(tmp_path):
    def _make(**overrides):
        files = {"/b/cow.png": png_bytes(), "/b/cow.mp4": mp4_bytes(tmp_path)}
        models = Models(embedder=FakeEmbedder(), detector=fake_detector, kb=KB, http=storage(files))
        for k, v in overrides.items():
            setattr(models, k, v)
        settings = Settings(internal_token=TOKEN, fetch_allowed_hosts=(STORAGE,))
        return TestClient(app_mod.create_app(settings=settings, models=models))

    return _make


# ---------------------------------------------------------------------------- 운영
def test_health_open_without_token(make_client):
    with make_client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert {"summary", "part_check", "behavior"} <= set(r.json()["models"])


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Token": "wrong"}])
def test_token_required(make_client, headers):
    with make_client() as c:
        r = c.post("/ai/summary", json=SUMMARY_REQ, headers=headers)
    assert r.status_code == 401
    assert r.json()["code"] == "LLM-FAIL" and r.json()["detail"]["reason"] == "unauthorized"


def test_empty_token_setting_rejects_everything():
    models = Models(kb=KB)
    with TestClient(app_mod.create_app(settings=Settings(internal_token=""), models=models)) as c:
        r = c.post("/ai/summary", json=SUMMARY_REQ, headers={"X-Internal-Token": ""})
    assert r.status_code == 401


def test_request_id_passed_through(make_client):
    with make_client() as c:
        r = c.post("/ai/summary", json=SUMMARY_REQ, headers={**H, "X-Request-Id": "req-123"})
    assert r.headers["x-request-id"] == "req-123"


# ---------------------------------------------------------------------------- 기능별 성공
def test_part_check_ok_matches_schema(make_client):
    with make_client() as c:
        r = c.post("/ai/part-check", json={"image_url": f"https://{STORAGE}/b/cow.png", "requested_part": "mouth"},
                   headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["match"] is True and r.json()["fallback"] is False
    jsonschema.validate(r.json(), schema("part_check", "response"))


def test_behavior_ok_matches_schema(make_client):
    with make_client() as c:
        r = c.post("/ai/behavior", json={"video_url": f"https://{STORAGE}/b/cow.mp4"}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["main_label"] == "stand"
    jsonschema.validate(r.json(), schema("behavior", "response"))


def test_guide_lookup_ok_matches_schema(make_client):
    with make_client() as c:
        r = c.post("/ai/guide", json={"disease": "FMD", "stage": "after_report"}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["texts"] == ["조항 원문 A1"]
    jsonschema.validate(r.json(), schema("guide", "response"))


def test_summary_blank_is_200_with_missing(make_client):
    with make_client() as c:
        r = c.post("/ai/summary", json=SUMMARY_REQ, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["missing"] == ["found_date"]
    jsonschema.validate(r.json(), schema("summary", "response"))


def test_stt_uploads_file_to_transcribe(make_client, monkeypatch):
    from farm_router import SttResponse

    seen = {}

    def fake_transcribe(path, client=None):
        seen["suffix"], seen["bytes"] = Path(path).suffix, Path(path).read_bytes()
        return SttResponse(text="소독했어요", duration_sec=1.2)

    monkeypatch.setattr(app_mod, "transcribe", fake_transcribe)
    with make_client() as c:
        r = c.post("/ai/stt", files={"file": ("a.m4a", b"audio", "audio/mp4")}, headers=H)
    assert r.status_code == 200, r.text
    assert seen == {"suffix": ".m4a", "bytes": b"audio"}
    jsonschema.validate(r.json(), schema("stt", "response"))


# ---------------------------------------------------------------------------- 상태 코드
def test_validation_error_is_422_with_feature_code_and_no_input(make_client):
    with make_client() as c:
        r = c.post("/ai/part-check", json={"image_url": "x", "requested_part": "elbow"}, headers=H)
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "PART-FAIL" and body["detail"]["reason"] == "bad_request"
    assert "elbow" not in json.dumps(body)  # 입력값은 돌려주지 않음


@pytest.mark.parametrize("url,reason", [
    ("https://evil.test/b/cow.png", "fetch_rejected"),   # 허용 호스트 밖
    (f"https://{STORAGE}.evil.test/b/cow.png", "fetch_rejected"),  # 허용 호스트로 시작만 하는 주소
    (f"https://{STORAGE}/b/expired.png", "fetch_rejected"),  # 서명 만료 → 403
    ("file:///etc/passwd", "fetch_rejected"),
])
def test_fetch_rejected_is_422(make_client, url, reason):
    with make_client() as c:
        r = c.post("/ai/part-check", json={"image_url": url, "requested_part": "mouth"}, headers=H)
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == reason
    assert "expired" not in r.text and "evil" not in r.text  # 주소(서명 포함)는 남기지 않음


def test_storage_down_is_503(make_client):
    def boom(req):
        raise httpx.ConnectError("down")

    with make_client(http=httpx.Client(transport=httpx.MockTransport(boom))) as c:
        r = c.post("/ai/part-check", json={"image_url": f"https://{STORAGE}/b/cow.png", "requested_part": "mouth"},
                   headers=H)
    assert r.status_code == 503 and r.json()["detail"]["reason"] == "fetch_failed"


def test_model_not_loaded_is_503(make_client):
    with make_client(detector=None, load_errors={"behavior": "BEHAVIOR_WEIGHTS 없음"}) as c:
        r = c.post("/ai/behavior", json={"video_url": f"https://{STORAGE}/b/cow.mp4"}, headers=H)
    assert r.status_code == 503
    assert r.json()["code"] == "BEHAVIOR-FAIL"
    assert r.json()["detail"] == {"reason": "not_loaded", "load_error": "BEHAVIOR_WEIGHTS 없음"}


def test_guide_no_rule_is_404(make_client):
    with make_client() as c:
        r = c.post("/ai/guide", json={"disease": "LSD", "stage": "after_report"}, headers=H)
    assert r.status_code == 404
    assert r.json()["code"] == "RAG-NONE" and r.json()["detail"]["reason"] == "no_rule"


@pytest.mark.parametrize("reason,status", [("silence", 422), ("api", 503), ("timeout", 504), ("config", 503)])
def test_stt_reason_to_status(make_client, monkeypatch, reason, status):
    from farm_router import SttError

    def fake(path, client=None):
        raise SttError("x", {"reason": reason})

    monkeypatch.setattr(app_mod, "transcribe", fake)
    with make_client() as c:
        r = c.post("/ai/stt", files={"file": ("a.webm", b"a", "audio/webm")}, headers=H)
    assert r.status_code == status and r.json()["code"] == "STT-FAIL"


@pytest.mark.parametrize("detail,status", [({"elapsed_sec": 9.1, "limit_sec": 8.0}, 504), ({"type": "APIError"}, 503)])
def test_route_timeout_is_504_and_other_failure_503(make_client, monkeypatch, detail, status):
    from farm_router import RouteError

    def fake(body, client=None):
        raise RouteError("x", detail)

    monkeypatch.setattr(app_mod, "route", fake)
    req = {"text": "소독했어요", "today": "2026-11-18", "farm_context": {"cows": [{"tag4": "1234"}]}}
    with make_client() as c:
        r = c.post("/ai/route", json=req, headers=H)
    assert r.status_code == status, r.text
    assert r.json()["code"] == "LLM-FAIL"


def test_server_timeout_is_504(make_client, monkeypatch):
    import time

    def slow(body):
        time.sleep(0.5)

    monkeypatch.setattr(app_mod, "summarize", slow)
    models = Models(kb=KB)
    settings = Settings(internal_token=TOKEN, timeout_sec=0.1)
    with TestClient(app_mod.create_app(settings=settings, models=models)) as c:
        r = c.post("/ai/summary", json=SUMMARY_REQ, headers=H)
    assert r.status_code == 504
    assert r.json()["detail"] == {"reason": "server_timeout", "limit_sec": 0.1}


def test_force_error_env_for_fallback_testing(make_client, monkeypatch):
    monkeypatch.setenv("AI_FORCE_ERROR", "summary")
    with make_client() as c:
        r = c.post("/ai/summary", json=SUMMARY_REQ, headers=H)
    assert r.status_code == 503 and r.json()["detail"] == {"forced": True}


def test_guide_failures_use_llm_fail_but_no_rule_stays_rag_none(make_client):
    with make_client() as c:
        bad = c.post("/ai/guide", json={"disease": "FMD"}, headers=H)          # stage 빠짐
        unauth = c.post("/ai/guide", json={"disease": "FMD", "stage": "after_report"})
        none = c.post("/ai/guide", json={"disease": "LSD", "stage": "after_report"}, headers=H)
    assert (bad.status_code, bad.json()["code"]) == (422, "LLM-FAIL")
    assert (unauth.status_code, unauth.json()["code"]) == (401, "LLM-FAIL")
    assert (none.status_code, none.json()["code"]) == (404, "RAG-NONE")


def test_virtual_hosted_storage_url_allowed(make_client):
    with make_client() as c:
        r = c.post("/ai/part-check", json={"image_url": f"https://bucket.{STORAGE}/b/cow.png", "requested_part": "mouth"},
                   headers=H)
    assert r.status_code == 200, r.text


def test_forced_rag_none_is_404(make_client, monkeypatch):
    monkeypatch.setenv("AI_FORCE_ERROR", "guide")
    with make_client() as c:
        r = c.post("/ai/guide", json={"disease": "FMD", "stage": "after_report"}, headers=H)
    assert (r.status_code, r.json()["code"]) == (404, "RAG-NONE")


def test_server_timeout_logs_once_with_failure_code(make_client, monkeypatch, caplog):
    import logging
    import time

    def slow(body, rules, client=None, index=None):
        time.sleep(0.4)
        raise RuntimeError("늦게 끝난 작업")  # 504 뒤에 끝나도 기록되면 안 됨

    monkeypatch.setattr(app_mod, "guide", slow)
    settings = Settings(internal_token=TOKEN, timeout_sec=0.1)
    with caplog.at_level(logging.INFO, logger="mnm_ai.calls"):
        with TestClient(app_mod.create_app(settings=settings, models=Models(kb=KB))) as c:
            r = c.post("/ai/guide", json={"disease": "FMD", "stage": "after_report"}, headers=H)
            time.sleep(0.6)  # 뒤에서 돌던 작업이 끝날 때까지
    assert (r.status_code, r.json()["code"]) == (504, "LLM-FAIL")
    lines = [json.loads(rec.getMessage()) for rec in caplog.records if rec.name == "mnm_ai.calls"]
    assert len(lines) == 1
    assert lines[0]["error"] == "LLM-FAIL" and lines[0]["result_summary"] == "error=server_timeout"


@pytest.mark.parametrize("load_error", ["GUIDE_KB_PATH 없음 (지식베이스 미적재 → 503 not_loaded)", "FileNotFoundError: kb.json"])
def test_guide_kb_not_loaded_is_503_llm_fail(make_client, load_error):
    with make_client(kb=[], load_errors={"guide": load_error}) as c:
        r = c.post("/ai/guide", json={"disease": "FMD", "stage": "after_report"}, headers=H)
    assert (r.status_code, r.json()["code"]) == (503, "LLM-FAIL")
    assert r.json()["detail"] == {"reason": "not_loaded", "load_error": load_error}


def test_guide_kb_loaded_but_empty_is_404_rag_none(make_client):
    with make_client(kb=[]) as c:  # 파일은 읽었는데 게시 조항 0개
        r = c.post("/ai/guide", json={"disease": "FMD", "stage": "after_report"}, headers=H)
    assert (r.status_code, r.json()["code"]) == (404, "RAG-NONE")


@pytest.mark.parametrize("token", ["토큰".encode(), "tést-token".encode("latin-1")])
def test_non_ascii_token_is_401_not_500(make_client, token):
    with make_client() as c:
        r = c.post("/ai/summary", json=SUMMARY_REQ, headers={"X-Internal-Token": token})
    assert r.status_code == 401 and r.json()["detail"]["reason"] == "unauthorized"
