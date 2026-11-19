"""mnm-ai 내부 API 서버 (FastAPI).

`services/api`(Spring)가 같은 Docker 네트워크 안에서 부른다. 외부(nginx)로는 열지 않는다.
요청·응답 형식은 `schemas/`, 주소·상태 코드·시간 제한은 mnm-dev `docs/api/ai-internal.yaml` 기준.

실행 (저장소 루트에서):
    uvicorn mlops.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# 기능 패키지는 agents/ · inference/ · 저장소 루트를 경로에 두고 불러온다 (각 tests/conftest.py 와 같은 방식)
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "agents", ROOT / "inference"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.ai_call import FEATURE_ERROR, UNEXPECTED_ERROR, AiCallLog, AiErrorResponse, call_ai  # noqa: E402
from farm_router import RouteRequest, route, transcribe  # noqa: E402
from farm_router import router as router_mod  # noqa: E402
from farm_router import stt as stt_mod  # noqa: E402
from guide_rag import GuideRequest, guide  # noqa: E402
guide_mod = importlib.import_module("guide_rag.guide")  # 패키지가 같은 이름의 함수를 내보내서 모듈로 직접 불러옴
from receiver_copilot import SummaryRequest, summarize  # noqa: E402
from receiver_copilot import summary as summary_mod  # noqa: E402
from behavior_yolo11s import analyze_video  # noqa: E402
from behavior_yolo11s import behavior as behavior_mod  # noqa: E402
from behavior_yolo11s.schema import BehaviorRequest  # noqa: E402
from part_check_siglip2 import check_part  # noqa: E402
from part_check_siglip2 import part_check as part_mod  # noqa: E402
from part_check_siglip2.schema import PartCheckRequest  # noqa: E402

from .config import Settings  # noqa: E402
from .errors import ServingError, status_for  # noqa: E402
from .fetch import FetchError, fetch, suffix_of, to_tempfile  # noqa: E402
from .models import Models, load_models  # noqa: E402

logger = logging.getLogger("mnm_ai.serving")
call_logger = logging.getLogger("mnm_ai.calls")

# 주소 → 기능 (요청 형식 오류를 기능별 오류 코드로 돌려주기 위해)
PATH_FEATURE = {
    "/ai/stt": "stt", "/ai/route": "route", "/ai/part-check": "part_check",
    "/ai/behavior": "behavior", "/ai/guide": "guide", "/ai/summary": "summary",
}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 60초 webm·m4a 여유 (제안값, 실측으로 조정)


def _code(feature: str, status: int) -> str:
    """서빙 계층 오류 코드. guide 는 근거 없음(404)만 RAG-NONE, 나머지(401·422·503·504)는 장애 코드 LLM-FAIL."""
    return FEATURE_ERROR[feature] if status == 404 else UNEXPECTED_ERROR[feature]


def _error(feature: str, status: int, reason: str, message: str, **extra) -> JSONResponse:
    body = {"code": _code(feature, status), "message": message, "detail": {"reason": reason, **extra}}
    return JSONResponse(status_code=status, content=body)


def create_app(settings: Optional[Settings] = None, models: Optional[Models] = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        if not settings.internal_token:
            logger.warning("AI_INTERNAL_TOKEN 이 비어 있어 /ai/* 요청을 모두 거절합니다")
        app.state.models = models if models is not None else await run_in_threadpool(load_models, settings)
        yield
        app.state.models.close()

    app = FastAPI(title="mnm-ai 내부 API", version="0.1.0", lifespan=lifespan,
                  docs_url="/docs", redoc_url=None)
    app.state.settings = settings

    # ------------------------------------------------------------------ 공통
    def require_token(request: Request) -> None:
        got = request.headers.get("x-internal-token", "")
        if not settings.internal_token or not hmac.compare_digest(got, settings.internal_token):
            feature = PATH_FEATURE.get(request.url.path, "route")
            raise _Unauthorized(feature)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        feature = PATH_FEATURE.get(request.url.path, "route")
        # 입력값은 개인정보가 있을 수 있어 넣지 않는다 (위치·종류만)
        errors = [{"loc": list(e.get("loc", [])), "type": e.get("type")} for e in exc.errors()]
        return _error(feature, 422, "bad_request", "요청 형식 오류", errors=errors)

    @app.exception_handler(_Unauthorized)
    async def _unauthorized(request: Request, exc: "_Unauthorized"):
        return _error(exc.feature, 401, "unauthorized", "내부 토큰이 없거나 맞지 않음")

    async def run(request: Request, feature: str, model_version: str, input_kind: str,
                  fn: Callable[[], Any], summarize_fn: Callable[[Any], str] = lambda r: "ok") -> JSONResponse:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]

        def sink(log: AiCallLog) -> None:
            call_logger.info(json.dumps({**log.model_dump(), "request_id": rid}, ensure_ascii=False))

        limit = settings.timeout_for(feature)
        try:
            result, _ = await asyncio.wait_for(
                run_in_threadpool(call_ai, feature, model_version, input_kind, fn, summarize_fn, sink), timeout=limit)
        except asyncio.TimeoutError:
            # 스레드 안의 작업은 멈추지 않고 끝까지 돈다. 응답만 먼저 돌려준다
            call_logger.info(json.dumps({"feature": feature, "model_version": model_version,
                                         "input_kind": input_kind, "result_summary": "error=server_timeout",
                                         "latency_ms": int(limit * 1000), "error": FEATURE_ERROR[feature],
                                         "request_id": rid}, ensure_ascii=False))
            return _error(feature, 504, "server_timeout", "시간 초과", limit_sec=limit)
        headers = {"X-Request-Id": rid}
        if isinstance(result, AiErrorResponse):
            status = status_for(feature, result.code, result.detail)
            return JSONResponse(status_code=status, content=result.model_dump(), headers=headers)
        return JSONResponse(status_code=200, content=result.model_dump(mode="json"), headers=headers)

    def m(request: Request) -> Models:
        return request.app.state.models

    def fetch_or_raise(models: Models, feature: str, url: str, max_bytes: int) -> bytes:
        try:
            return fetch(models.http, url, max_bytes, settings.fetch_allowed_hosts, settings.timeout_for(feature))
        except FetchError as e:
            raise ServingError(UNEXPECTED_ERROR[feature], e.reason, str(e)) from e

    def need(models: Models, obj: Any, feature: str, what: str) -> Any:
        if obj is None:
            extra = {"load_error": models.load_errors[feature]} if feature in models.load_errors else {}
            raise ServingError(UNEXPECTED_ERROR[feature], "not_loaded", f"{what}이 서버에 올라와 있지 않음", **extra)
        return obj

    # ------------------------------------------------------------------ 주소
    @app.get("/health", tags=["운영"])
    async def health(request: Request):
        models: Models = m(request)
        return {"status": "ok", "models": models.versions(), "unavailable": models.unavailable()}

    auth = [Depends(require_token)]

    @app.post("/ai/stt", dependencies=auth, tags=["AI 내부"])
    async def stt(request: Request, file: UploadFile = File(...)):
        models = m(request)
        data = await file.read(MAX_AUDIO_BYTES + 1)
        suffix = Path(file.filename or "").suffix.lower() or ".webm"

        def fn():
            if len(data) > MAX_AUDIO_BYTES:
                raise ServingError("STT-FAIL", "too_large", "음성 파일이 25MB를 넘음")
            path = to_tempfile(data, suffix)
            try:
                return transcribe(path, client=models.openai)
            finally:
                path.unlink(missing_ok=True)

        return await run(request, "stt", stt_mod.MODEL_VERSION, file.content_type or "audio", fn,
                         lambda r: f"duration_sec={r.duration_sec}")

    @app.post("/ai/route", dependencies=auth, tags=["AI 내부"])
    async def route_(request: Request, body: RouteRequest):
        models = m(request)
        return await run(request, "route", router_mod.MODEL_VERSION, "text",
                         lambda: route(body, client=models.openai),
                         lambda r: f"intent={r.intent} items={len(r.items)} missing={len(r.missing)}")

    @app.post("/ai/part-check", dependencies=auth, tags=["AI 내부"])
    async def part_check(request: Request, body: PartCheckRequest):
        models = m(request)

        def fn():
            embedder = need(models, models.embedder, "part_check", "부위 확인 모델")
            data = fetch_or_raise(models, "part_check", body.image_url, settings.max_image_bytes)
            with models.part_lock:
                return check_part(data, body.requested_part, embedder, vision=models.vision)

        return await run(request, "part_check", part_mod.MODEL_VERSION, "image", fn,
                         lambda r: f"match={r.match} fallback={r.fallback}")

    @app.post("/ai/behavior", dependencies=auth, tags=["AI 내부"])
    async def behavior(request: Request, body: BehaviorRequest):
        models = m(request)

        def fn():
            detector = need(models, models.detector, "behavior", "행동 인식 모델")
            data = fetch_or_raise(models, "behavior", body.video_url, settings.max_video_bytes)
            path = to_tempfile(data, suffix_of(body.video_url, ".mp4"))
            try:
                with models.behavior_lock:
                    return analyze_video(str(path), detector)
            finally:
                path.unlink(missing_ok=True)

        return await run(request, "behavior", behavior_mod.MODEL_VERSION, "video", fn,
                         lambda r: f"main={r.main_label} frames={len(r.frames)}")

    @app.post("/ai/guide", dependencies=auth, tags=["AI 내부"])
    async def guide_(request: Request, body: GuideRequest):
        models = m(request)
        return await run(request, "guide", guide_mod.MODEL_VERSION, "text",
                         lambda: guide(body, models.kb, client=models.openai, index=models.guide_index),
                         lambda r: f"rules={len(r.rule_ids)}")

    @app.post("/ai/summary", dependencies=auth, tags=["AI 내부"])
    async def summary(request: Request, body: SummaryRequest):
        return await run(request, "summary", summary_mod.MODEL_VERSION, "json", lambda: summarize(body),
                         lambda r: f"missing={len(r.missing)}")

    return app


class _Unauthorized(Exception):
    def __init__(self, feature: str):
        self.feature = feature


app = create_app()
