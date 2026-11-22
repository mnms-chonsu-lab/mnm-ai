"""서버가 켜질 때 한 번만 불러오는 모델·클라이언트.

불러오기에 실패한 기능은 서버를 멈추지 않고 비워 둔다. 그 기능 주소는 503 `not_loaded` 를 돌려주고,
`/health` 의 `unavailable` 에 이유가 나온다 (가중치·키를 하나씩 붙여 가며 스테이징에 올리기 위해).
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import Settings

logger = logging.getLogger("mnm_ai.serving")


@dataclass
class Models:
    openai: Any = None          # STT·라우터·방역 안내·Vision 대체 경로가 같이 쓰는 OpenAI 클라이언트
    embedder: Any = None        # 부위 확인 SigLIP 2
    vision: Any = None          # 부위 확인 대체 경로 (PART_FALLBACK_VISION=1 일 때만)
    detector: Any = None        # 행동 인식 YOLO11s
    kb: list = field(default_factory=list)  # 방역 안내 게시 조항
    guide_index: Any = None     # 조항 임베딩 캐시
    http: httpx.Client = field(default_factory=lambda: httpx.Client())
    load_errors: dict[str, str] = field(default_factory=dict)
    # CPU 2코어 서버라 무거운 추론은 한 번에 하나씩 (제안값, 실측으로 조정)
    part_lock: threading.Lock = field(default_factory=threading.Lock)
    behavior_lock: threading.Lock = field(default_factory=threading.Lock)

    def versions(self) -> dict[str, str]:
        from behavior_yolo11s import behavior as behavior_mod
        from farm_router import router as router_mod
        from farm_router import stt as stt_mod
        guide_mod = importlib.import_module("guide_rag.guide")
        from part_check_siglip2 import part_check as part_mod
        from receiver_copilot import summary as summary_mod

        out = {"summary": summary_mod.MODEL_VERSION}
        if self.openai is not None:
            out.update(stt=stt_mod.MODEL_VERSION, route=router_mod.MODEL_VERSION, guide=guide_mod.MODEL_VERSION)
        if self.embedder is not None:
            out["part_check"] = part_mod.MODEL_VERSION + ("+vision" if self.vision is not None else "")
        if self.detector is not None:
            out["behavior"] = behavior_mod.MODEL_VERSION
        return out

    def unavailable(self) -> dict[str, str]:
        return dict(self.load_errors)

    def close(self) -> None:
        self.http.close()


def _try(models: Models, feature: str, fn) -> Optional[Any]:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — 한 기능이 실패해도 서버는 켠다
        models.load_errors[feature] = f"{type(e).__name__}: {e}"[:200]
        logger.warning("모델 불러오기 실패 (%s): %s", feature, models.load_errors[feature])
        return None


def load_models(settings: Settings) -> Models:
    from behavior_yolo11s import load_detector
    from guide_rag import EmbeddingIndex, load_kb
    from part_check_siglip2 import load_embedder
    from part_check_siglip2.part_check import load_vision

    models = Models()

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        models.openai = _try(models, "openai", OpenAI)
    else:
        models.load_errors["openai"] = "OPENAI_API_KEY 없음 (stt·route·guide 질문 검색 불가)"

    if settings.load_part_check:
        models.embedder = _try(models, "part_check", load_embedder)
        models.vision = _try(models, "part_check_vision", load_vision)
    else:
        models.load_errors["part_check"] = "AI_LOAD_PART_CHECK=0"

    if settings.load_behavior:
        models.detector = _try(models, "behavior", load_detector)
    else:
        models.load_errors["behavior"] = "AI_LOAD_BEHAVIOR=0"

    if settings.guide_kb_path:
        models.kb = _try(models, "guide", lambda: load_kb(settings.guide_kb_path)) or []
    else:
        models.load_errors["guide"] = "GUIDE_KB_PATH 없음 (지식베이스 미적재 → 503 not_loaded)"
    if models.openai is not None:
        cache = Path(settings.guide_embed_cache) if settings.guide_embed_cache else None
        models.guide_index = _try(models, "guide_index", lambda: EmbeddingIndex(models.openai, cache_path=cache))

    logger.info("모델 준비: %s / 없음: %s", models.versions(), list(models.load_errors))
    return models
