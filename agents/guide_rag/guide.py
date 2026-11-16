"""방역 안내 (명세 REQ-AI-08): 근거 조항 선택.

- 게시된 지식베이스 조항만 쓴다.
- 화면 문구는 조항 원문 그대로. LLM 은 후보 조항 ID 중에서 고르기만 하고 문장을 만들지 않는다.
- 근거가 없으면 RAG-NONE (fallback = true).

질문(query)이 없으면: 질병·단계가 같은 조항을 순서대로 돌려준다 (LLM 미사용).
질문이 있으면: 임베딩으로 후보 TOP_K 개를 찾고, LLM 이 그중 근거가 되는 조항 ID 만 고른다.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model

from common.ai_call import mask_phone

from .schema import Citation, GuideErrorResponse, GuideRequest, GuideResponse, KbRule

PUBLISHED = {"게시", "published"}
# 질문 검색 대상. question·photo_guide·form_field 는 앱 질문·촬영 안내·신고서 칸 문구라 답이 될 수 없어 제외
# (제안, PM 확인 필요)
SEARCH_KINDS = {"clause", "action", "summary"}
EMBED_MODEL = os.getenv("GUIDE_EMBED_MODEL", "text-embedding-3-small")
SELECT_MODEL = os.getenv("GUIDE_MODEL", os.getenv("ROUTER_MODEL", "gpt-5.4-mini"))
TOP_K = 5  # 임베딩 후보 수 (제안값, 실측으로 조정)
TIMEOUT_SEC = 8.0  # 명세 REQ-AI-12 일반 타임아웃
PROMPT_VERSION = "guide-v1"
MODEL_VERSION = f"{EMBED_MODEL}+{SELECT_MODEL}+{PROMPT_VERSION}"  # 호출 로그(common)용


class GuideNone(Exception):
    """RAG-NONE. 근거 조항 없음 또는 검색·선택 실패."""

    code = "RAG-NONE"

    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.detail = {"fallback": True, **(detail or {})}

    def to_response(self) -> GuideErrorResponse:
        return GuideErrorResponse(code="RAG-NONE", message=self.message, detail=self.detail)


# ---------------------------------------------------------------------------
# 지식베이스
# ---------------------------------------------------------------------------
def published(rules: Iterable[KbRule | dict]) -> list[KbRule]:
    out = [r if isinstance(r, KbRule) else KbRule.model_validate(r) for r in rules]
    return [r for r in out if r.status in PUBLISHED]


def load_kb(path: str | Path) -> list[KbRule]:
    """JSON 배열 또는 JSONL 파일 → 게시된 조항."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    rows = json.loads(raw) if raw.startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    return published(rows)


def _scope(rules: Sequence[KbRule], disease: Optional[str], stage: str, exact: bool) -> list[KbRule]:
    """exact=True: 질병·단계가 정확히 같은 조항. False: 공통(null) 조항도 포함."""
    if exact:
        return [r for r in rules if r.disease == disease and r.stage == stage]
    return [r for r in rules if r.disease in (None, disease) and r.stage in (None, stage)]


def _respond(picked: Sequence[KbRule]) -> GuideResponse:
    return GuideResponse(
        rule_ids=[r.rule_id for r in picked],
        texts=[r.text for r in picked],
        citations=[Citation(doc=r.doc, clause=r.clause, revised_at=r.revised_at) for r in picked],
        fallback=False,
    )


# ---------------------------------------------------------------------------
# 임베딩 인덱스
# ---------------------------------------------------------------------------
def _key(r: KbRule) -> str:
    return hashlib.sha256(f"{r.rule_id}|{r.revised_at}|{r.text}".encode()).hexdigest()[:16]


class EmbeddingIndex:
    """조항 원문 임베딩. 조항이 개정되면(revised_at·text 변경) 그 조항만 다시 계산한다."""

    def __init__(self, client, model: str = EMBED_MODEL, cache_path: Optional[Path] = None):
        self.client, self.model, self.cache_path = client, model, cache_path
        self.vecs: dict[str, np.ndarray] = {}
        if cache_path and cache_path.exists():
            data = json.loads(cache_path.read_text())
            if data.get("model") == model:
                self.vecs = {k: np.asarray(v, dtype=np.float32) for k, v in data["vecs"].items()}

    def _embed(self, texts: list[str]) -> np.ndarray:
        res = self.client.with_options(timeout=TIMEOUT_SEC, max_retries=0).embeddings.create(model=self.model, input=texts)
        m = np.asarray([d.embedding for d in res.data], dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def ensure(self, rules: Sequence[KbRule]) -> None:
        for r in rules:  # DB 에 저장된 임베딩이 있으면 그대로 사용
            if r.embedding is not None and _key(r) not in self.vecs:
                v = np.asarray(r.embedding, dtype=np.float32)
                self.vecs[_key(r)] = v / np.linalg.norm(v)
        todo = [r for r in rules if _key(r) not in self.vecs]
        if not todo:
            return
        for r, v in zip(todo, self._embed([r.text for r in todo])):
            self.vecs[_key(r)] = v
        if self.cache_path:
            self.cache_path.write_text(json.dumps({"model": self.model, "vecs": {k: v.tolist() for k, v in self.vecs.items()}}))

    def top_k(self, query: str, rules: Sequence[KbRule], k: int = TOP_K) -> list[KbRule]:
        self.ensure(rules)
        q = self._embed([query])[0]
        scores = np.asarray([self.vecs[_key(r)] @ q for r in rules])
        return [rules[i] for i in np.argsort(-scores)[:k]]


# ---------------------------------------------------------------------------
# LLM 선택 (ID 만)
# ---------------------------------------------------------------------------
SELECT_INSTRUCTIONS = """너는 방역 안내 조항을 고르는 역할만 한다.
질문에 직접 근거가 되는 조항의 rule_id 만 고른다. 새 문장을 쓰지 않는다.
근거가 되는 조항이 없으면 빈 목록을 돌려준다.
질병을 판정하거나 추측하지 않는다."""


def _selection_model(ids: Sequence[str]) -> type[BaseModel]:
    """후보 ID 만 고를 수 있는 형식 (strict enum)."""
    Id = Literal[tuple(ids)]  # type: ignore[valid-type]
    return create_model("GuideSelection", __config__=ConfigDict(extra="forbid"), rule_ids=(list[Id], Field(...)))


def select(query: str, candidates: Sequence[KbRule], client, model: str = SELECT_MODEL) -> list[KbRule]:
    Sel = _selection_model([r.rule_id for r in candidates])
    listing = "\n".join(f"[{r.rule_id}] ({r.doc} {r.clause}) {r.text}" for r in candidates)
    t0 = time.perf_counter()
    try:
        resp = client.with_options(timeout=TIMEOUT_SEC, max_retries=0).responses.parse(
            model=model,
            instructions=SELECT_INSTRUCTIONS,
            input=f"질문: {mask_phone(query)}\n\n후보 조항:\n{listing}",
            text_format=Sel,
            store=False,
        )
    except Exception as e:
        raise GuideNone("조항 선택 호출 실패", {"reason": "api", "type": type(e).__name__}) from e
    if time.perf_counter() - t0 > TIMEOUT_SEC:
        raise GuideNone("시간 초과", {"reason": "timeout"})
    out = getattr(resp, "output_parsed", None)
    if getattr(resp, "status", "completed") != "completed" or out is None:
        raise GuideNone("조항 선택 결과 없음", {"reason": "no_parse"})
    by_id = {r.rule_id: r for r in candidates}
    seen, picked = set(), []
    for i in out.rule_ids:  # 중복 제거, 모델이 고른 순서 유지
        if i not in seen and i in by_id:
            seen.add(i)
            picked.append(by_id[i])
    return picked


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def guide(req: GuideRequest, rules: Sequence[KbRule | dict], client=None, index: Optional[EmbeddingIndex] = None) -> GuideResponse:
    kb = published(rules)
    if not req.query:
        picked = sorted(_scope(kb, req.disease, req.stage, exact=True), key=lambda r: (r.order, r.rule_id))
        if not picked:
            raise GuideNone("해당 질병·단계의 게시된 조항 없음", {"reason": "no_rule", "disease": req.disease, "stage": req.stage})
        return _respond(picked)

    pool = [r for r in _scope(kb, req.disease, req.stage, exact=False) if r.kind in SEARCH_KINDS]
    if not pool:
        raise GuideNone("검색할 게시된 조항 없음", {"reason": "no_rule"})
    if client is None:
        client = make_client()
    index = index or EmbeddingIndex(client)
    try:
        candidates = index.top_k(mask_phone(req.query), pool)
    except Exception as e:
        raise GuideNone("검색 실패", {"reason": "embed", "type": type(e).__name__}) from e
    picked = select(req.query, candidates, client)
    if not picked:
        raise GuideNone("질문에 맞는 근거 조항 없음", {"reason": "no_evidence"})
    return _respond(picked)


def make_client():
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise GuideNone("OPENAI_API_KEY 가 설정되지 않음", {"reason": "config"})
    return OpenAI()
