"""테스트용 지식베이스는 가상 문구다. 실제 문구는 mnm-data 게시 조항만 쓴다."""
from types import SimpleNamespace

import numpy as np
import pytest

from guide_rag import EmbeddingIndex, GuideNone, GuideRequest, guide

KB = [
    dict(rule_id="action.fmd.1", kind="action", disease="FMD", stage="after_report", order=1, text="조항 원문 A1",
         doc="문서 A", clause="제1조", revised_at="2026-01-02", status="게시"),
    dict(rule_id="action.fmd.2", kind="action", disease="FMD", stage="after_report", order=2, text="조항 원문 A2",
         doc="문서 A", clause="제2조", revised_at="2026-01-02", status="게시"),
    dict(rule_id="action.fmd.3", kind="action", disease="FMD", stage="after_report", order=3, text="조항 원문 A3 (검수 중)",
         doc="문서 A", clause="제3조", revised_at="2026-01-02", status="검수 중"),
    dict(rule_id="action.lsd.1", kind="action", disease="LSD", stage="after_report", order=1, text="조항 원문 L1",
         doc="문서 L", clause="제1조", revised_at="2026-03-04", status="게시"),
    dict(rule_id="common.1", kind="summary", disease=None, stage=None, order=1, text="조항 원문 C1 소독",
         doc="문서 C", clause="제9조", revised_at="2025-12-01", status="게시"),
]


class FakeClient:
    """embeddings.create: 텍스트 해시 기반 고정 벡터 / responses.parse: 정해 둔 ID 반환."""

    def __init__(self, pick=(), exc=None):
        self.pick, self.exc, self.embed_calls, self.parse_kwargs = list(pick), exc, 0, None
        self.embeddings = SimpleNamespace(create=self._embed)
        self.responses = SimpleNamespace(parse=self._parse)

    def with_options(self, **_):
        return self

    def _embed(self, model, input):
        self.embed_calls += 1
        def vec(t):
            return np.random.default_rng(abs(hash(t)) % 2**32).normal(size=8).tolist()

        return SimpleNamespace(data=[SimpleNamespace(embedding=vec(t)) for t in input])

    def _parse(self, **kw):
        self.parse_kwargs = kw
        if self.exc:
            raise self.exc
        return SimpleNamespace(status="completed", output_parsed=kw["text_format"](rule_ids=self.pick))


def test_lookup_without_query_is_verbatim_and_published_only():
    out = guide(GuideRequest(disease="FMD", stage="after_report"), KB)
    assert out.rule_ids == ["action.fmd.1", "action.fmd.2"]
    assert out.texts == ["조항 원문 A1", "조항 원문 A2"]
    assert out.citations[0].doc == "문서 A" and str(out.citations[0].revised_at) == "2026-01-02"
    assert out.fallback is False


def test_lookup_none_is_rag_none():
    with pytest.raises(GuideNone) as e:
        guide(GuideRequest(disease="LSD", stage="question"), KB)
    r = e.value.to_response()
    assert r.code == "RAG-NONE" and r.detail["fallback"] is True


def test_query_llm_picks_ids_texts_from_kb():
    c = FakeClient(pick=["common.1", "action.fmd.2", "common.1"])
    out = guide(GuideRequest(disease="FMD", stage="after_report", query="소독 어떻게 해요? 010-1111-2222"), KB, c)
    assert out.rule_ids == ["common.1", "action.fmd.2"]
    assert out.texts == ["조항 원문 C1 소독", "조항 원문 A2"]
    assert "010-1111-2222" not in c.parse_kwargs["input"]


def test_query_candidates_exclude_unpublished_and_other_disease():
    c = FakeClient(pick=[])
    with pytest.raises(GuideNone):
        guide(GuideRequest(disease="FMD", stage="after_report", query="질문"), KB, c)
    enum = c.parse_kwargs["text_format"].model_json_schema()["properties"]["rule_ids"]["items"]["enum"]
    assert set(enum) == {"action.fmd.1", "action.fmd.2", "common.1"}


def test_llm_cannot_return_unknown_id():
    c = FakeClient(pick=["action.lsd.1"])
    with pytest.raises(Exception):
        guide(GuideRequest(disease="FMD", stage="after_report", query="질문"), KB, c)


def test_api_failure_is_rag_none():
    with pytest.raises(GuideNone) as e:
        guide(GuideRequest(disease="FMD", stage="after_report", query="질문"), KB, FakeClient(exc=TimeoutError()))
    assert e.value.detail["reason"] == "api"


def test_index_cache_reembeds_only_changed(tmp_path):
    c = FakeClient(pick=["common.1"])
    path = tmp_path / "idx.json"
    guide(GuideRequest(disease="FMD", stage="after_report", query="질문"), KB, c, EmbeddingIndex(c, cache_path=path))
    first = c.embed_calls  # 조항 1번 + 질문 1번
    kb2 = [dict(r) for r in KB]
    kb2[0]["text"], kb2[0]["revised_at"] = "조항 원문 A1 개정", "2026-06-01"
    idx = EmbeddingIndex(c, cache_path=path)
    idx.ensure([r for r in __import__("guide_rag").published(kb2)])
    assert c.embed_calls == first + 1 and len(idx.vecs) == 5


def test_db_row_shape_accepted():
    """mnm-data biosecurity_rule 행 그대로 (컬럼 이름·대문자 질병 코드·빈 stage·추가 컬럼)."""
    from guide_rag.schema import KbRule

    r = KbRule.model_validate(dict(
        biosecurity_rule_id=301, kind="clause", disease="FMD", stage="", display_order=None,
        text="원문", source_doc="구제역 긴급행동지침(SOP)", clause="제4장 4", revised_at="2026-05-06",
        status="published", part_code=None, embedding=None, reviewer=None, version=1,
    ))
    assert (r.rule_id, r.disease, r.stage, r.order, r.doc) == ("301", "FMD", None, 0, "구제역 긴급행동지침(SOP)")


def test_query_excludes_question_and_form_rows():
    rows = [dict(rule_id="q1", kind="question", disease="FMD", stage="suspect", text="침을 흘리나요?",
                 doc="d", clause="c", revised_at="2026-05-06", status="published")]
    with pytest.raises(GuideNone):
        guide(GuideRequest(disease="FMD", stage="suspect", query="침"), rows, client=object())


def test_stored_db_embedding_used_without_api_call():
    from guide_rag.schema import KbRule

    r = KbRule.model_validate(dict(rule_id="e1", kind="clause", disease="FMD", text="원문", doc="d", clause="c",
                                   revised_at="2026-05-06", status="published", embedding=[1.0, 0.0]))

    class NoCall:
        def with_options(self, **_):
            raise AssertionError("저장된 임베딩이 있으면 API 를 부르지 않아야 함")

    idx = EmbeddingIndex(NoCall())
    idx.ensure([r])
    assert len(idx.vecs) == 1


def test_build_embeddings_sql():
    from guide_rag.build_embeddings import DIM, build_sql

    class C:
        class embeddings:
            @staticmethod
            def create(model, input):
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * DIM) for _ in input])

    sql = build_sql([{"biosecurity_rule_id": "7", "text": "원문"}], C())
    assert "WHERE biosecurity_rule_id = 7;" in sql and sql.startswith("BEGIN;")
