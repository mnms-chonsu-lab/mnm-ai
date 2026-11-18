"""방역 안내 (RAG) 입출력 형식 (명세 REQ-AI-08). mnm-dev 는 이 형식을 따른다."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

Disease = Literal["FMD", "LSD"]  # mnm-data biosecurity_rule.disease · rules/disease_branch 와 같은 코드


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KbRule(BaseModel):
    """지식베이스 조항 1개 (mnm-data `biosecurity_rule`, 게시된 것만).

    DB 컬럼 이름(`biosecurity_rule_id`, `source_doc`, `display_order`)과 질병 코드 대문자(`FMD`, `LSD`)를
    그대로 받는다. 쓰지 않는 컬럼(`part_code`, `embedding` 등)은 무시한다.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    rule_id: str = Field(validation_alias=AliasChoices("rule_id", "biosecurity_rule_id"))
    kind: str = Field(description="question · photo_guide · action · form_field · summary · clause")
    disease: Optional[Disease] = Field(default=None, description="질병 공통이면 null. 소문자로 와도 대문자로 바꿔 받음")
    stage: Optional[str] = Field(default=None, description="suspect · after_report · regional. 단계 공통이면 null")
    order: int = Field(default=0, validation_alias=AliasChoices("order", "display_order"), description="같은 묶음 안 순서")
    text: str = Field(description="조항 원문 그대로 (화면에 이 문구가 나감)")
    doc: str = Field(validation_alias=AliasChoices("doc", "source_doc"), description="근거 문서명")
    clause: str = Field(description="조항")
    revised_at: dt.date = Field(description="개정일")
    status: str = Field(description="draft · review · published")
    embedding: Optional[list[float]] = Field(
        default=None, description="biosecurity_rule.embedding (vector(1536)). 있으면 다시 계산하지 않음"
    )

    @field_validator("rule_id", mode="before")
    @classmethod
    def _id_to_str(cls, v):
        return str(v) if isinstance(v, int) else v

    @field_validator("disease", "stage", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v.upper() if v.upper() in ("FMD", "LSD") else (v or None)
        return v

    @field_validator("order", mode="before")
    @classmethod
    def _order_default(cls, v):
        return 0 if v in (None, "") else v


class GuideRequest(_Strict):
    disease: Optional[Disease] = None

    @field_validator("disease", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.upper() if isinstance(v, str) else v

    stage: str
    query: Optional[str] = Field(default=None, description="농가·접수자 질문. 없으면 단계별 조항 목록")


class Citation(_Strict):
    doc: str
    clause: str
    revised_at: dt.date


class GuideResponse(_Strict):
    rule_ids: list[str]
    texts: list[str] = Field(description="조항 원문 그대로. rule_ids 와 같은 순서")
    citations: list[Citation] = Field(description="rule_ids 와 같은 순서")
    fallback: bool = Field(description="근거가 없으면 true. 이때는 RAG-NONE 오류로 반환")


class GuideErrorResponse(_Strict):
    code: Literal["RAG-NONE", "LLM-FAIL"] = Field(
        description="RAG-NONE: 근거 조항 없음 (404). LLM-FAIL: 검색·선택 호출 실패 (503), 시간 초과 (504)"
    )
    message: str
    detail: Optional[dict] = None


def export(out: Path) -> None:
    doc = {
        "endpoint": "/ai/guide",
        "request": GuideRequest.model_json_schema(),
        "response": GuideResponse.model_json_schema(),
        "error": GuideErrorResponse.model_json_schema(),
        "kb_rule": KbRule.model_json_schema(),
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
