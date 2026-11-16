"""증상 일치 수 · 촬영 부위 · 질병 분기 입출력 형식 (명세 REQ-AI-06). 판정 아님."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SymptomAnswers(_Strict):
    """공식 증상 질문 4개에 대한 예(true)/아니요(false)."""

    drool: bool = Field(description="침 흘림")
    blister: bool = Field(description="물집")
    lump: bool = Field(description="혹")
    feed: bool = Field(description="사료 남김")


class MatchedSymptoms(_Strict):
    count: int = Field(description='"예"로 답한 공식 질문 수 (판정 아님)')
    items: list[str] = Field(description="그 질문의 증상 이름")


class RequiredPart(_Strict):
    part: Literal["mouth", "hoof", "flank", "udder"] = Field(
        description="mnm-data biosecurity_rule.part_code. 촬영 안내는 kind=photo_guide, part_code 가 같은 행"
    )


class DiseaseBranch(_Strict):
    disease: Literal["FMD", "LSD"] = Field(description="보여 줄 공식 지침 (mnm-data biosecurity_rule.disease): FMD=구제역, LSD=럼피스킨")
    stage: Literal["after_report"] = Field(
        default="after_report",
        description="조치 문구는 kind=action, disease·stage 가 같은 행을 display_order 순으로 (/ai/guide query 없이 호출)",
    )
