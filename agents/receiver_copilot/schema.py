"""접수자용 한 장 요약 · 신고서 6개 항목 입출력 형식 (명세 REQ-AI-09). mnm-dev 는 이 형식을 따른다."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from rules.disease_branch.schema import MatchedSymptoms, SymptomAnswers
from rules.report_state_machine.schema import Form6

Part = Literal["mouth", "udder", "hoof", "flank"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 입력: 서버가 승인된 소 상태 기록 ID 로 조회해 넘기는 구조화 데이터
# ---------------------------------------------------------------------------
class CowInfo(_Strict):
    tag4: str
    desc: Optional[str] = None
    barn: Optional[str] = None


class DiaryRecord(_Strict):
    cow_tag4: Optional[str] = Field(description="개체 기록이 아니면 null (예: 축사 소독)")
    category: Literal["사육현황", "접종", "소독", "관찰", "출입", "출생", "폐사", "이동", "신고", "메모"] = Field(
        description="mnm-data diary_entry.category. '신고'는 서버 자동 기록"
    )
    value: str
    date: dt.date


class PhotoResult(_Strict):
    part: Part = Field(description="요청 부위")
    match: bool = Field(description="/ai/part-check 결과")


class SummaryRequest(_Strict):
    record_id: str = Field(description="승인된 소 상태 기록 ID")
    found_date: dt.date = Field(description="발견일 (관련 일지 기준일)")
    cow: CowInfo
    symptom_answers: SymptomAnswers
    photos: list[PhotoResult] = Field(default_factory=list)
    diary: list[DiaryRecord] = Field(default_factory=list, description="농장 일지 (최근 14일 이상)")
    report_fields: dict = Field(description="의심 신고 상태 머신(rules/report_state_machine)의 fields, 빈칸 없음")
    farm_total_head: Optional[int] = Field(default=None, ge=1, description="농장 정보의 전체 사육 두수. 있으면 신고서 2호에 괄호로 붙임")


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
class LatestRecord(_Strict):
    date: dt.date
    value: str
    days_ago: int = Field(ge=0, description="found_date 기준 며칠 전")


class Card(_Strict):
    cow: CowInfo
    matched_symptoms: list[str] = Field(description='"예"로 답한 공식 증상 이름 (판정 아님)')
    diary_related: list[DiaryRecord] = Field(description="같은 개체, found_date 포함 최근 7일, 최근순")
    photos: list[PhotoResult]
    vaccine: Optional[LatestRecord] = Field(description="가장 최근 접종 기록")
    disinfection: Optional[LatestRecord] = Field(description="가장 최근 소독 기록")
    visit: Optional[LatestRecord] = Field(description="일지 분류 '출입' 중 가장 최근 1건")


class SummaryResponse(_Strict):
    card: Card
    form6: Form6
    form6_text: str = Field(description="인터넷 신고서 복사 텍스트 (F-S14·R-P02 '객체 + 텍스트')")


class SummaryErrorResponse(_Strict):
    code: Literal["LLM-FAIL"]
    message: str
    detail: Optional[dict] = None


def export(out: Path) -> None:
    doc = {
        "endpoint": "/ai/summary",
        "request": SummaryRequest.model_json_schema(),
        "response": SummaryResponse.model_json_schema(),
        "error": SummaryErrorResponse.model_json_schema(),
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["SummaryRequest", "SummaryResponse", "Card", "MatchedSymptoms"]
