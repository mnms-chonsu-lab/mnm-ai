"""접수자용 한 장 요약 · 신고서 6개 항목 (명세 REQ-AI-09).

모든 값은 서버가 넘긴 구조화 기록에서 규칙으로 채운다. 숫자·날짜를 바꾸지 않는다.
- matched_symptoms : 계산식 ② (rules/disease_branch)
- diary_related    : 같은 개체·최근 7일 (F-S11)
- vaccine 등       : 분류별 가장 최근 기록 (R-S02 "일지에서 분류별 가장 최근 값")
- form6            : 시행규칙 제13조 6개 항목 (rules/report_state_machine), 질병명 칸 "발견 당시 상태: …"
                     빈칸은 실패가 아니라 null + missing (mnm-dev PR #5 합의). 형식이 틀린 값만 LLM-FAIL
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

from pydantic import ValidationError

from rules.disease_branch.symptoms import matched_symptoms
from rules.report_state_machine.report_state import FORM6_LABELS, build_form6

from .schema import Card, DiaryRecord, Form6, LatestRecord, SummaryErrorResponse, SummaryRequest, SummaryResponse

RELATED_DAYS = 7  # 명세 F-S11 "같은 개체 최근 7일"
MODEL_VERSION = "summary-rules-v2"  # 호출 로그(common)용


class SummaryError(Exception):
    code = "LLM-FAIL"

    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_response(self) -> SummaryErrorResponse:
        return SummaryErrorResponse(code="LLM-FAIL", message=self.message, detail=self.detail)


def _latest(rows: Iterable, found: dt.date) -> Optional[LatestRecord]:
    """found_date 이전(당일 포함) 가장 최근 1건."""
    past = [r for r in rows if r.date <= found]
    if not past:
        return None
    r = max(past, key=lambda x: x.date)
    return LatestRecord(date=r.date, value=r.value, days_ago=(found - r.date).days)


def related_diary(diary: list[DiaryRecord], tag4: str, found: dt.date, days: int = RELATED_DAYS) -> list[DiaryRecord]:
    """같은 개체, found_date 포함 최근 days 일 (found-6 ~ found). 최근순, 같은 날은 입력 순서."""
    start = found - dt.timedelta(days=days - 1)
    rows = [r for r in diary if r.cow_tag4 == tag4 and start <= r.date <= found]
    return sorted(rows, key=lambda r: r.date, reverse=True)


def summarize(req: SummaryRequest) -> SummaryResponse:
    found = req.found_date
    card = Card(
        cow=req.cow,
        matched_symptoms=matched_symptoms(req.symptom_answers.model_dump()).items,
        diary_related=related_diary(req.diary, req.cow.tag4, found),
        photos=req.photos,
        vaccine=_latest((r for r in req.diary if r.category == "접종"), found),
        disinfection=_latest((r for r in req.diary if r.category == "소독"), found),
        visit=_latest((r for r in req.diary if r.category == "출입"), found),
    )
    form, missing = draft_form6(req.report_fields, req.farm_total_head)
    return SummaryResponse(card=card, form6=form, missing=missing, form6_text=draft_copy_text(form))


def draft_form6(fields: dict, farm_total_head: Optional[int] = None) -> tuple[Form6, list[str]]:
    """제13조 6개 항목. 칸을 만드는 규칙은 rules 의 build_form6 와 같고, 빈칸은 null + missing 으로 둔다."""
    species, head = fields.get("species"), fields.get("head_count")
    symptoms, found = fields.get("symptoms"), fields.get("found_date")
    try:
        found_date = dt.date.fromisoformat(str(found)) if found else None
    except ValueError as e:
        raise SummaryError("신고서 항목 형식 오류", {"reason": "report_fields", "field": "found_date"}) from e
    d = {
        "owner_and_location": fields.get("owner_and_location") or None,
        "species_and_head_count": (f"{species} {head}두" + (f" (전체 {farm_total_head}두)" if farm_total_head else ""))
        if species and head else None,
        "condition": ("발견 당시 상태: " + ", ".join(symptoms)) if symptoms else None,
        "found_date": found_date,
        "reporter": fields.get("reporter") or None,
        "other": fields.get("other") or "",
    }
    missing = [k for k, v in d.items() if v is None]
    if not missing:  # 빈칸이 없으면 rules 검증(질병명 칸 형식 등)까지 그대로 통과해야 함
        try:
            build_form6(fields, farm_total_head)
        except (ValueError, ValidationError) as e:
            raise SummaryError("신고서 항목 형식 오류", {"reason": "report_fields", "error": str(e)}) from e
    return Form6(**d), missing


def draft_copy_text(form: Form6) -> str:
    """인터넷 신고서에 붙여 넣을 복사 텍스트 (F-S14, R-P02). 빈칸은 '(빈칸)'."""
    d = form.model_dump()
    return "\n".join(f"{label}: {'(빈칸)' if d[key] is None else (d[key] or '-')}" for key, label in FORM6_LABELS)
