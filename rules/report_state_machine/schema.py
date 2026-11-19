"""신고서 상태 머신 입출력 형식 (명세 REQ-AI-05)."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportStateRequest(_Strict):
    report_kind: Literal["birth", "death", "sell", "buy", "move", "suspect"] = Field(
        description="birth=출생, death=폐사, sell=양도, buy=양수, move=이동, suspect=전염병 의심 신고"
    )
    fields: dict = Field(description="지금까지 채워진 항목. 정의되지 않은 키는 거부")


class ReportStateResponse(_Strict):
    fields: dict = Field(description="검사를 통과한 값. 형식이 틀린 값은 null")
    missing: list[str] = Field(description="비어 있는 필수 항목 (질문 순서)")
    next_question: Optional[str] = Field(description="빈칸 중 첫 번째 하나의 질문 (해요체). 다 차면 null")
    can_submit: bool = Field(description="필수 항목이 모두 차면 true")


class Form6(_Strict):
    """가축전염병 예방법 시행규칙 제13조 제1항 각 호."""

    owner_and_location: str = Field(description="1. 소유자 성명 및 사육 장소")
    species_and_head_count: str = Field(description="2. 가축의 종류 및 두수")
    condition: str = Field(description='3. 질병명 칸: 항상 "발견 당시 상태: …" (추정 병명 금지)')
    found_date: date = Field(description="4. 폐사일 또는 발견일")
    reporter: str = Field(description="5. 신고자 성명 및 주소")
    other: str = Field(default="", description="6. 그 밖의 사항")

    @field_validator("condition")
    @classmethod
    def _condition_prefix(cls, v: str) -> str:
        if not v.startswith("발견 당시 상태: "):
            raise ValueError('condition은 "발견 당시 상태: "로 시작해야 합니다')
        return v
