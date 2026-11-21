"""관할 농가 연락 우선순위 입출력 형식 (명세 REQ-AI-10)."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FarmIndicator(_Strict):
    farm_id: str
    unresponded: bool = Field(description="알림 미응답 (48시간)")
    days_left: Optional[int] = Field(default=None, description="가장 가까운 신고 기한까지 남은 일수, 없으면 null")
    head_count: int = Field(ge=0, description="사육 두수")
    last_record_date: Optional[date] = None


class PrioritizedFarm(_Strict):
    farm_id: str
    rank: int
    reasons: list[Literal["noresp", "due", "small"]] = Field(
        description="noresp=미응답, due=기한 7일 이내, small=50두 미만"
    )
