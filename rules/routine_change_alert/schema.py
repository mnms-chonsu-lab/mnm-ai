"""평소와 다름 · 미응답 입출력 형식 (명세 REQ-AI-07)."""

from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiaryEntry(_Strict):
    cow_tag4: Optional[str] = Field(description="귀표 끝 4자리")
    category: str = Field(description="일지 분류 (사육현황·접종·소독·관찰·출입·출생·폐사·이동·메모)")
    value: str = Field(description="기록 내용")
    date: dt.date


class RoutineChange(_Strict):
    """알림 문구에 쓸 값. '이상·질병' 같은 판정 표현은 만들지 않는다."""

    cow_tag4: str
    observation: str = Field(description="관찰 내용 (공백 정리)")
    dates: list[dt.date] = Field(description="기록 날짜들. 같은 날 2번 기록하면 2번 들어감")


class AlertState(_Strict):
    sent_at: dt.datetime
    viewed_at: Optional[dt.datetime] = None
    feedback: Optional[Literal["확인함", "문제 없음"]] = None
