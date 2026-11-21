"""기한 계산 입출력 형식 (명세 REQ-AI-04)."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeadlineRequest(_Strict):
    event_type: Literal["birth", "death", "move"]
    event_date: date = Field(description="출생·폐사·이동한 날")


class DeadlineResponse(_Strict):
    report_due: date = Field(description="신고 기한: 발생일 다음 날부터 토·일·공휴일 빼고 5번째 날")
    ear_tag_due: Optional[date] = Field(default=None, description="귀표 부착 기한 (출생만): 출생 신고일 + 30일")
    d_day: int = Field(description="신고 기한 − 오늘. 지나면 음수")
