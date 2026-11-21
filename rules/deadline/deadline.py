"""계산식 ① 신고 기한 · 귀표 기한 (REQ-AI-04). LLM 사용 금지.

규칙
- 신고 기한: 발생일 다음 날부터 세어 5번째 '셈하는 날'. 토요일·일요일·공휴일은 세지 않는다.
  (발생 당일은 세지 않음 = 초일 불산입. 일요일은 관공서 공휴일이므로 제외)
- 귀표 기한(출생만): 출생 신고일 + 30일 (달력 기준). 신고일을 모르면 오늘 신고한다고 보고 계산.
- D-day: 신고 기한 − 오늘 (지나면 음수)

공휴일 목록은 인교님 테이블(REQ-DATA-07)을 holidays 인자로 받는다.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Optional

from .schema import DeadlineResponse

REPORT_DAYS = 5
EAR_TAG_DAYS = 30


def is_counted_day(d: date, holidays: set[date]) -> bool:
    return d.weekday() < 5 and d not in holidays  # 월(0)~금(4), 공휴일 아님


def add_counted_days(start: date, n: int, holidays: set[date]) -> date:
    d = start
    counted = 0
    while counted < n:
        d += timedelta(days=1)
        if is_counted_day(d, holidays):
            counted += 1
    return d


def compute_deadline(
    event_type: str,
    event_date: date,
    today: date,
    holidays: Iterable[date] = (),
    reported_date: Optional[date] = None,
) -> DeadlineResponse:
    if event_type not in ("birth", "death", "move"):
        raise ValueError(f"알 수 없는 event_type: {event_type}")
    if event_date > today:
        raise ValueError("발생일이 오늘보다 뒤일 수 없습니다")

    hol = set(holidays)
    report_due = add_counted_days(event_date, REPORT_DAYS, hol)
    ear_tag_due = None
    if event_type == "birth":
        ear_tag_due = (reported_date or today) + timedelta(days=EAR_TAG_DAYS)

    return DeadlineResponse(
        report_due=report_due,
        ear_tag_due=ear_tag_due,
        d_day=(report_due - today).days,
    )
