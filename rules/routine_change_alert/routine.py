"""계산식 ⑤ 평소와 다름 · ⑥ 미응답 · 피드백 비율 (REQ-AI-07).

평소와 다름(MVP): 최근 7일(오늘 포함) 안에 같은 개체·같은 관찰 기록이 2회 이상이면
알림 값을 반환한다 (명세 계산식 ⑤). "이상·질병" 같은 판정 표현은 만들지 않는다.
- 기록 수로 센다. 같은 날 2번 기록해도 2회 (dates 에 같은 날짜가 2번 들어감)
- 관찰 내용 비교는 앞뒤·중간 공백을 정리한 문자열 일치
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Mapping, Optional

from .schema import AlertState, RoutineChange

WINDOW_DAYS = 7
MIN_COUNT = 2
NORESP_HOURS = 48


def _norm(s: str) -> str:
    return " ".join(s.split())


def routine_change(
    entries: Iterable[Mapping],
    cow_tag4: str,
    observation: str,
    today: date,
) -> Optional[RoutineChange]:
    """entries: 일지 항목 [{cow_tag4, category, value, date}] — 방금 저장한 항목 포함."""
    start = today - timedelta(days=WINDOW_DAYS - 1)
    target = _norm(observation)
    dates = sorted(
        e["date"]
        for e in entries
        if e.get("category") == "관찰"
        and e.get("cow_tag4") == cow_tag4
        and _norm(str(e.get("value", ""))) == target
        and start <= e["date"] <= today
    )
    if len(dates) < MIN_COUNT:
        return None
    return RoutineChange(cow_tag4=cow_tag4, observation=target, dates=dates)


def scan_routine_changes(entries: Iterable[Mapping], today: date) -> list[RoutineChange]:
    """전체 일지를 훑어 알림 대상(개체·관찰 조합)을 모두 반환."""
    entries = list(entries)
    keys = {
        (e["cow_tag4"], _norm(str(e["value"])))
        for e in entries
        if e.get("category") == "관찰" and e.get("cow_tag4")
    }
    out = []
    for cow, obs in sorted(keys):
        r = routine_change(entries, cow, obs, today)
        if r:
            out.append(r)
    return out


def is_unresponded(alert: AlertState, now: datetime) -> bool:
    """발송 후 48시간 동안 열람·피드백이 없으면 true."""
    if alert.viewed_at is not None or alert.feedback is not None:
        return False
    return now - alert.sent_at >= timedelta(hours=NORESP_HOURS)


def feedback_ratio(alerts: Iterable[AlertState]) -> dict:
    """어드민 A-S07: 농가 피드백(확인함·문제 없음) 비율. 비율의 분모는 피드백을 준 알림 수."""
    alerts = list(alerts)
    responded = [a for a in alerts if a.feedback is not None]
    confirmed = sum(1 for a in responded if a.feedback == "확인함")
    no_issue = sum(1 for a in responded if a.feedback == "문제 없음")
    n = len(responded)
    return {
        "total": len(alerts),
        "responded": n,
        "confirmed": confirmed,
        "no_issue": no_issue,
        "confirmed_ratio": round(confirmed / n, 3) if n else None,
        "no_issue_ratio": round(no_issue / n, 3) if n else None,
    }
