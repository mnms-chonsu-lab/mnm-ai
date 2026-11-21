"""REQ-AI-04 완료 기준: 토요일·공휴일 낀 경우 포함 테스트 20건 이상.

공휴일은 테스트용 입력값이다(실제 달력은 인교님 테이블 REQ-DATA-07 사용).
2026-11-12 = 목요일.
"""

from datetime import date as D

import pytest

from deadline import compute_deadline

XMAS = {D(2026, 12, 25), D(2027, 1, 1)}  # 테스트용: 금요일 공휴일 2개
OCT = {D(2026, 10, 5), D(2026, 10, 9)}  # 테스트용: 월요일 대체공휴일, 금요일 공휴일
LONG = {D(2026, 11, 16), D(2026, 11, 17), D(2026, 11, 18)}  # 가상 3일 연휴

TODAY = D(2026, 11, 12)

CASES = [
    # (설명, 발생일, 공휴일, 기대 신고 기한)
    ("월 발생, 주말 1번 낌", D(2026, 11, 9), set(), D(2026, 11, 16)),
    ("화 발생", D(2026, 11, 10), set(), D(2026, 11, 17)),
    ("수 발생", D(2026, 11, 11), set(), D(2026, 11, 18)),
    ("목 발생", D(2026, 11, 12), set(), D(2026, 11, 19)),
    ("금 발생", D(2026, 11, 13), set(), D(2026, 11, 20)),
    ("토 발생", D(2026, 11, 14), set(), D(2026, 11, 20)),
    ("일 발생", D(2026, 11, 15), set(), D(2026, 11, 20)),
    ("월말 → 다음 달", D(2026, 11, 25), set(), D(2026, 12, 2)),
    ("공휴일 없으면 금요일이 기한", D(2026, 12, 18), set(), D(2026, 12, 25)),
    ("기한 날이 공휴일 → 다음 평일", D(2026, 12, 18), XMAS, D(2026, 12, 28)),
    ("중간에 공휴일 1일", D(2026, 12, 21), XMAS, D(2026, 12, 29)),
    ("공휴일 없을 때 비교", D(2026, 12, 21), set(), D(2026, 12, 28)),
    ("연말 공휴일 2개 + 해 넘김", D(2026, 12, 24), XMAS, D(2027, 1, 4)),
    ("대체공휴일 월요일 + 금요일 공휴일", D(2026, 10, 2), OCT, D(2026, 10, 13)),
    ("다음 날이 공휴일", D(2026, 10, 8), OCT, D(2026, 10, 16)),
    ("발생일 자체가 공휴일", D(2026, 10, 9), OCT, D(2026, 10, 16)),
    ("주말 + 대체공휴일 붙은 연휴", D(2026, 10, 3), OCT, D(2026, 10, 13)),
    ("3일 연휴", D(2026, 11, 13), LONG, D(2026, 11, 25)),
    ("연휴 직전 수요일 발생", D(2026, 11, 11), LONG, D(2026, 11, 23)),
    ("연휴 중 발생", D(2026, 11, 17), LONG, D(2026, 11, 25)),
    ("공휴일 목록이 기한과 무관", D(2026, 11, 2), XMAS, D(2026, 11, 9)),
]


@pytest.mark.parametrize("desc,event,hol,expected", CASES, ids=[c[0] for c in CASES])
def test_report_due(desc, event, hol, expected):
    today = max(event, TODAY)
    r = compute_deadline("death", event, today=today, holidays=hol)
    assert r.report_due == expected


def test_case_count():
    assert len(CASES) >= 20


def test_d_day_future():
    r = compute_deadline("move", D(2026, 11, 12), today=D(2026, 11, 12))
    assert (r.report_due, r.d_day) == (D(2026, 11, 19), 7)


def test_d_day_today_is_due():
    r = compute_deadline("move", D(2026, 11, 5), today=D(2026, 11, 12))
    assert (r.report_due, r.d_day) == (D(2026, 11, 12), 0)


def test_d_day_overdue_negative():
    r = compute_deadline("death", D(2026, 11, 2), today=D(2026, 11, 12))
    assert (r.report_due, r.d_day) == (D(2026, 11, 9), -3)


def test_ear_tag_due_default_today():
    r = compute_deadline("birth", D(2026, 11, 10), today=D(2026, 11, 12))
    assert r.ear_tag_due == D(2026, 12, 12)


def test_ear_tag_due_from_reported_date():
    r = compute_deadline("birth", D(2026, 11, 10), today=D(2026, 11, 12), reported_date=D(2026, 11, 11))
    assert r.ear_tag_due == D(2026, 12, 11)


@pytest.mark.parametrize("et", ["death", "move"])
def test_no_ear_tag_for_non_birth(et):
    assert compute_deadline(et, D(2026, 11, 10), today=TODAY).ear_tag_due is None


def test_future_event_rejected():
    with pytest.raises(ValueError):
        compute_deadline("birth", D(2026, 11, 13), today=TODAY)


def test_unknown_event_rejected():
    with pytest.raises(ValueError):
        compute_deadline("sale", D(2026, 11, 10), today=TODAY)
