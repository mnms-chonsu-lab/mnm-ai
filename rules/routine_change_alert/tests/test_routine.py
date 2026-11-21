"""REQ-AI-07 완료 기준: 시드로 1234번 알림 1건 정상 생성."""

from datetime import date as D, datetime as DT, timedelta

import pytest

from routine_change_alert import feedback_ratio, is_unresponded, routine_change, scan_routine_changes
from routine_change_alert.schema import AlertState

TODAY = D(2026, 11, 12)


def e(cow, value, day, category="관찰"):
    return {"cow_tag4": cow, "category": category, "value": value, "date": day}


# 시드와 같은 모양의 테스트 일지 (실제 시드 값은 인교님 REQ-DATA-06 확정 후 교체)
SEED = [
    e("1234", "기침을 함", D(2026, 11, 9)),
    e("1234", "기침을 함", D(2026, 11, 12)),
    e("0871", "사료를 남김", D(2026, 11, 12)),
    e("5520", "기침을 함", D(2026, 11, 1)),
    e("5520", "기침을 함", D(2026, 11, 12)),
    e("1234", "구제역 접종", D(2026, 11, 10), category="접종"),
]


def test_seed_generates_one_alert_for_1234():
    alerts = scan_routine_changes(SEED, TODAY)
    assert len(alerts) == 1
    a = alerts[0]
    assert (a.cow_tag4, a.observation, a.dates) == ("1234", "기침을 함", [D(2026, 11, 9), D(2026, 11, 12)])


def test_same_day_twice_counts_as_two():
    r = routine_change([e("1234", "기침", TODAY), e("1234", "기침", TODAY)], "1234", "기침", TODAY)
    assert r and r.dates == [TODAY, TODAY]


def test_single_record_no_alert():
    assert routine_change([e("1234", "기침", TODAY)], "1234", "기침", TODAY) is None


def test_window_edge_included():  # 오늘 포함 7일 = 11/6~11/12
    assert routine_change([e("1234", "기침", D(2026, 11, 6)), e("1234", "기침", TODAY)], "1234", "기침", TODAY)


def test_outside_window_excluded():
    assert routine_change([e("1234", "기침", D(2026, 11, 5)), e("1234", "기침", TODAY)], "1234", "기침", TODAY) is None


def test_other_cow_not_mixed():
    assert routine_change([e("0871", "기침", D(2026, 11, 10)), e("1234", "기침", TODAY)], "1234", "기침", TODAY) is None


def test_non_observation_category_ignored():
    entries = [e("1234", "기침", D(2026, 11, 10), "메모"), e("1234", "기침", TODAY)]
    assert routine_change(entries, "1234", "기침", TODAY) is None


def test_whitespace_normalized():
    entries = [e("1234", " 기침을  함", D(2026, 11, 10)), e("1234", "기침을 함", TODAY)]
    r = routine_change(entries, "1234", "기침을 함 ", TODAY)
    assert r and r.observation == "기침을 함"


SENT = DT(2026, 11, 10, 9, 0)


@pytest.mark.parametrize(
    "hours,viewed,feedback,expected",
    [
        (47, None, None, False),
        (48, None, None, True),
        (72, None, None, True),
        (72, SENT + timedelta(hours=1), None, False),
        (72, None, "문제 없음", False),
    ],
)
def test_unresponded(hours, viewed, feedback, expected):
    a = AlertState(sent_at=SENT, viewed_at=viewed, feedback=feedback)
    assert is_unresponded(a, SENT + timedelta(hours=hours)) is expected


def test_feedback_ratio():
    alerts = [
        AlertState(sent_at=SENT, feedback="문제 없음"),
        AlertState(sent_at=SENT, feedback="문제 없음"),
        AlertState(sent_at=SENT, feedback="확인함"),
        AlertState(sent_at=SENT),
    ]
    assert feedback_ratio(alerts) == {"total": 4, "responded": 3, "confirmed": 1, "no_issue": 2,
                                     "confirmed_ratio": 0.333, "no_issue_ratio": 0.667}
    assert feedback_ratio([])["no_issue_ratio"] is None and feedback_ratio([])["confirmed_ratio"] is None
