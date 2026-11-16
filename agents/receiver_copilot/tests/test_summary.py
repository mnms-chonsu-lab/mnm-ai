import datetime as dt

import pytest

from receiver_copilot import SummaryError, SummaryRequest, summarize

D = dt.date


def base(**kw):
    req = dict(
        record_id="r-1",
        found_date="2026-11-15",
        cow={"tag4": "1234", "desc": "큰 암소", "barn": "A1"},
        symptom_answers={"drool": True, "blister": True, "lump": False, "feed": False},
        photos=[{"part": "mouth", "match": True}, {"part": "hoof", "match": False}],
        diary=[
            {"cow_tag4": "1234", "category": "관찰", "value": "침 흘림", "date": "2026-11-15"},
            {"cow_tag4": "1234", "category": "관찰", "value": "사료 남김", "date": "2026-11-09"},
            {"cow_tag4": "1234", "category": "관찰", "value": "잘 먹음", "date": "2026-11-08"},  # 8일 전 → 제외
            {"cow_tag4": "5678", "category": "관찰", "value": "다른 소", "date": "2026-11-14"},
            {"cow_tag4": "1234", "category": "접종", "value": "구제역 접종", "date": "2026-10-01"},
            {"cow_tag4": None, "category": "접종", "value": "럼피스킨 접종 10두", "date": "2026-11-02"},
            {"cow_tag4": None, "category": "소독", "value": "축사 소독", "date": "2026-11-13"},
            {"cow_tag4": None, "category": "소독", "value": "미래 기록", "date": "2026-11-20"},  # 발견일 뒤 → 제외
            {"cow_tag4": None, "category": "출입", "value": "사료 차량", "date": "2026-11-12"},
            {"cow_tag4": None, "category": "출입", "value": "미래 출입", "date": "2026-11-20"},  # 발견일 뒤 → 제외
        ],
        report_fields={
            "owner_and_location": "농장주, 경기 ○○군 ○○리",
            "species": "한우",
            "head_count": 23,
            "symptoms": ["침 흘림", "물집"],
            "found_date": "2026-11-15",
            "reporter": "농장주, 경기 ○○군 ○○리",
        },
    )
    req.update(kw)
    return SummaryRequest.model_validate(req)


def test_card_values_from_records():
    out = summarize(base())
    c = out.card
    assert c.matched_symptoms == ["침 흘림", "물집"]
    assert [(r.date, r.value) for r in c.diary_related] == [(D(2026, 11, 15), "침 흘림"), (D(2026, 11, 9), "사료 남김")]
    assert (c.vaccine.date, c.vaccine.value, c.vaccine.days_ago) == (D(2026, 11, 2), "럼피스킨 접종 10두", 13)
    assert (c.disinfection.date, c.disinfection.days_ago) == (D(2026, 11, 13), 2)
    assert (c.visit.value, c.visit.days_ago) == ("사료 차량", 3)
    assert [p.match for p in c.photos] == [True, False]


def test_form6_numbers_and_condition_unchanged():
    out = summarize(base())
    assert out.form6.species_and_head_count == "한우 23두"
    assert out.form6.condition == "발견 당시 상태: 침 흘림, 물집"
    assert out.form6.found_date == D(2026, 11, 15)
    assert "발견당시의 상태): 발견 당시 상태: 침 흘림, 물집" in out.form6_text


def test_no_records_gives_null():
    out = summarize(base(diary=[], photos=[]))
    assert out.card.vaccine is None and out.card.disinfection is None and out.card.visit is None
    assert out.card.diary_related == [] and out.card.photos == []


def test_seven_day_window_boundary():
    diary = [{"cow_tag4": "1234", "category": "관찰", "value": v, "date": d}
             for v, d in [("6일 전", "2026-11-09"), ("7일 전", "2026-11-08")]]
    assert [r.value for r in summarize(base(diary=diary)).card.diary_related] == ["6일 전"]


def test_blank_report_fields_is_error():
    rf = dict(base().report_fields, reporter="")
    with pytest.raises(SummaryError) as e:
        summarize(base(report_fields=rf))
    assert e.value.to_response().code == "LLM-FAIL"


def test_no_symptom_words_beyond_answers():
    out = summarize(base(symptom_answers={"drool": False, "blister": False, "lump": True, "feed": True}))
    assert out.card.matched_symptoms == ["사료 남김", "혹"]


def test_form6_farm_total_in_parentheses():
    out = summarize(base(farm_total_head=42))
    assert out.form6.species_and_head_count == "한우 23두 (전체 42두)"
    assert "2. 신고대상 가축의 종류 및 두수: 한우 23두 (전체 42두)" in out.form6_text
