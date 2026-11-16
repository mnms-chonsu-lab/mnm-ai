"""REQ-AI-05 완료 기준: 출생·의심 신고 각각 빈칸 조합 테스트 통과."""

from datetime import date as D
from itertools import product

import pytest

from report_state_machine import build_form6, form6_copy_text, report_state
from report_state_machine.schema import Form6

TODAY = D(2026, 11, 12)

BIRTH_FULL = {
    "mother_tag4": "1234",
    "birth_date": "2026-11-11",
    "sex": "암",
    "breed": "한우",
    "calf_tag": "002 1234 5678 9",
}
BIRTH_ORDER = list(BIRTH_FULL)

SUSPECT_FULL = {
    "owner_and_location": "김한우, 충남 홍성군 (가상)",
    "species": "소(한우)",
    "head_count": 1,
    "symptoms": ["침 흘림", "물집"],
    "found_date": "2026-11-12",
    "reporter": "김한우, 충남 홍성군 (가상)",
}
SUSPECT_ORDER = list(SUSPECT_FULL)


def _subsets(full):
    keys = list(full)
    for mask in product([0, 1], repeat=len(keys)):
        yield {k: full[k] for k, m in zip(keys, mask) if m}


OPTIONAL = {"calf_tag"}  # 출생 신고 송아지 귀표는 선택


@pytest.mark.parametrize("kind,full,order", [("birth", BIRTH_FULL, BIRTH_ORDER), ("suspect", SUSPECT_FULL, SUSPECT_ORDER)])
def test_every_blank_combination(kind, full, order):
    n = 0
    for given in _subsets(full):
        r = report_state(kind, given, TODAY)
        expected_missing = [k for k in order if k not in given and k not in OPTIONAL]
        assert r.missing == expected_missing
        assert r.can_submit == (not expected_missing)
        if expected_missing:
            assert r.next_question and r.next_question.endswith(("요.", "요?"))
        else:
            assert r.next_question is None
        n += 1
    assert n == 2 ** len(full)  # 출생 32가지, 의심 64가지


def test_next_question_is_first_missing_only():
    r = report_state("birth", {"mother_tag4": "1234"}, TODAY)
    assert r.next_question == "송아지가 태어난 날이 언제예요?"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("mother_tag4", "123"),
        ("mother_tag4", "12a4"),
        ("birth_date", "2026-11-13"),  # 미래
        ("birth_date", "어제"),  # 날짜로 변환 안 된 말
        ("sex", "몰라"),
        ("breed", "   "),
    ],
)
def test_invalid_value_is_blank(field, bad):
    r = report_state("birth", {**BIRTH_FULL, field: bad}, TODAY)
    assert r.missing == [field] and not r.can_submit


def test_birth_can_submit_without_calf_tag():
    r = report_state("birth", {k: v for k, v in BIRTH_FULL.items() if k != "calf_tag"}, TODAY)
    assert r.can_submit and r.fields["calf_tag"] is None


def test_sex_normalized():
    r = report_state("birth", {**BIRTH_FULL, "sex": "수컷"}, TODAY)
    assert r.fields["sex"] == "수" and r.can_submit


@pytest.mark.parametrize("hc", [0, -1, "두"])
def test_suspect_head_count_invalid(hc):
    r = report_state("suspect", {**SUSPECT_FULL, "head_count": hc}, TODAY)
    assert r.missing == ["head_count"]


def test_suspect_empty_symptom_list_is_blank():
    r = report_state("suspect", {**SUSPECT_FULL, "symptoms": []}, TODAY)
    assert r.missing == ["symptoms"]


def test_other_is_optional():
    r = report_state("suspect", SUSPECT_FULL, TODAY)
    assert r.can_submit and r.fields["other"] is None


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        report_state("birth", {"color": "검정"}, TODAY)


def test_form6_from_state():
    r = report_state("suspect", SUSPECT_FULL, TODAY)
    f = build_form6(r.fields)
    assert f.condition == "발견 당시 상태: 침 흘림, 물집"
    assert f.species_and_head_count == "소(한우) 1두"
    assert f.found_date == D(2026, 11, 12)
    text = form6_copy_text(f)
    assert text.count("\n") == 5 and "필요한 사항: -" in text


def test_form6_rejects_disease_name_in_condition():
    with pytest.raises(ValueError):
        Form6(
            owner_and_location="x", species_and_head_count="소 1마리", condition="구제역 의심",
            found_date=TODAY, reporter="x",
        )


def test_symptom_string_is_one_item():
    r = report_state("suspect", {**SUSPECT_FULL, "symptoms": "침 흘림"}, TODAY)
    assert r.fields["symptoms"] == ["침 흘림"]


def test_datetime_keeps_date_only():
    from datetime import datetime
    r = report_state("birth", {**BIRTH_FULL, "birth_date": datetime(2026, 11, 11, 9, 30)}, TODAY)
    assert r.fields["birth_date"] == "2026-11-11"
    r2 = report_state("birth", {**BIRTH_FULL, "birth_date": "2026-11-11T09:30:00"}, TODAY)
    assert r2.fields["birth_date"] == "2026-11-11"


@pytest.mark.parametrize("hc", [True, 1.7])
def test_head_count_rejects_bool_and_fraction(hc):
    assert report_state("suspect", {**SUSPECT_FULL, "head_count": hc}, TODAY).missing == ["head_count"]


def test_build_form6_with_blank_raises_clear_error():
    with pytest.raises(ValueError, match="head_count"):
        build_form6({**SUSPECT_FULL, "head_count": None})


def test_form6_head_count_with_farm_total():
    f = dict(owner_and_location="x", species="한우", head_count=1, symptoms=["침 흘림"],
             found_date="2026-11-15", reporter="x")
    assert build_form6(f, farm_total_head=42).species_and_head_count == "한우 1두 (전체 42두)"
    assert build_form6(f).species_and_head_count == "한우 1두"
