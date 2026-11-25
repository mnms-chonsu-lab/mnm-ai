from datetime import date
from types import SimpleNamespace

import pytest

from farm_router.router import LlmRoute, RouteError, route
from farm_router.schema import RouteRequest, RouteResponse

REQ = RouteRequest(
    text="x",
    today=date(2026, 11, 15),
    farm_context={"cows": [{"tag4": "1234", "desc": "큰 암소", "barn": "A1"}, {"tag4": "5678"}]},
)


class FakeClient:
    """responses.parse 만 흉내 낸다."""

    def __init__(self, parsed=None, status="completed", exc=None):
        self.parsed, self.status, self.exc, self.kwargs = parsed, status, exc, None
        self.responses = self

    def with_options(self, **_):
        return self

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return SimpleNamespace(status=self.status, output_parsed=self.parsed)


def llm(**kw):
    base = dict(intent="diary", items=[], report_type=None, birth=None, death=None, move=None)
    base.update(kw)
    return LlmRoute.model_validate(base)


def item(**kw):
    base = dict(category="소독", value="축사 소독", cow_tag4=None, vaccine_type=None, head_count=None)
    base.update(kw)
    return base


def test_diary_item():
    out = route(REQ, FakeClient(llm(items=[item()])))
    assert out.intent == "diary" and out.items[0].category == "소독"
    assert out.report_type is None and out.report_fields == {} and out.missing == []


def test_vaccine_with_known_cow():
    out = route(REQ, FakeClient(llm(items=[item(category="접종", value="구제역 접종", cow_tag4="1234", vaccine_type="구제역")])))
    assert out.items[0].cow_tag4 == "1234" and out.items[0].vaccine_type == "구제역"


def test_unknown_cow_is_null_and_missing():
    out = route(REQ, FakeClient(llm(items=[item(category="관찰", value="밥 안 먹음", cow_tag4="9999")])))
    assert out.items[0].cow_tag4 is None and out.missing == ["cow"]


def test_full_tag_reduced_to_last4():
    out = route(REQ, FakeClient(llm(items=[item(cow_tag4="002 1234 5678")])))
    assert out.items[0].cow_tag4 == "5678"


def test_birth_complete():
    p = llm(intent="history_report", items=[item(category="출생", value="송아지 출생", cow_tag4="1234")],
            report_type="birth", birth={"mother_tag4": "1234", "birth_date": "2026-11-14", "sex": "암"})
    out = route(REQ, FakeClient(p))
    assert out.report_fields.birth_date == date(2026, 11, 14) and out.missing == []


def test_birth_unsaid_field_is_null_not_missing():
    """빈칸 확인은 rules/report_state_machine 몫. missing 은 "cow" 만 (명세)."""
    p = llm(intent="history_report", report_type="birth", birth={"mother_tag4": "1234", "birth_date": "2026-11-15", "sex": None})
    out = route(REQ, FakeClient(p))
    assert out.report_fields.sex is None and out.missing == []


def test_birth_unknown_mother_only_cow():
    p = llm(intent="history_report", report_type="birth", birth={"mother_tag4": "4321", "birth_date": "2026-11-15", "sex": "수"})
    out = route(REQ, FakeClient(p))
    assert out.report_fields.mother_tag4 is None and out.missing == ["cow"]


def test_death_and_move():
    d = llm(intent="history_report", report_type="death", death={"cow_tag": "5678", "death_date": "2026-11-15"})
    assert route(REQ, FakeClient(d)).report_fields.cow_tag == "5678"
    m = llm(intent="history_report", report_type="move",
            move={"move_type": "sell", "cow_tag": "1234", "move_date": None, "counterpart_farm_no": None, "counterpart_name": "김"})
    out = route(REQ, FakeClient(m))
    assert out.report_fields.move_type == "sell" and out.missing == []


def test_bad_date_is_llm_fail():
    d = llm(intent="history_report", report_type="death", death={"cow_tag": "5678", "death_date": "어제"})
    with pytest.raises(RouteError):
        route(REQ, FakeClient(d))


def test_report_type_without_object_is_llm_fail():
    with pytest.raises(RouteError):
        route(REQ, FakeClient(llm(intent="history_report", report_type="birth")))


def test_negative_head_count_is_llm_fail():
    with pytest.raises(RouteError):
        route(REQ, FakeClient(llm(items=[item(category="사육현황", value="x", head_count=-1)])))


@pytest.mark.parametrize("client", [
    FakeClient(exc=TimeoutError()),
    FakeClient(parsed=None),
    FakeClient(parsed=llm(), status="incomplete"),
])
def test_failures_are_llm_fail(client):
    with pytest.raises(RouteError) as e:
        route(REQ, client)
    assert e.value.to_response().code == "LLM-FAIL"


def test_prompt_has_today_and_cows_no_personal_info():
    c = FakeClient(llm())
    route(REQ, c)
    msg = c.kwargs["input"]
    assert "2026-11-15" in msg and "tag4=1234" in msg and "큰 암소" in msg
    assert c.kwargs["text_format"] is LlmRoute and c.kwargs["store"] is False


def test_response_schema_rejects_fields_without_type():
    with pytest.raises(Exception):
        RouteResponse(intent="diary", items=[], report_type=None, report_fields={"sex": "암"})


def test_phone_masked_in_prompt():
    c = FakeClient(llm())
    route(REQ.model_copy(update={"text": "수의사 010-1234-5678 에 연락했어"}), c)
    assert "010-1234-5678" not in c.kwargs["input"] and "[전화번호]" in c.kwargs["input"]


def test_birth_item_gets_mother_tag_from_report_fields():
    p = llm(intent="history_report", items=[item(category="출생", value="송아지 출생")],
            report_type="birth", birth={"mother_tag4": "5678", "birth_date": "2026-11-14", "sex": "암"})
    out = route(REQ, FakeClient(p))
    assert out.items[0].cow_tag4 == "5678" and out.report_fields.mother_tag4 == "5678"


def test_spoken_tag_normalized_in_prompt():
    c = FakeClient(llm())
    route(REQ.model_copy(update={"text": "어젯밤 5, 6, 7, 8번이 송아지 낳았어요"}), c)
    assert "5678번" in c.kwargs["input"]


def test_report_type_forces_history_report_intent():
    p = llm(intent="diary", items=[item(category="출생", value="송아지 출생", cow_tag4="5678")],
            report_type="birth", birth={"mother_tag4": "5678", "birth_date": "2026-11-15", "sex": "수"})
    out = route(REQ, FakeClient(p))
    assert out.intent == "history_report" and out.report_type == "birth"


# ---------------------------------------------------------------------------
# 귀표 끝 4자리가 겹치는 소 (명세 변경 이력 11/20, mnm-ai#6). 번호는 가상
# ---------------------------------------------------------------------------
DUP = RouteRequest(
    text="x",
    today=date(2026, 11, 15),
    farm_context={"cows": [
        {"tag4": "0871", "tag": "900000000871", "barn": "1번 축사"},
        {"tag4": "0871", "tag": "900000010871", "barn": "2번 축사"},
        {"tag4": "5678", "tag": "900000005678"},
        {"tag4": "4321"},  # 12자리를 모르는 소
    ]},
)


def test_dup_tag4_only_is_null_and_missing():
    out = route(DUP, FakeClient(llm(items=[item(category="관찰", value="밥 안 먹음", cow_tag4="0871")])))
    assert out.items[0].cow_tag4 is None and out.items[0].cow_tag is None
    assert out.missing == ["cow"]


def test_dup_full_tag_picks_that_cow():
    out = route(DUP, FakeClient(llm(items=[item(category="관찰", value="밥 안 먹음", cow_tag4="9000 0001 0871")])))
    assert out.items[0].cow_tag4 == "0871" and out.items[0].cow_tag == "900000010871"
    assert out.missing == []


def test_full_tag_not_in_list_is_missing():
    out = route(DUP, FakeClient(llm(items=[item(cow_tag4="900000020871")])))
    assert out.items[0].cow_tag4 is None and out.missing == ["cow"]


def test_single_cow_tag4_fills_full_tag():
    out = route(DUP, FakeClient(llm(items=[item(cow_tag4="5678")])))
    assert out.items[0].cow_tag4 == "5678" and out.items[0].cow_tag == "900000005678"


def test_cow_without_full_tag_matches_by_last4():
    out = route(DUP, FakeClient(llm(items=[item(cow_tag4="4321")])))
    assert out.items[0].cow_tag4 == "4321" and out.items[0].cow_tag is None
    out = route(DUP, FakeClient(llm(items=[item(cow_tag4="002100004321")])))
    assert out.items[0].cow_tag4 == "4321" and out.items[0].cow_tag is None


def test_birth_mother_said_number_kept():
    p = llm(intent="history_report", report_type="birth", items=[item(category="출생", value="송아지 출생")],
            birth={"mother_tag4": "9000 0000 0871", "birth_date": "2026-11-14", "sex": "암"})
    out = route(DUP, FakeClient(p))
    assert out.report_fields.mother_tag4 == "900000000871"
    assert out.items[0].cow_tag4 == "0871" and out.items[0].cow_tag == "900000000871"
    p = llm(intent="history_report", report_type="birth", items=[], birth={"mother_tag4": "5678", "birth_date": None, "sex": None})
    assert route(DUP, FakeClient(p)).report_fields.mother_tag4 == "5678"


def test_birth_mother_dup_tag4_is_null_and_missing():
    p = llm(intent="history_report", report_type="birth", items=[item(category="출생", value="송아지 출생")],
            birth={"mother_tag4": "0871", "birth_date": "2026-11-14", "sex": "암"})
    out = route(DUP, FakeClient(p))
    assert out.report_fields.mother_tag4 is None and out.missing == ["cow"]
    assert out.items[0].cow_tag4 is None


def test_death_cow_full_tag_kept():
    d = llm(intent="history_report", report_type="death", items=[],
            death={"cow_tag": "900000010871", "death_date": "2026-11-14"})
    assert route(DUP, FakeClient(d)).report_fields.cow_tag == "900000010871"


def test_prompt_lists_full_tag():
    c = FakeClient(llm())
    route(DUP, c)
    assert "tag4=0871, tag=900000010871, barn=2번 축사" in c.kwargs["input"]


def test_cow_tag_must_end_with_tag4():
    with pytest.raises(ValueError):
        RouteRequest(text="x", today=date(2026, 11, 15),
                     farm_context={"cows": [{"tag4": "1234", "tag": "900000005678"}]})
