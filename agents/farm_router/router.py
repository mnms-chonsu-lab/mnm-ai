"""농가 라우터 (명세 REQ-AI-01).

발화 한 문장을 일지 항목·신고서 항목으로 구조화한다.
OpenAI Structured Outputs(`responses.parse`)로 형식을 강제하고,
소 목록 대조는 코드에서 한다. 신고서 빈칸 확인은 rules/report_state_machine 이 한다.
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.ai_call import mask_phone

from .tag_normalize import normalize_tags
from .schema import (
    Category,
    Intent,
    ReportType,
    RouteErrorResponse,
    RouteRequest,
    RouteResponse,
)

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-5.4-mini")
ROUTER_REASONING_EFFORT = os.getenv("ROUTER_REASONING_EFFORT") or None
TIMEOUT_SEC = 8.0  # 명세 REQ-AI-01·12: 8초 초과 시 LLM-FAIL
PROMPT_VERSION = "route-v3"
MODEL_VERSION = f"{ROUTER_MODEL}+{PROMPT_VERSION}"


class RouteError(Exception):
    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_response(self) -> RouteErrorResponse:
        return RouteErrorResponse(code="LLM-FAIL", message=self.message, detail=self.detail)


# ---------------------------------------------------------------------------
# 모델이 채우는 형식 (Structured Outputs strict 용: 모든 키 필수, 없으면 null)
# ---------------------------------------------------------------------------
class _O(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Item(_O):
    category: Category
    value: str = Field(description="기록 내용 한 줄")
    cow_tag4: Optional[str] = Field(description="발화에 나온 개체번호를 말한 그대로 (끝 4자리 또는 귀표 12자리). 없으면 null")
    vaccine_type: Optional[str] = Field(description="접종일 때 백신 종류. 아니면 null")
    head_count: Optional[int] = Field(description="두수. 말하지 않았으면 null")


class _Birth(_O):
    mother_tag4: Optional[str]
    birth_date: Optional[str] = Field(description="YYYY-MM-DD")
    sex: Optional[Literal["암", "수"]]


class _Death(_O):
    cow_tag: Optional[str]
    death_date: Optional[str] = Field(description="YYYY-MM-DD")


class _Move(_O):
    move_type: Optional[Literal["sell", "buy", "move"]]
    cow_tag: Optional[str]
    move_date: Optional[str] = Field(description="YYYY-MM-DD")
    counterpart_farm_no: Optional[str]
    counterpart_name: Optional[str]


class _Report(_O):
    report_type: ReportType
    birth: Optional[_Birth] = Field(description="report_type 이 birth 일 때만 채움")
    death: Optional[_Death] = Field(description="report_type 이 death 일 때만 채움")
    move: Optional[_Move] = Field(description="report_type 이 move 일 때만 채움")


class LlmRoute(_O):
    intent: Intent
    items: list[_Item]
    reports: list[_Report] = Field(description="신고서. 소 한 마리에 1장. 신고가 아니면 빈 목록")
    market_trade: bool = Field(description="가축시장(우시장)에서 사고판 거래이면 true")


INSTRUCTIONS = """너는 한우 농가의 말 한 문장을 축산 기록으로 정리한다.

[intent]
- diary: 사육현황·접종·소독·관찰·출입·메모 같은 일지 기록
- history_report: 출생·폐사·이동(판매·구입·옮김) 이력 신고
- symptom: 소가 아프거나 평소와 다른 상태를 말함
- 잘 먹음·잘 서 있음·활발함·컨디션 좋음 같은 정상 상태는 symptom 이 아니라 diary (category 관찰)
- 소가 죽었다는 말은 폐사 신고 → history_report, report_type death

[items] 말한 내용을 항목으로 나눈다.
- category 는 사육현황, 접종, 소독, 관찰, 출입, 출생, 폐사, 이동, 메모 중 하나
- 출입은 차량·사람·업자가 실제로 농장에 드나든 기록 (예: 사료 차량 방문, 수의사 방문)
- 앞으로의 예정이나 물품이 들어왔다는 말만 있으면 메모
- 한 문장에 소가 여러 마리 나오면 소마다 항목을 따로 만든다 (예: "1234번이랑 5678번 구제역 백신" → 접종 항목 2개)
- value 는 말한 내용을 짧게. 말하지 않은 내용을 지어내지 않는다
- cow_tag4 는 발화에 나온 개체번호를 말한 그대로 적는다. 끝 4자리만 말했으면 4자리, 귀표 12자리를 다 말했으면 12자리. 소 특징(예: 큰 암소)으로 말하면 소 목록의 desc 와 맞는 소의 tag4. 특정할 수 없으면 null
- 신고서의 mother_tag4, cow_tag 도 말한 번호를 그대로 적는다 (4자리 또는 12자리)
- 접종이면 vaccine_type 에 백신 종류. 두수를 말했으면 head_count

[reports] 신고서 목록
- 소 한 마리에 신고서 1장. 소가 여러 마리면 소마다 따로 만든다 (예: "3001번이랑 0000번이 죽었어" → 폐사 신고서 2장)
- 출생 → report_type birth, birth: mother_tag4, birth_date, sex(암 또는 수)
- 폐사 → report_type death, death: cow_tag, death_date
- 이동 → report_type move, move: move_type(sell=판매, buy=구입, move=옮김), cow_tag, move_date, counterpart_farm_no, counterpart_name
- report_type 에 해당하지 않는 객체는 null
- 신고가 아니면 빈 목록

[market_trade]
- 가축시장(우시장)에서 소를 팔거나 산 경우 true. 그 밖에는 false
- 이때도 이동 항목과 신고서는 평소처럼 채운다 (가축시장 거래 신고서는 코드가 뺀다)

[날짜]
- "어젯밤", "오늘 오후", "그저께" 같은 표현은 오늘 날짜 기준 YYYY-MM-DD 로 바꾼다
- 날짜를 말하지 않았으면 null. 추측하지 않는다

[모르는 값]
- 말하지 않은 값은 모두 null
"""


def _user_message(req: RouteRequest) -> str:
    cows = "\n".join(
        f"- tag4={c.tag4}" + (f", tag={c.tag}" if c.tag else "") + (f", desc={c.desc}" if c.desc else "")
        + (f", barn={c.barn}" if c.barn else "")
        for c in req.farm_context.cows
    ) or "- (없음)"
    return f"오늘: {req.today.isoformat()} ({'월화수목금토일'[req.today.weekday()]}요일)\n소 목록:\n{cows}\n\n발화: {normalize_tags(mask_phone(req.text))}"


def _to_date(v: Optional[str], name: str) -> Optional[date]:
    if v is None:
        return None
    try:
        return date.fromisoformat(v)
    except ValueError as e:
        raise RouteError("날짜 형식 오류", {"field": name, "value": v}) from e


def _digits(tag: Optional[str]) -> str:
    return "".join(ch for ch in tag if ch.isdigit()) if tag else ""


def resolve_cow(tag: Optional[str], cows) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """말한 번호 → (끝 4자리, 12자리, 신고서 칸에 넣을 번호). 정할 수 없으면 (None, None, None).

    규칙 (명세 변경 이력 11/20, mnm-ai#6)
    - 끝 4자리가 같은 소가 1마리 → 그 소
    - 2마리 이상 + 4자리만 말함 → 정할 수 없음
    - 12자리를 말함 → 소 목록 tag 와 같은 소. 12자리를 모르는 소가 1마리뿐이면 끝 4자리로 맞춤
    """
    d = _digits(tag)
    if len(d) < 4:
        return None, None, None
    t4 = d[-4:]
    cands = [c for c in cows if c.tag4 == t4]
    if len(d) == 12:
        exact = [c for c in cands if c.tag == d]
        if exact:
            return t4, d, d
        if len(cands) == 1 and cands[0].tag is None:
            return t4, None, d
        return None, None, None  # 12자리가 목록과 다르거나, 겹치는 소 중 어느 것도 아님
    if len(cands) == 1:
        return t4, cands[0].tag, t4
    return None, None, None  # 없음 또는 끝 4자리만으로는 겹침


def postprocess(out: LlmRoute, req: RouteRequest) -> RouteResponse:
    """모델 출력 → 고정 출력. 개체번호를 소 목록과 대조한다 (없거나 정할 수 없으면 null, missing 에 "cow")."""
    cows = req.farm_context.cows
    missing: list[str] = []

    def check(tag: Optional[str]):
        if tag is None:
            return None, None, None
        r = resolve_cow(tag, cows)
        if r[0] is None and "cow" not in missing:
            missing.append("cow")
        return r

    items = []
    for it in out.items:
        d = it.model_dump()
        t4, t12, _ = check(it.cow_tag4)
        d["cow_tag4"], d["cow_tag"] = t4, t12
        items.append(d)

    # 가축시장에서 사고판 거래는 시장(축협)이 신고하므로 양도·양수 신고서를 만들지 않는다 (mnm-ai#20)
    notice = None
    src_reports = list(out.reports)
    if out.market_trade:
        kept = [r for r in src_reports if not (r.report_type == "move" and r.move and r.move.move_type in ("sell", "buy"))]
        if len(kept) != len(src_reports):
            notice = "market_trade"
        src_reports = kept

    reports: list[dict] = []
    subjects: list[tuple] = []
    for r in src_reports:
        src = getattr(r, r.report_type)
        if src is None:
            raise RouteError("report_type 과 항목이 맞지 않음", {"report_type": r.report_type})
        fields = src.model_dump()
        for k in ("birth_date", "death_date", "move_date"):
            if k in fields:
                fields[k] = _to_date(fields[k], k)
        subject = (None, None, None)
        for k in ("mother_tag4", "cow_tag"):
            if k in fields:
                subject = check(fields[k])
                fields[k] = subject[2]  # 말한 번호 그대로 (4자리 또는 12자리)
        reports.append({"report_type": r.report_type, "report_fields": fields})
        subjects.append((r.report_type, subject))

    # 신고서 칸에만 번호가 있고 같은 종류의 일지 항목에는 빠진 경우 채운다 (신고서 1장일 때만: 여러 장이면 어느 항목인지 정할 수 없음)
    if len(subjects) == 1:
        rt, subject = subjects[0]
        cat = {"birth": "출생", "death": "폐사", "move": "이동"}[rt]
        if subject[0]:
            for d in items:
                if d["category"] == cat and d["cow_tag4"] is None:
                    d["cow_tag4"], d["cow_tag"] = subject[0], subject[1]

    # 신고서를 만들었으면 신고 흐름으로 보낸다 (모델이 intent 를 diary 로 둔 경우, 실측 3/50).
    # 가축시장 거래로 신고서가 모두 빠졌으면 일지로 남긴다.
    if reports:
        intent = "history_report"
    elif notice and out.intent == "history_report":
        intent = "diary"
    else:
        intent = out.intent
    first = reports[0] if reports else {"report_type": None, "report_fields": {}}

    try:
        return RouteResponse(
            intent=intent,
            items=items,
            report_type=first["report_type"],
            report_fields=first["report_fields"],
            reports=reports,
            notice=notice,
            missing=missing,
        )
    except ValidationError as e:
        raise RouteError("출력 형식 오류", {"errors": e.errors(include_url=False, include_context=False)}) from e


def route(req: RouteRequest, client=None, model: str = ROUTER_MODEL, timeout_sec: float = TIMEOUT_SEC) -> RouteResponse:
    """발화 → 구조화. 실패·형식 오류·시간 초과는 RouteError(LLM-FAIL)."""
    if client is None:
        client = make_client()
    kwargs = dict(
        model=model,
        instructions=INSTRUCTIONS,
        input=_user_message(req),
        text_format=LlmRoute,
        store=False,
    )
    if ROUTER_REASONING_EFFORT:
        kwargs["reasoning"] = {"effort": ROUTER_REASONING_EFFORT}

    t0 = time.perf_counter()
    try:
        resp = client.with_options(timeout=timeout_sec, max_retries=0).responses.parse(**kwargs)
    except Exception as e:  # 네트워크·인증·시간 초과 등
        raise RouteError("모델 호출 실패", {"type": type(e).__name__}) from e
    elapsed = time.perf_counter() - t0
    if elapsed > timeout_sec:
        raise RouteError("시간 초과", {"elapsed_sec": round(elapsed, 2), "limit_sec": timeout_sec})

    if getattr(resp, "status", "completed") != "completed":
        raise RouteError("응답 미완료", {"status": resp.status})
    out = getattr(resp, "output_parsed", None)
    if out is None:
        raise RouteError("구조화 결과 없음 (거절 또는 형식 오류)")
    return postprocess(out, req)


def make_client():
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RouteError("OPENAI_API_KEY 가 설정되지 않음")
    return OpenAI()
