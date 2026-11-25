"""신고서 상태 머신 (REQ-AI-05). LLM 사용 금지.

입력된 값을 검사해 빈칸을 찾고, 빈칸 중 첫 번째 하나만 물어본다(해요체).
값이 형식에 맞지 않으면 빈칸으로 본다(잘못된 값을 그대로 제출하지 않음).

신고서 종류
- birth   : 출생 신고 (어미 소, 출생일, 성별, 품종, 송아지 귀표)
- death   : 폐사 신고 (소, 폐사일)
- sell    : 양도 신고 (소, 보낸 날, 받는 농장 식별번호)
- buy     : 양수 신고 (소, 받은 날, 보낸 농장 식별번호)
- move    : 이동 신고 (소, 옮긴 날. 상대 농장은 선택)
  폐사·양도·양수·이동 칸 이름·필수 여부·질문은 mnm-dev FarmerRules(F-S04Q)와 같다.
- suspect : 전염병 의심 신고 — 가축전염병 예방법 시행규칙 제13조 제1항 6개 항목
            1 소유자 성명·사육 장소 / 2 가축 종류·두수 / 3 질병명(미진단 시 발견 당시 상태)
            4 폐사일 또는 발견일 / 5 신고자 성명·주소 / 6 그 밖의 사항(선택)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, Mapping, Optional

from .schema import Form6, ReportStateResponse


def _tag4(v: Any) -> Optional[str]:
    """소 칸: 끝 4자리 또는 귀표 12자리 (라우터가 말한 번호를 그대로 넘김, mnm-ai#6)."""
    s = str(v).strip() if v is not None else ""
    return s if len(s) in (4, 12) and s.isdigit() else None


def _date(today: date) -> Callable[[Any], Optional[date]]:
    def f(v: Any) -> Optional[date]:
        if v is None:
            return None
        if isinstance(v, datetime):  # 시각이 붙어 오면 날짜만
            v = v.date()
        try:
            d = v if isinstance(v, date) else date.fromisoformat(str(v)[:10])
        except ValueError:
            return None
        return d if d <= today else None  # 미래 날짜는 빈칸 처리

    return f


_SEX = {"암": "암", "암컷": "암", "암소": "암", "수": "수", "수컷": "수", "수소": "수"}


def _sex(v: Any) -> Optional[str]:
    return _SEX.get(str(v).strip()) if v is not None else None


def _text(v: Any) -> Optional[str]:
    s = " ".join(str(v).split()) if v is not None else ""
    return s or None


def _pos_int(v: Any) -> Optional[int]:
    if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _symptoms(v: Any) -> Optional[list[str]]:
    if not v:
        return None
    if isinstance(v, str):  # 문자열 하나로 오면 항목 1개
        v = [v]
    items = [_text(x) for x in v]
    items = [x for x in items if x]
    return items or None


# (필드, 필수 여부, 검사 함수 생성기, 질문)
def _spec(kind: str, today: date):
    if kind == "birth":
        return [
            ("mother_tag4", True, _tag4, "어미 소 귀표 번호 끝 4자리를 말씀해 주세요."),
            ("birth_date", True, _date(today), "송아지가 태어난 날이 언제예요?"),
            ("sex", True, _sex, "송아지가 암컷이에요, 수컷이에요?"),
            ("breed", True, _text, "송아지 품종을 알려 주세요. 예를 들면 한우예요."),
            ("calf_tag", False, _text, "송아지 귀표 번호를 알려 주세요."),  # 선택: 번호는 출생 신고 후 위탁기관이 부여
        ]
    if kind == "death":
        return [
            ("cow_tag", True, _tag4, "어느 소인가요? 귀표 끝 4자리를 알려 주세요."),
            ("death_date", True, _date(today), "소가 죽은 날을 알려 주세요."),
        ]
    if kind in ("sell", "buy", "move"):
        return [
            ("cow_tag", True, _tag4, "어느 소인가요? 귀표 끝 4자리를 알려 주세요."),
            ("move_date", True, _date(today), "소를 보내거나 받은 날을 알려 주세요."),
            # 상대 농장식별번호: 양도·양수는 필수, 이동은 선택 (FarmerRules와 같음)
            ("counterpart_farm_no", kind != "move", _text, "상대 농장식별번호를 알려 주세요."),
            ("counterpart_name", False, _text, None),
        ]
    if kind == "suspect":
        return [
            ("owner_and_location", True, _text, "소 주인 성함과 소를 키우는 곳 주소를 알려 주세요."),
            ("species", True, _text, "어떤 가축이에요?"),
            ("head_count", True, _pos_int, "그런 모습을 보이는 소가 몇 마리예요?"),
            ("symptoms", True, _symptoms, "소가 어떤 모습인지 말씀해 주세요."),
            ("found_date", True, _date(today), "그 모습을 처음 본 날이 언제예요?"),
            ("reporter", True, _text, "신고하시는 분 성함과 주소를 알려 주세요."),
            ("other", False, _text, None),
        ]
    raise ValueError(f"알 수 없는 신고서 종류: {kind}")


def report_state(kind: str, fields: Mapping[str, Any], today: date) -> ReportStateResponse:
    spec = _spec(kind, today)
    known = {name for name, *_ in spec}
    extra = set(fields) - known
    if extra:
        raise ValueError(f"정의되지 않은 항목: {sorted(extra)}")

    clean: dict[str, Any] = {}
    missing: list[str] = []
    next_q: Optional[str] = None
    for name, required, check, question in spec:
        val = check(fields.get(name))
        clean[name] = val
        if val is None and required:
            missing.append(name)
            if next_q is None:
                next_q = question

    return ReportStateResponse(
        fields={k: (v.isoformat() if isinstance(v, date) else v) for k, v in clean.items()},
        missing=missing,
        next_question=next_q,
        can_submit=not missing,
    )


def build_form6(fields: Mapping[str, Any], farm_total_head: Optional[int] = None) -> Form6:
    """빈칸이 없는 의심 신고 상태(fields)를 제13조 6개 항목으로 바꾼다. 숫자·날짜는 그대로.

    2호 두수는 증상이 있거나 죽은 소의 수(head_count). 농장 정보의 전체 사육 두수를 알면 괄호로 붙인다.
    예: "한우 1두 (전체 42두)"
    """
    required = ("owner_and_location", "species", "head_count", "symptoms", "found_date", "reporter")
    missing = [k for k in required if not fields.get(k)]
    if missing:
        raise ValueError(f"빈칸이 남아 6개 항목을 만들 수 없습니다: {missing}")
    return Form6(
        owner_and_location=fields["owner_and_location"],
        species_and_head_count=f"{fields['species']} {fields['head_count']}두"
        + (f" (전체 {farm_total_head}두)" if farm_total_head else ""),
        condition="발견 당시 상태: " + ", ".join(fields["symptoms"]),
        found_date=date.fromisoformat(str(fields["found_date"])),
        reporter=fields["reporter"],
        other=fields.get("other") or "",
    )


# 칸 이름은 mnm-data biosecurity_rule kind=form_field 행(가축전염병 예방법 시행규칙 제13조) 원문 그대로
FORM6_LABELS = (
    ("owner_and_location", "1. 신고대상 가축의 소유자등의 성명(축산계열화사업자의 경우에는 회사ㆍ법인명으로 한다) 및 신고대상 가축의 사육장소 또는 발견장소"),
    ("species_and_head_count", "2. 신고대상 가축의 종류 및 두수"),
    ("condition", "3. 질병명(수의사의 진단을 받지 아니한 때에는 신고자가 추정하는 병명 또는 발견당시의 상태)"),
    ("found_date", "4. 죽은 연월일(죽은 연월일이 분명하지 아니한 때에는 발견 연월일)"),
    ("reporter", "5. 신고자의 성명 및 주소"),
    ("other", "6. 그 밖에 가축이 죽거나 병든 원인 등 신고에 관하여 필요한 사항"),
)


def form6_copy_text(form: Form6) -> str:
    """인터넷 신고서에 붙여 넣을 복사 텍스트 (F-S14, R-P02)."""
    d = form.model_dump()
    return "\n".join(f"{label}: {d[key] or '-'}" for key, label in FORM6_LABELS)
