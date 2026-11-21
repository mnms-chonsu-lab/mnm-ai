"""농가 라우터·음성 인식 입출력 형식 (명세 REQ-AI-01, REQ-AI-02). mnm-dev 는 이 형식을 따른다."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Category = Literal["사육현황", "접종", "소독", "관찰", "출입", "출생", "폐사", "이동", "메모"]
Intent = Literal["diary", "history_report", "symptom"]
ReportType = Literal["birth", "death", "move"]

_TAG4 = re.compile(r"^\d{4}$")


def _tag4(v: Optional[str]) -> Optional[str]:
    if v is not None and not _TAG4.match(v):
        raise ValueError("개체번호 끝 4자리는 숫자 4개여야 합니다")
    return v


_TAG12 = re.compile(r"^\d{12}$")


def _tag12(v: Optional[str]) -> Optional[str]:
    if v is not None and not _TAG12.match(v):
        raise ValueError("귀표 12자리는 숫자 12개여야 합니다")
    return v


def _said_tag(v: Optional[str]) -> Optional[str]:
    """신고서 소 칸: 말한 번호 그대로 (끝 4자리 또는 12자리)."""
    if v is not None and not (_TAG4.match(v) or _TAG12.match(v)):
        raise ValueError("개체번호는 숫자 4개(끝 4자리) 또는 12개(귀표 전체)여야 합니다")
    return v


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# REQ-AI-01 /ai/route
# ---------------------------------------------------------------------------
class Cow(_Strict):
    tag4: str = Field(description="개체번호 끝 4자리")
    tag: Optional[str] = Field(default=None, description="귀표 12자리. 끝 4자리가 겹치는 소를 12자리로 구분할 때 씀. 모르면 null")
    desc: Optional[str] = Field(default=None, description="농가가 부르는 특징 (예: 큰 암소)")
    barn: Optional[str] = Field(default=None, description="축사 칸")

    _v = field_validator("tag4")(_tag4)
    _v12 = field_validator("tag")(_tag12)

    @model_validator(mode="after")
    def _tag_ends_with_tag4(self):
        if self.tag is not None and self.tag[-4:] != self.tag4:
            raise ValueError("tag 의 끝 4자리가 tag4 와 다릅니다")
        return self


class FarmContext(_Strict):
    cows: list[Cow]


class RouteRequest(_Strict):
    """이름·전화번호는 넣지 않는다."""

    text: str = Field(min_length=1, description="농가 발화 (STT 결과 또는 직접 입력)")
    today: date = Field(description="상대 날짜 변환 기준일")
    farm_context: FarmContext


class RouteItem(_Strict):
    category: Category
    value: str
    cow_tag4: Optional[str] = Field(default=None, description="소 목록에 없거나, 끝 4자리가 같은 소가 2마리 이상이라 정할 수 없으면 null")
    cow_tag: Optional[str] = Field(default=None, description="소가 정해졌고 소 목록에 12자리(tag)가 있으면 그 12자리. 아니면 null")
    vaccine_type: Optional[str] = None
    head_count: Optional[int] = Field(default=None, ge=0)

    _v = field_validator("cow_tag4")(_tag4)
    _v12 = field_validator("cow_tag")(_tag12)


class BirthFields(_Strict):
    mother_tag4: Optional[str] = Field(default=None, description="어미 소. 말한 번호 그대로 (끝 4자리 또는 12자리)")
    birth_date: Optional[date] = None
    sex: Optional[Literal["암", "수"]] = None

    _v = field_validator("mother_tag4")(_said_tag)


class DeathFields(_Strict):
    cow_tag: Optional[str] = Field(default=None, description="말한 번호 그대로 (끝 4자리 또는 12자리)")
    death_date: Optional[date] = None

    _v = field_validator("cow_tag")(_said_tag)


class MoveFields(_Strict):
    move_type: Optional[Literal["sell", "buy", "move"]] = None
    cow_tag: Optional[str] = Field(default=None, description="말한 번호 그대로 (끝 4자리 또는 12자리)")
    move_date: Optional[date] = None
    counterpart_farm_no: Optional[str] = None
    counterpart_name: Optional[str] = None

    _v = field_validator("cow_tag")(_said_tag)


REPORT_FIELDS: dict[str, type[_Strict]] = {"birth": BirthFields, "death": DeathFields, "move": MoveFields}


class RouteResponse(_Strict):
    intent: Intent
    items: list[RouteItem]
    report_type: Optional[ReportType] = None
    report_fields: Union[BirthFields, DeathFields, MoveFields, dict] = Field(
        default_factory=dict, description="report_type 에 맞는 항목. 신고가 아니면 {}"
    )
    missing: list[str] = Field(default_factory=list, description='말한 개체번호가 소 목록에 없거나, 끝 4자리만 말했는데 같은 소가 2마리 이상이면 "cow"')

    @model_validator(mode="before")
    @classmethod
    def _fields_by_type(cls, data):
        if not isinstance(data, dict):
            return data
        rt = data.get("report_type")
        rf = data.get("report_fields") or {}
        if rt is None:
            if rf:
                raise ValueError("report_type 이 없으면 report_fields 는 비어 있어야 합니다")
            return {**data, "report_fields": {}}
        model = REPORT_FIELDS[rt]
        return {**data, "report_fields": rf if isinstance(rf, model) else model.model_validate(rf)}


# ---------------------------------------------------------------------------
# REQ-AI-02 /ai/stt
# ---------------------------------------------------------------------------
class SttResponse(_Strict):
    text: str = Field(min_length=1)
    duration_sec: float = Field(ge=0, le=60)


# ---------------------------------------------------------------------------
# 오류 (REQ-AI-12 공통 형식 { code, message, detail })
# ---------------------------------------------------------------------------
class RouteErrorResponse(_Strict):
    code: Literal["LLM-FAIL"]
    message: str
    detail: Optional[dict] = None


class SttErrorResponse(_Strict):
    code: Literal["STT-FAIL"]
    message: str
    detail: Optional[dict] = None


def export(out_dir: Path) -> None:
    docs = {
        "router.json": {
            "endpoint": "/ai/route",
            "request": RouteRequest.model_json_schema(),
            "response": RouteResponse.model_json_schema(),
            "error": RouteErrorResponse.model_json_schema(),
        },
        "stt.json": {
            "endpoint": "/ai/stt",
            "request": {"type": "multipart/form-data", "file": "audio/webm | audio/mp4(m4a), 60초 이하"},
            "response": SttResponse.model_json_schema(),
            "error": SttErrorResponse.model_json_schema(),
        },
    }
    for name, doc in docs.items():
        (out_dir / name).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
