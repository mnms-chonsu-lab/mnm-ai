"""부위 일치 검증 입출력 형식 (명세 REQ-AI-03). mnm-dev 는 이 형식을 따른다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Part = Literal["mouth", "udder", "hoof", "flank"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PartCheckRequest(_Strict):
    image_url: str = Field(description="사진 주소 (jpg·png·heic)")
    requested_part: Part = Field(description="앱이 촬영을 요청한 부위")


class Quality(_Strict):
    dark: bool = Field(description="평균 밝기 < 60/255")
    blurry: bool = Field(description="선명도(라플라시안 분산) < 기준값")
    brightness: float = Field(ge=0, le=255, description="평균 밝기 (0–255)")


class PartCheckResponse(_Strict):
    requested_part: Part
    predicted_part: Literal["mouth", "udder", "hoof", "flank", "none"] = Field(
        description='모델이 고른 부위. "none" = 소 부위가 아님·알아볼 수 없음 (OpenAI Vision 만 냄). 로그·어드민·접수자용, 농가 화면에 노출하지 않음'
    )
    match: bool = Field(description='predicted_part == requested_part 그리고 score >= 기준값. "none" 이면 false')
    score: float = Field(ge=0, le=1, description="predicted_part 의 점수. SigLIP 2 는 softmax(100 × 코사인), fallback = true 면 Vision 이 매긴 확신도")
    quality: Quality
    model_version: str
    fallback: bool = Field(default=False, description="OpenAI Vision 결과 여부 (SigLIP 2 실패·시간 초과 때 대체, 또는 SigLIP 2 가 다른 부위라고 해서 재확인)")


class PartCheckErrorResponse(_Strict):
    code: Literal["PART-FAIL"]
    message: str
    detail: Optional[dict] = None


def export(path: Path) -> None:
    doc = {
        "endpoint": "/ai/part-check",
        "request": PartCheckRequest.model_json_schema(),
        "response": PartCheckResponse.model_json_schema(),
        "error": PartCheckErrorResponse.model_json_schema(),
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
