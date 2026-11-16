"""행동 인식 입출력 형식 (명세 REQ-AI-11). mnm-dev 는 이 형식을 따른다.

`python inference/behavior_yolo11s/export_schema.py` 로 `schemas/behavior.json` 을 다시 만든다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

BehaviorLabel = Literal["stand", "lie", "eat"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BehaviorRequest(_Strict):
    video_url: str = Field(description="스마트폰 영상 주소 (mp4·mov, HEVC 포함). 처음 10초만 사용")


class BehaviorFrame(_Strict):
    t: float = Field(ge=0, description="영상 내 시각(초)")
    label: BehaviorLabel
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float] = Field(
        description="방향 보정 후 화면 기준 픽셀 [x1, y1, x2, y2]"
    )


class BehaviorSummary(_Strict):
    stand: float = Field(ge=0, le=100, description="서기 비율(%), 박스 수 기준")
    lie: float = Field(ge=0, le=100, description="눕기 비율(%)")
    eat: float = Field(ge=0, le=100, description="먹기 비율(%)")


class BehaviorResponse(_Strict):
    frames: list[BehaviorFrame] = Field(description="검출된 소 1마리(박스 1개)당 1줄")
    summary: BehaviorSummary
    main_label: Optional[BehaviorLabel] = Field(description="가장 많은 행동, 검출이 없으면 null")
    model_version: str


class BehaviorErrorResponse(_Strict):
    code: Literal["BEHAVIOR-FAIL"]
    message: str
    detail: Optional[dict] = None


def export(path: Path) -> None:
    doc = {
        "endpoint": "/ai/behavior",
        "request": BehaviorRequest.model_json_schema(),
        "response": BehaviorResponse.model_json_schema(),
        "error": BehaviorErrorResponse.model_json_schema(),
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
