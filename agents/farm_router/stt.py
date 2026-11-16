"""음성 인식 (명세 REQ-AI-02).

농가 음성(webm·m4a, 60초 이하)을 문장으로 바꾼다. 공급자: OpenAI.
축산 용어는 전사 요청에 키워드·프롬프트로 넣어 보정한다.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from .schema import SttErrorResponse, SttResponse

STT_MODEL = os.getenv("STT_MODEL", "gpt-transcribe")
TIMEOUT_SEC = 8.0  # 명세 REQ-AI-12 일반 타임아웃
MAX_SEC = 60.0  # 명세 REQ-AI-02 입력 60초 이하
ALLOWED_EXT = {".webm", ".m4a"}
SILENCE_MAX_DB = -50.0  # 최대 음량이 이보다 낮으면 무음 (제안값, 실측으로 조정)
MODEL_VERSION = f"{STT_MODEL}+terms-v1"

# 명세에 적힌 축산 용어 + 농가 기록에 자주 나오는 말
LIVESTOCK_TERMS = [
    "소독", "접종", "구제역", "럼피스킨", "귀표", "축사", "송아지", "사료",
    "한우", "암소", "수소", "분만", "폐사", "출하", "백신", "침흘림", "수포", "절뚝",
]
PROMPT = "한우 농가가 축사 일과를 기록하는 말입니다. " + ", ".join(LIVESTOCK_TERMS) + " 같은 축산 용어가 나옵니다."


class SttError(Exception):
    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_response(self) -> SttErrorResponse:
        return SttErrorResponse(code="STT-FAIL", message=self.message, detail=self.detail)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=20)


def probe_duration(path: Path) -> float:
    """길이(초). 브라우저 녹음 webm 은 헤더에 길이가 없을 수 있어 끝까지 읽어 계산한다."""
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    try:
        return float(r.stdout.strip())
    except ValueError:
        pass
    r = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-f", "null", "-"])
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not times:
        raise SttError("음성 파일을 읽을 수 없음", {"reason": "unreadable"})
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def max_volume_db(path: Path) -> Optional[float]:
    r = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"max_volume:\s*(-?inf|-?\d+(?:\.\d+)?) dB", r.stderr)
    if not m:
        return None
    return float("-inf") if "inf" in m.group(1) else float(m.group(1))


def transcribe(path: str | Path, client=None, model: str = STT_MODEL, timeout_sec: float = TIMEOUT_SEC) -> SttResponse:
    """음성 → { text, duration_sec }. 무음·인식 실패·형식 오류는 SttError(STT-FAIL)."""
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXT:
        raise SttError("지원하지 않는 형식", {"reason": "format", "ext": path.suffix})
    if not path.exists() or path.stat().st_size == 0:
        raise SttError("빈 파일", {"reason": "empty"})

    duration = probe_duration(path)
    if duration > MAX_SEC:
        raise SttError("60초 초과", {"reason": "too_long", "duration_sec": round(duration, 2)})
    vol = max_volume_db(path)
    if vol is not None and vol < SILENCE_MAX_DB:
        raise SttError("무음", {"reason": "silence", "max_volume_db": vol})

    if client is None:
        client = make_client()
    kwargs = dict(model=model, language="ko", prompt=PROMPT, response_format="json")
    if model.startswith("gpt-transcribe"):
        kwargs["keywords"] = LIVESTOCK_TERMS

    t0 = time.perf_counter()
    try:
        with path.open("rb") as f:
            res = client.with_options(timeout=timeout_sec, max_retries=0).audio.transcriptions.create(file=f, **kwargs)
    except Exception as e:
        raise SttError("인식 호출 실패", {"reason": "api", "type": type(e).__name__}) from e
    elapsed = time.perf_counter() - t0
    if elapsed > timeout_sec:
        raise SttError("시간 초과", {"reason": "timeout", "elapsed_sec": round(elapsed, 2)})

    text = (getattr(res, "text", None) or "").strip()
    if not text:
        raise SttError("인식 결과 없음", {"reason": "no_text"})
    return SttResponse(text=text, duration_sec=round(duration, 2))


def make_client():
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SttError("OPENAI_API_KEY 가 설정되지 않음", {"reason": "config"})
    return OpenAI()
