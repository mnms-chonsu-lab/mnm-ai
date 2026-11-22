"""최근 5분 AI 호출 수·실패 수 (계산식 ⑪ AI 오류율의 알람용, `/health` 의 `calls_5m`).

- 모델 처리까지 간 호출만 센다 (`record()` 에서 더함). 요청 형식 오류(422 bad_request)·토큰 오류(401)는 세지 않는다.
- 실패 = 기록의 `error` 가 있는 호출 (서버 시간 초과 504 포함).
- 메모리에만 둔다. 서버를 재시작하면 0부터 다시 센다 (24시간 오류율은 DB `inference_log` 로 계산).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

WINDOW_SEC = 300  # 명세 ⑪ "알람은 5분 창"


class CallWindow:
    def __init__(self, seconds: int = WINDOW_SEC):
        self.seconds = seconds
        self._calls: deque[tuple[float, bool]] = deque()
        self._lock = threading.Lock()

    def add(self, failed: bool, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._calls.append((now, failed))
            self._trim(now)

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._trim(now)
            calls = len(self._calls)
            failures = sum(1 for _, f in self._calls if f)
        return {"window_sec": self.seconds, "calls": calls, "failures": failures,
                "error_rate": round(failures / calls, 4) if calls else None}

    def _trim(self, now: float) -> None:
        while self._calls and now - self._calls[0][0] > self.seconds:
            self._calls.popleft()
