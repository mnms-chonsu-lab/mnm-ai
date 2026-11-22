"""AI 호출 기록을 DB `inference_log` 에 남긴다 (A-S07 AI 호출 로그 · 계산식 ⑪ AI 오류율의 원천).

- 요청을 붙잡지 않는다: 응답 경로에서는 큐에 넣기만 하고, 쓰기는 백그라운드 스레드 1개가 한다.
- DB 가 죽어도 AI 응답은 그대로 나간다. 못 쓴 기록은 버리고 개수만 센다 (`/health` 에 표시).
- 원문(발화·사진·영상·주소)은 넣지 않는다. `AiCallLog` 의 요약·오류 코드만 쓴다.
- 테이블 정의는 mnm-data `migrations/0001_init_operational.sql` + `0004_inference_log_http_status.sql`(`http_status`) 기준.
  컬럼 길이에 맞춰 자른다.
- `http_status`: AI 서버 응답 상태. 24시간 오류율(계산식 ⑪)은 `http_status IN (503, 504)`만 실패로 센다 (mnm-ai#14).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger("mnm_ai.inference_log")

INSERT_SQL = (
    "INSERT INTO inference_log (feature, model_version, input_kind, result_summary, latency_ms, error, http_status, at) "
    "VALUES (%(feature)s, %(model_version)s, %(input_kind)s, %(result_summary)s, %(latency_ms)s, %(error)s, "
    "%(http_status)s, %(at)s)"
)
# 컬럼 길이 (mnm-data migrations): feature 50 · model_version 50 · input_kind 30 · result_summary 500
LIMITS = {"feature": 50, "model_version": 50, "input_kind": 30, "result_summary": 500}
QUEUE_MAX = 1000  # 쌓이는 최대 건수 (제안값, 실측으로 조정). 넘으면 버린다
BATCH_MAX = 50
RETRY_SEC = (1, 2, 5, 10, 30)  # DB 연결 실패 뒤 기다리는 시간


def to_row(log: dict[str, Any], at: Optional[datetime] = None) -> dict[str, Any]:
    """`AiCallLog` 형태(dict) → INSERT 값. 모르는 키(request_id 등)는 버린다."""
    row: dict[str, Any] = {
        "feature": log.get("feature"),
        "model_version": log.get("model_version"),
        "input_kind": log.get("input_kind"),
        "result_summary": log.get("result_summary"),
        "latency_ms": log.get("latency_ms"),
        "error": log.get("error"),
        "http_status": log.get("http_status"),
        "at": at or datetime.now(timezone.utc),
    }
    for key, n in LIMITS.items():
        if isinstance(row[key], str):
            row[key] = row[key][:n]
    return row


class InferenceLogWriter:
    """큐 + 백그라운드 스레드. `dsn` 이 비면 꺼진 상태로 아무것도 하지 않는다."""

    def __init__(self, dsn: str = "", connect: Optional[Callable[[str], Any]] = None,
                 queue_max: int = QUEUE_MAX, retry_sec: tuple[float, ...] = RETRY_SEC):
        self.dsn = dsn
        self._connect = connect or _psycopg_connect
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._retry_sec = retry_sec
        self._stop = threading.Event()
        self._conn = None
        self.written = 0
        self.dropped = 0
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, name="inference-log", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    def put(self, log: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(to_row(log))
        except queue.Full:
            self.dropped += 1

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"enabled": self.enabled}
        if self.enabled:
            out.update(written=self.written, dropped=self.dropped, queued=self._q.qsize())
            if self.last_error:
                out["last_error"] = self.last_error
        return out

    def flush(self, timeout: float = 5.0) -> bool:
        """큐가 빌 때까지 기다린다 (테스트·종료용)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    def close(self, timeout: float = 5.0) -> None:
        if not self.enabled:
            return
        self.flush(timeout)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self._close_conn()

    # ------------------------------------------------------------------ 내부
    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < BATCH_MAX:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                self._write(batch)
                self.written += len(batch)
                failures = 0
            except Exception as e:  # noqa: BLE001 — DB 오류로 서버가 멈추면 안 된다
                self.dropped += len(batch)
                # 오류 문구에 접속 정보가 섞일 수 있어 클래스 이름만 남긴다
                self.last_error = type(e).__name__
                logger.warning("inference_log 기록 실패 %d건 (%s)", len(batch), type(e).__name__)
                self._close_conn()
                wait = self._retry_sec[min(failures, len(self._retry_sec) - 1)] if self._retry_sec else 0
                failures += 1
                self._stop.wait(wait)
            finally:
                for _ in batch:
                    self._q.task_done()

    def _write(self, rows: list[dict[str, Any]]) -> None:
        if self._conn is None:
            self._conn = self._connect(self.dsn)
        with self._conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
        self._conn.commit()

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


def _psycopg_connect(dsn: str):
    import psycopg  # 기록을 켤 때만 필요

    return psycopg.connect(dsn, connect_timeout=5, application_name="mnm-ai")
