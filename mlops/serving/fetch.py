"""사진·영상 주소(Object Storage 서명 URL)에서 파일을 받아 온다."""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx


class FetchError(Exception):
    """reason: fetch_rejected | too_large | fetch_failed"""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _check_url(url: str, allowed_hosts: tuple[str, ...]) -> None:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise FetchError("fetch_rejected", "http(s) 주소만 받을 수 있음")
    if allowed_hosts and u.hostname.lower() not in allowed_hosts:
        raise FetchError("fetch_rejected", "허용하지 않은 호스트")


def fetch(client: httpx.Client, url: str, max_bytes: int, allowed_hosts: tuple[str, ...], timeout: float) -> bytes:
    """주소 → 바이트. 크기 상한을 넘으면 받는 도중에 끊는다. 주소는 로그·오류에 남기지 않는다(서명 포함)."""
    _check_url(url, allowed_hosts)
    try:
        with client.stream("GET", url, timeout=timeout, follow_redirects=False) as r:
            if 400 <= r.status_code < 500:
                raise FetchError("fetch_rejected", f"파일을 받을 수 없음 (HTTP {r.status_code})")
            if r.status_code != 200:
                raise FetchError("fetch_failed", f"저장소 응답 오류 (HTTP {r.status_code})")
            declared = r.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise FetchError("too_large", f"파일이 {max_bytes // (1024 * 1024)}MB를 넘음")
            buf = bytearray()
            for chunk in r.iter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise FetchError("too_large", f"파일이 {max_bytes // (1024 * 1024)}MB를 넘음")
            return bytes(buf)
    except httpx.HTTPError as e:
        raise FetchError("fetch_failed", f"저장소 연결 실패: {type(e).__name__}") from e


def suffix_of(url: str, default: str) -> str:
    """서명 URL 의 경로에서 확장자만 뽑는다 (쿼리 제외)."""
    s = Path(urlparse(url).path).suffix.lower()
    return s if s else default


def to_tempfile(data: bytes, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(prefix="mnm-ai-", suffix=suffix, delete=False)
    with f:
        f.write(data)
    return Path(f.name)
