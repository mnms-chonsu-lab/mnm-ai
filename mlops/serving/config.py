"""서빙 설정. 모두 환경변수로 받고, 값은 코드에 두지 않는다."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _hosts(raw: str | None) -> tuple[str, ...]:
    return tuple(h.strip().lower() for h in (raw or "").split(",") if h.strip())


@dataclass(frozen=True)
class Settings:
    # 서버 간 공유 토큰. 비어 있으면 /ai/* 를 모두 거절한다 (닫힌 상태로 시작)
    internal_token: str = ""
    # 서버 시간 제한 (mnm-dev docs/api/ai-internal.yaml). 일반 8초, 부위 확인 10초
    timeout_sec: float = 8.0
    part_check_timeout_sec: float = 10.0
    # 사진·영상을 받아 올 수 있는 호스트. 비우면 제한 없음 (로컬 개발용)
    fetch_allowed_hosts: tuple[str, ...] = field(default_factory=tuple)
    # 받아 오는 파일 크기 상한 (제안값, 실측으로 조정)
    max_image_bytes: int = 15 * 1024 * 1024
    max_video_bytes: int = 200 * 1024 * 1024
    # 시작할 때 불러올 모델. 0이면 건너뛴다 (메모리 확인·로컬 개발용)
    load_part_check: bool = True
    load_behavior: bool = True
    # 방역 안내 지식베이스 (JSON 배열 또는 JSONL). 비우면 guide 는 503 not_loaded
    guide_kb_path: str = ""
    guide_embed_cache: str = ""
    # AI 호출 기록 DB (`inference_log` INSERT 전용 계정). 비우면 기록하지 않음 (로그 줄만 남김)
    log_db_url: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            internal_token=os.getenv("AI_INTERNAL_TOKEN", ""),
            timeout_sec=_float("AI_TIMEOUT_SEC", 8.0),
            part_check_timeout_sec=_float("AI_PART_CHECK_TIMEOUT_SEC", 10.0),
            fetch_allowed_hosts=_hosts(os.getenv("AI_FETCH_ALLOWED_HOSTS", "kr.object.ncloudstorage.com")),
            max_image_bytes=_int("AI_MAX_IMAGE_BYTES", 15 * 1024 * 1024),
            max_video_bytes=_int("AI_MAX_VIDEO_BYTES", 200 * 1024 * 1024),
            load_part_check=os.getenv("AI_LOAD_PART_CHECK", "1") not in ("0", "false", "off"),
            load_behavior=os.getenv("AI_LOAD_BEHAVIOR", "1") not in ("0", "false", "off"),
            guide_kb_path=os.getenv("GUIDE_KB_PATH", ""),
            guide_embed_cache=os.getenv("GUIDE_EMBED_CACHE", ""),
            log_db_url=os.getenv("AI_LOG_DB_URL", ""),
        )

    def timeout_for(self, feature: str) -> float:
        return self.part_check_timeout_sec if feature == "part_check" else self.timeout_sec
