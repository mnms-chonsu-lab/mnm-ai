"""계산식 ⑩ 관할 농가 연락 우선순위 (REQ-AI-10).

정렬: 알림 미응답 먼저 → 기한 남은 일수 적은 순(기한 없으면 뒤로) → 사육 두수 적은 순 → farm_id
이유 코드: noresp(미응답), due(기한 7일 이내), small(50두 미만)
"""

from __future__ import annotations

from typing import Iterable

from .schema import FarmIndicator, PrioritizedFarm

DUE_WITHIN_DAYS = 7
SMALL_HEAD = 50


def _reasons(f: FarmIndicator) -> list[str]:
    r = []
    if f.unresponded:
        r.append("noresp")
    if f.days_left is not None and f.days_left <= DUE_WITHIN_DAYS:
        r.append("due")
    if f.head_count < SMALL_HEAD:
        r.append("small")
    return r


def contact_priority(farms: Iterable[FarmIndicator]) -> list[PrioritizedFarm]:
    big = 10**9
    ordered = sorted(
        farms,
        key=lambda f: (
            not f.unresponded,
            f.days_left if f.days_left is not None else big,
            f.head_count,
            f.farm_id,
        ),
    )
    return [
        PrioritizedFarm(farm_id=f.farm_id, rank=i + 1, reasons=_reasons(f))
        for i, f in enumerate(ordered)
    ]
