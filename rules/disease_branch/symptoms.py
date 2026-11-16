"""계산식 ② 공식 증상 일치 수 · ③ 촬영 부위 결정 · ④ 질병별 안내 분기 (REQ-AI-06).

판정이 아니다. "예"로 답한 공식 질문을 세고, 정해진 표대로 부위·지침을 고를 뿐이다.

공식 증상 질문 4개 (코드 → 증상 이름은 지식베이스 값으로 교체 가능)
- drool   : 침 흘림
- blister : 물집
- feed    : 사료 남김
- lump    : 혹
"""

from __future__ import annotations

from typing import Mapping, Optional

from .schema import DiseaseBranch, MatchedSymptoms, RequiredPart

QUESTIONS = ("drool", "blister", "feed", "lump")  # 질문 순서 고정 (PM 11/16, mnm-data PR#5): 침 → 물집 → 사료 → 혹

DEFAULT_SYMPTOM_NAMES = {
    "drool": "침 흘림",
    "blister": "물집",
    "feed": "사료 남김",
    "lump": "혹",
}

PART_ORDER = ("mouth", "hoof", "flank", "udder")


def _validate(answers: Mapping[str, bool]) -> dict[str, bool]:
    missing = [q for q in QUESTIONS if q not in answers]
    extra = [k for k in answers if k not in QUESTIONS]
    if missing or extra:
        raise ValueError(f"답변 키 오류 — 빠짐: {missing}, 모르는 키: {extra}")
    return {q: bool(answers[q]) for q in QUESTIONS}


def matched_symptoms(
    answers: Mapping[str, bool], names: Optional[Mapping[str, str]] = None
) -> MatchedSymptoms:
    a = _validate(answers)
    names = names or DEFAULT_SYMPTOM_NAMES
    items = [names[q] for q in QUESTIONS if a[q]]
    return MatchedSymptoms(count=len(items), items=items)


def required_parts(answers: Mapping[str, bool]) -> list[RequiredPart]:
    """침 → 입 / 물집 → 입 + 발굽 / 혹 → 몸 옆면 / 사료만 → 몸 옆면 / 모두 아니요 → 없음."""
    a = _validate(answers)
    parts: set[str] = set()
    if a["drool"]:
        parts.add("mouth")
    if a["blister"]:
        parts |= {"mouth", "hoof"}
    if a["lump"]:
        parts.add("flank")
    only_feed = a["feed"] and not (a["drool"] or a["blister"] or a["lump"])
    if only_feed:
        parts.add("flank")
    return [RequiredPart(part=p) for p in PART_ORDER if p in parts]


def disease_branch(answers: Mapping[str, bool]) -> Optional[DiseaseBranch]:
    """어느 공식 지침의 조치 안내를 보여 줄지 고르는 분기 (판정 아님).

    모두 아니요 → None (안내 없음) / 혹 예 + 침·물집 아니요 → 럼피스킨(LSD) / 그 외 → 구제역(FMD).
    """
    a = _validate(answers)
    if not any(a.values()):
        return None
    disease = "LSD" if (a["lump"] and not a["drool"] and not a["blister"]) else "FMD"
    return DiseaseBranch(disease=disease)
