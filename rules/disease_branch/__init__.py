"""증상 일치 수 · 촬영 부위 · 질병 분기 (명세 계산식 ②③④). SOP 기준 분기만 수행, 판정 아님."""

from .symptoms import disease_branch, matched_symptoms, required_parts

__all__ = ["matched_symptoms", "required_parts", "disease_branch"]
