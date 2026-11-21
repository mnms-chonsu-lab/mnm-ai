"""REQ-AI-10 완료 기준: 시드 7곳 정렬 결과 확인."""

from contact_priority import contact_priority
from contact_priority.schema import FarmIndicator


def test_contact_priority_seven_farms():
    farms = [
        FarmIndicator(farm_id="F1", unresponded=False, days_left=None, head_count=42),
        FarmIndicator(farm_id="F2", unresponded=True, days_left=10, head_count=120),
        FarmIndicator(farm_id="F3", unresponded=False, days_left=2, head_count=30),
        FarmIndicator(farm_id="F4", unresponded=True, days_left=3, head_count=80),
        FarmIndicator(farm_id="F5", unresponded=False, days_left=2, head_count=15),
        FarmIndicator(farm_id="F6", unresponded=False, days_left=None, head_count=200),
        FarmIndicator(farm_id="F7", unresponded=False, days_left=-1, head_count=60),
    ]
    out = contact_priority(farms)
    assert [f.farm_id for f in out] == ["F4", "F2", "F7", "F5", "F3", "F1", "F6"]
    reasons = {f.farm_id: f.reasons for f in out}
    assert reasons["F4"] == ["noresp", "due"]
    assert reasons["F2"] == ["noresp"]
    assert reasons["F7"] == ["due"]
    assert reasons["F5"] == ["due", "small"]
    assert reasons["F1"] == ["small"]
    assert reasons["F6"] == []
    assert [f.rank for f in out] == list(range(1, 8))
