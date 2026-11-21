"""REQ-AI-06 완료 기준: 답변 조합 16가지 전부 테스트.

표는 계산식 ③·④를 손으로 옮긴 정답지다 (코드 로직을 다시 쓰지 않음).
M=입(mouth), H=발굽(hoof), B=몸 옆면(flank)
"""

import pytest

from disease_branch import disease_branch, matched_symptoms, required_parts

M, H, B = "mouth", "hoof", "flank"

# (침, 물집, 혹, 사료) → (부위, 질병, 증상 수)
TABLE = {
    (0, 0, 0, 0): ([], None, 0),  # 모두 아니요 → 안내 없음
    (0, 0, 0, 1): ([B], "FMD", 1),
    (0, 0, 1, 0): ([B], "LSD", 1),
    (0, 0, 1, 1): ([B], "LSD", 2),
    (0, 1, 0, 0): ([M, H], "FMD", 1),
    (0, 1, 0, 1): ([M, H], "FMD", 2),
    (0, 1, 1, 0): ([M, H, B], "FMD", 2),
    (0, 1, 1, 1): ([M, H, B], "FMD", 3),
    (1, 0, 0, 0): ([M], "FMD", 1),
    (1, 0, 0, 1): ([M], "FMD", 2),
    (1, 0, 1, 0): ([M, B], "FMD", 2),
    (1, 0, 1, 1): ([M, B], "FMD", 3),
    (1, 1, 0, 0): ([M, H], "FMD", 2),
    (1, 1, 0, 1): ([M, H], "FMD", 3),
    (1, 1, 1, 0): ([M, H, B], "FMD", 3),
    (1, 1, 1, 1): ([M, H, B], "FMD", 4),
}


def _ans(k):
    return dict(zip(("drool", "blister", "lump", "feed"), map(bool, k)))


def test_table_is_complete():
    assert len(TABLE) == 16


@pytest.mark.parametrize("key", sorted(TABLE), ids=lambda k: "".join(map(str, k)))
def test_all_16_combinations(key):
    parts, disease, count = TABLE[key]
    a = _ans(key)
    assert [p.part for p in required_parts(a)] == parts
    r = disease_branch(a)
    assert (r.disease if r else None) == disease
    assert matched_symptoms(a).count == count


def test_matched_items_names_in_order():
    assert matched_symptoms(_ans((1, 0, 1, 0))).items == ["침 흘림", "혹"]


def test_branch_points_to_after_report_actions():
    r = disease_branch(_ans((0, 0, 1, 0)))
    assert (r.disease, r.stage) == ("LSD", "after_report")


def test_missing_answer_rejected():
    with pytest.raises(ValueError):
        required_parts({"drool": True, "blister": False, "lump": False})


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        matched_symptoms({"drool": 1, "blister": 0, "lump": 0, "feed": 0, "fever": 1})
