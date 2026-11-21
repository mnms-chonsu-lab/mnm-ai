import pytest

from farm_router.tag_normalize import normalize_tags

# 왼쪽: 휴대폰 녹음 10개 STT 실측 결과에 나온 표기 (2026-11-17)
CASES = [
    ("일 이 삼 사 번 소 구제역 백신", "1234 번 소 구제역 백신"),
    ("천 이백 삼십사 번 소 구제역", "1234 번 소 구제역"),
    ("어젯밤 5, 6, 7, 8번이 송아지", "어젯밤 5678번이 송아지"),
    ("이 사 육 팔 번이 아침부터", "2468 번이 아침부터"),
    ("1,234번이랑 5,678번 구제역", "1234번이랑 5678번 구제역"),
    ("천삼백오십칠 번이 다리를", "1357 번이 다리를"),
    ("오육칠팔번 소가 분만", "5678번 소가 분만"),
]

# 번호가 아닌 말은 그대로
UNCHANGED = ["1234번 구제역", "송아지 두 마리", "오늘 오후 3시에", "이번 주 기록", "일 년에 한 번", "두 번 소독", "C2 축사에 21357번이"]


@pytest.mark.parametrize("src,want", CASES)
def test_normalize(src, want):
    assert normalize_tags(src) == want


@pytest.mark.parametrize("src", UNCHANGED)
def test_unchanged(src):
    assert normalize_tags(src) == src
