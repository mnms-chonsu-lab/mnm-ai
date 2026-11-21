"""개체번호 표기 정리 (라우터 입력 전).

STT 가 같은 번호를 여러 형식으로 내보낸다 (휴대폰 녹음 10개 실측, 2026-11-17).
  "5, 6, 7, 8번" · "1,234번" · "일 이 삼 사 번" · "이사육팔번" · "천 이백 삼십사 번"
'번' 앞에 붙은 숫자 표현만 1234 형식으로 바꾼다. 그 밖의 문장은 건드리지 않는다.
"""

from __future__ import annotations

import re

_DIGIT = {"공": 0, "영": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
_UNIT = {"십": 10, "백": 100, "천": 1000}
_D = "공영일이삼사오육륙칠팔구"

# 1) 숫자 사이 쉼표·공백: "5, 6, 7, 8번", "1,234번", "12 34번"
_NUM_SPACED = re.compile(r"(?<![\d,])(\d(?:[,\s]*\d){2,11})(?=\s*번)")
# 2) 한 자리씩 읽은 한글: "일 이 삼 사 번", "이사육팔번" (3~12자리)
_KO_DIGITS = re.compile(rf"(?<![가-힣])([{_D}](?:\s?[{_D}]){{2,11}})(?=\s*번)")
# 3) 자릿수로 읽은 한글: "천 이백 삼십사 번", "천삼백오십칠 번" (천·백·십 중 하나 이상 포함)
_KO_NUMBER = re.compile(rf"(?<![가-힣])((?:[{_D}]?\s?[천백십]\s?)+[{_D}]?)(?=\s*번)")


def _ko_number(s: str) -> int | None:
    s = s.replace(" ", "")
    total, cur = 0, None
    for ch in s:
        if ch in _DIGIT:
            if cur is not None:
                return None  # 숫자 두 개가 연달아 오면 자릿수 읽기가 아님
            cur = _DIGIT[ch]
        elif ch in _UNIT:
            total += (1 if cur is None else cur) * _UNIT[ch]
            cur = None
        else:
            return None
    return total + (cur or 0)


def normalize_tags(text: str) -> str:
    text = _NUM_SPACED.sub(lambda m: re.sub(r"[,\s]", "", m.group(1)), text)
    text = _KO_DIGITS.sub(lambda m: "".join(str(_DIGIT[c]) for c in m.group(1) if c != " "), text)

    def _num(m):
        n = _ko_number(m.group(1))
        return str(n) if n is not None and n >= 10 else m.group(1)

    return _KO_NUMBER.sub(_num, text)
