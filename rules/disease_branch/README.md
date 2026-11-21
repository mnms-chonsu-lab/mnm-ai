# rules/disease_branch

SOP 기준 분기 규칙입니다 (명세 REQ-AI-06, 계산식 ②③④). 농가의 공식 증상 질문 답변으로 **어느 공식 지침의 안내를 보여 줄지, 어느 부위를 찍을지**만 고릅니다. 판정·확률·자동 신고로 읽히는 로직과 문구는 넣지 않습니다.

- **담당**: @parkmyeonghun
- **쓰이는 곳**: F-S10 촬영 부위, F-S11 공식 증상 일치, F-S14 질병별 지금 할 일, R-S02 관련 지침
- **입출력 형식**: [`schemas/rules.json`](../../schemas/rules.json) `disease_branch`

## 입력: 공식 증상 질문 4개

| 코드 | 질문 |
|---|---|
| `drool` | 침 흘림 |
| `blister` | 물집 |
| `feed` | 사료 남김 |
| `lump` | 혹 |

- 질문 순서는 침 → 물집 → 사료 → 혹 (PM 확정, `mnm-data` PR#5). 증상 이름은 `mnm-frontend` 화면 표기와 같습니다.

## 규칙

| 함수 | 규칙 |
|---|---|
| `matched_symptoms` ② | "예"로 답한 질문 수와 증상 이름 |
| `required_parts` ③ | 침 → 입 / 물집 → 입 + 발굽 / 혹 → 몸 옆면 / 사료만 → 몸 옆면 / 모두 아니요 → 촬영 없음 |
| `disease_branch` ④ | 모두 아니요 → 안내 없음(`None`) / 혹 예 + 침·물집 아니요 → 럼피스킨(`LSD`) / 그 외 → 구제역(`FMD`) |

| 증상 | 공식 자료 |
|---|---|
| 침 흘림·물집·사료 거부 | 농촌진흥청: 한우·젖소 구제역 주요증상 "입, 혀, 발굽 또는 젖꼭지 등에 물집", "사료섭취 거부, 거품섞인 심한 침흘림" |
| 혹 | 농림축산식품부: 럼피스킨은 "피부 및 내부점막에 혹덩어리(Lumpy)를 형성" |

- 코드 값은 `mnm-data` `biosecurity_rule`과 같습니다. 부위 = `part_code`(`mouth`·`hoof`·`flank`·`udder`), 질병 = `disease`(`FMD`·`LSD`).
- 촬영 안내 문구: `kind=photo_guide`이고 `part_code`가 같은 행.
- 조치 문구: `kind=action`, `disease`·`stage=after_report`가 같은 행을 `display_order` 순으로. `/ai/guide`를 `query` 없이 부르면 이 목록이 나옵니다.

## 사용법

```python
import sys; sys.path.insert(0, "rules")
from disease_branch import disease_branch, required_parts

answers = {"drool": False, "blister": True, "lump": False, "feed": False}
required_parts(answers)   # [mouth, hoof]
disease_branch(answers)   # DiseaseBranch(disease='FMD', stage='after_report')
```

## 테스트

```bash
python -m pytest -q rules/disease_branch/tests   # 22개 (답변 조합 16가지 전부)
```
