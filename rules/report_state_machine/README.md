# rules/report_state_machine

신고서 상태 머신입니다 (명세 REQ-AI-05). 채워진 값을 검사해 빈칸을 찾고, 빈칸 중 첫 번째 하나만 해요체로 묻습니다. LLM을 쓰지 않습니다.

- **담당**: @parkmyeonghun
- **쓰이는 곳**: F-S04 신고서 작성, F-S05 직접 제출, F-P02·R-P02 신고서 6개 항목
- **입출력 형식**: [`schemas/rules.json`](../../schemas/rules.json) `report_state`, `form6`

## 신고서 항목

**출생 신고 (`birth`)**

| 항목 | 필수 | 검사 |
|---|---|---|
| `mother_tag4` 어미 소 귀표 끝 4자리 | O | 숫자 4개 |
| `birth_date` 출생일 | O | 날짜, 오늘 이후 불가 |
| `sex` 성별 | O | 암·암컷·암소 → 암 / 수·수컷·수소 → 수 |
| `breed` 품종 | O | 빈 문자열 불가 |
| `calf_tag` 송아지 귀표 | 선택 | 번호는 출생 신고 후 위탁기관(지역축협 등)이 부여 (이력제 매뉴얼) |

**폐사 신고 (`death`)**

| 항목 | 필수 | 검사 |
|---|---|---|
| `cow_tag` 소 귀표 끝 4자리 | O | 숫자 4개 |
| `death_date` 폐사일 | O | 날짜, 오늘 이후 불가 |

**양도 (`sell`) · 양수 (`buy`) · 이동 (`move`) 신고**

| 항목 | 필수 | 검사 |
|---|---|---|
| `cow_tag` 소 귀표 끝 4자리 | O | 숫자 4개 |
| `move_date` 보낸 날·받은 날 | O | 날짜, 오늘 이후 불가 |
| `counterpart_farm_no` 상대 농장식별번호 | 양도·양수 O, 이동 선택 | 빈 문자열 불가 |
| `counterpart_name` 상대 이름 | 선택 | 빈 문자열 불가 |

- 칸 이름·필수 여부·질문은 `mnm-dev` `FarmerRules`(화면설계서 F-S04Q)와 같습니다.
- 라우터(`agents/farm_router`)의 `move_type`(sell·buy·move)을 그대로 `report_kind`로 씁니다.

**전염병 의심 신고 (`suspect`)** — 가축전염병 예방법 시행규칙 제13조 제1항

| 항목 | 필수 | 6개 항목 |
|---|---|---|
| `owner_and_location` | O | 1. 소유자등의 성명 및 사육장소 또는 발견장소 |
| `species`, `head_count` | O | 2. 신고대상 가축의 종류 및 두수 (두수는 1 이상 정수) |
| `symptoms` | O | 3. 질병명 칸 → 항상 "발견 당시 상태: …" (추정 병명 금지) |
| `found_date` | O | 4. 죽은 연월일 (불분명하면 발견 연월일) |
| `reporter` | O | 5. 신고자의 성명 및 주소 |
| `other` | 선택 | 6. 그 밖에 신고에 관하여 필요한 사항 |

- 복사 텍스트(`form6_copy_text`)의 칸 이름은 `mnm-data` `kind=form_field` 행(시행규칙 제13조) 원문 그대로입니다.
- 형식이 틀린 값은 빈칸으로 봅니다 (잘못된 값을 그대로 제출하지 않음).
- `build_form6`은 빈칸이 없는 의심 신고 상태를 6개 항목으로 바꾸고, `form6_copy_text`는 인터넷 신고서용 복사 텍스트를 만듭니다.

## 사용법

```python
import sys; sys.path.insert(0, "rules")
from datetime import date
from report_state_machine import report_state

r = report_state("birth", {"mother_tag4": "1234"}, today=date(2026, 11, 12))
# missing=['birth_date', 'sex', 'breed'], next_question='송아지가 태어난 날이 언제예요?', can_submit=False
```

## 테스트

```bash
python -m pytest -q rules/report_state_machine/tests   # 출생·의심·폐사·양도·양수·이동 빈칸 조합 포함
```
