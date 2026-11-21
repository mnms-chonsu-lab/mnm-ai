# rules/routine_change_alert

평소와 다름 알림 · 미응답 판정 규칙입니다 (명세 REQ-AI-07, 계산식 ⑤⑥). 알림 문구에 쓸 값(개체, 관찰 내용, 날짜)만 돌려주고 "이상·질병" 같은 판정 표현은 만들지 않습니다.

- **담당**: @parkmyeonghun
- **쓰이는 곳**: F-S03 평소와 다름 안내, F-S16 알림, R-S02 최근 14일 일지, R-S03 미응답, A-S07 피드백 비율
- **입출력 형식**: [`schemas/rules.json`](../../schemas/rules.json) `routine_change`

## 규칙

| 함수 | 규칙 |
|---|---|
| `routine_change` ⑤ | 최근 7일(오늘 포함) 안에 같은 개체·같은 관찰 **기록**이 2회 이상 → `{cow_tag4, observation, dates}`, 아니면 `None` |
| `scan_routine_changes` | 전체 일지에서 알림 대상을 모두 찾음 |
| `is_unresponded` ⑥ | 알림 발송 후 48시간 동안 열람·피드백 없음 → `true` |
| `feedback_ratio` | 피드백(확인함·문제 없음)별 개수와 비율. 분모는 피드백을 준 알림 수 (어드민 A-S07) |

- 기록 수로 셉니다. 같은 날 2번 기록해도 2회입니다.
- 관찰 내용은 앞뒤·중간 공백을 정리한 뒤 같은 문장일 때 같은 관찰로 봅니다.
- 일지 항목 형식: `{cow_tag4, category, value, date}` (`category == "관찰"`만 사용)

## 사용법

```python
import sys; sys.path.insert(0, "rules")
from routine_change_alert import routine_change

routine_change(diary_entries, cow_tag4="1234", observation="기침을 함", today=date(2026, 11, 12))
# RoutineChange(cow_tag4='1234', observation='기침을 함', dates=[2026-11-09, 2026-11-12])
```

## 테스트

```bash
python -m pytest -q rules/routine_change_alert/tests   # 14개 (시드 1234번 알림 1건 생성 포함)
```
