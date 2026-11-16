# rules/deadline

신고 기한·귀표 부착 기한 계산 규칙입니다 (명세 REQ-AI-04, 계산식 ①). LLM을 쓰지 않습니다.

- **담당**: @parkmyeonghun
- **쓰이는 곳**: F-S01 오늘 할 일, F-S03·F-S04 신고 기한 안내, F-S07, F-S16
- **입출력 형식**: [`schemas/rules.json`](../../schemas/rules.json) `deadline`

## 규칙

| 값 | 계산 | 근거 |
|---|---|---|
| `report_due` | 발생일 다음 날부터 세어 5번째 날. 토요일·일요일·공휴일은 세지 않음 | 이력제 시행규칙: "출생ㆍ폐사한 날부터 5일 이내(공휴일과 토요일은 제외한다)" / 관공서 공휴일 규정 제2조 1호 "일요일" / 민법 제157조 초일 불산입 |
| `ear_tag_due` | 출생 신고일 + 30일 (출생만, 신고일을 모르면 오늘 신고로 계산) | 명세 계산식 ① |
| `d_day` | `report_due` − 오늘 (지나면 음수) | 명세 계산식 ① |

- 공휴일·대체공휴일 목록은 인교님 테이블(REQ-DATA-07)을 `holidays` 인자로 받습니다.

## 사용법

```python
import sys; sys.path.insert(0, "rules")
from datetime import date
from deadline import compute_deadline

compute_deadline("birth", date(2026, 11, 12), today=date(2026, 11, 12), holidays=holiday_list)
# DeadlineResponse(report_due=2026-11-19, ear_tag_due=2026-12-12, d_day=7)
```

## 테스트

```bash
python -m pytest -q rules/deadline/tests   # 31개 (토요일·공휴일 낀 경우 21건 포함)
```
