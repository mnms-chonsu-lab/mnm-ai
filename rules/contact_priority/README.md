# rules/contact_priority

관할 농가 연락 우선순위 규칙입니다 (명세 REQ-AI-10, 계산식 ⑩).

- **담당**: @parkmyeonghun
- **쓰이는 곳**: R-S03 관할 농가 표 + 연락 순서
- **입출력 형식**: [`schemas/rules.json`](../../schemas/rules.json) `contact_priority`

## 규칙

| 순서 | 정렬 기준 |
|---|---|
| 1 | 알림 미응답 농가 먼저 |
| 2 | 신고 기한까지 남은 일수가 적은 순 (기한 없으면 뒤로) |
| 3 | 사육 두수가 적은 순 |
| 4 | `farm_id` (동점 정리) |

| 이유 코드 | 조건 |
|---|---|
| `noresp` | 알림 미응답 |
| `due` | 기한 7일 이내 |
| `small` | 50두 미만 |

## 사용법

```python
import sys; sys.path.insert(0, "rules")
from contact_priority import contact_priority
from contact_priority.schema import FarmIndicator

contact_priority([FarmIndicator(farm_id="F1", unresponded=True, days_left=3, head_count=42), ...])
# [PrioritizedFarm(farm_id='F1', rank=1, reasons=['noresp', 'due', 'small']), ...]
```

## 테스트

```bash
python -m pytest -q rules/contact_priority/tests   # 시드 7곳 정렬
```
