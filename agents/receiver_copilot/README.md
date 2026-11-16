# agents/receiver_copilot

접수자(공무원)가 보는 한 장 요약 카드와 인터넷 신고서 6개 항목을 만듭니다.

- **담당**: @parkmyeonghun
- **명세**: REQ-AI-09 `/ai/summary` (화면 R-S02, R-P02, F-S13, F-S14)
- **입출력 형식**: `schemas/summary.json` (`schema.py`에서 생성)

## 동작

서버가 승인된 소 상태 기록 ID로 구조화 데이터를 조회해 넘기면, 모든 값을 규칙으로 채웁니다. LLM을 부르지 않습니다. 그래서 숫자·날짜가 바뀌지 않고, 원본 기록과 항상 같습니다.

| 출력 | 채우는 방법 | 근거 |
| --- | --- | --- |
| `card.cow` | 입력 그대로 | |
| `card.matched_symptoms` | "예"로 답한 공식 증상 이름 (`rules/disease_branch` 계산식 ②) | REQ-AI-06 |
| `card.diary_related` | 같은 개체, 발견일 포함 최근 7일, 최근순 | F-S11 |
| `card.photos` | `/ai/part-check` 결과 `{ part, match }` | REQ-AI-03 |
| `card.vaccine` · `disinfection` | 일지에서 분류(접종·소독)별 가장 최근 1건 + 며칠 전 | R-S02 |
| `card.visit` | 일지 분류 `출입` 중 가장 최근 1건 + 며칠 전 | R-S02 |
| `form6` · `form6_text` | 시행규칙 제13조 6개 항목. 질병명 칸은 `"발견 당시 상태: …"` 고정 (`rules/report_state_machine`) | REQ-AI-05 |

- 발견일 뒤의 기록은 쓰지 않습니다.
- 신고서 2호 두수는 증상이 있거나 죽은 소의 수입니다. 요청에 `farm_total_head`(농장 정보의 전체 사육 두수)가 있으면 괄호로 붙입니다. 예: `한우 1두 (전체 42두)`.
- 외부 출입은 일지 분류 `출입`(9개 분류 확정, `mnm-data` PR#6)에서 가져옵니다. 별도 `visits` 입력은 없습니다.
- 의심 신고 항목에 빈칸이 있으면 `LLM-FAIL`(`detail.reason = "report_fields"`)을 반환합니다.

## 실행

저장소 루트와 `agents/`가 `PYTHONPATH`에 있어야 합니다. (`rules/`를 불러옴)

```python
from receiver_copilot import summarize, SummaryRequest
out = summarize(SummaryRequest.model_validate(payload))   # SummaryResponse
```

```bash
python -m pytest -q agents/receiver_copilot   # 6개
```
