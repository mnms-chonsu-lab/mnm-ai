# common

AI 기능이 함께 쓰는 오류 코드·호출 로그·개인정보 가림입니다. (명세 REQ-AI-12)

- **담당**: @parkmyeonghun
- **받는 사람**: 보경(서버 연동), 인서(모니터링 A-S07)

## 오류 코드

실패 응답 형식은 `{ code, message, detail }` 하나입니다.

| 기능 (`feature`) | 엔드포인트 | 오류 코드 | 서버 대체 경로 (명세) |
| --- | --- | --- | --- |
| `stt` | `/ai/stt` | `STT-FAIL` | 글로 쓰기 화면 (F-S02T) |
| `route` | `/ai/route` | `LLM-FAIL` | 선택형 입력 화면 (F-S03E) |
| `part_check` | `/ai/part-check` | `PART-FAIL` | 판정 없이 진행 |
| `behavior` | `/ai/behavior` | `BEHAVIOR-FAIL` | 영상만 저장 후 진행 |
| `guide` | `/ai/guide` | `RAG-NONE` | [바로 신고]·담당 축협 연결 |
| `summary` | `/ai/summary` | `LLM-FAIL` | 명세에 없음 → 보경님 확인 |

타임아웃 기준값은 `TIMEOUT_SEC`에 있습니다. 일반 8초, 부위 검증 1.5초입니다.

## 호출 로그

호출 1번에 1줄씩 남습니다. 로거 이름은 `mnm_ai.calls`이고 JSON 형식입니다.

| 필드 | 예 |
| --- | --- |
| `feature` | `route` |
| `model_version` | `gpt-5.4-mini+route-v1` |
| `input_kind` | `text`, `audio/webm`, `image/jpeg`, `video/mp4` |
| `result_summary` | `intent=history_report items=1 report=birth missing=1` |
| `latency_ms` | `1830` |
| `error` | `null` 또는 오류 코드 |

- 발화 원문·사진·영상은 로그에 넣지 않습니다. `result_summary`는 코드와 개수로만 만듭니다.
- 전화번호는 `mask_phone()`으로 `[전화번호]`로 바꿉니다. 라우터 프롬프트와 로그 메시지에 적용합니다.
- 이름은 입력 형식에서 받지 않습니다. 라우터 입력은 `tag4`·`desc`·`barn`만 받습니다.

## 사용

```python
from common import call_ai
from farm_router import route

result, log = call_ai(
    "route", MODEL_VERSION, "text",
    lambda: route(req),
    summarize=lambda r: f"intent={r.intent} items={len(r.items)} report={r.report_type} missing={len(r.missing)}",
)
# result: RouteResponse 또는 AiErrorResponse(code="LLM-FAIL", ...). 예외는 밖으로 나오지 않음
```

## 오류 강제 발생 (완료 기준 확인용)

```bash
AI_FORCE_ERROR=stt,route,part_check,behavior   # 기능 이름 또는 오류 코드, all 가능
```

설정하면 해당 기능은 모델을 부르지 않고 바로 오류 코드를 돌려줍니다. 서버가 대체 경로로 넘어가는지 확인할 때 씁니다.

| 값 | guide 결과 |
| --- | --- |
| `guide`, `RAG-NONE`, `all` | `RAG-NONE` (근거 없음, 404 경로) |
| `LLM-FAIL` | `LLM-FAIL` (장애, 503 재시도 경로). route·summary도 같이 걸림 |

## 테스트

```bash
python -m pytest -q common   # 20개
```
