# agents/farm_router

농가가 말한 한 문장을 일지 항목·신고서 항목으로 정리합니다. 음성은 먼저 문장으로 바꿉니다.

- **담당**: @parkmyeonghun
- **명세**: REQ-AI-01 농가 라우터 `/ai/route`, REQ-AI-02 음성 인식 `/ai/stt`
- **입출력 형식**: `schemas/router.json`, `schemas/stt.json` (`schema.py`에서 생성)

```
음성(webm·m4a) ──/ai/stt──▶ { text, duration_sec } ──/ai/route──▶ { intent, items, report_type, report_fields, missing }
                                                                         └─ report_fields → rules/report_state_machine
```

---

## 1. 음성 인식 `/ai/stt` (REQ-AI-02)

| 항목 | 내용 |
| --- | --- |
| 공급자·모델 | OpenAI `gpt-transcribe` (`STT_MODEL`로 변경) |
| 입력 | webm·m4a, 60초 이하 |
| 출력 | `{ text, duration_sec }` |
| 축산 용어 보정 | 전사 요청에 `keywords`(용어 사전) + `prompt`(문맥) + `language="ko"` |
| 용어 사전 | 소독, 접종, 구제역, 럼피스킨, 귀표, 축사, 송아지, 사료 (명세) + 한우, 암소, 수소, 분만, 폐사, 출하, 백신, 침흘림, 수포, 절뚝 |
| 타임아웃 | 8초 (REQ-AI-12 일반) |

`STT-FAIL` 반환 조건 (`detail.reason`):

| reason | 조건 |
| --- | --- |
| `format` | 확장자가 webm·m4a 가 아님 |
| `empty` / `unreadable` | 빈 파일, 디코딩 불가 |
| `too_long` | 60초 초과 |
| `silence` | 최대 음량 < -50 dB (제안값, 실측으로 조정) |
| `api` / `timeout` | 호출 실패, 8초 초과 |
| `no_text` | 인식 결과가 빈 문자열 |

- 브라우저 녹음 webm은 헤더에 길이가 없을 수 있습니다. 이때는 `ffmpeg`로 끝까지 읽어 길이를 계산합니다.
- 서버 이미지에 `ffmpeg`·`ffprobe`가 있어야 합니다.

## 2. 농가 라우터 `/ai/route` (REQ-AI-01)

| 항목 | 내용 |
| --- | --- |
| 모델 | OpenAI `gpt-5.4-mini` (`ROUTER_MODEL`로 변경) |
| 형식 강제 | Structured Outputs (`responses.parse` + `text_format`, strict) |
| 입력 | `{ text, today, farm_context: { cows: [{ tag4, desc, barn }] } }` (이름·전화번호 없음) |
| 출력 | `{ intent, items[], report_type, report_fields, missing[] }` |
| 타임아웃 | 8초, 재시도 없음 |
| 저장 | `store=False` (OpenAI 쪽에 응답 저장 안 함) |

처리 순서:

1. 모델이 `intent`, `items`, `report_type`, 신고 종류별 항목을 채웁니다. 상대 날짜("어젯밤", "그저께")는 `today` 기준 `YYYY-MM-DD`로 바꿉니다.
2. 코드가 개체번호를 소 목록과 대조합니다. 끝 4자리가 목록에 없으면 `null`로 바꾸고 `missing`에 `"cow"`를 넣습니다. 항목의 `cow_tag4`와 신고 항목의 `mother_tag4`·`cow_tag`에 같은 규칙을 씁니다.
3. 말하지 않은 값은 `null`입니다. 신고서 빈칸 확인과 되묻기는 `rules/report_state_machine`이 합니다.

| report_type | report_fields |
| --- | --- |
| `birth` | `mother_tag4`, `birth_date`, `sex`(암·수) |
| `death` | `cow_tag`, `death_date` |
| `move` | `move_type`(sell·buy·move), `cow_tag`, `move_date`, `counterpart_farm_no?`, `counterpart_name?` |
| `null` | `{}` |

`LLM-FAIL` 반환 조건: 호출 실패, 8초 초과, 응답 미완료·거절, 날짜 형식 오류, `report_type`과 항목 불일치, 출력 스키마 위반.

## 3. 평가

| 파일 | 내용 |
| --- | --- |
| `eval/route_eval_50.jsonl` | 평가 문장 50개 + 정답 (일지 25 · 이력 신고 15 · 증상 10, 기준일 2026-11-15) |
| `eval/eval_route.py` | 정확도·p50/p95 지연·스키마 통과율 → `eval/route_eval_result.json` |
| `eval/eval_stt.py` | 시연 음성 정확 인식·p95 지연 → `eval/stt_eval_result.json` |

라우터 정답 판정은 문장 단위입니다. `intent`·`report_type`·`report_fields`·`missing`이 같고, 항목 묶음 `(category, cow_tag4, head_count)`가 같고, 백신 종류가 맞아야 정답입니다.

| 명세 완료 기준 | 측정 스크립트 |
| --- | --- |
| 항목 추출 정확도 ≥ 0.9 (50문장) | `eval_route.py` |
| 응답 p95 ≤ 4초 | `eval_route.py` |
| 스키마 검증 100% | `eval_route.py` |
| 15초 발화 p95 ≤ 3초 | `eval_stt.py` |
| 시연 문장 3종 정확 인식 | `eval_stt.py` |

## 4. 실행

```bash
pip install -r agents/farm_router/requirements.txt   # + apt install ffmpeg
export OPENAI_API_KEY=...                              # .env, 커밋 금지
python -m pytest -q agents/farm_router/tests           # 25개, 키 없이 실행
python agents/farm_router/eval/eval_route.py
python agents/farm_router/eval/eval_stt.py <음성+정답 txt 폴더>
```

```python
from farm_router import route, transcribe, RouteRequest, RouteError, SttError

stt = transcribe("voice.webm")                    # SttResponse
res = route(RouteRequest(text=stt.text, today="2026-11-15",
                         farm_context={"cows": [{"tag4": "1234", "desc": "큰 암소", "barn": "A1"}]}))
```

| 환경변수 | 기본값 | 용도 |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | 필수 |
| `ROUTER_MODEL` | `gpt-5.4-mini` | 라우터 모델 |
| `ROUTER_REASONING_EFFORT` | (없음) | 추론 강도 지정 시 전달 |
| `STT_MODEL` | `gpt-transcribe` | 음성 인식 모델 |
