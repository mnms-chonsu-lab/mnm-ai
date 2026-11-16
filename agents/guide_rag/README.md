# agents/guide_rag

질병·단계·질문에 맞는 공식 지침 조항을 골라, 조항 원문을 그대로 돌려줍니다.

- **담당**: @parkmyeonghun
- **명세**: REQ-AI-08 `/ai/guide` (화면 F-S09, F-S14, R-S04)
- **입출력 형식**: `schemas/guide.json` (`schema.py`에서 생성)
- **지식베이스**: `mnm-data` `biosecurity_rule`의 게시 조항 (REQ-DATA-05, @tjdlsry)

## 원칙

| 원칙 | 구현 |
| --- | --- |
| 게시된(검수 완료) 조항만 검색 | `status`가 `게시`인 조항만 씀. 초안·검수 중은 버림 |
| 화면 문구는 조항 원문 그대로 | `texts`는 지식베이스 `text`를 복사. 모델 출력 문장을 쓰지 않음 |
| LLM은 고르기만 함 | 모델 출력 형식이 후보 `rule_id` 목록(enum) 하나뿐. 문장 필드가 없음 |
| 근거가 없으면 fallback | `RAG-NONE` 오류, `detail.fallback = true` |

## 동작

| 입력 | 처리 | LLM |
| --- | --- | --- |
| `query` 없음 | `disease`·`stage`가 같은 게시 조항을 `order` 순으로 반환 | 안 씀 |
| `query` 있음 | ① 질병·단계가 맞거나 공통인 게시 조항 중 임베딩 유사도 상위 5개 ② 모델이 그중 근거 조항 ID만 고름 | 씀 |

- 질병 안내 6건(구제역 3·럼피스킨 3), 증상 질문 4건은 `query` 없이 부르면 됩니다. LLM을 거치지 않아서 원문과 항상 같습니다.
- 임베딩은 `text-embedding-3-small`(1536차원 = `mnm-data` `migrations/0003`의 `vector(1536)`)을 씁니다(`GUIDE_EMBED_MODEL`로 변경). DB `embedding`이 있으면 그 값을 쓰고, 없을 때만 계산합니다.
- DB `embedding` 채우기: `python -m guide_rag.build_embeddings <행.csv|json> <out.sql>` → `UPDATE` 문 SQL. DB 적용은 `mnm-data` 담당.
- 후보 수 5개는 제안값이며 실측으로 조정합니다.

`RAG-NONE` 반환 조건 (`detail.reason`): `no_rule`(해당 게시 조항 없음), `no_evidence`(모델이 고른 조항 없음), `embed`·`api`·`timeout`·`no_parse`(호출 실패), `config`(키 없음).

## 지식베이스 형식 (`KbRule`)

| 필드 | 설명 |
| --- | --- |
| `rule_id` | 조항 ID. DB 컬럼 `biosecurity_rule_id`도 받음 |
| `kind` | question · photo_guide · action · form_field · summary · clause. `query` 검색은 `clause`·`action`·`summary`만 (제안, PM 확인 필요) |
| `disease` | `FMD` · `LSD` · `null`(공통). DB 값 그대로 |
| `stage` | 안내 단계 · `null`(공통). 값은 `mnm-data` 확정값: `suspect`(질문·촬영 안내), `after_report`(신고 후 조치), `regional`(지역 발생 안내). 코드는 값을 검증하지 않고 그대로 비교 |
| `order` | 같은 묶음 안 순서. DB 컬럼 `display_order`도 받음 |
| `embedding` | DB `biosecurity_rule.embedding`(`vector(1536)`). 있으면 다시 계산하지 않음 |
| `text` | 조항 원문 |
| `doc` · `clause` · `revised_at` | 근거 문서·조항·개정일 |
| `status` | 초안 · 검수 중 · 게시 |

서버가 게시 뷰에서 읽은 목록을 넘기거나, `load_kb(path)`로 JSON·JSONL 파일을 읽습니다.

## 실행

```python
from guide_rag import guide, GuideRequest, load_kb
kb = load_kb("biosecurity_rule_published.jsonl")
guide(GuideRequest(disease="FMD", stage="after_report"), kb)                       # 조항 목록
guide(GuideRequest(disease="FMD", stage="after_report", query="소독은 얼마나 자주"), kb)  # 질문
```

```bash
python -m pytest -q agents/guide_rag   # 7개, 키 없이 실행 (테스트 문구는 가상)
```
