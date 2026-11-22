# mlops/serving

AI 기능 6개를 내부 API로 여는 FastAPI 서버입니다. `services/api`(Spring)가 같은 Docker 네트워크 안에서 부릅니다. 외부(nginx)로는 열지 않습니다.

- **담당**: @dooooing
- **명세**: mnm-dev `docs/api/ai-internal.yaml` (주소·상태 코드·시간 제한) · `schemas/` (요청·응답 형식)

## 구조

| 파일 | 내용 |
| --- | --- |
| `app.py` | 주소 7개, 토큰 검사, 서버 시간 제한, 오류 → 상태 코드 |
| `models.py` | 시작할 때 1회 불러오는 모델·OpenAI 클라이언트·지식베이스 |
| `fetch.py` | 사진·영상 주소(Object Storage 서명 URL)에서 파일 받기 |
| `errors.py` | 기능별 `detail.reason` → HTTP 상태 코드 |
| `config.py` | 환경변수 |

기능 로직은 새로 만들지 않고 `agents/`·`inference/`의 진입 함수를 `common.call_ai`로 감싸 부릅니다.

| 주소 | 부르는 함수 | 입력 처리 |
| --- | --- | --- |
| `POST /ai/stt` | `farm_router.transcribe` | 업로드 파일 → 임시 파일 |
| `POST /ai/route` | `farm_router.route` | JSON 그대로 |
| `POST /ai/part-check` | `part_check_siglip2.check_part` | `image_url` → 바이트 |
| `POST /ai/behavior` | `behavior_yolo11s.analyze_video` | `video_url` → 임시 파일 |
| `POST /ai/guide` | `guide_rag.guide` | `GUIDE_KB_PATH` 조항 목록 사용 |
| `POST /ai/summary` | `receiver_copilot.summarize` | JSON 그대로 |
| `GET /health` | — | 토큰 없이 열림. 불러온 모델 버전과 못 불러온 이유 |

## 실행

```bash
# 저장소 루트에서
pip install -r mlops/serving/requirements.txt
export AI_INTERNAL_TOKEN=로컬용_아무_문자열
uvicorn mlops.serving.app:app --host 0.0.0.0 --port 8000
curl localhost:8000/health
```

- torch·가중치 없이 띄우려면 `AI_LOAD_PART_CHECK=0 AI_LOAD_BEHAVIOR=0`. 그 기능은 503 `not_loaded`를 돌려줍니다.
- 로컬에서 `image_url`을 다른 호스트로 시험하려면 `AI_FETCH_ALLOWED_HOSTS=`(빈 값)로 제한을 끕니다.
- 자동 문서: `localhost:8000/docs`

## 환경변수

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `AI_INTERNAL_TOKEN` | 없음 | `X-Internal-Token` 헤더 값. **비어 있으면 `/ai/*`를 모두 거절** |
| `AI_TIMEOUT_SEC` | `8` | 서버 시간 제한 (부위 확인 제외) |
| `AI_PART_CHECK_TIMEOUT_SEC` | `10` | 부위 확인 서버 시간 제한 (모델 1.5초 + Vision 대체 경로) |
| `AI_FETCH_ALLOWED_HOSTS` | `kr.object.ncloudstorage.com` | 파일을 받을 수 있는 호스트 (쉼표 구분). 하위 도메인(`버킷.kr.object.ncloudstorage.com`)도 허용 |
| `AI_MAX_IMAGE_BYTES` · `AI_MAX_VIDEO_BYTES` | 15MB · 200MB | 받는 파일 크기 상한 (제안값, 실측으로 조정) |
| `AI_LOAD_PART_CHECK` · `AI_LOAD_BEHAVIOR` | `1` | `0`이면 그 모델을 불러오지 않음 |
| `GUIDE_KB_PATH` | 없음 | 방역 안내 게시 조항 파일 (JSON·JSONL). 없거나 못 읽으면 `/ai/guide`는 503 `LLM-FAIL` (`not_loaded`), 파일은 있는데 조항 0개면 404 `RAG-NONE` |
| `GUIDE_EMBED_CACHE` | 없음 | 조항 임베딩 캐시 파일 |

기능 모듈이 쓰는 키(`OPENAI_API_KEY`, `BEHAVIOR_WEIGHTS`, `PART_FALLBACK_VISION` 등)는 각 폴더 README를 봅니다.

## 상태 코드

| 상태 | 뜻 | 예 (`detail.reason`) |
| --- | --- | --- |
| 200 | 성공 | — |
| 401 | 내부 토큰 없음·불일치 | `unauthorized` |
| 404 | 근거 조항 없음 (`RAG-NONE`, 강제 오류 포함) | `no_rule`, `no_evidence` |
| 422 | 입력 때문에 실패. 다시 불러도 같음 | `bad_request`, `silence`, `format`, `open`, `fetch_rejected`(만료된 서명 URL·허용 밖 호스트), `too_large`, `report_fields` |
| 503 | 외부 API·모델 실패 | `api`, `inference`, `fallback`, `fetch_failed`, `not_loaded`(모델·guide 지식베이스), `config`, 강제 오류(`AI_FORCE_ERROR`) |
| 504 | 시간 초과 | `timeout`, `server_timeout` |

- 오류 본문은 모두 `{ code, message, detail }`. 입력값과 서명 URL은 오류·로그에 넣지 않습니다.
- 재시도는 호출하는 서버가 엔드포인트별로 정합니다 (mnm-dev#5).
- `X-Request-Id`를 보내면 응답 헤더와 호출 로그(`mnm_ai.calls`)에 그대로 남깁니다.
- 서버 시간 제한으로 504를 돌려준 호출은 기록 1건만 남깁니다 (뒤에서 끝난 작업은 기록하지 않음).
- 호출 기록은 지금 로그로만 남깁니다. `inference_log` 테이블 기록은 다음 PR.

## 테스트

```bash
pip install -r mlops/serving/requirements-dev.txt
pytest mlops/serving/tests
```

모델·OpenAI는 가짜로 바꾸고 주소·토큰·상태 코드·응답 형식(`schemas/*.json`)만 확인합니다. 모델 정확도·속도는 각 모듈 테스트와 스테이징 측정에서 봅니다.

## 이미지 (GHCR)

- `.github/workflows/build-inference.yml`: develop 머지 → 테스트 → `ghcr.io/mnms-chonsu-lab/mnm-ai-inference:{커밋 SHA, develop}` 푸시
- 직접 빌드: `docker build -f mlops/serving/Dockerfile -t mnm-ai-inference .` (저장소 루트에서)
- torch는 CPU 전용 휠. 이미지에 가중치는 넣지 않습니다.

| 실행할 때 붙이는 것 | 위치 |
| --- | --- |
| 행동 인식 가중치 `best.pt` | `/models/behavior/best.pt` 마운트 (`BEHAVIOR_WEIGHTS`) |
| SigLIP 2 가중치 | 첫 실행 때 Hugging Face에서 받아 `/models/hf` 볼륨에 저장 (`HF_HOME`). 두 번째부터는 볼륨에서 읽음 |
| 키·토큰 | `env_file` |
