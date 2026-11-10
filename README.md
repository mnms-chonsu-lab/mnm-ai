# mnm-ai

촌수의사 — 한우농가 신고·기록·방역 업무 지원 AI. 에이전트, 규칙 코드, 인식 모델 추론, 학습·평가·MLflow를 담당하는 레포입니다. (M&M's, 공모전 본선 MVP)

## 담당
- 전체 — 박명훈 (AI)
- `mlops/**` — 이인서(MLOps) 공동

## 구성
| 경로 | 내용 |
|---|---|
| `agents/farm_router/` | 농가 발화 라우팅 에이전트 |
| `agents/guide_rag/` | SOP·법령 기반 안내 RAG 에이전트 |
| `agents/receiver_copilot/` | 접수자 보조 에이전트 |
| `rules/report_state_machine/` | 신고서 상태 전이 규칙 |
| `rules/deadline/` | 신고 기한 규칙 |
| `rules/disease_branch/` | 질병 분기 규칙 (판정·확률 표현 없이 SOP 기준 분기만) |
| `rules/routine_change_alert/` | 사양 변경 알림 규칙 |
| `inference/part_check_siglip2/` | SigLIP 2 기반 부위 확인 추론 |
| `inference/behavior_yolo11s/` | YOLO11s 기반 행동 탐지 추론 |
| `mlops/train/`, `mlops/eval/`, `mlops/mlflow/` | 학습·평가·MLflow 트래킹 |
| `schemas/` | 에이전트/추론 입출력 JSON 형식 — **mnm-dev가 이 형식을 따름** |

## 로컬 실행 방법
1. `.env.example`을 복사해 `.env` 생성 후 키 값 채우기 (`OPENAI_API_KEY` 등)
2. 각 `agents/*`, `inference/*`는 자체 디렉터리에서 의존성 설치 후 실행 (세부 사항은 각 README)
3. 학습·평가는 `mlops/train`, `mlops/eval`에서, 트래킹은 `mlops/mlflow` 설정을 따름

> 모델 가중치·원본 데이터는 git에 넣지 않습니다. S3 / MLflow에서 가져와 사용합니다.

## 브랜치·PR 규칙
- 브랜치: `YYYYMMDD_#이슈번호_요약` (develop에서 분기, 한글 가능, 공백은 `_`)
- 1 이슈 = 1 브랜치 = 1 PR, **develop**으로 PR
- PR 제목: `type(영역): 요약 (#이슈번호)`
- squash merge만 허용, 머지 후 브랜치 자동 삭제
- 자세한 규칙은 [CONTRIBUTING.md](./CONTRIBUTING.md) 참고

## 연결 방식
- **mnm-ai → mnm-dev**: `main` 머지 시 GitHub Actions(`image.yml`)로 추론·에이전트 서비스 이미지를 GHCR(`ghcr.io/mnms-chonsu-lab/mnm-ai-*`)에 올립니다. mnm-dev는 태그로 그 이미지를 가져다 씁니다. 입출력 형식은 `schemas/`가 기준이며, mnm-dev는 이 형식을 따릅니다.
- **mnm-data → mnm-ai**: 지식베이스(`knowledge-base/`)·공공데이터는 mnm-data가 관리합니다. `guide_rag` 등에서 참조만 하고 원문을 직접 수정하지 않습니다.
- `schemas/` 변경은 mnm-dev 담당자(@dooooing, @kkamret)를 PR 리뷰어로 지정합니다.
- 다른 레포에 영향 주는 PR은 본문에 관련 레포 PR·이슈 링크를 반드시 포함합니다.
