<div align="center">
  <h3>촌수의사 · mnm-ai</h3>
  <p>고령·소규모 한우농가가 말로 남긴 농장 일지를, 신고할 때 접수자(축협·방역기관)에게 한 장 요약으로 함께 전달하는 신고·기록·방역 업무 지원 AI</p>
  <p><b>LLM 에이전트 · 규칙 코드 · 인식 모델 · MLOps</b></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-YOLO11s · SigLIP2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/OpenAI-STT·LLM·TTS-412991?logo=openai&logoColor=white" alt="OpenAI" />
    <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow" />
    <img src="https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white" alt="Docker" />
  </p>
  <sub>팀 M&amp;M's · 공모전 본선 MVP · 조직 <a href="https://github.com/mnms-chonsu-lab">mnms-chonsu-lab</a></sub>
</div>

---

## 목차

- [이 레포가 맡는 것](#이-레포가-맡는-것)
- [빠른 시작](#빠른-시작)
- [환경변수](#환경변수)
- [기술 스택](#기술-스택)
- [디렉터리 구조](#디렉터리-구조)
- [서비스 구성과 데이터 흐름](#서비스-구성과-데이터-흐름)
- [다른 레포와의 연결 방식](#다른-레포와의-연결-방식)
- [팀 소개](#팀-소개)
- [작업 흐름](#작업-흐름)
- [데이터 출처·라이선스](#데이터-출처라이선스)

---

## 이 레포가 맡는 것

| 구분 | 구성 | 담당 부분 |
| --- | --- | --- |
| LLM 에이전트 3개 | 농가 라우터, 방역 안내 RAG, 접수자 코파일럿 | ① 말로 쓰는 일지 구조화, ② 신고서 초안, ④ 접수자 한 장 요약 |
| 규칙 코드 | 신고서 상태 머신, 기한 계산, 질병별 안내 분기, 요약 카드, 미응답 판정, 평소와 다름 알림 | ② 기한 알림, ③ 안내 분기 |
| 인식 모델 2개 | SigLIP 2 부위 일치 검증(정확도 93.8%), YOLO11s 소 행동 인식(Macro F1 0.857) | 사진 검증, 평소와 다름 알림의 입력 |
| MLOps | 학습·평가·MLflow | 모델 성능 기록, 가중치 관리 |
| 입출력 형식 | `schemas/` | **mnm-dev가 따르는 기준** |

> 모델 지표는 팀이 측정한 값입니다. 측정 조건(데이터 분할·버전)은 🚧 `mlops/eval/`에 정리 예정입니다.
> 현재 상태: 디렉터리 골격과 README만 있고 코드는 아직 없습니다. 아래 🚧 표시는 모두 구현 예정입니다.
> 요약 카드·미응답 판정 규칙은 아직 전용 디렉터리가 없습니다 (`rules/` 아래 4개만 존재). 위치는 🚧 정해지면 추가합니다.

---

## 빠른 시작

```bash
git clone https://github.com/mnms-chonsu-lab/mnm-ai.git
cd mnm-ai
cp .env.example .env      # OPENAI_API_KEY 등 채우기
```

| 단계 | 명령 | 상태 |
| --- | --- | --- |
| 에이전트 실행 | `agents/*` 의존성 설치·실행 | 🚧 구현 예정 |
| 규칙 코드 테스트 | `rules/*` 테스트 (`pytest`) | 🚧 구현 예정 |
| 추론 서비스 | `inference/*` 실행 | 🚧 구현 예정 |
| 학습·평가 | `mlops/train`, `mlops/eval` | 🚧 구현 예정 |
| MLflow | `mlops/mlflow` 설정 후 `MLFLOW_TRACKING_URI` 지정 | 🚧 구현 예정 |
| 이미지 빌드·GHCR push | `main` 머지 시 GitHub Actions | 🚧 구현 예정 (워크플로는 @dooooing이 추가) |

> 모델 가중치·원본 데이터는 git에 넣지 않습니다. S3 / MLflow에서 가져와 사용합니다.

---

## 환경변수

값은 `.env.example`을 복사해 `.env`에 채웁니다. 실제 키는 커밋하지 않습니다.

| 변수 | 위치 | 용도 | 없으면 |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | `agents/*` | STT·구조화 출력·TTS 호출 | 에이전트 동작 불가 |
| `AWS_ACCESS_KEY_ID` | `mlops/*`, `inference/*` | S3의 모델 가중치·데이터 접근 | 가중치·데이터를 받지 못함 |
| `AWS_SECRET_ACCESS_KEY` | `mlops/*`, `inference/*` | S3 접근 | 위와 같음 |
| `AWS_REGION` | `mlops/*`, `inference/*` | AWS 리전 | 위와 같음 |
| `S3_BUCKET_NAME` | `mlops/*`, `inference/*` | 가중치·데이터 버킷 | 위와 같음 |
| `MLFLOW_TRACKING_URI` | `mlops/*` | MLflow 서버 주소 | 실험·모델 기록 불가 |
| `DATABASE_URL` | `agents/*` | PostgreSQL 접속 (읽기 용도 예정) | DB 조회 기능 불가 |

---

## 기술 스택

| 영역 | 기술 | 용도 |
| --- | --- | --- |
| LLM·음성 | OpenAI API | STT, 구조화 출력, TTS |
| 에이전트 | Python | 농가 라우터, 방역 안내 RAG, 접수자 코파일럿 |
| 검색 | RAG (`mnm-data/knowledge-base` 참조) | SOP·법령 원문 근거 안내 |
| 인식 모델 | PyTorch, SigLIP 2, YOLO11s | 부위 일치 검증, 소 행동 인식 |
| MLOps | MLflow, AWS S3 | 학습·평가 기록, 가중치 저장 |
| 배포 | Docker, GHCR, GitHub Actions | 이미지 빌드·배포 |
| 품질 | ruff, pytest | lint·테스트 |

---

## 디렉터리 구조

```
mnm-ai/
├── agents/
│   ├── farm_router/            # 농가 발화 → 일지·신고서 구조화 라우터
│   ├── guide_rag/              # SOP·법령 원문 기반 방역 안내 RAG
│   └── receiver_copilot/       # 접수자 코파일럿
├── rules/
│   ├── report_state_machine/   # 신고서 상태 전이
│   ├── deadline/               # 신고 기한 계산
│   ├── disease_branch/         # SOP 기준 안내 분기 (판정·확률 표현 없음)
│   └── routine_change_alert/   # 평소와 다름 알림
├── inference/
│   ├── part_check_siglip2/     # SigLIP 2 부위 일치 검증
│   └── behavior_yolo11s/       # YOLO11s 소 행동 인식
├── mlops/
│   ├── train/                  # 학습
│   ├── eval/                   # 평가
│   └── mlflow/                 # MLflow 트래킹 설정
├── schemas/                    # 입출력 JSON 형식 — mnm-dev가 따름
├── .github/                    # 이슈·PR 템플릿, CODEOWNERS, 라벨
├── .env.example
├── CONTRIBUTING.md · CLAUDE.md · AGENTS.md
└── README.md
```

---

## 서비스 구성과 데이터 흐름

이 레포는 흐름의 **STT·LLM 구조화, 부위 일치 검증, 기한·상태 규칙, 접수자 요약**을 맡습니다. 입구·출구 화면과 API는 `mnm-dev`가 맡습니다.

```
농가 음성·사진
  → STT · LLM 구조화
  → 일지 · 신고서 초안
  → 부위 일치 검증
  → 농가 승인
  → 요약 카드
  → 선택한 신고 경로로 전달 (전화 · 사이트 · 축협)
  → 접수자 웹 열람
```

**지키는 원칙**

| 원칙 | 내용 |
| --- | --- |
| 판정하지 않음 | 질병·신고 여부를 판정하지 않는다. 확률·자동 신고로 읽히는 표현을 쓰지 않는다 |
| `[바로 신고]` 상시 노출 | 모든 화면에 항상 보인다 |
| 공식 문구만 | 안내 문구는 공식 지침(SOP·법령) 원문만 사용한다 |
| 승인된 기록만 전달 | 농가가 승인한 기록만 접수자에게 넘어간다 |

---

## 다른 레포와의 연결 방식

| 방향 | 규칙 |
| --- | --- |
| mnm-ai → mnm-dev | `main` 머지 시 GitHub Actions로 추론·에이전트 서비스 이미지를 GHCR(`ghcr.io/mnms-chonsu-lab/mnm-ai-*`)에 올립니다. mnm-dev는 태그로 그 이미지를 가져다 씁니다. **입출력 형식은 `schemas/`가 기준**입니다 |
| mnm-data → mnm-ai | 지식베이스·공공데이터는 mnm-data가 관리합니다. `agents/guide_rag` 등은 참조만 하고 원문을 직접 수정하지 않습니다 |
| `schemas/` 변경 | 입출력 형식 변경은 @dooooing 승인, mnm-dev 담당자(@dooooing, @kkamret)를 PR 리뷰어로 지정합니다 |
| PR 링크 | 다른 레포에 영향을 주는 PR은 본문에 관련 레포 PR·이슈 링크를 반드시 넣습니다 |

- [mnm-dev](https://github.com/mnms-chonsu-lab/mnm-dev) — 농가 앱·접수자 웹·백엔드·배포
- [mnm-data](https://github.com/mnms-chonsu-lab/mnm-data) — DB 마이그레이션·공공데이터·지식베이스

---

## 팀 소개

| 이름 | 역할 | GitHub |
| --- | --- | --- |
| 이유민 | PM · Frontend | [@kkamret](https://github.com/kkamret) |
| 이인서 | Backend · Infra · MLOps | [@dooooing](https://github.com/dooooing) |
| 박명훈 | AI | [@parkmyeonghun](https://github.com/parkmyeonghun) |
| 서인교 | Data | [@tjdlsry](https://github.com/tjdlsry) |

**일정**: 11/21 기능 개발 마감(이후 버그 수정만) → 11/22 운영 배포 → 11/22~27 QA·안정화 → 11/28 제출

---

## 작업 흐름

```
이슈 (템플릿 선택) → 브랜치 → develop으로 PR → 리뷰 → squash merge
```

| 항목 | 규칙 |
| --- | --- |
| 이슈 | `.github/ISSUE_TEMPLATE/`의 템플릿 5개 중 선택 (빈 이슈 금지) |
| 브랜치 | `YYYYMMDD_#이슈번호_요약` — develop에서 분기, 예) `20261112_#12_농가_라우터_STT` |
| PR | `.github/pull_request_template.md` 사용, 대상은 `develop` |
| PR 제목 | `type(영역): 요약 (#이슈번호)` — type: `feat` `fix` `docs` `chore` `test` `refactor` |
| 머지 | squash merge만, 머지 후 브랜치 자동 삭제 |
| 원칙 | 1 이슈 = 1 브랜치 = 1 PR |

자세한 규칙: [CONTRIBUTING.md](./CONTRIBUTING.md) · AI 도구 사용 시: [CLAUDE.md](./CLAUDE.md), [AGENTS.md](./AGENTS.md)


---

## 데이터 출처·라이선스

| 데이터 | 출처 | 이 서비스에서의 용도 | 이용 조건 |
| --- | --- | --- | --- |
| 가축 질병 발생 정보 | 민관협력 지원 플랫폼 공공데이터 | 발생 현황 참고 | 제공처 이용 조건 확인 필요 |
| 사육규모별 농장 통계 | 민관협력 지원 플랫폼 공공데이터 | 농가 규모 정보 | 제공처 이용 조건 확인 필요 |
| 거점소독시설 | 민관협력 지원 플랫폼 공공데이터 | 소독 시설 안내 | 제공처 이용 조건 확인 필요 |
| 구제역·럼피스킨 SOP, 가축전염병 예방법 | 공식 지침·법령 | 안내 문구 원문 | 공문서·법령 원문 (출처 표기) |
| 한우 행동 데이터 | AI-Hub | 소 행동 인식 모델(YOLO11s) 학습·평가 | **AI-Hub 이용약관 준수** — 원본 재배포 금지, git 커밋 금지 |
| 부위 판정 이미지 | FMD Cattle, Cattle side/back view | 부위 일치 검증 모델(SigLIP 2) 학습·평가 | FMD Cattle: **CC BY 4.0** (출처 표기). Cattle side/back view: 🚧 라이선스 확인 중 |

> 🚧 레포 자체 라이선스는 아직 정하지 않았습니다. 정해지면 이 절에 추가합니다.
> 원본 데이터·모델 가중치는 git에 넣지 않고 S3 / MLflow에서 관리합니다.
