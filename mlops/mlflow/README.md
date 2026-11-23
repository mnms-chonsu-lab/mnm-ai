# mlops/mlflow

MLflow 트래킹 서버와 실험·모델 버전 기록입니다.

- **담당**: @dooooing, @parkmyeonghun

## 무엇을 기록하나

| 실험 (experiment) | run | 모델 버전 (serving) |
| --- | --- | --- |
| `behavior_yolo11s` | 기본 학습(농장 B·C) · 농장 맞춤 학습(+농장 A 253장) | `behavior_yolo11s` v1 = 맞춤 학습, 가중치 `best.pt` |
| `part_check_siglip2` | 1차 2개(3부위) · 2차 모델 비교 3개(4부위) | `part_check_siglip2` v1 = SigLIP 2 base-384, 설정 파일(제로샷이라 가중치 없음) |

- 값은 `inference/*/README.md` 표를 그대로 옮겼습니다 (다시 계산하지 않음). 출처는 run 태그 `mnm.source`, 측정 조건은 `mnm.condition`
- 모델 버전 별칭(`@yolo11s-960-adaptA` 등)과 태그 `model_version`은 `inference_log.model_version`과 같은 값입니다. 호출 기록 → `models:/<모델>@<model_version>` → 실험(run)으로 바로 이어집니다
- 모델 버전 태그 `serving_image`는 그 모델을 서비스하는 AI 서버 이미지입니다

## 서버 (스테이징)

| 항목 | 값 |
| --- | --- |
| 위치 | `/root/mnm-mlflow/compose.yml` (이 폴더 `compose.yml`) |
| 이미지 | `ghcr.io/mlflow/mlflow:v3.18.0` (공식 이미지, 서버에서 빌드하지 않음) |
| 저장 | 볼륨 `mlflow-data`: 기록 `mlflow.db`(SQLite) · 파일 `artifacts/` |
| 열린 곳 | 서버 안 `127.0.0.1:5000`만. 외부·nginx로 열지 않음 |
| 메모리 | 상한 768MB (제안값). 작업 실행기는 끔 (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`) |

### 화면 보기 (SSH 터널)

내 PC에서 실행한 뒤 브라우저로 `http://localhost:5000`을 엽니다.

```bash
ssh -i "$HOME\.ssh\mnm_staging" -N -L 5000:127.0.0.1:5000 inseo@211.233.223.189
```

- 서버 접속 창이 아니라 **노트북의 새 창**에서 실행합니다. 아무것도 안 나오고 멈춘 것처럼 보이면 정상이고, 창을 열어 둔 동안만 화면이 열립니다
- 노트북에서 5000번을 못 쓰면(Windows 예약 등) 앞 번호만 바꿉니다: `-L 15000:127.0.0.1:5000` → `http://localhost:15000`
- 화면 왼쪽 위 `GenAI | Model training`에서 **Model training**을 골라야 run 목록·모델 버전이 보입니다

### 기존 결과 기록

서버에서 MLflow 이미지로 한 번 실행합니다. 여러 번 실행해도 같은 기록이 늘지 않습니다 (`mnm.key`로 확인).

```bash
sudo docker run --rm --network mnm-mlflow_default \
  -v /root/mnm/models/behavior:/models/behavior:ro \
  -v "$PWD/log_existing.py:/app/log_existing.py:ro" \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 -e GIT_PYTHON_REFRESH=quiet \
  -e BEHAVIOR_WEIGHTS=/models/behavior/best.pt \
  -e SERVING_IMAGE=ghcr.io/mnms-chonsu-lab/mnm-ai-inference:<커밋 SHA> \
  ghcr.io/mlflow/mlflow:v3.18.0 python /app/log_existing.py
```

- 가중치 sha256이 `inference/behavior_yolo11s/README.md` 기록(`5485f12b8d9cf044…`)과 다르면 기록하지 않고 멈춥니다
- `GIT_PYTHON_REFRESH=quiet`: 컨테이너 안에 git이 없어서 나오는 경고를 끕니다. 커밋 정보는 run 태그 `mnm.source`에 남습니다

## 확장 (MVP 밖)

- 학습 스크립트(`mlops/train`)가 직접 MLflow에 기록
- 새 모델 버전을 만들면 기존 버전과 지표 비교 → 통과하면 `serving` 별칭 옮기기 (기획서 6.4.6)
