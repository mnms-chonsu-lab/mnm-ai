# inference/part_check_siglip2

SigLIP 2 제로샷 기반 부위 일치 검증 모듈입니다. 농가가 찍은 사진이 앱이 요청한 부위(입·유방·발굽·몸 옆면)가 맞는지 확인하고, 사진 화질(어두움·흐림)을 함께 판정합니다.

- **담당**: @parkmyeonghun
- **쓰이는 곳**: 농가 F-S10 촬영 화면, 접수자 R-S02·R-P03, 어드민 A-S07 (명세 REQ-AI-03)
- **입출력 형식**: [`schemas/part_check.json`](../../schemas/part_check.json)

---

## 처리 흐름

| 단계 | 내용 |
|---|---|
| 1. 사진 열기 | jpg·png·heic (아이폰 기본 형식 포함), EXIF 방향 보정 |
| 2. 화질 | 어두움: 평균 밝기 < 60/255 · 흐림: 라플라시안 분산 < 100 (제안값, 실측으로 조정) |
| 3. 표준화 | 512×512 BICUBIC 변환 + JPEG 품질 95 (평가 데이터와 같은 전처리) |
| 4. 임베딩 | `google/siglip2-base-patch16-384` 이미지 임베딩 |
| 5. 점수 | 부위별 문장 임베딩과 코사인 유사도 → softmax(100 × 코사인) |
| 6. 판정 | 점수 1등 = `predicted_part`, `match` = 예측 부위가 요청 부위와 같고 점수 ≥ 기준값 |
| 실패 | 사진을 열 수 없음·추론 오류·1.5초 초과 → `PART-FAIL` (서버는 판정 없이 진행) |
| 대체 경로 | 추론 오류·1.5초 초과 때 `PART_FALLBACK_VISION=1`이면 OpenAI Vision으로 판정, `fallback: true`, 출력 형식 동일. 대체 경로도 실패하면 `PART-FAIL` |

## 출력 예시

```json
{
  "requested_part": "hoof",
  "predicted_part": "hoof",
  "match": true,
  "score": 0.9132,
  "quality": {"dark": false, "blurry": false, "brightness": 118.4},
  "model_version": "siglip2-b384-zs-A_simple-v1",
  "fallback": false
}
```

| 필드 | 설명 |
|---|---|
| `predicted_part` | 4부위 중 점수 1등. 로그·어드민용 (농가 화면에는 노출하지 않음) |
| `score` | `predicted_part`의 점수 (0–1). 4부위 softmax라 항상 0.25 이상 |
| `quality` | 화질 판정. 부위 판정과 별개로 항상 함께 반환 |
| `fallback` | 대체 경로(OpenAI Vision) 사용 여부. `true`일 때 `score`는 모델이 스스로 매긴 확신도(0–1)이며 SigLIP 2 점수와 같은 척도가 아님 |

---

## 모델

| 항목 | 내용 |
|---|---|
| 모델 | `google/siglip2-base-patch16-384` (375M 파라미터), 추가 학습 없음 (제로샷) |
| 부위 | `mouth`(입) · `udder`(유방) · `hoof`(발굽) · `flank`(몸 옆면) |
| 프롬프트 | 부위마다 영어 문장 2개의 임베딩 평균. 후보 3세트 중 검증 세트 Macro-F1이 가장 높은 세트 선택 |

### 모델 선정 과정

| 방식 | 데이터 | 결과 | 판단 |
|---|---|---|---|
| YOLOv8n-cls 직접 학습 | 1차 실험: 3부위(입·발굽·해당 없음), 시험 131장 | 교차검증 96.6%, 외부 사진(럼피스킨 피부 197장) "해당 없음" 판정 0.48 | 학습 데이터와 다른 사진에 약함 |
| CLIP ViT-L/14 제로샷 | 1차 실험: 같은 3부위 | 시험 0.95, 5-fold 교차검증 82% | 데이터 분할에 따라 흔들림 |
| **SigLIP 2 제로샷 (채택)** | 2차 실험: 4부위, 시험 81장 | 시험 0.938, 같은 조건 3개 모델 중 정확도·Macro-F1 최고 | 학습 없이 4부위 확장, 서비스 코드로 재현 확인 |

| 프롬프트 세트 (SigLIP 2, 4부위) | 검증 Macro-F1 |
|---|---|
| **A_simple (채택)** | **0.837** |
| B_descriptive | 0.730 |
| C_minimal | 0.715 |

| 부위 | 문장 |
|---|---|
| 입 | a photo of the head and mouth of a cow / a close-up photo of a cow's muzzle, tongue or mouth |
| 유방 | a photo of the udder and teats of a cow / a close-up photo of a cow's udder |
| 발굽 | a photo of the hoof and foot of a cow / a close-up photo of a cow's hoof |
| 몸 옆면 | a photo of the side view of a whole cow body / a photo of a cow standing, seen from the side |

### 성능

| 지표 | 값 |
|---|---|
| 정확도 | **0.938** (76/81), 95% CI [0.86, 0.97] |
| Macro-F1 | 0.930 |
| 요청 부위 정통과율 | 0.938 |
| 다른 부위 오통과율 | 0.021 |

| 부위 | n | 정밀도 | 재현율 |
|---|---|---|---|
| 입 | 43 | 1.000 | 0.884 |
| 유방 | 15 | 1.000 | 1.000 |
| 발굽 | 9 | 0.692 | 1.000 |
| 몸 옆면 | 14 | 0.933 | 1.000 |

- 유방·발굽·몸 옆면은 재현율 1.000, 입은 정밀도 1.000입니다.
- 오답 5장은 모두 병변이 있는 입 사진(혀·치판·주둥이 근접)이며, 그중 4장을 발굽으로 판정했습니다.

측정 조건: 시험 세트 81장 (전체 404장을 비슷한 사진끼리 묶어 train 242 / val 81 / test 81로 분할, 프롬프트는 val로 선택), Colab T4 GPU float16

**서비스 코드 재현** (이 폴더 코드 그대로, Colab CPU float32, 2026-11-14)

| 항목 | 결과 |
|---|---|
| 전처리 일치 | 512×512 JPEG md5 404/404 일치 |
| 시험 정확도 | 0.938 (76/81) — 실험과 같음, 부위별 정밀도·재현율과 통과율도 모두 같음 |
| CPU 처리 시간 (파일 → 결과) | p50 2.41초 · p95 3.61초 (Colab CPU 2코어, 40장) |

시간 제한은 명세대로 1.5초입니다. 위 CPU 실측(p95 3.61초)으로는 넘기 때문에, 명세 p95 1.5초를 맞추려면 GPU 서빙이 필요합니다 (T4 모델 추론 13.4 ms/장, 파일 → 결과 전체 시간은 GPU에서 미측정). 서버는 비동기로 호출하고 결과를 저장합니다 (명세 REQ-OPS-09).

| 모델 비교 (같은 조건) | 정확도 | Macro-F1 | 처리 시간 (T4) |
|---|---|---|---|
| **SigLIP 2 base-384 (채택)** | **0.938** | **0.930** | 13.4 ms/장 |
| CLIP ViT-B/16 | 0.926 | 0.904 | 6.7 ms/장 |
| SigLIP base-224 | 0.901 | 0.894 | 6.7 ms/장 |

### 판정 기준값

- `MATCH_THRESHOLD = 0.0` — 예측 부위가 요청 부위와 같으면 통과.
- 4부위 softmax에서 1등 점수는 항상 0.25 이상이므로, 0.25 이하 기준값은 판정을 바꾸지 않습니다.
- 위 정통과율 0.938 · 오통과율 0.021이 이 기준의 시험 결과입니다.
- 기준값을 올리면 틀린 사진을 거르는 만큼 맞는 사진도 비슷하게 떨어뜨려서 0을 씁니다 (이미지별 점수로 계산).

| 기준값 | val 맞는 사진 통과 | val 틀린 사진 걸러짐 | test 맞는 사진 통과 | test 틀린 사진 걸러짐 |
|---|---|---|---|---|
| **0 (채택)** | 69/69 | 0/12 | 76/76 | 0/5 |
| 0.6 | 67/69 | 4/12 | 74/76 | 1/5 |
| 0.7 | 64/69 | 5/12 | 72/76 | 3/5 |
| 0.9 | 57/69 | 8/12 | 62/76 | 3/5 |

### 데이터

| 데이터 | 부위 | 라이선스 |
|---|---|---|
| FMD Cattle (Osborn, 2023, [Zenodo 10.5281/zenodo.7779246](https://zenodo.org/records/7779246)) | 입·유방·발굽 | CC BY 4.0 |
| Cattle side view and back view dataset (Bai, 2025, [Mendeley 10.17632/h2s22wr5py.3](https://data.mendeley.com/datasets/h2s22wr5py)) | 몸 옆면 | CC BY 4.0 |

---

## 사용법

```bash
pip install -r inference/part_check_siglip2/requirements.txt
```

```python
import sys; sys.path.insert(0, "inference")
from part_check_siglip2 import check_part, load_embedder, PartCheckError

embedder = load_embedder()          # 서버 시작 시 1회 (모델 로드 + 문장 임베딩 미리 계산)
try:
    result = check_part("/tmp/photo.heic", "hoof", embedder)   # PartCheckResponse
    body = result.model_dump(mode="json")
except PartCheckError as e:
    body = e.to_response()          # {"code": "PART-FAIL", ...}
```

- 입력은 로컬 파일 경로 또는 바이트입니다. `image_url`에서 파일을 받는 일은 호출하는 서버가 맡습니다.

## 테스트

```bash
python -m pytest -q inference/part_check_siglip2/tests   # 모델 없이 실행 (가짜 임베더)
python inference/part_check_siglip2/export_schema.py    # schemas/part_check.json 다시 생성
```

## 파일

| 파일 | 내용 |
|---|---|
| `part_check.py` | 전처리·임베딩·점수·판정 |
| `quality.py` | 화질 판정 (어두움·흐림) |
| `schema.py` | 입출력 형식 (pydantic) |
| `export_schema.py` | `schemas/part_check.json` 생성 |
| `tests/` | 처리 흐름·형식 테스트 17개 |
| `requirements.txt` | 실험 노트북 기준 버전 |
