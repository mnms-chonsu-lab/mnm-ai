# inference/behavior_yolo11s

YOLO11s 기반 소 행동 인식 추론 모듈입니다. 스마트폰 영상에서 소를 찾아 행동(서기·눕기·먹기)을 박스 단위로 인식하고, 행동 비율을 돌려줍니다.

- **담당**: @parkmyeonghun
- **쓰이는 곳**: 농가 F-S10V "영상에서 본 모습", F-S11·R-S02 비율 표시 (명세 REQ-AI-11)
- **입출력 형식**: [`schemas/behavior.json`](../../schemas/behavior.json)

---

## 처리 흐름

| 단계 | 내용 |
|---|---|
| 1. 영상 열기 | mp4·mov (HEVC 포함). 세로 영상은 회전 정보로 방향 보정 |
| 2. 프레임 추출 | 처음 10초(짧으면 영상 길이)에서 5장 균등 추출 → 10초 영상이면 0·2·4·6·8초 |
| 3. 모델 입력 | YOLO11s, 960px (학습과 같은 크기) |
| 4. 임계값 | 서기 0.5 · 눕기 0.4 · 먹기 0.5 |
| 5. 출력 | 박스 1개당 1줄 + 박스 수 기준 비율(%) + 가장 많은 행동 |

## 출력 예시

```json
{
  "frames": [
    {"t": 0.0, "label": "stand", "confidence": 0.874, "bbox": [674.5, 525.3, 1169.8, 929.6]}
  ],
  "summary": {"stand": 88.9, "lie": 0.0, "eat": 11.1},
  "main_label": "stand",
  "model_version": "yolo11s-960-adaptA"
}
```

| 필드 | 설명 |
|---|---|
| `frames` | 검출된 소 1마리(박스 1개)당 1줄. 같은 `t`가 여러 줄일 수 있음 |
| `t` | 영상 내 시각(초) |
| `bbox` | 방향 보정 후 화면 기준 픽셀 `[x1, y1, x2, y2]` |
| `summary` | 전체 박스 중 행동별 비율(%), 소수 1자리, 합 100 |
| `main_label` | 가장 많은 행동. 검출이 없으면 `null` |
| 오류 | 영상을 열 수 없음·프레임을 읽지 못함·추론 오류·8초 초과(REQ-AI-12 일반 타임아웃) 시 `BEHAVIOR-FAIL` |

---

## 모델

| 항목 | 내용 |
|---|---|
| 구조 | YOLO11s (객체 검출) |
| 클래스 | `standing` · `lying` · `eating` |
| 학습 데이터 | AI-Hub 소(한우·젖소) 및 돼지 발정행동 데이터(71471) 한우 이미지, 1920×1080, 학습 10,537장 (농장 B·C) |
| 농장 맞춤 학습 | 농장 A 253장 추가 (카메라당 30장 이하) |
| 입력 크기 | 960px |
| 임계값 | 서기 0.5 · 눕기 0.4 · 먹기 0.5 (시험 전에 고정) |
| 가중치 | `best.pt` (19.2MB, sha256 `5485f12b8d9cf044…`). git에 넣지 않음 → 환경변수 `BEHAVIOR_WEIGHTS`로 경로 지정 |

### 성능 — 학습에 쓰지 않은 영상

| 행동 | 정밀도 | 재현율 | F1 |
|---|---|---|---|
| 서기 | **0.937** | 0.816 | 0.872 |
| 눕기 | **0.976** | 0.695 | 0.812 |
| 먹기 | **0.962** | 0.824 | 0.888 |
| **Macro F1** | | | **0.857** |

측정 조건: 학습·맞춤 학습에 쓰지 않은 농장 A 영상 14개(카메라 9대, 1,274장, 정답 박스 7,344개), 임계값은 위 값으로 고정해 1회 평가

- 정밀도가 세 행동 모두 0.93 이상입니다. 모델이 출력한 행동은 대부분 정답과 일치합니다 (축사 CCTV 영상 기준).

### 농장 맞춤(적응) 학습

새 농장의 영상을 소량 추가로 학습해 그 농장 성능을 높이고, 기존 농장 성능은 유지하는 구조입니다. 서비스를 쓰는 농가가 늘어날수록 농장별 데이터를 더해 모델을 개선합니다 (MLOps 재학습 경로).

| 행동 | 새 농장(A) 재현율: 전 → 후 | 기존 농장(B·C) 재현율: 전 → 후 |
|---|---|---|
| 서기 | 0.765 → **0.840** | 0.821 → 0.807 |
| 눕기 | 0.798 → **0.933** | 0.806 → 0.805 |
| 먹기 | 0.968 → **0.977** | 0.798 → 0.789 |

| 행동 | 새 농장(A) 정밀도: 전 → 후 | 기존 농장(B·C) 정밀도: 전 → 후 |
|---|---|---|
| 서기 | 0.892 → 0.889 | 0.813 → 0.822 |
| 눕기 | 0.990 → 0.925 | 0.811 → 0.813 |
| 먹기 | 0.944 → 0.956 | 0.862 → 0.857 |

측정 조건: 새 농장은 맞춤 학습에 쓰지 않은 농장 A 확인용 영상 5개(130장), 기존 농장은 B·C 검증 세트(1,178장)

- 253장(카메라당 30장 이하)만으로 새 농장 재현율이 최대 0.135 올랐고, 기존 농장 변화는 ±0.014 이내입니다.

### 스마트폰 영상 실행

아이폰 기본 설정 영상(HEVC, 1920×1080, 세로·가로, 9–11초) 4개로 전체 흐름을 실행했습니다.

| 항목 | 결과 |
|---|---|
| 영상 열기·방향 보정 | 4개 모두 성공 |
| 처리 시간 (프레임 추출 + 추론) | 4.4–4.7초 (CPU 2코어 환경, 첫 호출 제외) |

---

## 사용법

```bash
pip install -r inference/behavior_yolo11s/requirements.txt
export BEHAVIOR_WEIGHTS=/path/to/best.pt
```

```python
import sys; sys.path.insert(0, "inference")
from behavior_yolo11s import analyze_video, load_detector, BehaviorError

detector = load_detector()            # 서버 시작 시 1회
try:
    result = analyze_video("/tmp/video.mp4", detector)   # BehaviorResponse
    body = result.model_dump(mode="json")
except BehaviorError as e:
    body = e.to_response()            # {"code": "BEHAVIOR-FAIL", ...}
```

- 입력은 로컬 파일 경로입니다. `video_url`에서 파일을 받는 일은 호출하는 서버가 맡습니다.

## 테스트

```bash
python -m pytest -q inference/behavior_yolo11s/tests   # 가중치 없이 실행 (가짜 검출기)
python inference/behavior_yolo11s/export_schema.py    # schemas/behavior.json 다시 생성
```

## 파일

| 파일 | 내용 |
|---|---|
| `behavior.py` | 영상 처리·추론·출력 |
| `schema.py` | 입출력 형식 (pydantic) |
| `export_schema.py` | `schemas/behavior.json` 생성 |
| `tests/` | 영상 처리·형식 테스트 11개 |
| `requirements.txt` | 동작 확인 버전 |
