"""이미 끝난 실험 결과를 MLflow에 기록하고 서비스 중인 모델을 모델 버전으로 등록한다.

값은 다시 계산하지 않는다. 아래 README의 표를 그대로 옮긴다 (출처는 각 run 태그 `mnm.source`).
- inference/behavior_yolo11s/README.md
- inference/part_check_siglip2/README.md

같은 run(`mnm.key`)이 이미 있으면 건너뛴다. 여러 번 실행해도 기록이 늘지 않는다.

    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
    BEHAVIOR_WEIGHTS=/models/behavior/best.pt \
    SERVING_IMAGE=ghcr.io/mnms-chonsu-lab/mnm-ai-inference:<커밋 SHA> \
    python mlops/mlflow/log_existing.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

BEHAVIOR_SRC = "inference/behavior_yolo11s/README.md"
PART_SRC = "inference/part_check_siglip2/README.md"
# behavior.py · part_check.py 의 MODEL_VERSION. inference_log.model_version 에 남는 값과 같다
BEHAVIOR_MODEL_VERSION = "yolo11s-960-adaptA"
PART_MODEL_VERSION = "siglip2-b384-zs-A_simple-v1"
BEHAVIOR_SHA256_PREFIX = "5485f12b8d9cf044"  # README 기록값

BEHAVIOR_COMMON = {
    "arch": "yolo11s", "task": "detect", "classes": "standing,lying,eating", "imgsz": 960,
    "data": "AI-Hub 71471 한우 이미지 1920x1080",
}

# (key, run 이름, params, metrics, 측정 조건)
BEHAVIOR_RUNS = [
    ("behavior/base", "기본 학습 (농장 B·C)",
     {**BEHAVIOR_COMMON, "train_images": 10537, "train_farms": "B,C"},
     {"farmA_recall_standing": 0.765, "farmA_recall_lying": 0.798, "farmA_recall_eating": 0.968,
      "farmA_precision_standing": 0.892, "farmA_precision_lying": 0.990, "farmA_precision_eating": 0.944,
      "farmBC_recall_standing": 0.821, "farmBC_recall_lying": 0.806, "farmBC_recall_eating": 0.798,
      "farmBC_precision_standing": 0.813, "farmBC_precision_lying": 0.811, "farmBC_precision_eating": 0.862},
     "새 농장: 맞춤 학습에 쓰지 않은 농장 A 확인용 영상 5개(130장) / 기존 농장: B·C 검증 세트(1,178장)"),
    ("behavior/adaptA", "농장 맞춤 학습 (+농장 A 253장) — 서비스 중",
     {**BEHAVIOR_COMMON, "train_images": 10537, "train_farms": "B,C", "adapt_farm": "A",
      "adapt_images": 253, "adapt_max_per_camera": 30,
      "threshold_standing": 0.5, "threshold_lying": 0.4, "threshold_eating": 0.5},
     {"farmA_recall_standing": 0.840, "farmA_recall_lying": 0.933, "farmA_recall_eating": 0.977,
      "farmA_precision_standing": 0.889, "farmA_precision_lying": 0.925, "farmA_precision_eating": 0.956,
      "farmBC_recall_standing": 0.807, "farmBC_recall_lying": 0.805, "farmBC_recall_eating": 0.789,
      "farmBC_precision_standing": 0.822, "farmBC_precision_lying": 0.813, "farmBC_precision_eating": 0.857,
      "test_precision_standing": 0.937, "test_recall_standing": 0.816, "test_f1_standing": 0.872,
      "test_precision_lying": 0.976, "test_recall_lying": 0.695, "test_f1_lying": 0.812,
      "test_precision_eating": 0.962, "test_recall_eating": 0.824, "test_f1_eating": 0.888,
      "test_macro_f1": 0.857},
     ("farmA·farmBC: 위 run과 같은 세트 / test: 학습·맞춤 학습에 쓰지 않은 농장 A 영상 14개"
      "(카메라 9대, 1,274장, 정답 박스 7,344개), 임계값 고정 1회 평가")),
]

PART_COMMON = {"parts": "mouth,udder,hoof,flank", "finetune": "none (zero-shot)"}
PART_TEST = "시험 세트 81장 (404장을 비슷한 사진끼리 묶어 train 242 / val 81 / test 81), Colab T4 float16"
PART_RUNS = [
    ("part/1st/yolov8n-cls", "1차: YOLOv8n-cls 직접 학습 (3부위)",
     {"model": "yolov8n-cls", "parts": "mouth,hoof,none", "finetune": "yes"},
     {"cv_accuracy": 0.966, "external_none_rate": 0.48},
     "1차 실험 3부위, 시험 131장. external_none_rate: 외부 사진(럼피스킨 피부 197장)을 '해당 없음'으로 판정한 비율"),
    ("part/1st/clip-l14", "1차: CLIP ViT-L/14 제로샷 (3부위)",
     {"model": "CLIP ViT-L/14", "parts": "mouth,hoof,none", "finetune": "none (zero-shot)"},
     {"test_accuracy": 0.95, "cv5_accuracy": 0.82},
     "1차 실험 3부위. cv5_accuracy: 5-fold 교차검증"),
    ("part/2nd/clip-b16", "2차: CLIP ViT-B/16 (4부위)",
     {**PART_COMMON, "model": "CLIP ViT-B/16"},
     {"test_accuracy": 0.926, "test_macro_f1": 0.904, "latency_ms_t4": 6.7}, PART_TEST),
    ("part/2nd/siglip-b224", "2차: SigLIP base-224 (4부위)",
     {**PART_COMMON, "model": "SigLIP base-224"},
     {"test_accuracy": 0.901, "test_macro_f1": 0.894, "latency_ms_t4": 6.7}, PART_TEST),
    ("part/2nd/siglip2-b384", "2차: SigLIP 2 base-384 (4부위) — 서비스 중",
     {**PART_COMMON, "model": "google/siglip2-base-patch16-384", "prompt_set": "A_simple", "match_threshold": 0.0},
     {"test_accuracy": 0.938, "test_macro_f1": 0.930, "latency_ms_t4": 13.4,
      "val_macro_f1_prompt_A_simple": 0.837, "val_macro_f1_prompt_B_descriptive": 0.730,
      "val_macro_f1_prompt_C_minimal": 0.715,
      "pass_rate_requested_part": 0.938, "false_pass_rate_other_part": 0.021,
      "precision_mouth": 1.000, "recall_mouth": 0.884, "precision_udder": 1.000, "recall_udder": 1.000,
      "precision_hoof": 0.692, "recall_hoof": 1.000, "precision_flank": 0.933, "recall_flank": 1.000},
     PART_TEST + ". 프롬프트는 val 81장으로 선택"),
]

PART_CONFIG = {  # 제로샷이라 가중치를 따로 만들지 않았다. 서비스가 쓰는 설정이 곧 모델이다
    "model": "google/siglip2-base-patch16-384",
    "prompt_set": "A_simple",
    "prompts": {
        "mouth": ["a photo of the head and mouth of a cow",
                  "a close-up photo of a cow's muzzle, tongue or mouth"],
        "udder": ["a photo of the udder and teats of a cow", "a close-up photo of a cow's udder"],
        "hoof": ["a photo of the hoof and foot of a cow", "a close-up photo of a cow's hoof"],
        "flank": ["a photo of the side view of a whole cow body",
                  "a photo of a cow standing, seen from the side"],
    },
    "match_threshold": 0.0,
    "model_version": PART_MODEL_VERSION,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_run(client: MlflowClient, exp_id: str, key: str):
    runs = client.search_runs([exp_id], filter_string=f"tags.`mnm.key` = '{key}'", max_results=1)
    return runs[0] if runs else None


def log_runs(client: MlflowClient, exp_name: str, src: str, runs: list, commit: str) -> dict[str, str]:
    exp = client.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp else client.create_experiment(exp_name)
    ids = {}
    for key, name, params, metrics, condition in runs:
        found = find_run(client, exp_id, key)
        if found:
            print(f"  건너뜀 (이미 있음): {name}")
            ids[key] = found.info.run_id
            continue
        with mlflow.start_run(experiment_id=exp_id, run_name=name) as run:
            mlflow.set_tags({"mnm.key": key, "mnm.source": f"{src}@{commit}",
                             "mnm.condition": condition, "mnm.recorded": "기존 실험 결과 옮겨 적음 (재계산 아님)"})
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            ids[key] = run.info.run_id
        print(f"  기록: {name} (지표 {len(metrics)}개)")
    return ids


def register(client: MlflowClient, name: str, run_id: str, artifact: str, tags: dict, desc: str) -> None:
    try:
        client.create_registered_model(name, description=desc)
    except mlflow.exceptions.MlflowException:
        pass  # 이미 있음
    mv = next((v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == run_id), None)
    if mv:
        print(f"  건너뜀 (이미 등록): {name} v{mv.version}")
    else:
        mv = client.create_model_version(name, f"runs:/{run_id}/{artifact}", run_id=run_id, tags=tags)
        print(f"  등록: {name} v{mv.version}")
    # 별칭 2개: serving(지금 서비스 중) + inference_log.model_version 과 같은 이름
    for alias in ("serving", tags["model_version"]):
        client.set_registered_model_alias(name, alias, mv.version)
    print(f"  별칭: @serving, @{tags['model_version']} → v{mv.version}")


def main() -> int:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("MLFLOW_TRACKING_URI 가 비어 있어요", file=sys.stderr)
        return 2
    image = os.environ.get("SERVING_IMAGE", "")
    commit = os.environ.get("SOURCE_COMMIT", image.rsplit(":", 1)[-1][:7] if image else "develop")
    weights = Path(os.environ.get("BEHAVIOR_WEIGHTS", ""))
    if not weights.is_file():
        print(f"BEHAVIOR_WEIGHTS 파일이 없어요: {weights}", file=sys.stderr)
        return 2
    digest = sha256(weights)
    if not digest.startswith(BEHAVIOR_SHA256_PREFIX):
        print(f"가중치 sha256 이 README 기록과 달라요: {digest[:16]} != {BEHAVIOR_SHA256_PREFIX}", file=sys.stderr)
        return 3
    client = MlflowClient()

    print("[행동 인식] behavior_yolo11s")
    b = log_runs(client, "behavior_yolo11s", BEHAVIOR_SRC, BEHAVIOR_RUNS, commit)
    run_id = b["behavior/adaptA"]
    if not client.list_artifacts(run_id, "weights"):
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(weights), artifact_path="weights")
            mlflow.set_tag("weights_sha256", digest)
        print(f"  가중치 올림: {weights.name} ({weights.stat().st_size / 1e6:.1f}MB)")
    register(client, "behavior_yolo11s", run_id, "weights",
             {"model_version": BEHAVIOR_MODEL_VERSION, "weights_sha256": digest, "serving_image": image},
             "소 행동 인식 (서기·눕기·먹기). inference_log.model_version 과 버전 태그 model_version 이 같다")

    print("[부위 확인] part_check_siglip2")
    p = log_runs(client, "part_check_siglip2", PART_SRC, PART_RUNS, commit)
    run_id = p["part/2nd/siglip2-b384"]
    if not client.list_artifacts(run_id, "config"):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "part_check_config.json"
            cfg.write_text(json.dumps(PART_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(cfg), artifact_path="config")
        print("  설정 올림: part_check_config.json (제로샷: 모델 이름·프롬프트·기준값)")
    register(client, "part_check_siglip2", run_id, "config",
             {"model_version": PART_MODEL_VERSION, "hf_model": PART_CONFIG["model"], "serving_image": image},
             "촬영 부위 확인 (입·유방·발굽·몸 옆면). 제로샷이라 가중치 대신 설정을 등록")
    return 0


if __name__ == "__main__":
    sys.exit(main())
