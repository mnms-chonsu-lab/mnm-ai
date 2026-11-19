# mnm-ai 내부 API 서버 이미지 (GHCR: ghcr.io/mnms-chonsu-lab/mnm-ai-inference)
# 빌드 (저장소 루트에서): docker build -f mlops/serving/Dockerfile -t mnm-ai-inference .
#
# 이미지에 넣지 않는 것 (실행할 때 붙임)
# - 행동 인식 가중치 best.pt → /models/behavior/best.pt 로 마운트 (BEHAVIOR_WEIGHTS)
# - SigLIP 2 가중치 → 첫 실행 때 Hugging Face 에서 받아 /models/hf 볼륨에 저장 (HF_HOME)
# - 키·토큰 → env_file

FROM python:3.12-slim

# ffmpeg: STT 음성 길이·무음 확인 (agents/farm_router) · libglib2.0-0: opencv
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 는 CPU 전용 휠을 먼저 설치 (기본 휠은 CUDA 포함이라 수 GB)
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 의존성 먼저 (코드만 바뀌면 이 층은 다시 받지 않음)
COPY agents/farm_router/requirements.txt agents/farm_router/
COPY agents/guide_rag/requirements.txt agents/guide_rag/
COPY agents/receiver_copilot/requirements.txt agents/receiver_copilot/
COPY inference/part_check_siglip2/requirements.txt inference/part_check_siglip2/
COPY inference/behavior_yolo11s/requirements.txt inference/behavior_yolo11s/
COPY mlops/serving/requirements.txt mlops/serving/
RUN pip install --no-cache-dir -r mlops/serving/requirements.txt

COPY common/ common/
COPY rules/ rules/
COPY agents/ agents/
COPY inference/ inference/
COPY mlops/serving/ mlops/serving/

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models/hf /models/behavior \
 && chown -R app:app /models
USER app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    BEHAVIOR_WEIGHTS=/models/behavior/best.pt \
    OMP_NUM_THREADS=2

EXPOSE 8000

# 모델을 불러오는 동안(첫 실행은 SigLIP 다운로드 포함) 헬스 체크가 실패하지 않게 여유를 둔다
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# 모델을 프로세스마다 따로 올리므로 메모리 4GB 서버에서는 worker 1개
CMD ["uvicorn", "mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
