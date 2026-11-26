"""SigLIP 2 제로샷 부위 일치 검증 (명세 REQ-AI-03).

전처리·프롬프트·점수는 실험 노트북(dataset_work, 셀 14·17·18·21)과 같게 둔다.
같아야 시험 결과(4부위 81장 정확도 0.938)가 이 코드의 근거가 된다.

처리 순서
1. 사진 열기 (jpg·png·heic) → EXIF 방향 보정 → RGB
2. 화질 검사 (어두움·흐림)
3. 512×512 로 늘려 맞춤 (BICUBIC, 비율 유지 안 함) → JPEG 품질 95 로 다시 저장 (노트북 셀 14)
4. SigLIP 2 이미지 임베딩 (정규화)
5. 부위별 문장 임베딩(문장 2개 평균, 정규화)과 코사인 → softmax(100 × 코사인) (셀 21)
6. 점수가 가장 높은 부위 = predicted_part, 그 점수 = score
7. match = predicted_part == requested_part 그리고 score >= MATCH_THRESHOLD

OpenAI Vision 사용 (대체 경로가 켜져 있을 때만. 켜기/끄기: 환경변수 PART_FALLBACK_VISION=1, 어드민 설정 A-S07)
- 대체: SigLIP 2 추론 실패·시간 초과 → Vision 으로 판정 (fallback = true). Vision 도 실패하면 PART-FAIL
- 재확인: SigLIP 2 결과가 나오면 Vision 도 항상 본다. Vision 이 실패하면 SigLIP 2 결과 그대로 (PART-FAIL 아님)
  - SigLIP 2 가 요청 부위와 같다고 함 → Vision 이 none 이면 불통과(Vision 결과), 아니면 SigLIP 2 결과 그대로
  - SigLIP 2 가 요청 부위와 다르다고 함 → Vision 결과를 따른다 (fallback = true)
- Vision 은 4부위 또는 none(소 부위가 아님·알아볼 수 없음)을 고른다. none 이면 match = false
- 두 단계 합쳐 TOTAL_BUDGET_SEC 안에 끝나도록 Vision 제한 시간을 남은 시간으로 줄인다
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Literal, Optional, Protocol, Sequence, Union

import numpy as np
from PIL import Image, ImageOps

from .quality import check_quality
from .schema import PartCheckResponse

try:  # 아이폰 기본 형식(HEIC) 열기
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

MODEL_ID = "google/siglip2-base-patch16-384"
MODEL_VERSION = "siglip2-b384-zs-A_simple-v1"
PARTS = ("mouth", "udder", "hoof", "flank")  # 노트북 CLASSES4 순서

# 노트북 셀 17 A_simple (검증 세트로 선택된 프롬프트)
PROMPTS = {
    "mouth": ["a photo of the head and mouth of a cow", "a close-up photo of a cow's muzzle, tongue or mouth"],
    "udder": ["a photo of the udder and teats of a cow", "a close-up photo of a cow's udder"],
    "hoof": ["a photo of the hoof and foot of a cow", "a close-up photo of a cow's hoof"],
    "flank": ["a photo of the side view of a whole cow body", "a photo of a cow standing, seen from the side"],
}

STD_SIZE = 512
JPEG_QUALITY = 95
TEMPERATURE = 100.0
# 1안: 예측 부위 = 요청 부위면 통과. 4부위 softmax 에서 1등 점수는 항상 0.25 이상이라
# 0.25 이하 값은 결과를 바꾸지 않는다. 이미지별 점수로 다시 정하면 이 값만 바꾼다.
MATCH_THRESHOLD = 0.0
TIMEOUT_SEC = 1.5  # 명세 REQ-AI-03·12: 부위 검증 1.5초 초과 시 PART-FAIL. (Colab CPU 실측 p95 3.61초 → GPU 서빙 필요)
VISION_MODEL = os.getenv("PART_VISION_MODEL", "gpt-5.4-mini")
VISION_TIMEOUT_SEC = 8.0  # 대체 경로는 명세 REQ-AI-12 일반 타임아웃
# SigLIP 2 + Vision 합계 상한. 백엔드 호출 제한(mnm-dev HttpAiClient.PART_CHECK_TIMEOUT 11초)에서
# 사진 내려받기·네트워크 여유를 뺀 값 (제안값, 실측으로 조정)
TOTAL_BUDGET_SEC = 9.0
MIN_VISION_SEC = 1.0  # 남은 시간이 이보다 적으면 Vision 을 부르지 않는다
NONE_PART = "none"  # Vision 이 소 부위를 찾지 못했을 때


class PartCheckError(Exception):
    """PART-FAIL. reason: open | inference | timeout(1.5초 초과) | bad_request | fallback(대체 경로 실패·시간 부족)"""

    code = "PART-FAIL"

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason

    def to_response(self) -> dict:
        return {"code": self.code, "message": str(self), "detail": {"reason": self.reason}}


class Embedder(Protocol):
    """이미지 1장 → 정규화된 임베딩 벡터. text_matrix 는 PARTS 순서로 (4, d)."""

    text_matrix: np.ndarray

    def encode_image(self, img: Image.Image) -> np.ndarray: ...


def open_image(src: Union[str, bytes]) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        raise PartCheckError("open", f"사진을 열 수 없습니다: {type(e).__name__}") from e


def standardize(img: Image.Image) -> Image.Image:
    """노트북 셀 14 와 같은 512×512 변환 + JPEG 품질 95 저장본."""
    im = img.resize((STD_SIZE, STD_SIZE), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def part_scores(emb: np.ndarray, text_matrix: np.ndarray) -> np.ndarray:
    """softmax(100 × 코사인), PARTS 순서."""
    logits = TEMPERATURE * (text_matrix @ emb)
    logits = logits - logits.max()
    p = np.exp(logits)
    return p / p.sum()


def check_part(
    src: Union[str, bytes],
    requested_part: str,
    embedder: Embedder,
    threshold: float = MATCH_THRESHOLD,
    timeout_sec: Optional[float] = TIMEOUT_SEC,
    vision: Optional["VisionFallback"] = None,
    second_opinion: bool = True,
) -> PartCheckResponse:
    """사진(파일 경로 또는 바이트) + 요청 부위 → PartCheckResponse. 실패 시 PartCheckError.

    vision 이 주어지면
    - SigLIP 2 추론 실패·시간 초과 때 Vision 으로 판정한다 (대체, fallback = true)
    - second_opinion 이면 모든 사진을 Vision 으로 다시 본다. SigLIP 2 가 다르다고 했거나 Vision 이 none 이면
      Vision 결과를 따른다 (재확인, fallback = true)
    """
    if requested_part not in PARTS:
        raise PartCheckError("bad_request", f"알 수 없는 부위: {requested_part}")
    t0 = time.perf_counter()
    img = open_image(src)
    quality = check_quality(img)
    std = standardize(img)
    try:
        try:
            emb = embedder.encode_image(std)
            p = part_scores(emb, embedder.text_matrix)
        except Exception as e:
            raise PartCheckError("inference", f"부위 인식 중 오류: {type(e).__name__}") from e
        if timeout_sec is not None and time.perf_counter() - t0 > timeout_sec:
            raise PartCheckError("timeout", f"{timeout_sec}초를 넘었습니다")
    except PartCheckError:
        if vision is None:
            raise
        # 대체: SigLIP 2 결과가 없으므로 Vision 실패는 그대로 PART-FAIL
        predicted, score = vision.classify(std, timeout_sec=_vision_time_left(t0))
        return _response(requested_part, predicted, score, threshold, quality, vision.model_version, fallback=True)

    k = int(np.argmax(p))
    siglip = _response(requested_part, PARTS[k], float(p[k]), threshold, quality, MODEL_VERSION, fallback=False)
    if vision is None or not second_opinion:
        return siglip
    # 재확인: 모든 사진을 Vision 이 다시 본다. Vision 이 실패하면 SigLIP 2 결과 유지
    try:
        predicted, score = vision.classify(std, timeout_sec=_vision_time_left(t0))
    except PartCheckError:
        return siglip
    if siglip.match and predicted != NONE_PART:
        # SigLIP 2 가 맞다고 했고 Vision 도 소 부위로 봤으면 SigLIP 2 결과 그대로 (Vision 이 다른 부위를 골라도 통과)
        return siglip
    # SigLIP 2 가 다르다고 했거나, Vision 이 소 부위가 아니라고(none) 했으면 Vision 결과를 따른다
    return _response(requested_part, predicted, score, threshold, quality, vision.model_version, fallback=True)


def _vision_time_left(t0: float) -> float:
    """Vision 제한 시간 = min(VISION_TIMEOUT_SEC, 전체 예산 - 지금까지 쓴 시간). 너무 적으면 부르지 않는다."""
    left = min(VISION_TIMEOUT_SEC, TOTAL_BUDGET_SEC - (time.perf_counter() - t0))
    if left < MIN_VISION_SEC:
        raise PartCheckError("fallback", f"대체 경로를 부를 시간이 부족합니다 (남은 시간 {max(left, 0):.1f}초)")
    return left


def _response(requested, predicted, score, threshold, quality, version, fallback) -> PartCheckResponse:
    return PartCheckResponse(
        requested_part=requested,
        predicted_part=predicted,
        match=(predicted != NONE_PART and predicted == requested and score >= threshold),
        score=round(score, 4),
        quality=quality,
        model_version=version,
        fallback=fallback,
    )


VISION_INSTRUCTIONS = (
    "You classify which body part of a cow the photo mainly shows. "
    "Choose exactly one: mouth (head, mouth, muzzle, lips, tongue, gums, dental pad, inside of the mouth, "
    "including close-ups of the mouth), udder (udder, teats), "
    "hoof (hoof, foot, lower leg), flank (side of the whole body). "
    "Choose none only if the photo does not show a cow, or no cow body part can be recognized "
    "(for example too dark, too blurry, or only the floor, a wall, feed or a person). "
    "If a cow body part is visible, choose the part even if the photo is not ideal. "
    "confidence is your own estimate between 0 and 1. Do not describe disease."
)


class VisionFallback:
    """OpenAI Vision 으로 4부위 또는 none 중 하나를 고른다 (대체·재확인 공용).

    score 는 모델이 스스로 매긴 확신도(0~1)라 SigLIP 2 의 softmax 점수와 같은 척도가 아니다.
    """

    def __init__(self, client, model: str = VISION_MODEL, timeout_sec: float = VISION_TIMEOUT_SEC):
        from pydantic import BaseModel, ConfigDict

        class _Pick(BaseModel):
            model_config = ConfigDict(extra="forbid")
            part: Literal["mouth", "udder", "hoof", "flank", "none"]
            confidence: float

        self.client, self.model, self.timeout_sec, self._Pick = client, model, timeout_sec, _Pick
        self.model_version = f"{model}+vision-v2"

    def classify(self, img: Image.Image, timeout_sec: Optional[float] = None) -> tuple[str, float]:
        """timeout_sec 를 주면 그 값과 self.timeout_sec 중 작은 값으로 제한한다."""
        limit = self.timeout_sec if timeout_sec is None else min(self.timeout_sec, timeout_sec)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        t0 = time.perf_counter()
        try:
            resp = self.client.with_options(timeout=limit, max_retries=0).responses.parse(
                model=self.model,
                instructions=VISION_INSTRUCTIONS,
                input=[{"role": "user", "content": [{"type": "input_image", "image_url": data_url}]}],
                text_format=self._Pick,
                store=False,
            )
        except Exception as e:
            raise PartCheckError("fallback", f"대체 경로 호출 실패: {type(e).__name__}") from e
        if time.perf_counter() - t0 > limit:
            raise PartCheckError("fallback", f"대체 경로 {limit:.1f}초 초과")
        out = getattr(resp, "output_parsed", None)
        if getattr(resp, "status", "completed") != "completed" or out is None:
            raise PartCheckError("fallback", "대체 경로 결과 없음")
        return out.part, min(max(float(out.confidence), 0.0), 1.0)


def load_vision() -> Optional[VisionFallback]:
    """PART_FALLBACK_VISION=1 이고 OPENAI_API_KEY 가 있으면 대체 경로를 켠다. 아니면 None."""
    if os.getenv("PART_FALLBACK_VISION", "0") not in ("1", "true", "on") or not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    return VisionFallback(OpenAI())


class SiglipEmbedder:
    """transformers SigLIP 2 임베더 (노트북 셀 18 과 같은 방식). 서버 시작 시 1회 생성."""

    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.text_matrix = np.stack([self._encode_texts(PROMPTS[p]).mean(0) for p in PARTS])
        self.text_matrix /= np.linalg.norm(self.text_matrix, axis=1, keepdims=True)
        self.encode_image(Image.new("RGB", (STD_SIZE, STD_SIZE)))  # 워밍업

    @staticmethod
    def _to_tensor(o):
        import torch

        if torch.is_tensor(o):
            return o
        if getattr(o, "pooler_output", None) is not None:
            return o.pooler_output
        return o[0]

    def _normalize(self, e) -> np.ndarray:
        e = self.torch.nn.functional.normalize(e.float(), dim=-1)
        return e.cpu().numpy()

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tok = getattr(self.proc, "tokenizer", self.proc)
        enc = tok([t.lower() for t in texts], padding="max_length", max_length=64,
                  truncation=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            e = self._to_tensor(self.model.get_text_features(input_ids=enc["input_ids"]))
        return self._normalize(e)

    def encode_image(self, img: Image.Image) -> np.ndarray:
        x = self.proc(images=[img], return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        with self.torch.no_grad():
            e = self._to_tensor(self.model.get_image_features(pixel_values=x))
        return self._normalize(e)[0]


def load_embedder(model_id: str = MODEL_ID) -> SiglipEmbedder:
    return SiglipEmbedder(model_id)
