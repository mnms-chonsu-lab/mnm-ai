"""YOLO11s 소 행동 인식 (명세 REQ-AI-11).

처리 순서
1. 영상 열기 (mp4·mov, HEVC 포함). 세로 영상은 회전 정보로 방향 보정
2. 처음 10초(짧으면 영상 길이)에서 5장을 같은 간격으로 추출 → 10초 영상이면 0·2·4·6·8초
3. YOLO11s 에 960px 로 입력 (학습과 같은 크기)
4. 클래스별 임계값을 넘은 박스만 남김
5. 박스 1개당 1줄 + 박스 수 기준 비율(%) + 가장 많은 행동
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .schema import BehaviorFrame, BehaviorResponse, BehaviorSummary

MAX_SEC = 10.0
N_FRAMES = 5
IMGSZ = 960
LABELS = ("stand", "lie", "eat")
MODEL_VERSION = "yolo11s-960-adaptA"

# 가중치에 저장된 클래스 이름 → 출력 라벨
MODEL_NAMES = {0: "standing", 1: "lying", 2: "eating"}
NAME_TO_LABEL = {"standing": "stand", "lying": "lie", "eating": "eat"}
# 클래스별 임계값 (검증 세트로 정한 값)
CLASS_THRESHOLD = {"stand": 0.5, "lie": 0.4, "eat": 0.5}
TIMEOUT_SEC = 8.0  # 명세 REQ-AI-12 일반 타임아웃. 넘으면 BEHAVIOR-FAIL (REQ-AI-11)


@dataclass
class Detection:
    label: str  # stand | lie | eat
    confidence: float
    bbox: tuple[float, float, float, float]  # 픽셀 x1, y1, x2, y2


# 프레임 여러 장 → 프레임별 검출 목록. 모델을 바꿔도 이 모양만 지키면 된다.
Detector = Callable[[Sequence[np.ndarray]], list[list[Detection]]]


class BehaviorError(Exception):
    """BEHAVIOR-FAIL. reason: open | no_frames | inference | timeout"""

    code = "BEHAVIOR-FAIL"

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason

    def to_response(self) -> dict:
        return {"code": self.code, "message": str(self), "detail": {"reason": self.reason}}


def sample_times(duration_sec: float, n: int = N_FRAMES, max_sec: float = MAX_SEC) -> list[float]:
    """처음 max_sec 초(짧으면 영상 길이)를 n 칸으로 나눈 각 칸의 시작 시각.

    길이를 알 수 없으면(0 이하) max_sec 기준으로 뽑고, 읽히지 않는 시각은 건너뛴다.
    """
    span = min(duration_sec, max_sec) if duration_sec > 0 else max_sec
    return [round(i * span / n, 3) for i in range(n)]


def read_frames(path: str) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise BehaviorError("open", "영상 파일을 열 수 없습니다")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # 세로 영상 방향 보정 (기본값이지만 명시)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = count / fps if fps > 0 and count > 0 else 0.0
        frames = []
        for t in sample_times(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((t, frame))
    finally:
        cap.release()
    if not frames:
        raise BehaviorError("no_frames", "영상에서 프레임을 읽지 못했습니다")
    return frames


def summarize(dets: Sequence[Detection]) -> tuple[BehaviorSummary, Optional[str]]:
    """박스 수 기준 비율(%), 소수 1자리, 합 100. main_label 은 가장 많은 행동(같으면 신뢰도 합이 큰 쪽)."""
    if not dets:
        return BehaviorSummary(stand=0, lie=0, eat=0), None
    cnt = Counter(d.label for d in dets)
    pct = {k: round(cnt[k] * 100 / len(dets), 1) for k in LABELS}
    diff = round(100 - sum(pct.values()), 1)
    if diff:
        top = max(LABELS, key=lambda k: pct[k])
        pct[top] = round(pct[top] + diff, 1)
    conf_sum = {k: sum(d.confidence for d in dets if d.label == k) for k in LABELS}
    main = max(LABELS, key=lambda k: (cnt[k], conf_sum[k]))
    return BehaviorSummary(**pct), main


def analyze_video(
    path: str, detector: Detector, model_version: str = MODEL_VERSION, timeout_sec: Optional[float] = TIMEOUT_SEC
) -> BehaviorResponse:
    """로컬 영상 파일 경로 → BehaviorResponse. 실패·시간 초과 시 BehaviorError."""
    t0 = time.perf_counter()
    frames = read_frames(path)
    try:
        per_frame = detector([f for _, f in frames])
    except Exception as e:
        raise BehaviorError("inference", f"행동 인식 중 오류: {type(e).__name__}") from e
    if timeout_sec is not None and time.perf_counter() - t0 > timeout_sec:
        raise BehaviorError("timeout", f"{timeout_sec}초를 넘었습니다")

    rows: list[BehaviorFrame] = []
    kept: list[Detection] = []
    for (t, _), dets in zip(frames, per_frame):
        for d in dets:
            thr = CLASS_THRESHOLD.get(d.label)
            if thr is None or d.confidence < thr:
                continue
            kept.append(d)
            rows.append(
                BehaviorFrame(
                    t=t,
                    label=d.label,
                    confidence=round(d.confidence, 3),
                    bbox=tuple(round(v, 1) for v in d.bbox),
                )
            )
    summary, main = summarize(kept)
    return BehaviorResponse(frames=rows, summary=summary, main_label=main, model_version=model_version)


def load_detector(weights_path: Optional[str] = None) -> Detector:
    """가중치(best.pt)를 불러 Detector 를 만든다. 경로는 인자 또는 환경변수 BEHAVIOR_WEIGHTS."""
    from ultralytics import YOLO

    path = weights_path or os.environ.get("BEHAVIOR_WEIGHTS")
    if not path:
        raise ValueError("가중치 경로가 없습니다 (인자 또는 BEHAVIOR_WEIGHTS)")
    model = YOLO(path)
    if dict(model.names) != MODEL_NAMES:
        raise ValueError(f"가중치 클래스가 예상과 다릅니다: {model.names}")
    min_thr = min(CLASS_THRESHOLD.values())

    def detect(frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        results = model.predict(list(frames), imgsz=IMGSZ, conf=min_thr, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            out.append(
                [
                    Detection(
                        label=NAME_TO_LABEL[MODEL_NAMES[int(c)]],
                        confidence=float(p),
                        bbox=tuple(float(v) for v in xyxy),
                    )
                    for c, p, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())
                ]
            )
        return out

    return detect
