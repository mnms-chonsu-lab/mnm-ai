"""사진 화질 검사 (명세 계산식 ⑨). 앱에서 따로 계산하지 않는다.

- 어두움: 흑백 평균 밝기 < 60/255 (명세 값)
- 흐림: 라플라시안 응답 분산 < BLUR_THRESHOLD (제안값, 실측으로 조정)
- 측정은 방향 보정한 원본을 긴 변 512px 로 줄여서 한다 (해상도에 따른 차이를 줄임)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .schema import Quality

DARK_THRESHOLD = 60.0
BLUR_THRESHOLD = 100.0  # 제안값, 실측으로 조정
MEASURE_SIDE = 512


def _gray(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    w, h = g.size
    s = MEASURE_SIDE / max(w, h)
    if s < 1:
        g = g.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32)


def laplacian_var(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    c = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * c
    return float(lap.var())


def check_quality(img: Image.Image) -> Quality:
    g = _gray(img)
    brightness = float(g.mean())
    return Quality(
        dark=brightness < DARK_THRESHOLD,
        blurry=laplacian_var(g) < BLUR_THRESHOLD,
        brightness=round(brightness, 1),
    )
