"""처리 흐름·출력 형식 테스트. 모델 없이 가짜 임베더로 실행한다."""

import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from part_check_siglip2 import part_check as pc
from part_check_siglip2.quality import check_quality
from part_check_siglip2.schema import PartCheckResponse


class FakeEmbedder:
    """사진 평균색으로 부위를 정하는 가짜 임베더. 빨강=입, 초록=유방, 파랑=발굽, 회색=몸 옆면."""

    text_matrix = np.eye(4, dtype=np.float32)

    def __init__(self):
        self.seen = []

    def encode_image(self, img):
        self.seen.append(img.size)
        r, g, b = np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(0)
        if max(r, g, b) - min(r, g, b) < 20:
            v = np.array([0, 0, 0, 1.0])
        else:
            v = np.zeros(4)
            v[int(np.argmax([r, g, b]))] = 1.0
        v = v + 0.05
        return v / np.linalg.norm(v)


def _jpg(color, size=(400, 300), noise=True):
    a = np.full((size[1], size[0], 3), color, np.uint8)
    if noise:  # 흐림 판정에 걸리지 않도록 질감 추가
        a = np.clip(a.astype(int) + np.random.default_rng(0).integers(-40, 40, a.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_prompts_and_parts_match_notebook():
    assert pc.PARTS == ("mouth", "udder", "hoof", "flank")
    assert pc.PROMPTS["hoof"] == ["a photo of the hoof and foot of a cow", "a close-up photo of a cow's hoof"]
    assert pc.MODEL_ID == "google/siglip2-base-patch16-384"
    assert (pc.STD_SIZE, pc.JPEG_QUALITY, pc.TEMPERATURE) == (512, 95, 100.0)


@pytest.mark.parametrize(
    "color,requested,predicted,match",
    [
        ((200, 60, 60), "mouth", "mouth", True),
        ((60, 200, 60), "mouth", "udder", False),
        ((60, 60, 200), "hoof", "hoof", True),
        ((128, 128, 128), "flank", "flank", True),
    ],
)
def test_match_rule(color, requested, predicted, match):
    r = pc.check_part(_jpg(color), requested, FakeEmbedder())
    assert (r.predicted_part, r.match) == (predicted, match)
    assert 0.25 <= r.score <= 1.0  # 4부위 softmax 1등 점수는 0.25 이상
    PartCheckResponse.model_validate(r.model_dump())


def test_threshold_applies_on_top_of_argmax():
    class Unsure(FakeEmbedder):  # 입이 1등이지만 점수가 낮은 경우
        def encode_image(self, img):
            v = np.array([0.51, 0.5, 0.5, 0.5])
            return v / np.linalg.norm(v)

    r = pc.check_part(_jpg((200, 60, 60)), "mouth", Unsure(), threshold=0.5)
    assert r.predicted_part == "mouth" and r.score < 0.5 and r.match is False
    r0 = pc.check_part(_jpg((200, 60, 60)), "mouth", Unsure())  # 1안(기준값 0)이면 통과
    assert r0.match is True


def test_model_input_is_512_square_regardless_of_aspect():
    emb = FakeEmbedder()
    pc.check_part(_jpg((200, 60, 60), size=(900, 1600)), "mouth", emb)
    assert emb.seen == [(512, 512)]


def test_exif_orientation_applied():
    a = np.zeros((100, 200, 3), np.uint8)  # 가로 200 × 세로 100
    im = Image.fromarray(a)
    exif = Image.Exif()
    exif[0x0112] = 6  # 90도 회전 정보
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif.tobytes())
    assert pc.open_image(buf.getvalue()).size == (100, 200)


def test_heic_opens():
    try:
        import pillow_heif  # noqa: F401
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (200, 60, 60)).save(buf, "HEIF")
    except Exception:
        pytest.skip("HEIF 저장을 지원하지 않는 환경")
    assert pc.open_image(buf.getvalue()).size == (64, 48)


def test_broken_file_is_part_fail():
    with pytest.raises(pc.PartCheckError) as e:
        pc.check_part(b"not an image", "hoof", FakeEmbedder())
    assert e.value.to_response() == {"code": "PART-FAIL", "message": str(e.value), "detail": {"reason": "open"}}


def test_unknown_part_is_rejected():
    with pytest.raises(pc.PartCheckError) as e:
        pc.check_part(_jpg((200, 60, 60)), "tail", FakeEmbedder())
    assert e.value.reason == "bad_request"


def test_model_error_is_part_fail():
    class Boom(FakeEmbedder):
        def encode_image(self, img):
            raise RuntimeError("x")

    with pytest.raises(pc.PartCheckError) as e:
        pc.check_part(_jpg((200, 60, 60)), "hoof", Boom())
    assert e.value.reason == "inference"


def test_timeout_is_part_fail():
    with pytest.raises(pc.PartCheckError) as e:
        pc.check_part(_jpg((200, 60, 60)), "hoof", FakeEmbedder(), timeout_sec=0.0)
    assert e.value.reason == "timeout"


def test_scores_are_softmax_100_cosine():
    p = pc.part_scores(np.array([1.0, 0, 0, 0]), np.eye(4))
    assert np.isclose(p.sum(), 1.0) and p.argmax() == 0 and p[0] > 0.99


def test_quality_dark_and_blurry():
    q = check_quality(Image.new("RGB", (640, 480), (20, 20, 20)))
    assert q.dark and q.blurry and q.brightness == 20.0


def test_quality_sharp_bright():
    a = (np.indices((256, 256)).sum(axis=0) // 8 % 2 * 200 + 30).astype(np.uint8)
    q = check_quality(Image.fromarray(a))
    assert not q.dark and not q.blurry


def test_quality_blurred():
    a = (np.indices((256, 256)).sum(axis=0) // 8 % 2 * 200 + 30).astype(np.uint8)
    q = check_quality(Image.fromarray(a).filter(ImageFilter.GaussianBlur(6)))
    assert q.blurry and not q.dark


class FakeVision:
    model_version = "vision-fake"

    def __init__(self, part="hoof", conf=0.8):
        self.part, self.conf, self.called = part, conf, 0

    def classify(self, img):
        self.called += 1
        assert img.size == (pc.STD_SIZE, pc.STD_SIZE)
        return self.part, self.conf


def test_default_timeout_is_spec_1_5s():
    assert pc.TIMEOUT_SEC == 1.5


def test_fallback_on_timeout_same_format():
    v = FakeVision("hoof", 0.8)
    out = pc.check_part(_jpg((200, 60, 60)), "hoof", FakeEmbedder(), timeout_sec=0.0, vision=v)
    assert v.called == 1 and out.fallback is True and out.predicted_part == "hoof" and out.match is True
    assert out.model_version == "vision-fake" and set(out.model_dump()) == set(PartCheckResponse.model_fields)


def test_fallback_on_model_error():
    class Boom(FakeEmbedder):
        def encode_image(self, img):
            raise RuntimeError("x")

    out = pc.check_part(_jpg((200, 60, 60)), "hoof", Boom(), vision=FakeVision("mouth", 0.9))
    assert out.fallback is True and out.match is False


def test_no_fallback_when_siglip_ok_or_photo_broken():
    v = FakeVision()
    assert pc.check_part(_jpg((60, 60, 200)), "hoof", FakeEmbedder(), timeout_sec=None, vision=v).fallback is False
    with pytest.raises(pc.PartCheckError):
        pc.check_part(b"not an image", "hoof", FakeEmbedder(), vision=v)
    assert v.called == 0


def test_vision_fallback_client_call():
    from types import SimpleNamespace

    class C:
        def with_options(self, **_):
            return self

        def __init__(self):
            self.responses = self

        def parse(self, **kw):
            self.kw = kw
            return SimpleNamespace(status="completed", output_parsed=kw["text_format"](part="udder", confidence=1.7))

    c = C()
    part, score = pc.VisionFallback(c).classify(Image.new("RGB", (512, 512)))
    assert (part, score) == ("udder", 1.0) and c.kw["store"] is False
    assert c.kw["input"][0]["content"][0]["image_url"].startswith("data:image/jpeg;base64,")


def test_load_vision_off_by_default(monkeypatch):
    monkeypatch.delenv("PART_FALLBACK_VISION", raising=False)
    assert pc.load_vision() is None
