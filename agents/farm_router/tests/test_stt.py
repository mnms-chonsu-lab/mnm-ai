import subprocess
from types import SimpleNamespace

import pytest

from farm_router.stt import LIVESTOCK_TERMS, SttError, transcribe


def make_audio(path, sec, silent=False):
    src = "anullsrc=r=16000:cl=mono" if silent else "sine=frequency=440:sample_rate=16000"
    codec = ["-c:a", "libopus"] if path.suffix == ".webm" else ["-c:a", "aac"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src, "-t", str(sec), *codec, str(path)], check=True)
    return path


class FakeClient:
    def __init__(self, text="축사 소독했어요", exc=None):
        self.text, self.exc, self.kwargs = text, exc, None
        self.audio = SimpleNamespace(transcriptions=self)

    def with_options(self, **_):
        return self

    def create(self, file, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return SimpleNamespace(text=self.text)


@pytest.mark.parametrize("ext", [".webm", ".m4a"])
def test_ok(tmp_path, ext):
    c = FakeClient()
    out = transcribe(make_audio(tmp_path / f"a{ext}", 3), c)
    assert out.text == "축사 소독했어요" and 2.9 <= out.duration_sec <= 3.2
    assert c.kwargs["language"] == "ko" and c.kwargs["keywords"] == LIVESTOCK_TERMS
    assert all(t in c.kwargs["prompt"] for t in ["소독", "접종", "구제역", "럼피스킨", "귀표", "축사", "송아지", "사료"])


def test_too_long(tmp_path):
    with pytest.raises(SttError) as e:
        transcribe(make_audio(tmp_path / "a.m4a", 61), FakeClient())
    assert e.value.detail["reason"] == "too_long"


def test_silence(tmp_path):
    with pytest.raises(SttError) as e:
        transcribe(make_audio(tmp_path / "a.webm", 2, silent=True), FakeClient())
    assert e.value.to_response().code == "STT-FAIL" and e.value.detail["reason"] == "silence"


def test_bad_format(tmp_path):
    p = tmp_path / "a.mp3"
    p.write_bytes(b"x")
    with pytest.raises(SttError):
        transcribe(p, FakeClient())


@pytest.mark.parametrize("client", [FakeClient(text="  "), FakeClient(exc=ConnectionError())])
def test_no_text_or_api_error(tmp_path, client):
    with pytest.raises(SttError):
        transcribe(make_audio(tmp_path / "a.webm", 2), client)


def test_keywords_only_for_gpt_transcribe(tmp_path):
    c = FakeClient()
    transcribe(make_audio(tmp_path / "a.webm", 2), c, model="whisper-1")
    assert "keywords" not in c.kwargs and "prompt" in c.kwargs
