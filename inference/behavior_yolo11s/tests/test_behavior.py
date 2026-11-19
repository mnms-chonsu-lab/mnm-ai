"""영상 처리·출력 형식 테스트. 가중치 없이 가짜 검출기로 실행한다."""

import cv2
import numpy as np
import pytest

from behavior_yolo11s.behavior import BehaviorError, Detection, analyze_video, sample_times, summarize
from behavior_yolo11s.schema import BehaviorResponse


def _make_video(path, seconds, fps=10, w=320, h=240):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        vw.write(np.full((h, w, 3), i % 255, np.uint8))
    vw.release()


def fake(dets):
    return lambda frames: [list(dets) for _ in frames]


def D(label, conf):
    return Detection(label, conf, (10.0, 20.0, 110.0, 220.0))


@pytest.mark.parametrize(
    "dur,expected",
    [
        (10, [0, 2, 4, 6, 8]),
        (11.4, [0, 2, 4, 6, 8]),  # 10초 넘으면 처음 10초
        (30, [0, 2, 4, 6, 8]),
        (9.2669, [0, 1.853, 3.707, 5.56, 7.414]),  # 10초 미만이면 영상 길이 안에서
        (0, [0, 2, 4, 6, 8]),  # 길이를 모르면 10초 기준
    ],
)
def test_sample_times(dur, expected):
    assert sample_times(dur) == expected


def test_long_video_uses_first_10s(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, 12)
    r = analyze_video(str(p), fake([D("stand", 0.9)]))
    assert sorted({f.t for f in r.frames}) == [0, 2, 4, 6, 8]


def test_one_row_per_box_and_class_threshold(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, 10)
    dets = [D("stand", 0.9), D("lie", 0.45), D("lie", 0.39), D("eat", 0.49)]  # 0.39 눕기·0.49 먹기는 임계값 미만
    r = analyze_video(str(p), fake(dets))
    assert len(r.frames) == 5 * 2
    assert (r.summary.stand, r.summary.lie, r.summary.eat) == (50.0, 50.0, 0)
    assert r.frames[0].bbox == (10.0, 20.0, 110.0, 220.0)
    BehaviorResponse.model_validate(r.model_dump())  # 고정 형식 통과


def test_no_detection_returns_empty_result(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, 3)
    r = analyze_video(str(p), fake([]))
    assert r.frames == [] and r.main_label is None and r.summary.stand == 0


def test_unreadable_file(tmp_path):
    p = tmp_path / "broken.mp4"
    p.write_bytes(b"not a video")
    with pytest.raises(BehaviorError) as e:
        analyze_video(str(p), fake([]))
    assert e.value.to_response()["code"] == "BEHAVIOR-FAIL"


def test_detector_error(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, 3)

    def boom(frames):
        raise RuntimeError("x")

    with pytest.raises(BehaviorError) as e:
        analyze_video(str(p), boom)
    assert e.value.reason == "inference"


def test_summary_sums_to_100_and_tie_break():
    s, main = summarize([D("stand", 0.6), D("lie", 0.9), D("eat", 0.7)])
    assert round(s.stand + s.lie + s.eat, 1) == 100.0
    assert main == "lie"  # 1:1:1 이면 신뢰도 합이 큰 쪽


def test_timeout_is_behavior_fail(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, 3)
    with pytest.raises(BehaviorError) as e:
        analyze_video(str(p), fake([D("stand", 0.9)]), timeout_sec=0.0)
    assert e.value.reason == "timeout" and e.value.to_response()["code"] == "BEHAVIOR-FAIL"
