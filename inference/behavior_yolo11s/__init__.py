"""YOLO11s 소 행동 인식 추론 모듈."""

from .behavior import BehaviorError, analyze_video, load_detector

__all__ = ["BehaviorError", "analyze_video", "load_detector"]
