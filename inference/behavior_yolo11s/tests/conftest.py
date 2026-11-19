import sys
from pathlib import Path

# inference/ 를 경로에 넣어 `behavior_yolo11s` 패키지로 불러온다
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
