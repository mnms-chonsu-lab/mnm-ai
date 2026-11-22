import sys
from pathlib import Path

# 저장소 루트를 넣어 `mlops.serving` 으로 불러온다 (app.py 가 agents/·inference/ 경로를 직접 추가)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
