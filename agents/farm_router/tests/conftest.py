import sys
from pathlib import Path

# agents/ 를 경로에 넣어 `farm_router` 패키지로, 저장소 루트를 넣어 `common` 을 불러온다
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "agents"), str(ROOT)]
