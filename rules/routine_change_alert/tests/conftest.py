import sys
from pathlib import Path

# rules/ 를 경로에 넣어 각 규칙 폴더를 패키지로 불러온다
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
