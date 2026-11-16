"""`schemas/behavior.json` 생성. 실행: python inference/behavior_yolo11s/export_schema.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from behavior_yolo11s.schema import export  # noqa: E402

out = HERE.parents[2] / "schemas" / "behavior.json"
export(out)
print(f"저장: {out}")
