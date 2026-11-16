"""`schemas/part_check.json` 생성. 실행: python inference/part_check_siglip2/export_schema.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from part_check_siglip2.schema import export  # noqa: E402

out = HERE.parents[2] / "schemas" / "part_check.json"
export(out)
print(f"저장: {out}")
