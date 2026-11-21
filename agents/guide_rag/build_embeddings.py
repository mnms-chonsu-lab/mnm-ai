"""biosecurity_rule.embedding 채우는 SQL 생성 (mnm-data migrations/0003, vector(1536)).

입력: DB 에서 내보낸 행 (CSV 또는 JSON/JSONL). 필요한 컬럼: biosecurity_rule_id, text
출력: UPDATE 문 SQL 파일. DB 적용은 mnm-data 담당자가 한다.

    python -m guide_rag.build_embeddings rows.csv out.sql
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from .guide import EMBED_MODEL

DIM = 1536  # text-embedding-3-small 출력 차원 = migrations/0003 vector(1536)
BATCH = 100


def read_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.suffix == ".csv":
        with p.open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    raw = p.read_text(encoding="utf-8").strip()
    return json.loads(raw) if raw.startswith("[") else [json.loads(x) for x in raw.splitlines() if x.strip()]


def build_sql(rows: list[dict], client, model: str = EMBED_MODEL) -> str:
    out = ["BEGIN;"]
    for i in range(0, len(rows), BATCH):
        part = rows[i : i + BATCH]
        res = client.embeddings.create(model=model, input=[r["text"] for r in part])
        for r, d in zip(part, res.data):
            if len(d.embedding) != DIM:
                raise ValueError(f"임베딩 차원 {len(d.embedding)} ≠ {DIM} (model={model})")
            rid = int(r["biosecurity_rule_id"])
            vec = "[" + ",".join(f"{x:.7f}" for x in d.embedding) + "]"
            out.append(f"UPDATE biosecurity_rule SET embedding = '{vec}' WHERE biosecurity_rule_id = {rid};")
    out.append("COMMIT;")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.exit("사용법: python -m guide_rag.build_embeddings <rows.csv|json> <out.sql>")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY 가 설정되지 않음")
    from openai import OpenAI

    rows = read_rows(argv[0])
    Path(argv[1]).write_text(build_sql(rows, OpenAI()), encoding="utf-8")
    print(f"{len(rows)}행 → {argv[1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
