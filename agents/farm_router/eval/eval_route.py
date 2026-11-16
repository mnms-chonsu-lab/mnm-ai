"""라우터 평가 (명세 REQ-AI-01 완료 기준).

정확도: 문장 단위. intent·report_type·report_fields·missing 이 정답과 같고,
        items 의 (category, cow_tag4, head_count) 묶음이 같고, vaccine_type 정답이 예측에 포함되면 정답.
지연: 호출 시작~구조화 결과까지(초). p50·p95.
스키마 통과율: RouteResponse 검증을 통과한 비율 (LLM-FAIL 은 실패로 셈).
실행: OPENAI_API_KEY=... python agents/farm_router/eval/eval_route.py
"""
import json
import statistics
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[2])]

from farm_router.router import MODEL_VERSION, RouteError, make_client, route  # noqa: E402
from farm_router.schema import RouteRequest, RouteResponse  # noqa: E402

TODAY = date(2026, 11, 15)
COWS = [
    {"tag4": "1234", "desc": "큰 암소", "barn": "A1"},
    {"tag4": "5678", "desc": None, "barn": "B2"},
    {"tag4": "2468", "desc": "얼룩이", "barn": "C1"},
    {"tag4": "1357", "desc": None, "barn": "C2"},
]


def item_key(i):
    return (i["category"], i["cow_tag4"], i["head_count"])


def correct(pred: dict, gold: dict) -> tuple[bool, list[str]]:
    why = []
    for k in ("intent", "report_type"):
        if pred[k] != gold[k]:
            why.append(k)
    pf = {k: (str(v) if v is not None else None) for k, v in (pred["report_fields"] or {}).items()}
    for k, v in gold["report_fields"].items():
        if pf.get(k) != v:
            why.append(f"report_fields.{k}")
    if set(pred["missing"]) != set(gold["missing"]):
        why.append("missing")
    if Counter(map(item_key, pred["items"])) != Counter(map(item_key, gold["items"])):
        why.append("items")
    for g in gold["items"]:
        if g["vaccine_type"] and not any(g["vaccine_type"] in (p["vaccine_type"] or "") for p in pred["items"]):
            why.append("vaccine_type")
    return not why, why


def main(path=HERE / "route_eval_50.jsonl"):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    client = make_client()
    lat, ok, schema_ok, fails = [], 0, 0, []
    for n, g in enumerate(rows, 1):
        req = RouteRequest(text=g["text"], today=TODAY, farm_context={"cows": COWS})
        t0 = time.perf_counter()
        try:
            out = route(req, client)
            lat.append(time.perf_counter() - t0)
            RouteResponse.model_validate(out.model_dump())
            schema_ok += 1
            pred = json.loads(out.model_dump_json())
            good, why = correct(pred, g)
        except RouteError as e:
            lat.append(time.perf_counter() - t0)
            good, why, pred = False, [f"LLM-FAIL: {e.message}"], None
        ok += good
        if not good:
            fails.append({"n": n, "text": g["text"], "why": why, "pred": pred})
        print(f"[{n:02d}] {'O' if good else 'X'} {lat[-1]:.2f}s {g['text']}")

    q = statistics.quantiles(lat, n=20)
    result = {
        "model_version": MODEL_VERSION,
        "n": len(rows),
        "accuracy": round(ok / len(rows), 3),
        "schema_pass": round(schema_ok / len(rows), 3),
        "latency_p50_sec": round(statistics.median(lat), 2),
        "latency_p95_sec": round(q[18], 2),
        "fails": fails,
    }
    (HERE / "route_eval_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "fails"}, ensure_ascii=False, indent=2))
    print("기준: 정확도 ≥ 0.9 · p95 ≤ 4초 · 스키마 1.0")


if __name__ == "__main__":
    main()
