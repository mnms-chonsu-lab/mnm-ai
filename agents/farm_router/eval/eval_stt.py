"""음성 인식 평가 (명세 REQ-AI-02 완료 기준).

입력 폴더에 음성 파일(webm·m4a)과 같은 이름의 .txt(정답 문장)를 둔다.
정확 인식: 띄어쓰기·문장부호를 빼고 정답과 같으면 O.
지연: transcribe() 전체(길이 확인 포함) 시간. p95.
실행: OPENAI_API_KEY=... python agents/farm_router/eval/eval_stt.py <폴더>
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[2])]

from farm_router.stt import MODEL_VERSION, SttError, make_client, transcribe  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[\s\.,!?~·]", "", s)


def main(folder: Path, repeat: int = 5):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".webm", ".m4a"})
    client = make_client()
    rows, lat = [], []
    for p in files:
        gold = p.with_suffix(".txt").read_text(encoding="utf-8").strip()
        for r in range(repeat):  # 같은 파일을 여러 번 불러 지연 분포를 본다
            t0 = time.perf_counter()
            try:
                out = transcribe(p, client)
                text, dur = out.text, out.duration_sec
            except SttError as e:
                text, dur = f"STT-FAIL: {e.message}", None
            lat.append(time.perf_counter() - t0)
            if r == 0:
                rows.append({"file": p.name, "duration_sec": dur, "gold": gold, "pred": text, "exact": norm(text) == norm(gold)})
                print(f"{'O' if rows[-1]['exact'] else 'X'} {p.name} ({dur}s) → {text}")
    result = {
        "model_version": MODEL_VERSION,
        "files": len(files),
        "calls": len(lat),
        "exact": sum(r["exact"] for r in rows),
        "latency_p50_sec": round(statistics.median(lat), 2),
        "latency_p95_sec": round(statistics.quantiles(lat, n=20)[18], 2) if len(lat) >= 2 else round(lat[0], 2),
        "rows": rows,
    }
    (HERE / "stt_eval_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print("기준: 15초 발화 p95 ≤ 3초 · 시연 문장 3종 정확 인식")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
