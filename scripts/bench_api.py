"""API latency benchmark with the in-process FastAPI TestClient.

No network, no server process: numbers measure handler plus serialization
cost only. 300 requests per endpoint after 20 warmup calls, single
thread. Writes results/bench_api.json.
Usage: .venv/bin/python scripts/bench_api.py
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from trialscope.api import create_app

RESULTS = Path(__file__).resolve().parents[1] / "results"

ENDPOINTS = [
    "/studies",
    "/studies/STU-001/metrics",
    "/signals?method=prr",
    "/quality",
]


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main() -> None:
    app = create_app(seed=7, n_signals=25)
    client = TestClient(app)
    out = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "note": "in-process TestClient, single thread, Apple silicon, no network",
        "requests_per_endpoint": 300,
        "endpoints": {},
    }
    for ep in ENDPOINTS:
        for _ in range(20):
            assert client.get(ep).status_code == 200
        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            r = client.get(ep)
            times.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200
        out["endpoints"][ep] = {
            "p50_ms": round(percentile(times, 0.50), 3),
            "p95_ms": round(percentile(times, 0.95), 3),
            "mean_ms": round(statistics.mean(times), 3),
        }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bench_api.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
