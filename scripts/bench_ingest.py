"""Ingestion throughput benchmark.

Times ingest() over a seeded dirty batch, best and median of 5 runs,
single thread, in process. Writes results/bench_ingest.json.
Usage: .venv/bin/python scripts/bench_ingest.py
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialscope.generator import GeneratorConfig, default_signals, dirty_batch, generate
from trialscope.quality import ingest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    config = GeneratorConfig(seed=7, n_studies=4, sites_per_study=15, subjects_per_site=50)
    config.signals = default_signals(config, n_signals=25)
    batch = generate(config)
    dirty, _ = dirty_batch(batch, seed=8)
    tables = dirty.tables()
    total_rows = dirty.total_rows()

    ingest(tables)  # warmup
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        ingest(tables)
        times.append(time.perf_counter() - t0)

    best = min(times)
    median = statistics.median(times)
    out = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "note": "single thread, in process, Apple silicon",
        "total_rows": total_rows,
        "runs": 5,
        "best_seconds": round(best, 4),
        "median_seconds": round(median, 4),
        "rows_per_sec_best": round(total_rows / best, 1),
        "rows_per_sec_median": round(total_rows / median, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bench_ingest.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
