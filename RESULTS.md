# Benchmark and validation notes

Everything below was measured on my machine: Apple silicon (arm64), Python 3.9.6, single thread, no network. Raw outputs are committed under `results/`.

## Quality gate exactness

Command:

```
.venv/bin/python scripts/run_eval.py 7
```

The eval batch is 15,835 rows (4 studies, 60 sites, 3,000 subjects plus visits and AEs) with 117 corrupted rows injected by a seeded ledger: 12 missing-field, 10 invalid-value, 8 referential-orphan and 9 duplicate rows in each of subjects, visits and adverse_events. All 12 per-table, per-error-type detected counts matched the injected counts exactly (`all_counts_exact: true` in `results/quality_gates.json`). The late-record counter is reported separately as a warning and is not part of quarantine.

## Signal detection against injected truth

Same command as above, output in `results/eval_signals.json`. Universe: 360 drug and event pairs (12 drugs x 30 events), 25 injected signals at multiplier 6.0 over a 0.02 base event rate, 2,329 clean AE reports after quarantine.

| criterion | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| PRR (PRR>=2, chi2>=4, n>=3) | 25 | 3 | 0 | 332 | 0.8929 | 1.0000 | 0.9434 |
| ROR (CI low > 1, n>=3) | 25 | 3 | 0 | 332 | 0.8929 | 1.0000 | 0.9434 |

The two criteria produced identical confusion counts on this seed. That is seed and effect-size specific; they diverge on smaller datasets (the API default dataset with 1,000 subjects shows different FP sets per method).

Formula correctness is separately pinned by hand-computed fixtures in `tests/fixtures/two_by_two_cases.json` (PRR, ROR, Yates chi-square, both CIs, including zero-cell cases) and a scipy `chi2_contingency` cross-check.

## Ingestion throughput

```
.venv/bin/python scripts/bench_ingest.py
```

15,835 rows through all gates, warmup plus 5 timed runs: best 0.1655 s, median 0.1667 s, which is 95,689 rows/s best and 95,004 rows/s median. Single thread, in process. Output: `results/bench_ingest.json`.

## API latency (in-process TestClient)

```
.venv/bin/python scripts/bench_api.py
```

300 requests per endpoint after 20 warmup calls, in-process FastAPI TestClient, so this measures handler and serialization cost only, no network or server process.

| endpoint | p50 ms | p95 ms |
|---|---|---|
| /studies | 7.115 | 7.885 |
| /studies/STU-001/metrics | 17.995 | 18.695 |
| /signals?method=prr | 1.317 | 1.379 |
| /quality | 0.937 | 1.037 |

The per-study metrics endpoint recomputes site risk on every request, which is why it is the slow one. Output: `results/bench_api.json`.

## Tests and coverage

```
.venv/bin/python -m pytest --color=no --cov=trialscope --cov-report=term
```

93 passed in about 1.5 s. Coverage over the `trialscope` package: 100 percent (438 statements, 0 missed).
