# TrialScope

I wanted to understand how pharmacovigilance signal detection actually works under the hood, so I built a small clinical operations data product around it: a seeded synthetic trial dataset, an ingestion pipeline with strict quality gates, an analytics API, and a disproportionality engine that I can score against known ground truth because I injected the signals myself.

Everything in here is synthetic. The drug names are tokens like DRG-01, the event terms are tokens like EVT-01, and no real patient data, real trial data, or real MedDRA dictionary is used anywhere.

## How the pieces fit

1. `trialscope/generator.py` produces five tables (studies, sites, subjects, visits, adverse events) from a single seed. A config lists injected drug and event pairs whose adverse event probability is multiplied by a factor, and those pairs are returned as ground truth labels. A second function corrupts a copy of the batch with an exact, seeded ledger of defects: blanked required fields, illegal enum and date values, foreign keys pointed at parents that do not exist, and duplicated primary keys. It also simulates operationally late records, where the received date trails the source date by weeks.
2. `trialscope/quality.py` is the ingestion gate. Every row gets assigned at most one error type, the first gate it fails, in a fixed order: missing field, invalid value, referential orphan, duplicate. Bad rows land in a quarantine frame with their error type attached, and the report counts errors per table and per type. Because the corruption ledger is exact and the gates assign one error per row, I can test that detected counts equal injected counts exactly, not approximately.
3. `trialscope/signals.py` implements PRR and ROR on standard 2x2 contingency tables. The PRR criterion (PRR >= 2, Yates chi-square >= 4, at least 3 cases) follows Evans, Waller and Davis 2001, Pharmacoepidemiology and Drug Safety 10:483-486. The ROR criterion (lower 95 percent confidence bound above 1, at least 3 cases) follows van Puijenbroek et al 2002, same journal, 11:3-10. Zero cells get the Haldane-Anscombe 0.5 correction for ROR. The formulas are verified against hand-computed fixtures checked into `tests/fixtures/`, and the chi-square is cross-checked against scipy.
4. `trialscope/metrics.py` computes enrollment velocity per study and per-site risk indicators (enrollment shortfall, AE reports per subject, severe AE share, late record share) rolled into a normalized composite score.
5. `trialscope/api.py` serves it all with FastAPI: `/studies`, `/studies/{id}/metrics`, `/signals`, and `/quality`.

## What the numbers came out to

On the evaluation dataset (3,000 subjects, 360 drug and event pairs, 25 injected signals at 6x the 2 percent base rate, seed 7):

- Quality gates: all 12 per-table, per-error-type detected counts matched the injected ledger exactly (117 corrupted rows across subjects, visits and adverse events).
- PRR criterion: precision 0.893, recall 1.000, F1 0.943 (25 TP, 3 FP, 0 FN, 332 TN).
- ROR criterion: identical confusion counts on this seed, so also 0.893 / 1.000 / 0.943.
- Ingestion throughput: about 95k rows per second single threaded on Apple silicon.
- API latency via the in-process TestClient: p50 under 8 ms on `/studies`, around 18 ms on the heaviest per-study metrics endpoint.

Exact numbers, machine notes and reproduce commands are in RESULTS.md, and the raw JSON lives in `results/`.

## Running it

```
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/run_eval.py
.venv/bin/python scripts/bench_ingest.py
.venv/bin/python scripts/bench_api.py
.venv/bin/uvicorn "trialscope.api:create_app" --factory
```

## Limitations

- The data is synthetic and generous to the detector. Injected signals are clean multiplicative rate bumps on independent events; real spontaneous reporting data has confounding, stimulated reporting, duplicates that are not exact copies, and terms that overlap. Precision and recall here say the implementation is correct, not that PRR would perform this well in the wild.
- The two criteria agreeing exactly on my eval seed is a property of this dataset size and effect size, not a general fact. On a smaller dataset the two methods do diverge, which is visible in the per-pair output.
- Event terms are flat tokens. There is no MedDRA hierarchy, so nothing here handles term grouping, which matters a lot in real signal detection.
- The quality gates assign one error per row by design. A row with two defects is counted under the first failing gate only, which keeps counts exact for testing but understates total defect volume.
- Referential checks validate against raw parent tables, so a quarantined parent does not cascade orphan errors onto its children. That is a deliberate attribution choice, and the other convention is defensible too.
- Latency numbers are in-process TestClient measurements on my machine (Apple silicon, single thread). They exclude network and server process overhead, so treat them as handler cost, not deployment numbers.
- No auth, no dashboard, no persistence layer. The API rebuilds its dataset at startup from a seed.
