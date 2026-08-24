"""End-to-end evaluation: gate exactness plus signal precision/recall/F1.

Generates a seeded eval dataset (larger than the API default), dirties it
with a known corruption ledger, ingests it, verifies the quality report
matches the ledger exactly, then scores PRR and ROR criteria against the
injected drug-event signal ground truth over the full drug x event
universe.

Writes results/quality_gates.json and results/eval_signals.json.
Usage: .venv/bin/python scripts/run_eval.py [seed]
"""

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialscope.generator import GeneratorConfig, default_signals, dirty_batch, generate
from trialscope.quality import ingest
from trialscope.signals import detect_signals, score_against_truth

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(seed: int = 7) -> None:
    config = GeneratorConfig(seed=seed, n_studies=4, sites_per_study=15, subjects_per_site=50)
    config.signals = default_signals(config, n_signals=25)
    batch = generate(config)
    dirty, ledger = dirty_batch(batch, seed=seed + 1)
    result = ingest(dirty.tables())

    gate_rows = []
    all_exact = True
    for (table, err), injected in sorted(ledger.counts.items()):
        detected = result.report.tables[table].error_counts.get(err, 0)
        exact = detected == injected
        all_exact = all_exact and exact
        gate_rows.append(
            {"table": table, "error_type": err, "injected": injected,
             "detected": detected, "exact": exact}
        )

    quality_out = {
        "seed": seed,
        "machine": platform.machine(),
        "rows_in_batch": dirty.total_rows(),
        "per_error_type": gate_rows,
        "all_counts_exact": all_exact,
        "report": result.report.to_dict(),
    }

    stats = detect_signals(result.clean["adverse_events"],
                           drugs=config.drugs, events=config.events)
    eval_out = {
        "seed": seed,
        "n_subjects": len(batch.subjects),
        "n_ae_reports_clean": len(result.clean["adverse_events"]),
        "universe_pairs": len(stats),
        "injected_signals": len(batch.truth),
        "base_event_rate": config.base_event_rate,
        "signal_multiplier": 6.0,
        "prr": score_against_truth(stats, batch.truth, "prr"),
        "ror": score_against_truth(stats, batch.truth, "ror"),
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "quality_gates.json").write_text(json.dumps(quality_out, indent=2))
    (RESULTS / "eval_signals.json").write_text(json.dumps(eval_out, indent=2))
    print(json.dumps({"gates_all_exact": all_exact,
                      "prr": eval_out["prr"], "ror": eval_out["ror"]}, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
