import pandas as pd
import pytest

from trialscope.signals import contingency_tables, detect_signals, evaluate_pair, score_against_truth


@pytest.fixture()
def tiny_aes():
    rows = []
    spec = [("D1", "E1", 4), ("D1", "E2", 6), ("D2", "E1", 1), ("D2", "E2", 9)]
    i = 0
    for drug, event, n in spec:
        for _ in range(n):
            i += 1
            rows.append(
                {
                    "ae_id": f"AE-{i:04d}",
                    "subject_id": f"SUBJ-{i:04d}",
                    "drug": drug,
                    "event_term": event,
                    "severity": "mild",
                    "onset_date": "2024-02-01",
                    "received_date": "2024-02-03",
                }
            )
    return pd.DataFrame(rows)


def test_contingency_known_cells(tiny_aes):
    tables = contingency_tables(tiny_aes)
    assert tables[("D1", "E1")] == (4, 6, 1, 9)
    assert tables[("D2", "E2")] == (9, 1, 6, 4)


def test_contingency_cells_sum_to_total(tiny_aes):
    tables = contingency_tables(tiny_aes)
    for cells in tables.values():
        assert sum(cells) == len(tiny_aes)


def test_contingency_explicit_universe_includes_zero_pairs(tiny_aes):
    tables = contingency_tables(tiny_aes, drugs=["D1", "D2", "D3"], events=["E1", "E2"])
    assert tables[("D3", "E1")] == (0, 0, 5, 15)


def test_detect_signals_covers_full_universe(tiny_aes):
    stats = detect_signals(tiny_aes, drugs=["D1", "D2", "D3"], events=["E1", "E2", "E3"])
    assert len(stats) == 9


def test_score_confusion_counts():
    stats = [
        evaluate_pair("D", "E1", 10, 90, 20, 880),
        evaluate_pair("D", "E2", 3, 97, 30, 1870),
        evaluate_pair("D", "E3", 0, 100, 50, 850),
    ]
    truth = [("D", "E1"), ("D", "E2")]
    scored = score_against_truth(stats, truth, "prr")
    assert scored["tp"] == 1 and scored["fn"] == 1 and scored["fp"] == 0 and scored["tn"] == 1
    assert scored["precision"] == 1.0
    assert scored["recall"] == 0.5


def test_score_counts_false_positive():
    stats = [evaluate_pair("D", "E1", 10, 90, 20, 880)]
    scored = score_against_truth(stats, [], "prr")
    assert scored["fp"] == 1 and scored["tp"] == 0 and scored["precision"] == 0.0


def test_score_zero_division_guarded():
    stats = [evaluate_pair("D", "E", 0, 100, 50, 850)]
    scored = score_against_truth(stats, [], "ror")
    assert scored["precision"] == 0.0 and scored["recall"] == 0.0 and scored["f1"] == 0.0


def test_end_to_end_detects_strong_injected_signal():
    from trialscope.generator import GeneratorConfig, SignalSpec, generate

    cfg = GeneratorConfig(seed=17, n_studies=2, sites_per_study=5,
                          subjects_per_site=30, n_drugs=5)
    cfg.signals = [SignalSpec("DRG-01", "EVT-01", 12.0)]
    batch = generate(cfg)
    stats = detect_signals(batch.adverse_events)
    flagged = {(s.drug, s.event) for s in stats if s.prr_signal or s.ror_signal}
    assert ("DRG-01", "EVT-01") in flagged
