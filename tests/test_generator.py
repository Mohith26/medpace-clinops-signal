import pandas as pd
import pytest

from trialscope.generator import (
    GeneratorConfig,
    SignalSpec,
    default_signals,
    dirty_batch,
    generate,
)

from .conftest import small_config


def test_same_seed_is_reproducible():
    a = generate(small_config(seed=21))
    b = generate(small_config(seed=21))
    for name in a.tables():
        pd.testing.assert_frame_equal(a.tables()[name], b.tables()[name])


def test_different_seed_differs():
    a = generate(small_config(seed=21))
    b = generate(small_config(seed=22))
    assert not a.subjects.equals(b.subjects)


def test_row_counts_match_config(small_batch):
    cfg = small_config()
    assert len(small_batch.studies) == cfg.n_studies
    assert len(small_batch.sites) == cfg.n_studies * cfg.sites_per_study
    assert len(small_batch.subjects) == cfg.n_studies * cfg.sites_per_study * cfg.subjects_per_site


def test_subject_fields_are_valid(small_batch):
    assert small_batch.subjects["age"].between(18, 84).all()
    assert set(small_batch.subjects["sex"]) <= {"F", "M"}
    assert set(small_batch.subjects["drug"]) <= set(small_config().drugs)


def test_primary_keys_unique(small_batch):
    for name, df in small_batch.tables().items():
        pk = df.columns[0]
        assert df[pk].is_unique, f"{name}.{pk} not unique"


def test_injected_signal_rate_is_elevated():
    cfg = GeneratorConfig(seed=5, n_studies=2, sites_per_study=6, subjects_per_site=40)
    cfg.signals = [SignalSpec("DRG-02", "EVT-03", 10.0)]
    batch = generate(cfg)
    on_drug = batch.subjects[batch.subjects["drug"] == "DRG-02"]["subject_id"]
    hits = batch.adverse_events[
        (batch.adverse_events["event_term"] == "EVT-03")
        & (batch.adverse_events["subject_id"].isin(on_drug))
    ]
    rate = len(hits) / len(on_drug)
    assert rate > 3 * cfg.base_event_rate


def test_truth_matches_config_signals(small_batch):
    assert small_batch.truth == [("DRG-01", "EVT-01"), ("DRG-03", "EVT-05")]


def test_default_signals_count_and_distinct():
    cfg = GeneratorConfig(seed=7)
    sigs = default_signals(cfg, n_signals=25)
    pairs = {(s.drug, s.event) for s in sigs}
    assert len(sigs) == 25 and len(pairs) == 25


def test_default_signals_deterministic():
    cfg = GeneratorConfig(seed=7)
    a = [(s.drug, s.event) for s in default_signals(cfg, n_signals=10)]
    b = [(s.drug, s.event) for s in default_signals(cfg, n_signals=10)]
    assert a == b


def test_dirty_ledger_counts(small_dirty):
    _, ledger = small_dirty
    for table in ("subjects", "visits", "adverse_events"):
        assert ledger.counts[(table, "missing_field")] == 5
        assert ledger.counts[(table, "invalid_value")] == 4
        assert ledger.counts[(table, "referential_orphan")] == 3
        assert ledger.counts[(table, "duplicate")] == 3
    assert ledger.total() == 45


def test_dirty_batch_row_growth(small_batch, small_dirty):
    dirty, _ = small_dirty
    assert len(dirty.subjects) == len(small_batch.subjects) + 3
    assert len(dirty.visits) == len(small_batch.visits) + 3
    assert len(dirty.adverse_events) == len(small_batch.adverse_events) + 3


def test_dirty_batch_does_not_mutate_original():
    batch = generate(small_config(seed=31))
    before = batch.subjects.copy(deep=True)
    dirty_batch(batch, seed=32, n_missing=5, n_invalid=4, n_orphan=3, n_duplicate=3)
    pd.testing.assert_frame_equal(batch.subjects, before)


def test_total_rows_sums_tables(small_batch):
    assert small_batch.total_rows() == sum(len(df) for df in small_batch.tables().values())
