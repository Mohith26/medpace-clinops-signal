import pandas as pd
import pytest

from trialscope.generator import generate
from trialscope.quality import ingest

from .conftest import small_config


@pytest.fixture(scope="module")
def clean_result(small_batch):
    return ingest(small_batch.tables())


@pytest.fixture(scope="module")
def dirty_result(small_dirty):
    dirty, ledger = small_dirty
    return ingest(dirty.tables()), ledger


def test_clean_batch_has_zero_errors(clean_result):
    for treport in clean_result.report.tables.values():
        assert treport.error_counts == {}


def test_clean_batch_preserves_all_rows(clean_result, small_batch):
    for name, df in small_batch.tables().items():
        assert len(clean_result.clean[name]) == len(df)


def test_missing_field_count_exact(dirty_result):
    result, ledger = dirty_result
    for table in ("subjects", "visits", "adverse_events"):
        assert result.report.tables[table].error_counts["missing_field"] == ledger.counts[(table, "missing_field")]


def test_invalid_value_count_exact(dirty_result):
    result, ledger = dirty_result
    for table in ("subjects", "visits", "adverse_events"):
        assert result.report.tables[table].error_counts["invalid_value"] == ledger.counts[(table, "invalid_value")]


def test_referential_orphan_count_exact(dirty_result):
    result, ledger = dirty_result
    for table in ("subjects", "visits", "adverse_events"):
        assert result.report.tables[table].error_counts["referential_orphan"] == ledger.counts[(table, "referential_orphan")]


def test_duplicate_count_exact(dirty_result):
    result, ledger = dirty_result
    for table in ("subjects", "visits", "adverse_events"):
        assert result.report.tables[table].error_counts["duplicate"] == ledger.counts[(table, "duplicate")]


def test_full_ledger_matches_report(dirty_result):
    result, ledger = dirty_result
    for (table, err), injected in ledger.counts.items():
        assert result.report.tables[table].error_counts.get(err, 0) == injected


def test_duplicate_keeps_first_occurrence():
    batch = generate(small_config(seed=41))
    df = batch.subjects
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    tables = batch.tables()
    tables["subjects"] = dup
    result = ingest(tables)
    assert result.report.tables["subjects"].error_counts.get("duplicate", 0) == 1
    assert (result.clean["subjects"]["subject_id"] == df.iloc[0]["subject_id"]).sum() == 1


def test_quarantine_carries_error_type(dirty_result):
    result, _ = dirty_result
    q = result.quarantine["adverse_events"]
    assert "error_type" in q.columns
    assert set(q["error_type"]) <= {"missing_field", "invalid_value", "referential_orphan", "duplicate"}


def test_clean_plus_quarantine_partition(dirty_result):
    result, _ = dirty_result
    for name, treport in result.report.tables.items():
        assert len(result.clean[name]) + len(result.quarantine[name]) == treport.rows_in


def test_late_records_counted():
    batch = generate(small_config(seed=42))
    tables = batch.tables()
    v = tables["visits"].copy()
    v.iloc[0, v.columns.get_loc("visit_date")] = "2024-03-01"
    v.iloc[0, v.columns.get_loc("received_date")] = "2024-03-30"
    tables["visits"] = v
    base = ingest(batch.tables()).report.tables["visits"].late_records
    got = ingest(tables).report.tables["visits"].late_records
    assert got >= 1
    assert got != base or base > 0


def test_invalid_date_format_flagged():
    batch = generate(small_config(seed=43))
    tables = batch.tables()
    a = tables["adverse_events"].copy()
    a.iloc[0, a.columns.get_loc("onset_date")] = "03/15/2024"
    tables["adverse_events"] = a
    result = ingest(tables)
    assert result.report.tables["adverse_events"].error_counts.get("invalid_value", 0) == 1


def test_invalid_enum_flagged():
    batch = generate(small_config(seed=44))
    tables = batch.tables()
    a = tables["adverse_events"].copy()
    a.iloc[0, a.columns.get_loc("severity")] = "fatal"
    tables["adverse_events"] = a
    result = ingest(tables)
    assert result.report.tables["adverse_events"].error_counts.get("invalid_value", 0) == 1


def test_age_out_of_range_flagged():
    batch = generate(small_config(seed=45))
    tables = batch.tables()
    s = tables["subjects"].copy()
    s.iloc[0, s.columns.get_loc("age")] = 150
    tables["subjects"] = s
    result = ingest(tables)
    assert result.report.tables["subjects"].error_counts.get("invalid_value", 0) == 1


def test_non_numeric_age_flagged():
    batch = generate(small_config(seed=47))
    tables = batch.tables()
    s = tables["subjects"].copy()
    s["age"] = s["age"].astype(object)
    s.iloc[0, s.columns.get_loc("age")] = "unknown"
    tables["subjects"] = s
    result = ingest(tables)
    assert result.report.tables["subjects"].error_counts.get("invalid_value", 0) == 1


def test_missing_field_beats_other_errors():
    batch = generate(small_config(seed=46))
    tables = batch.tables()
    s = tables["subjects"].copy()
    s.iloc[0, s.columns.get_loc("drug")] = None
    s.iloc[0, s.columns.get_loc("sex")] = "X"
    tables["subjects"] = s
    result = ingest(tables)
    counts = result.report.tables["subjects"].error_counts
    assert counts.get("missing_field", 0) == 1
    assert counts.get("invalid_value", 0) == 0


def test_report_totals_consistent(dirty_result):
    result, ledger = dirty_result
    d = result.report.to_dict()
    assert d["total_quarantined"] == ledger.total()
    assert d["total_rows_in"] == sum(t["rows_in"] for t in d["tables"].values())


def test_missing_table_is_skipped(small_batch):
    tables = small_batch.tables()
    tables.pop("visits")
    result = ingest(tables)
    assert "visits" not in result.clean
    assert result.report.tables["subjects"].error_counts == {}
