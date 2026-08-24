"""Ingestion pipeline with explicit data quality gates.

Gates run per table in a fixed order and each bad row is assigned exactly
one error type, the first gate it fails:

  1. missing_field: any required column is null or empty
  2. invalid_value: enum outside its allowed set, non-parseable date,
     or numeric field outside its allowed range
  3. referential_orphan: foreign key not present in the raw parent table
  4. duplicate: primary key already seen earlier in the same table
     (first occurrence is kept, later ones are quarantined)

Referential checks validate against the raw parent tables on purpose, not
the post-quarantine ones. That keeps error attribution local: a corrupted
parent row does not cascade orphan errors onto otherwise clean children.

Late records are not quarantined. A record whose received_date trails its
source date by more than late_days_threshold is counted as a warning in
the report, since late data is an operational finding, not a defect.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd

DATE_FORMAT = "%Y-%m-%d"

ENUMS = {
    ("subjects", "sex"): {"F", "M"},
    ("subjects", "study_id"): None,
    ("visits", "visit_name"): {"screening", "baseline", "week4", "week8", "week12"},
    ("adverse_events", "severity"): {"mild", "moderate", "severe"},
}

RANGES = {("subjects", "age"): (18, 120)}

SCHEMAS = {
    "studies": {
        "pk": "study_id",
        "required": ["study_id", "phase", "therapeutic_area", "target_enrollment", "start_date"],
        "dates": ["start_date"],
        "fks": [],
        "late": None,
    },
    "sites": {
        "pk": "site_id",
        "required": ["site_id", "study_id", "country", "activation_date"],
        "dates": ["activation_date"],
        "fks": [("study_id", "studies", "study_id")],
        "late": None,
    },
    "subjects": {
        "pk": "subject_id",
        "required": ["subject_id", "site_id", "study_id", "drug", "enroll_date", "age", "sex"],
        "dates": ["enroll_date"],
        "fks": [("site_id", "sites", "site_id"), ("study_id", "studies", "study_id")],
        "late": None,
    },
    "visits": {
        "pk": "visit_id",
        "required": ["visit_id", "subject_id", "visit_name", "visit_date", "received_date"],
        "dates": ["visit_date", "received_date"],
        "fks": [("subject_id", "subjects", "subject_id")],
        "late": ("visit_date", "received_date"),
    },
    "adverse_events": {
        "pk": "ae_id",
        "required": ["ae_id", "subject_id", "drug", "event_term", "severity", "onset_date", "received_date"],
        "dates": ["onset_date", "received_date"],
        "fks": [("subject_id", "subjects", "subject_id")],
        "late": ("onset_date", "received_date"),
    },
}

ERROR_TYPES = ["missing_field", "invalid_value", "referential_orphan", "duplicate"]


@dataclass
class TableReport:
    table: str
    rows_in: int = 0
    rows_clean: int = 0
    error_counts: Dict[str, int] = field(default_factory=dict)
    late_records: int = 0

    def to_dict(self) -> Dict:
        return {
            "table": self.table,
            "rows_in": self.rows_in,
            "rows_clean": self.rows_clean,
            "rows_quarantined": self.rows_in - self.rows_clean,
            "error_counts": dict(self.error_counts),
            "late_records": self.late_records,
        }


@dataclass
class QualityReport:
    tables: Dict[str, TableReport] = field(default_factory=dict)

    def total_quarantined(self) -> int:
        return sum(t.rows_in - t.rows_clean for t in self.tables.values())

    def to_dict(self) -> Dict:
        return {
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "total_rows_in": sum(t.rows_in for t in self.tables.values()),
            "total_quarantined": self.total_quarantined(),
            "total_late_records": sum(t.late_records for t in self.tables.values()),
        }


@dataclass
class IngestResult:
    clean: Dict[str, pd.DataFrame]
    quarantine: Dict[str, pd.DataFrame]
    report: QualityReport


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or value == "" or pd.isna(value)


def _valid_date(value) -> bool:
    try:
        datetime.strptime(str(value), DATE_FORMAT)
        return True
    except (ValueError, TypeError):
        return False


def _row_error(table: str, row: Dict, parent_keys: Dict[str, Set],
               seen_pks: Set) -> Optional[str]:
    """Return the first failing error type for a row, or None if clean."""
    schema = SCHEMAS[table]

    for col in schema["required"]:
        if _is_missing(row[col]):
            return "missing_field"

    for col in schema["dates"]:
        if not _valid_date(row[col]):
            return "invalid_value"
    for (tbl, col), allowed in ENUMS.items():
        if tbl == table and allowed is not None and row[col] not in allowed:
            return "invalid_value"
    for (tbl, col), (lo, hi) in RANGES.items():
        if tbl == table:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                return "invalid_value"
            if not (lo <= v <= hi):
                return "invalid_value"

    for fk_col, parent, _parent_col in schema["fks"]:
        if row[fk_col] not in parent_keys[parent]:
            return "referential_orphan"

    pk = row[schema["pk"]]
    if pk in seen_pks:
        return "duplicate"
    seen_pks.add(pk)
    return None


def ingest(tables: Dict[str, pd.DataFrame],
           late_days_threshold: int = 14) -> IngestResult:
    """Run all quality gates over a batch of tables.

    Returns clean tables, quarantined rows (with an error_type column) and
    a QualityReport with per-table, per-error-type counts.
    """
    parent_keys: Dict[str, Set] = {}
    for name in SCHEMAS:
        df = tables.get(name, pd.DataFrame(columns=SCHEMAS[name]["required"]))
        parent_keys[name] = set(df[SCHEMAS[name]["pk"]].dropna())

    clean: Dict[str, pd.DataFrame] = {}
    quarantine: Dict[str, pd.DataFrame] = {}
    report = QualityReport()

    for name in SCHEMAS:
        df = tables.get(name)
        if df is None:
            continue
        schema = SCHEMAS[name]
        treport = TableReport(table=name, rows_in=len(df))
        seen_pks: Set = set()
        clean_rows: List[int] = []
        quar_rows: List[int] = []
        quar_errors: List[str] = []
        records = df.to_dict("records")

        for i, row in enumerate(records):
            err = _row_error(name, row, parent_keys, seen_pks)
            if err is None:
                clean_rows.append(i)
                if schema["late"] is not None:
                    src, recv = schema["late"]
                    delta = (datetime.strptime(row[recv], DATE_FORMAT)
                             - datetime.strptime(row[src], DATE_FORMAT)).days
                    if delta > late_days_threshold:
                        treport.late_records += 1
            else:
                quar_rows.append(i)
                quar_errors.append(err)
                treport.error_counts[err] = treport.error_counts.get(err, 0) + 1

        clean[name] = df.iloc[clean_rows].reset_index(drop=True)
        qdf = df.iloc[quar_rows].reset_index(drop=True)
        qdf = qdf.assign(error_type=quar_errors)
        quarantine[name] = qdf
        treport.rows_clean = len(clean_rows)
        report.tables[name] = treport

    return IngestResult(clean=clean, quarantine=quarantine, report=report)
