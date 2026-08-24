"""Seeded synthetic CTMS and safety data generator.

Everything here is synthetic. No real patient data, no real drug names,
no real MedDRA terms. Drug and event vocabularies are made-up tokens.

The generator produces five tables as pandas DataFrames:
  studies, sites, subjects, visits, adverse_events

Injected signals: a list of (drug, event) pairs whose per-subject AE
probability is multiplied by a configured factor. The injected pairs are
returned as ground truth so a detector can be scored against them.

dirty_batch() takes a clean batch and applies a fixed, seeded set of
corruptions to disjoint rows, returning both the dirty batch and an exact
ledger of how many rows of each corruption type were injected per table.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEVERITIES = ["mild", "moderate", "severe"]
SEXES = ["F", "M"]
PHASES = ["1", "2", "3"]
THERAPEUTIC_AREAS = ["oncology", "cardiology", "neurology", "metabolic"]
VISIT_NAMES = ["screening", "baseline", "week4", "week8", "week12"]


@dataclass
class SignalSpec:
    """One injected drug-event signal with an elevated AE rate."""

    drug: str
    event: str
    multiplier: float = 6.0


@dataclass
class GeneratorConfig:
    seed: int = 7
    n_studies: int = 4
    sites_per_study: int = 10
    subjects_per_site: int = 25
    n_drugs: int = 12
    n_events: int = 30
    base_event_rate: float = 0.02
    late_fraction: float = 0.08
    late_days_threshold: int = 14
    start_date: date = date(2024, 1, 8)
    signals: List[SignalSpec] = field(default_factory=list)

    @property
    def drugs(self) -> List[str]:
        return [f"DRG-{i:02d}" for i in range(1, self.n_drugs + 1)]

    @property
    def events(self) -> List[str]:
        return [f"EVT-{i:02d}" for i in range(1, self.n_events + 1)]


@dataclass
class Batch:
    """A generated batch of clean tables plus signal ground truth."""

    studies: pd.DataFrame
    sites: pd.DataFrame
    subjects: pd.DataFrame
    visits: pd.DataFrame
    adverse_events: pd.DataFrame
    truth: List[Tuple[str, str]]

    def tables(self) -> Dict[str, pd.DataFrame]:
        return {
            "studies": self.studies,
            "sites": self.sites,
            "subjects": self.subjects,
            "visits": self.visits,
            "adverse_events": self.adverse_events,
        }

    def total_rows(self) -> int:
        return sum(len(t) for t in self.tables().values())


def default_signals(config: GeneratorConfig, n_signals: int, seed: Optional[int] = None,
                    multiplier: float = 6.0) -> List[SignalSpec]:
    """Pick n_signals distinct (drug, event) pairs uniformly at random."""
    rng = np.random.default_rng(config.seed if seed is None else seed)
    pairs = [(d, e) for d in config.drugs for e in config.events]
    idx = rng.choice(len(pairs), size=n_signals, replace=False)
    return [SignalSpec(pairs[i][0], pairs[i][1], multiplier) for i in sorted(idx)]


def generate(config: GeneratorConfig) -> Batch:
    """Generate a deterministic clean batch from the seed in config."""
    rng = np.random.default_rng(config.seed)
    signal_map = {(s.drug, s.event): s.multiplier for s in config.signals}

    studies = []
    sites = []
    subjects = []
    visits = []
    aes = []

    subject_counter = 0
    ae_counter = 0
    visit_counter = 0

    for st in range(1, config.n_studies + 1):
        study_id = f"STU-{st:03d}"
        studies.append(
            {
                "study_id": study_id,
                "phase": PHASES[int(rng.integers(0, len(PHASES)))],
                "therapeutic_area": THERAPEUTIC_AREAS[int(rng.integers(0, len(THERAPEUTIC_AREAS)))],
                "target_enrollment": int(config.sites_per_study * config.subjects_per_site * 1.2),
                "start_date": config.start_date.isoformat(),
            }
        )
        for si in range(1, config.sites_per_study + 1):
            site_id = f"{study_id}-S{si:02d}"
            activation = config.start_date + timedelta(days=int(rng.integers(0, 30)))
            sites.append(
                {
                    "site_id": site_id,
                    "study_id": study_id,
                    "country": ["US", "DE", "JP", "BR"][int(rng.integers(0, 4))],
                    "activation_date": activation.isoformat(),
                }
            )
            for _ in range(config.subjects_per_site):
                subject_counter += 1
                subject_id = f"SUBJ-{subject_counter:06d}"
                drug = config.drugs[int(rng.integers(0, len(config.drugs)))]
                enroll = activation + timedelta(days=int(rng.integers(0, 120)))
                subjects.append(
                    {
                        "subject_id": subject_id,
                        "site_id": site_id,
                        "study_id": study_id,
                        "drug": drug,
                        "enroll_date": enroll.isoformat(),
                        "age": int(rng.integers(18, 85)),
                        "sex": SEXES[int(rng.integers(0, 2))],
                    }
                )

                n_visits = int(rng.integers(2, len(VISIT_NAMES) + 1))
                for v in range(n_visits):
                    visit_counter += 1
                    vdate = enroll + timedelta(days=28 * v)
                    delay = int(rng.integers(0, 5))
                    if rng.random() < config.late_fraction:
                        delay = config.late_days_threshold + int(rng.integers(1, 30))
                    visits.append(
                        {
                            "visit_id": f"VIS-{visit_counter:07d}",
                            "subject_id": subject_id,
                            "visit_name": VISIT_NAMES[v],
                            "visit_date": vdate.isoformat(),
                            "received_date": (vdate + timedelta(days=delay)).isoformat(),
                        }
                    )

                for event in config.events:
                    p = config.base_event_rate * signal_map.get((drug, event), 1.0)
                    if rng.random() < p:
                        ae_counter += 1
                        onset = enroll + timedelta(days=int(rng.integers(1, 100)))
                        delay = int(rng.integers(0, 7))
                        if rng.random() < config.late_fraction:
                            delay = config.late_days_threshold + int(rng.integers(1, 45))
                        aes.append(
                            {
                                "ae_id": f"AE-{ae_counter:07d}",
                                "subject_id": subject_id,
                                "drug": drug,
                                "event_term": event,
                                "severity": SEVERITIES[int(rng.integers(0, 3))],
                                "onset_date": onset.isoformat(),
                                "received_date": (onset + timedelta(days=delay)).isoformat(),
                            }
                        )

    return Batch(
        studies=pd.DataFrame(studies),
        sites=pd.DataFrame(sites),
        subjects=pd.DataFrame(subjects),
        visits=pd.DataFrame(visits),
        adverse_events=pd.DataFrame(aes),
        truth=[(s.drug, s.event) for s in config.signals],
    )


@dataclass
class CorruptionLedger:
    """Exact injected corruption counts, keyed by (table, error_type)."""

    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def add(self, table: str, error_type: str, n: int) -> None:
        self.counts[(table, error_type)] = self.counts.get((table, error_type), 0) + n

    def total(self) -> int:
        return sum(self.counts.values())


def dirty_batch(batch: Batch, seed: int = 99,
                n_missing: int = 12, n_invalid: int = 10,
                n_orphan: int = 8, n_duplicate: int = 9) -> Tuple[Batch, CorruptionLedger]:
    """Apply seeded corruptions to disjoint rows of a copy of the batch.

    Corruption types (all applied to subjects, visits and adverse_events):
      missing_field: a required field is blanked to NaN
      invalid_value: an enum or numeric field is set to an illegal value
      referential_orphan: a foreign key is pointed at a nonexistent parent
      duplicate: an existing row is appended again with the same primary key

    Rows touched by the first three types are disjoint, so a gate that
    assigns one error per row can be checked for exact per-type counts.
    Duplicates are appended copies of untouched rows.
    """
    rng = np.random.default_rng(seed)
    subjects = batch.subjects.copy(deep=True)
    visits = batch.visits.copy(deep=True)
    aes = batch.adverse_events.copy(deep=True)
    ledger = CorruptionLedger()

    plans = [
        ("subjects", subjects, "subject_id",
         [("enroll_date", None), ("drug", None)],
         [("sex", "X"), ("age", -5)],
         ("site_id", "STU-999-S99")),
        ("visits", visits, "visit_id",
         [("visit_date", None), ("visit_name", None)],
         [("visit_name", "week99"), ("visit_date", "not-a-date")],
         ("subject_id", "SUBJ-999999")),
        ("adverse_events", aes, "ae_id",
         [("onset_date", None), ("event_term", None)],
         [("severity", "catastrophic"), ("onset_date", "13/45/2024")],
         ("subject_id", "SUBJ-999999")),
    ]

    dirty = {}
    for name, df, pk, missing_opts, invalid_opts, (fk_col, fk_bad) in plans:
        n_rows = len(df)
        needed = n_missing + n_invalid + n_orphan
        idx = rng.choice(n_rows, size=needed, replace=False)
        miss_idx = idx[:n_missing]
        inv_idx = idx[n_missing:n_missing + n_invalid]
        orph_idx = idx[n_missing + n_invalid:]

        for i, row in enumerate(miss_idx):
            col, val = missing_opts[i % len(missing_opts)]
            df.iloc[row, df.columns.get_loc(col)] = val
        ledger.add(name, "missing_field", n_missing)

        for i, row in enumerate(inv_idx):
            col, val = invalid_opts[i % len(invalid_opts)]
            df.iloc[row, df.columns.get_loc(col)] = val
        ledger.add(name, "invalid_value", n_invalid)

        for row in orph_idx:
            df.iloc[row, df.columns.get_loc(fk_col)] = fk_bad
        ledger.add(name, "referential_orphan", n_orphan)

        clean_mask = np.ones(n_rows, dtype=bool)
        clean_mask[idx] = False
        clean_rows = np.flatnonzero(clean_mask)
        dup_idx = rng.choice(clean_rows, size=n_duplicate, replace=False)
        dup_rows = df.iloc[dup_idx].copy()
        df = pd.concat([df, dup_rows], ignore_index=True)
        ledger.add(name, "duplicate", n_duplicate)
        dirty[name] = df

    out = Batch(
        studies=batch.studies.copy(deep=True),
        sites=batch.sites.copy(deep=True),
        subjects=dirty["subjects"],
        visits=dirty["visits"],
        adverse_events=dirty["adverse_events"],
        truth=list(batch.truth),
    )
    return out, ledger
