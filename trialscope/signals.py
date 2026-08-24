"""Disproportionality signal detection: PRR and ROR on 2x2 tables.

For a drug D and event E over a set of AE reports, the standard 2x2 is

              event E    all other events
  drug D         a              b
  other drugs    c              d

PRR (proportional reporting ratio), from Evans, Waller and Davis (2001),
"Use of proportional reporting ratios (PRRs) for signal generation from
spontaneous adverse drug reaction reports", Pharmacoepidemiol Drug Saf
10(6):483-486:

  PRR = (a / (a + b)) / (c / (c + d))

with their signal criterion: PRR >= 2, chi-square >= 4 (Yates corrected),
and a >= 3.

ROR (reporting odds ratio), from van Puijenbroek et al. (2002), "A
comparison of measures of disproportionality for signal detection in
spontaneous reporting systems", Pharmacoepidemiol Drug Saf 11(1):3-10:

  ROR = (a * d) / (b * c)

with the common criterion: lower bound of the 95% CI of ROR > 1 and
a >= 3. The CI uses ln(ROR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d).

When any cell is zero, ROR and its CI use a Haldane-Anscombe correction
(0.5 added to every cell), a standard practice for sparse 2x2 tables.

Counting convention: a report here is one adverse_events row, and cells
count reports (a subject with two AE rows contributes two reports). This
matches spontaneous-reporting practice where the unit is the report.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

Z_95 = 1.959963984540054


@dataclass
class SignalStats:
    drug: str
    event: str
    a: int
    b: int
    c: int
    d: int
    prr: Optional[float]
    prr_ci_low: Optional[float]
    prr_ci_high: Optional[float]
    ror: Optional[float]
    ror_ci_low: Optional[float]
    ror_ci_high: Optional[float]
    chi2: Optional[float]
    prr_signal: bool
    ror_signal: bool

    def to_dict(self) -> Dict:
        return {
            "drug": self.drug,
            "event": self.event,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "prr": self.prr,
            "prr_ci_low": self.prr_ci_low,
            "prr_ci_high": self.prr_ci_high,
            "ror": self.ror,
            "ror_ci_low": self.ror_ci_low,
            "ror_ci_high": self.ror_ci_high,
            "chi2": self.chi2,
            "prr_signal": self.prr_signal,
            "ror_signal": self.ror_signal,
        }


def prr(a: int, b: int, c: int, d: int) -> Optional[float]:
    """Proportional reporting ratio. None when undefined (no exposure or no comparator events)."""
    if a + b == 0 or c + d == 0 or c == 0:
        return None
    return (a / (a + b)) / (c / (c + d))


def prr_ci(a: int, b: int, c: int, d: int) -> Tuple[Optional[float], Optional[float]]:
    """95% CI for PRR on the log scale: ln(PRR) +/- 1.96 * sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))."""
    value = prr(a, b, c, d)
    if value is None or a == 0:
        return None, None
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    log_p = math.log(value)
    return math.exp(log_p - Z_95 * se), math.exp(log_p + Z_95 * se)


def ror(a: int, b: int, c: int, d: int) -> Optional[float]:
    """Reporting odds ratio with Haldane-Anscombe 0.5 correction when any cell is zero."""
    if min(a, b, c, d) == 0:
        a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        return (a2 * d2) / (b2 * c2)
    return (a * d) / (b * c)


def ror_ci(a: int, b: int, c: int, d: int) -> Tuple[Optional[float], Optional[float]]:
    """95% CI for ROR: exp(ln(ROR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d))."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    value = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_r = math.log(value)
    return math.exp(log_r - Z_95 * se), math.exp(log_r + Z_95 * se)


def chi2_yates(a: int, b: int, c: int, d: int) -> Optional[float]:
    """Chi-square with Yates continuity correction for a 2x2 table.

    chi2 = sum over cells of (|observed - expected| - 0.5)^2 / expected,
    with the |o - e| - 0.5 term floored at zero.
    """
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if n == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return None
    total = 0.0
    for obs, er, ec in ((a, row1, col1), (b, row1, col2), (c, row2, col1), (d, row2, col2)):
        expected = er * ec / n
        adj = max(abs(obs - expected) - 0.5, 0.0)
        total += adj * adj / expected
    return total


def evaluate_pair(drug: str, event: str, a: int, b: int, c: int, d: int,
                  prr_threshold: float = 2.0, chi2_threshold: float = 4.0,
                  min_cases: int = 3) -> SignalStats:
    """Compute all statistics and both signal criteria for one drug-event pair."""
    p = prr(a, b, c, d)
    p_lo, p_hi = prr_ci(a, b, c, d)
    r = ror(a, b, c, d)
    r_lo, r_hi = ror_ci(a, b, c, d)
    x2 = chi2_yates(a, b, c, d)

    prr_signal = (
        a >= min_cases
        and p is not None
        and p >= prr_threshold
        and x2 is not None
        and x2 >= chi2_threshold
    )
    ror_signal = a >= min_cases and r_lo is not None and r_lo > 1.0

    return SignalStats(
        drug=drug, event=event, a=a, b=b, c=c, d=d,
        prr=p, prr_ci_low=p_lo, prr_ci_high=p_hi,
        ror=r, ror_ci_low=r_lo, ror_ci_high=r_hi,
        chi2=x2, prr_signal=prr_signal, ror_signal=ror_signal,
    )


def contingency_tables(adverse_events: pd.DataFrame,
                       drugs: Optional[List[str]] = None,
                       events: Optional[List[str]] = None) -> Dict[Tuple[str, str], Tuple[int, int, int, int]]:
    """Build (a, b, c, d) report counts for every drug-event pair in the universe.

    The universe defaults to drugs and events observed in the data; passing
    explicit lists lets the caller score pairs with zero observed reports.
    """
    counts = adverse_events.groupby(["drug", "event_term"]).size()
    drug_totals = adverse_events.groupby("drug").size()
    event_totals = adverse_events.groupby("event_term").size()
    n = len(adverse_events)

    drug_list = drugs if drugs is not None else sorted(drug_totals.index)
    event_list = events if events is not None else sorted(event_totals.index)

    tables: Dict[Tuple[str, str], Tuple[int, int, int, int]] = {}
    for d_ in drug_list:
        dt = int(drug_totals.get(d_, 0))
        for e_ in event_list:
            a = int(counts.get((d_, e_), 0))
            b = dt - a
            c = int(event_totals.get(e_, 0)) - a
            d2 = n - a - b - c
            tables[(d_, e_)] = (a, b, c, d2)
    return tables


def detect_signals(adverse_events: pd.DataFrame,
                   drugs: Optional[List[str]] = None,
                   events: Optional[List[str]] = None) -> List[SignalStats]:
    """Score every pair in the universe and return the full stats list."""
    tables = contingency_tables(adverse_events, drugs=drugs, events=events)
    return [evaluate_pair(d_, e_, *cells) for (d_, e_), cells in sorted(tables.items())]


def score_against_truth(stats: List[SignalStats], truth: List[Tuple[str, str]],
                        method: str) -> Dict:
    """Precision, recall, F1 and confusion counts for one criterion vs injected truth."""
    truth_set = set(truth)
    tp = fp = fn = tn = 0
    for s in stats:
        flagged = s.prr_signal if method == "prr" else s.ror_signal
        is_true = (s.drug, s.event) in truth_set
        if flagged and is_true:
            tp += 1
        elif flagged and not is_true:
            fp += 1
        elif not flagged and is_true:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "method": method,
        "pairs_evaluated": len(stats),
        "injected_signals": len(truth_set),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }
