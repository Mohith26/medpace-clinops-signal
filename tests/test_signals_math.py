import math

import pytest
from scipy.stats import chi2_contingency

from trialscope.signals import chi2_yates, evaluate_pair, prr, prr_ci, ror, ror_ci

CASE_NAMES = ["clear_signal", "boundary_below_threshold", "zero_cell_b", "zero_cell_a", "strong_small"]


def cells(case):
    return case["a"], case["b"], case["c"], case["d"]


def close(x, y, tol=1e-9):
    if x is None or y is None:
        return x is None and y is None
    return math.isclose(x, y, rel_tol=tol, abs_tol=tol)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_prr_matches_hand_computed(two_by_two_cases, name):
    case = two_by_two_cases[name]
    assert close(prr(*cells(case)), case["prr"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_ror_matches_hand_computed(two_by_two_cases, name):
    case = two_by_two_cases[name]
    assert close(ror(*cells(case)), case["ror"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_chi2_matches_hand_computed(two_by_two_cases, name):
    case = two_by_two_cases[name]
    assert close(chi2_yates(*cells(case)), case["chi2"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_ror_ci_matches_hand_computed(two_by_two_cases, name):
    case = two_by_two_cases[name]
    lo, hi = ror_ci(*cells(case))
    assert close(lo, case["ror_ci_low"]) and close(hi, case["ror_ci_high"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_prr_ci_matches_hand_computed(two_by_two_cases, name):
    case = two_by_two_cases[name]
    lo, hi = prr_ci(*cells(case))
    assert close(lo, case["prr_ci_low"]) and close(hi, case["prr_ci_high"])


@pytest.mark.parametrize("name", ["clear_signal", "boundary_below_threshold", "strong_small"])
def test_chi2_cross_checked_against_scipy(two_by_two_cases, name):
    case = two_by_two_cases[name]
    a, b, c, d = cells(case)
    stat, _, _, _ = chi2_contingency([[a, b], [c, d]], correction=True)
    assert math.isclose(chi2_yates(a, b, c, d), stat, rel_tol=1e-9)


def test_prr_criterion_flags_clear_signal(two_by_two_cases):
    s = evaluate_pair("D", "E", *cells(two_by_two_cases["clear_signal"]))
    assert s.prr_signal and s.ror_signal


def test_prr_criterion_rejects_boundary_case(two_by_two_cases):
    case = two_by_two_cases["boundary_below_threshold"]
    s = evaluate_pair("D", "E", *cells(case))
    assert case["prr"] < 2 and case["chi2"] < 4
    assert not s.prr_signal


def test_zero_a_cell_is_never_a_signal(two_by_two_cases):
    s = evaluate_pair("D", "E", *cells(two_by_two_cases["zero_cell_a"]))
    assert not s.prr_signal and not s.ror_signal


def test_min_cases_gate_blocks_small_a():
    s = evaluate_pair("D", "E", 2, 8, 5, 985)
    assert not s.prr_signal and not s.ror_signal


def test_prr_undefined_when_no_comparator_events():
    assert prr(5, 95, 0, 900) is None
    lo, hi = prr_ci(5, 95, 0, 900)
    assert lo is None and hi is None


def test_chi2_none_on_degenerate_margin():
    assert chi2_yates(0, 0, 5, 95) is None


def test_prr_criterion_is_conjunction_of_all_three_gates():
    s = evaluate_pair("D", "E", 3, 7, 30, 160)
    assert s.prr_signal == (s.a >= 3 and s.prr >= 2 and s.chi2 >= 4)
    assert not s.prr_signal


def test_ror_haldane_applied_only_with_zero_cell():
    assert ror(5, 0, 10, 985) == pytest.approx((5.5 * 985.5) / (0.5 * 10.5))
    assert ror(10, 90, 20, 880) == pytest.approx((10 * 880) / (90 * 20))
