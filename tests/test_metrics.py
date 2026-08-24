import pandas as pd
import pytest

from trialscope.metrics import enrollment_velocity, site_risk, study_summaries


@pytest.fixture(scope="module")
def crafted():
    studies = pd.DataFrame(
        [{"study_id": "STU-901", "phase": "2", "therapeutic_area": "oncology",
          "target_enrollment": 40, "start_date": "2024-01-08"}]
    )
    subjects = pd.DataFrame(
        [
            {"subject_id": "S1", "site_id": "A", "study_id": "STU-901", "drug": "DRG-01",
             "enroll_date": "2024-01-08", "age": 30, "sex": "F"},
            {"subject_id": "S2", "site_id": "A", "study_id": "STU-901", "drug": "DRG-01",
             "enroll_date": "2024-01-10", "age": 40, "sex": "M"},
            {"subject_id": "S3", "site_id": "B", "study_id": "STU-901", "drug": "DRG-02",
             "enroll_date": "2024-01-22", "age": 50, "sex": "F"},
        ]
    )
    visits = pd.DataFrame(
        [
            {"visit_id": "V1", "subject_id": "S1", "visit_name": "baseline",
             "visit_date": "2024-01-08", "received_date": "2024-01-09"},
            {"visit_id": "V2", "subject_id": "S3", "visit_name": "baseline",
             "visit_date": "2024-01-22", "received_date": "2024-02-22"},
        ]
    )
    aes = pd.DataFrame(
        [
            {"ae_id": "A1", "subject_id": "S1", "drug": "DRG-01", "event_term": "EVT-01",
             "severity": "severe", "onset_date": "2024-01-15", "received_date": "2024-01-16"},
            {"ae_id": "A2", "subject_id": "S3", "drug": "DRG-02", "event_term": "EVT-02",
             "severity": "mild", "onset_date": "2024-01-25", "received_date": "2024-01-26"},
        ]
    )
    return studies, subjects, visits, aes


def test_velocity_total_and_weekly_sum(crafted):
    _, subjects, _, _ = crafted
    v = enrollment_velocity(subjects, "STU-901")
    assert v["total_enrolled"] == 3
    assert sum(w["enrolled"] for w in v["weekly"]) == 3


def test_velocity_rate_uses_span(crafted):
    _, subjects, _, _ = crafted
    v = enrollment_velocity(subjects, "STU-901")
    assert v["subjects_per_week"] == pytest.approx(3 / 2.0)


def test_velocity_empty_study(crafted):
    _, subjects, _, _ = crafted
    v = enrollment_velocity(subjects, "STU-999")
    assert v["total_enrolled"] == 0 and v["weekly"] == [] and v["subjects_per_week"] == 0.0


def test_site_risk_covers_all_sites(crafted):
    studies, subjects, visits, aes = crafted
    rows = site_risk(subjects, visits, aes, studies)
    assert {r["site_id"] for r in rows} == {"A", "B"}


def test_site_risk_scores_in_unit_range(small_batch):
    rows = site_risk(small_batch.subjects, small_batch.visits,
                     small_batch.adverse_events, small_batch.studies)
    assert all(0.0 <= r["risk_score"] <= 1.0 for r in rows)
    assert rows == sorted(rows, key=lambda r: r["risk_score"], reverse=True)


def test_site_risk_late_share_crafted(crafted):
    studies, subjects, visits, aes = crafted
    rows = {r["site_id"]: r for r in site_risk(subjects, visits, aes, studies)}
    assert rows["B"]["late_record_share"] == pytest.approx(0.5)
    assert rows["A"]["late_record_share"] == 0.0


def test_site_risk_ae_per_subject(crafted):
    studies, subjects, visits, aes = crafted
    rows = {r["site_id"]: r for r in site_risk(subjects, visits, aes, studies)}
    assert rows["A"]["ae_per_subject"] == pytest.approx(0.5)
    assert rows["B"]["ae_per_subject"] == pytest.approx(1.0)


def test_study_summaries_fields(crafted):
    studies, subjects, _, aes = crafted
    out = study_summaries(subjects, studies, aes)
    assert len(out) == 1
    row = out[0]
    assert row["total_enrolled"] == 3
    assert row["enrollment_pct"] == pytest.approx(3 / 40)
    assert row["ae_reports"] == 2
