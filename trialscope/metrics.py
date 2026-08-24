"""Operational analytics over clean CTMS tables.

Enrollment velocity: subjects enrolled per ISO week, per study, plus a
simple overall subjects-per-week rate between first and last enrollment.

Site risk: per-site indicators that a monitor might triage on, computed
only from the synthetic tables: enrollment against an even share of the
study target, AE reports per subject, share of severe AEs, and share of
late-arriving records. The composite score is the mean of the min-max
normalized indicators (higher means more attention needed).
"""

from typing import Dict, List

import pandas as pd


def enrollment_velocity(subjects: pd.DataFrame, study_id: str) -> Dict:
    """Weekly enrollment counts and overall rate for one study."""
    sub = subjects[subjects["study_id"] == study_id]
    if sub.empty:
        return {"study_id": study_id, "total_enrolled": 0, "weekly": [], "subjects_per_week": 0.0}
    dates = pd.to_datetime(sub["enroll_date"])
    iso = dates.dt.isocalendar()
    weekly = (
        pd.DataFrame({"year": iso.year, "week": iso.week})
        .groupby(["year", "week"]).size().reset_index(name="enrolled")
    )
    span_days = (dates.max() - dates.min()).days
    weeks = max(span_days / 7.0, 1.0)
    return {
        "study_id": study_id,
        "total_enrolled": int(len(sub)),
        "weekly": [
            {"year": int(r.year), "week": int(r.week), "enrolled": int(r.enrolled)}
            for r in weekly.itertuples()
        ],
        "subjects_per_week": round(len(sub) / weeks, 3),
    }


def _late_share(df: pd.DataFrame, src: str, recv: str, threshold: int) -> pd.Series:
    delta = (pd.to_datetime(df[recv]) - pd.to_datetime(df[src])).dt.days
    return delta > threshold


def site_risk(subjects: pd.DataFrame, visits: pd.DataFrame,
              adverse_events: pd.DataFrame, studies: pd.DataFrame,
              late_days_threshold: int = 14) -> List[Dict]:
    """Per-site risk indicators plus a composite 0..1 score."""
    rows = []
    targets = studies.set_index("study_id")["target_enrollment"].to_dict()
    sites_per_study = subjects.groupby("study_id")["site_id"].nunique().to_dict()

    visits = visits.assign(_late=_late_share(visits, "visit_date", "received_date", late_days_threshold))
    aes = adverse_events.assign(_late=_late_share(adverse_events, "onset_date", "received_date", late_days_threshold))
    ae_by_subject = aes.merge(subjects[["subject_id", "site_id"]], on="subject_id", how="inner")
    visits_by_subject = visits.merge(subjects[["subject_id", "site_id"]], on="subject_id", how="inner")

    for site_id, grp in subjects.groupby("site_id"):
        study_id = grp["study_id"].iloc[0]
        n_sites = max(sites_per_study.get(study_id, 1), 1)
        share = targets.get(study_id, 0) / n_sites
        enrolled = len(grp)
        site_aes = ae_by_subject[ae_by_subject["site_id"] == site_id]
        site_visits = visits_by_subject[visits_by_subject["site_id"] == site_id]
        late_n = int(site_aes["_late"].sum()) + int(site_visits["_late"].sum())
        record_n = len(site_aes) + len(site_visits)
        rows.append(
            {
                "site_id": site_id,
                "study_id": study_id,
                "enrolled": enrolled,
                "enrollment_shortfall": max(0.0, 1.0 - enrolled / share) if share else 0.0,
                "ae_per_subject": len(site_aes) / enrolled if enrolled else 0.0,
                "severe_ae_share": float((site_aes["severity"] == "severe").mean()) if len(site_aes) else 0.0,
                "late_record_share": late_n / record_n if record_n else 0.0,
            }
        )

    indicators = ["enrollment_shortfall", "ae_per_subject", "severe_ae_share", "late_record_share"]
    df = pd.DataFrame(rows)
    normed = pd.DataFrame(index=df.index)
    for col in indicators:
        lo, hi = df[col].min(), df[col].max()
        normed[col] = 0.0 if hi == lo else (df[col] - lo) / (hi - lo)
    df["risk_score"] = normed.mean(axis=1).round(4)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df.to_dict(orient="records")


def study_summaries(subjects: pd.DataFrame, studies: pd.DataFrame,
                    adverse_events: pd.DataFrame) -> List[Dict]:
    """One summary row per study: enrollment, velocity, AE volume."""
    out = []
    for r in studies.itertuples():
        vel = enrollment_velocity(subjects, r.study_id)
        study_subjects = subjects[subjects["study_id"] == r.study_id]
        study_aes = adverse_events.merge(
            study_subjects[["subject_id"]], on="subject_id", how="inner"
        )
        out.append(
            {
                "study_id": r.study_id,
                "phase": r.phase,
                "therapeutic_area": r.therapeutic_area,
                "target_enrollment": int(r.target_enrollment),
                "total_enrolled": vel["total_enrolled"],
                "enrollment_pct": round(vel["total_enrolled"] / r.target_enrollment, 4) if r.target_enrollment else 0.0,
                "subjects_per_week": vel["subjects_per_week"],
                "ae_reports": int(len(study_aes)),
            }
        )
    return out
