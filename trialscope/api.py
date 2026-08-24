"""FastAPI service over an ingested dataset.

The app is built by create_app(), which generates a seeded batch, dirties
it, runs it through the quality gates, and serves analytics computed from
the clean tables. Endpoints:

  GET /studies                      study summaries with enrollment velocity
  GET /studies/{study_id}/metrics   velocity detail plus site risk for one study
  GET /signals?method=prr|ror|all   disproportionality results (flagged only by default)
  GET /quality                      quality gate report from the last ingest
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from .generator import Batch, GeneratorConfig, default_signals, dirty_batch, generate
from .metrics import enrollment_velocity, site_risk, study_summaries
from .quality import ingest
from .signals import detect_signals


def build_dataset(seed: int = 7, n_signals: int = 25):
    """Generate, dirty, and ingest a batch. Returns (clean_tables, report, truth, config)."""
    config = GeneratorConfig(seed=seed)
    config.signals = default_signals(config, n_signals=n_signals)
    batch = generate(config)
    dirty, _ledger = dirty_batch(batch, seed=seed + 1)
    result = ingest(dirty.tables())
    return result.clean, result.report, batch.truth, config


def create_app(seed: int = 7, n_signals: int = 25) -> FastAPI:
    clean, report, truth, config = build_dataset(seed=seed, n_signals=n_signals)
    app = FastAPI(title="TrialScope", version="0.1.0")

    signal_stats = detect_signals(
        clean["adverse_events"], drugs=config.drugs, events=config.events
    )

    @app.get("/studies")
    def studies():
        return {
            "studies": study_summaries(
                clean["subjects"], clean["studies"], clean["adverse_events"]
            )
        }

    @app.get("/studies/{study_id}/metrics")
    def study_metrics(study_id: str):
        known = set(clean["studies"]["study_id"])
        if study_id not in known:
            raise HTTPException(status_code=404, detail=f"unknown study {study_id}")
        risk = [
            r for r in site_risk(
                clean["subjects"], clean["visits"], clean["adverse_events"], clean["studies"]
            )
            if r["study_id"] == study_id
        ]
        return {
            "velocity": enrollment_velocity(clean["subjects"], study_id),
            "site_risk": risk,
        }

    @app.get("/signals")
    def signals(method: str = Query("all", pattern="^(prr|ror|all)$"),
                include_negative: bool = False,
                drug: Optional[str] = None):
        rows = signal_stats
        if drug is not None:
            rows = [s for s in rows if s.drug == drug]
        if not include_negative:
            if method == "prr":
                rows = [s for s in rows if s.prr_signal]
            elif method == "ror":
                rows = [s for s in rows if s.ror_signal]
            else:
                rows = [s for s in rows if s.prr_signal or s.ror_signal]
        return {"method": method, "count": len(rows), "signals": [s.to_dict() for s in rows]}

    @app.get("/quality")
    def quality():
        return report.to_dict()

    return app
