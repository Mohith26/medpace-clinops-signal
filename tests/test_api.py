def test_studies_returns_summaries(api_client):
    r = api_client.get("/studies")
    assert r.status_code == 200
    studies = r.json()["studies"]
    assert len(studies) == 4
    for row in studies:
        assert {"study_id", "total_enrolled", "subjects_per_week", "ae_reports"} <= set(row)


def test_study_metrics_ok(api_client):
    r = api_client.get("/studies/STU-001/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["velocity"]["study_id"] == "STU-001"
    assert len(body["site_risk"]) == 10
    assert all(row["study_id"] == "STU-001" for row in body["site_risk"])


def test_study_metrics_unknown_study_404(api_client):
    r = api_client.get("/studies/STU-999/metrics")
    assert r.status_code == 404


def test_signals_prr_returns_only_flagged(api_client):
    r = api_client.get("/signals?method=prr")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(body["signals"])
    assert all(s["prr_signal"] for s in body["signals"])


def test_signals_ror_returns_only_flagged(api_client):
    r = api_client.get("/signals?method=ror")
    assert r.status_code == 200
    assert all(s["ror_signal"] for s in r.json()["signals"])


def test_signals_all_returns_union_of_flags(api_client):
    r = api_client.get("/signals?method=all")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    assert all(s["prr_signal"] or s["ror_signal"] for s in body["signals"])


def test_signals_full_universe_size(api_client):
    r = api_client.get("/signals?method=all&include_negative=true")
    assert r.status_code == 200
    assert r.json()["count"] == 12 * 30


def test_signals_invalid_method_rejected(api_client):
    r = api_client.get("/signals?method=bogus")
    assert r.status_code == 422


def test_signals_drug_filter(api_client):
    r = api_client.get("/signals?method=all&include_negative=true&drug=DRG-01")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 30
    assert all(s["drug"] == "DRG-01" for s in body["signals"])


def test_quality_report_shape(api_client):
    r = api_client.get("/quality")
    assert r.status_code == 200
    body = r.json()
    assert set(body["tables"]) == {"studies", "sites", "subjects", "visits", "adverse_events"}
    assert body["total_quarantined"] > 0
    assert body["total_rows_in"] > 5000
