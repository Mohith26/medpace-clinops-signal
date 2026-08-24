import json
from pathlib import Path

import pytest

from trialscope.generator import GeneratorConfig, SignalSpec, default_signals, dirty_batch, generate

FIXTURES = Path(__file__).parent / "fixtures"


def small_config(seed: int = 11) -> GeneratorConfig:
    cfg = GeneratorConfig(seed=seed, n_studies=2, sites_per_study=4, subjects_per_site=15)
    cfg.signals = [SignalSpec("DRG-01", "EVT-01", 8.0), SignalSpec("DRG-03", "EVT-05", 8.0)]
    return cfg


@pytest.fixture(scope="session")
def small_batch():
    return generate(small_config())


@pytest.fixture(scope="session")
def small_dirty():
    batch = generate(small_config())
    return dirty_batch(batch, seed=12, n_missing=5, n_invalid=4, n_orphan=3, n_duplicate=3)


@pytest.fixture(scope="session")
def two_by_two_cases():
    return json.loads((FIXTURES / "two_by_two_cases.json").read_text())["cases"]


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient

    from trialscope.api import create_app

    return TestClient(create_app(seed=7, n_signals=25))
