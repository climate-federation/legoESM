"""Fail-closed controls for the Round-135 tracer subfamily discriminator."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
           / "ocean_fidelity" / "testcases")
YEAR = SCRIPTS / "nemo_testcase_l2_gyre_year_fromrest.py"
OWNERS = SCRIPTS / "nemo_testcase_l2_gyre_year_owners.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def year():
    return _load("round135_year", YEAR)


@pytest.fixture(scope="module")
def owners():
    return _load("round135_owners", OWNERS)


def _card():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return build_nemo_testcase_card("GYRE-zco")


def _payload(card):
    shape = card.recipe.initial_state.T.data.shape
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return {"tn": values + 1.0, "sn": values + 2.0}


@pytest.mark.parametrize("plant", [
    "daily-source-ulp", "daily-family-registry", "daily-cadence-registry",
])
def test_reset_plants_exit_nonzero(plant):
    result = subprocess.run(
        [sys.executable, str(YEAR), "--daily-reset-self-check",
         "--plant", plant], capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"STATUS PLANT-FIRED: {plant}" in result.stdout
    assert f"REFUSE Round-134 {plant}" in result.stderr


def test_temperature_and_salinity_resets_mutate_only_the_named_tracer(year):
    card = _card()
    state = card.recipe.initial_state
    payload = _payload(card)

    temperature, pending = year.apply_daily_reset(
        state, payload, "temperature", card)
    assert pending is None
    assert not np.array_equal(np.asarray(temperature.T.data),
                              np.asarray(state.T.data))
    assert np.array_equal(np.asarray(temperature.S.data),
                          np.asarray(state.S.data))
    assert np.array_equal(np.asarray(temperature.eta.data),
                          np.asarray(state.eta.data))

    salinity, pending = year.apply_daily_reset(state, payload, "salinity", card)
    assert pending is None
    assert np.array_equal(np.asarray(salinity.T.data), np.asarray(state.T.data))
    assert not np.array_equal(np.asarray(salinity.S.data),
                              np.asarray(state.S.data))
    assert np.array_equal(np.asarray(salinity.eta.data),
                          np.asarray(state.eta.data))


def test_subfamily_and_cadence_registries_are_exact(year):
    year.validate_daily_reset_registry()
    assert year.DAILY_RESET_VARIABLES["temperature"] == ("tn",)
    assert year.DAILY_RESET_VARIABLES["salinity"] == ("sn",)
    for interval_days in (1, 2, 4, 8):
        year.validate_daily_reset_interval(interval_days)
    with pytest.raises(year.GateError, match="outside"):
        year.validate_daily_reset_interval(0)
    with pytest.raises(year.GateError, match="integer"):
        year.validate_daily_reset_interval(True)


def test_score_registry_rejects_a_missing_or_duplicate_real_key(owners):
    labels = ("temperature", "salinity")
    rows = [
        {"family": label, "day": day, "field": field}
        for label in labels
        for day in owners.DAILY_ATTRIBUTION_DAYS
        for field in owners.FIELDS
    ]
    owners._validate_daily_metric_registry(rows, labels)
    with pytest.raises(owners.GateError, match="79 rows"):
        owners._validate_daily_metric_registry(rows[:-1], labels)
    duplicate = rows[:-1] + [rows[0]]
    with pytest.raises(owners.GateError, match="duplicate"):
        owners._validate_daily_metric_registry(duplicate, labels)
