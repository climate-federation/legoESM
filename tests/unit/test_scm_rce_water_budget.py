"""Gates for scripts/validate/scm_rce_water_budget.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "_scm_h2o_budget",
        REPO_ROOT / "scripts" / "validate" / "scm_rce_water_budget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load()


def test_column_water_of_a_uniform_profile_is_q_times_column_mass(probe):
    """A uniform q over a full column integrates to q * p_sfc / g [kg/m^2]."""
    from legoesm import constants
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import WING_P_SFC
    n = 40
    ref = SimpleNamespace(mass_weights=np.full(n, 1.0 / n))
    q = 0.01
    got = probe.column_water(ref, np.full(n, q))
    assert got == pytest.approx(q * WING_P_SFC / constants.g, rel=1e-12)


def test_column_water_is_zero_for_a_dry_column(probe):
    ref = SimpleNamespace(mass_weights=np.full(10, 0.1))
    assert probe.column_water(ref, np.zeros(10)) == 0.0


def test_column_water_uses_the_campaign_weights_not_a_second_discretisation(probe):
    """Weighting must come from ref.mass_weights: a probe that rebuilt the
    vertical grid could disagree with the campaign it is diagnosing."""
    ref = SimpleNamespace(mass_weights=np.array([0.9, 0.1]))
    top_heavy = probe.column_water(ref, np.array([1.0, 0.0]))
    bot_heavy = probe.column_water(ref, np.array([0.0, 1.0]))
    assert top_heavy == pytest.approx(9.0 * bot_heavy, rel=1e-12)


def test_the_two_readings_are_stated_in_the_source(probe):
    """The probe must not print a verdict, but it MUST say what each branch
    would mean — an operator reading only the log has to be able to act."""
    src = (REPO_ROOT / "scripts" / "validate"
           / "scm_rce_water_budget.py").read_text()
    assert "column is CLOSED" in src
    assert "WITHOUT being counted as precipitation" in src
    assert "does not choose between them" in src
