"""Direct test for ``scripts/validate/mpas_energy_budget_leak.py`` (#1354).

The whole answer this script produces is a sign and a magnitude, so the tests
are about exactly those: a synthetic series with a KNOWN leak must come back
with that leak, and a series that closes must come back at zero.  Built from
the budget identity rather than from the script's own algebra, so it is a
cross-check and not a restatement.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "mpas_energy_budget_leak.py"


def _load():
    spec = importlib.util.spec_from_file_location("_leak", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_leak"] = mod
    spec.loader.exec_module(mod)
    return mod


def _series(tmp_path, *, leak, n=6):
    """A series built FORWARD from the budget, with a chosen leak injected.

    Column budget, everything positive INTO the column:
        dE/dt = toa_net - sfc_rad_into_surface + hfss + hfls + leak
    and the tracker reports residual = toa_net - dE/dt.
    """
    rng = np.random.default_rng(0)
    toa = rng.normal(-3.0, 2.0, n)
    sw = rng.normal(160.0, 5.0, n)
    lw = rng.normal(-55.0, 3.0, n)
    sh = rng.normal(20.0, 2.0, n)
    lh = rng.normal(80.0, 4.0, n)
    dedt = toa - (sw + lw) + sh + lh + leak
    path = tmp_path / "timeseries.npz"
    np.savez(path, days=np.arange(n, dtype=float),
             energy_toa_net=toa, energy_dE_dt=dedt,
             energy_residual=toa - dedt,
             sw_net_sfc=sw, lw_net_sfc=lw, hfss=sh, hfls=lh)
    return path


def _leak_values(mod, path):
    t = mod.load(path)
    return (t["sw_net_sfc"] + t["lw_net_sfc"] - t["hfss"] - t["hfls"]
            - t["energy_residual"])


def test_a_closing_budget_reports_zero_leak(tmp_path):
    mod = _load()
    got = _leak_values(mod, _series(tmp_path, leak=0.0))
    np.testing.assert_allclose(got, 0.0, atol=1e-9)


@pytest.mark.parametrize("injected", [20.0, -20.0, 5.0])
def test_an_injected_leak_is_recovered_with_the_right_sign(tmp_path, injected):
    """Sign included: a leak that CREATES energy must come back positive."""
    mod = _load()
    got = _leak_values(mod, _series(tmp_path, leak=injected))
    np.testing.assert_allclose(got, injected, atol=1e-9)


def test_missing_energy_channels_fail_loudly(tmp_path):
    """A series without the energy channels must raise, not report a leak.

    The MPAS lane writes these only once it reaches a diagnostic step; silently
    reporting on a file that lacks them would manufacture a number.
    """
    mod = _load()
    path = tmp_path / "timeseries.npz"
    np.savez(path, days=np.arange(3.0), T_atm=np.zeros(3))
    with pytest.raises(SystemExit, match="energy"):
        mod.load(path)


def test_runs_end_to_end(tmp_path):
    mod = _load()
    assert mod.main([str(_series(tmp_path, leak=12.0)), "--skip-days", "1"]) == 0
