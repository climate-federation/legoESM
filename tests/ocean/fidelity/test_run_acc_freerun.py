"""Unit tests for scripts/validate/ocean_fidelity/run_acc_freerun.py.

The script lives outside ``src/`` so we add ``scripts/`` to sys.path (same
pattern as test_run_comparison_script.py). The heavy end-to-end run (legoESM +
Veros) is exercised by the script itself / a @slow test; here we lock the bulk
diagnostics on the rest IC and a tiny forced integration.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def freerun_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.run_acc_freerun as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_bulk_stats_on_rest_ic(freerun_module):
    """At rest (the ACC IC), bulk diagnostics must be well-defined: zero ACC
    transport + zero KE (u=v=0), and T within the [0, 15] degC IC range."""
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe
    recipe = build_acc_recipe(with_surface_forcing=True)
    stats = freerun_module._bulk_stats(
        recipe.initial_state, recipe.z_coord, recipe.grid)
    expected_keys = {
        "ACC_transport_Sv", "ACC_transport_drake_band_Sv", "total_KE_J",
        "vol_mean_T_C", "T_min_C", "T_max_C", "max_abs_u_ms",
    }
    assert set(stats) == expected_keys
    assert all(np.isfinite(v) for v in stats.values())
    # rest state -> no flow (both the full-domain and Drake-band ranges)
    assert abs(stats["ACC_transport_Sv"]) < 1e-6
    assert abs(stats["ACC_transport_drake_band_Sv"]) < 1e-6
    assert stats["total_KE_J"] == pytest.approx(0.0, abs=1e-3)
    assert stats["max_abs_u_ms"] == pytest.approx(0.0, abs=1e-12)
    # T within the Veros-exact IC band: a cold abyss (~-1.15 degC, the deepest
    # cell of temp=(1-zt/zw[0])*15 with zw[0]=top face of the bottom cell) up to
    # ~15 degC at the surface. Commit 32e1dec41 replaced the old linear [0,15] IC
    # with this Veros transcription; the earlier [-0.5, 15.5] band predated it.
    assert -1.5 <= stats["T_min_C"] <= 0.0
    assert 0.0 <= stats["T_max_C"] <= 15.5
    assert 0.0 <= stats["vol_mean_T_C"] <= 15.0


def test_acc_transport_reports_full_domain_range(freerun_module, monkeypatch):
    """ACC_transport_Sv must report the FULL-DOMAIN ψ range (#431), not the
    Drake-band restriction. With a synthetic ψ whose extrema lie OUTSIDE the
    Drake band, the primary number must equal the full-domain range while the
    separately-labelled band number sees only the in-band range."""
    import legoesm.ocean.diagnostics_streamfunction as dsf
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe

    recipe = build_acc_recipe(with_surface_forcing=True)
    lat = np.degrees(np.asarray(recipe.grid.lat))
    n_lon = int(np.asarray(recipe.initial_state.land_mask.data).shape[1])

    in_band = (lat >= freerun_module._ACC_DRAKE_LAT_S) & (
        lat <= freerun_module._ACC_DRAKE_LAT_N)
    north = lat > freerun_module._ACC_DRAKE_LAT_N
    assert in_band.any() and north.any()  # the recipe grid spans both regions

    psi = np.zeros((lat.shape[0], n_lon))
    psi[in_band, :] = np.linspace(0.0, 10.0, int(in_band.sum()))[:, None]
    psi[north, :] = 100.0  # a full-domain extremum the Drake band misses

    monkeypatch.setattr(dsf, "barotropic_streamfunction", lambda *a, **k: psi)
    stats = freerun_module._bulk_stats(
        recipe.initial_state, recipe.z_coord, recipe.grid)

    assert stats["ACC_transport_Sv"] == pytest.approx(100.0)            # full-domain
    assert stats["ACC_transport_drake_band_Sv"] == pytest.approx(10.0)  # band only
    assert stats["ACC_transport_Sv"] > stats["ACC_transport_drake_band_Sv"]


@pytest.mark.slow
def test_short_freerun_is_stable_and_spins_up(freerun_module):
    """A 2-day forced free run stays finite and develops an eastward ACC
    (nonzero transport, positive max|u|)."""
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        years = 2.0 / 365.0
        state, recipe = freerun_module._run_legoesm(years, 4800.0)
        stats = freerun_module._bulk_stats(state, recipe.z_coord, recipe.grid)
    finally:
        set_policy(_prev)
    assert all(np.isfinite(v) for v in stats.values())
    assert stats["max_abs_u_ms"] > 0.0          # flow developed
    assert abs(stats["ACC_transport_Sv"]) > 0.0  # circulation developed
    # full-domain range >= Drake-band range by construction (superset extrema)
    assert (stats["ACC_transport_Sv"]
            >= stats["ACC_transport_drake_band_Sv"] - 1e-9)
