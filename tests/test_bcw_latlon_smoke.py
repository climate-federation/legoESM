"""Smoke test for the latlon C-grid BCW init + one-step dispatch path.

The slopbuster 2026-05-03 review flagged that the BCW benchmark's
latlon dispatch had no end-to-end coverage.  This test exercises
the underlying model path (init + one step) at a tiny resolution
so a regression in any of the symbols the BCW script's latlon
branch references is caught immediately, without paying the cost
of a full benchmark.

Symbols verified:
* ``legoesm.grids.latlon.create_latlon_grid``
* ``tests.test_cases.baroclinic_wave.baroclinic_wave_init_latlon``
* ``legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid.{
    CGridLatLonPrimitiveEquationModel,
    CGridLatLonPrimitiveEquationConfig,
    hydrostatic_to_cgrid,
    cgrid_to_hydrostatic,
}``
* ``legoesm.core.cfl.estimate_min_dx_latlon``
"""

from __future__ import annotations

import jax.numpy as jnp


def test_bcw_latlon_init_and_one_step():
    """Construct BCW init on L8, build the CGridLatLon model, and
    step once.  Pins the latlon dispatch path of the BCW benchmark.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
        cgrid_to_hydrostatic,
    )
    from legoesm.core.cfl import estimate_min_dx_latlon
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon

    n_lat = 8
    n_lev = 8
    grid = create_latlon_grid(n_lat=n_lat)  # n_lon = 2*n_lat = 16
    sigma = create_sigma_coordinate(n_lev)

    # The dx estimate must be a positive scalar.
    dx_min = estimate_min_dx_latlon(n_lat)
    assert dx_min > 0.0

    state_cc = baroclinic_wave_init_latlon(grid, sigma, perturbed=True)
    state = hydrostatic_to_cgrid(state_cc, grid)

    config = CGridLatLonPrimitiveEquationConfig(
        A_h=1.0e4,
        time_integrator="ssp_rk3",
        fix_mass=False,
        zero_mean_ps_tendency=False,
        use_ppm_transport=True,
        use_polar_filter=True,
        polar_filter_cutoff_deg=60.0,
        polar_filter_max_wave_speed=300.0,
    )
    dt = 100.0  # well within polar CFL at L8
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)

    state_next = model.step(state, dt=dt)

    # All prognostic fields must remain finite after one step.
    assert jnp.all(jnp.isfinite(state_next.p_s)), "p_s went non-finite"
    assert jnp.all(jnp.isfinite(state_next.u)), "u went non-finite"
    assert jnp.all(jnp.isfinite(state_next.v)), "v went non-finite"
    assert jnp.all(jnp.isfinite(state_next.T)), "T went non-finite"

    # Round-trip back to cell-centred to verify cgrid_to_hydrostatic
    # is wired correctly (the script uses it for diagnostics).
    state_cc_back = cgrid_to_hydrostatic(state_next, grid)
    assert jnp.all(jnp.isfinite(state_cc_back.T.data))


def test_bcw_script_latlon_branch_imports():
    """If the BCW benchmark script declares a ``latlon`` grid choice,
    every dispatch-table entry must be wired (resolution parser, dt
    default, scan-steps).  Skips silently when ``latlon`` is not in
    ``GRID_CHOICES`` (e.g. on a branch where the latlon dispatch
    has not yet been merged).
    """
    import importlib.util
    from pathlib import Path
    import pytest

    here = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_bcw_benchmark",
        here / "scripts" / "run" / "run_baroclinic_wave_benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if "latlon" not in mod.GRID_CHOICES:
        pytest.skip("BCW script does not declare a 'latlon' grid choice on this branch")

    # Dispatch tables must agree
    assert "latlon" in mod.AUTO_SCAN_STEPS, (
        "latlon is in GRID_CHOICES but missing from AUTO_SCAN_STEPS"
    )
    # Resolution + dt defaults must be defined for latlon
    assert mod._default_resolution("latlon").startswith("L")
    assert mod._default_dt("latlon") > 0
    # Resolution parsing must accept the L<int> form
    assert mod._parse_resolution("L8", "latlon") == 8
    assert mod._parse_resolution("L64", "latlon") == 64
