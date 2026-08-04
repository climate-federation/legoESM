"""#1226 zu_frc structural-question support: closure + bit-identity for the
momentum-tendency diagnostics UNDER the exact DINO ``nemo_dino_kamm_mlf``
operator selections (``lateral_viscosity_operator="nemo_div_curl"``,
``vertical_momentum_scheme="nemo_advective"``).

``test_momentum_diagnostics_closure.py`` already proves the general
``Σ components == total`` contract, but only under the DEFAULT
``lateral_viscosity_operator``/``vertical_momentum_scheme`` — it never
exercises the NEMO-faithful branches that expose ``Ah_lap_u/v`` (dyn_ldf,
``ocean_pe_latlon_cgrid.py:2671-2733``) and ``vertadv_u/v`` (dyn_adv ZAD,
``ocean_pe_latlon_cgrid.py:2523-2548``) via
``nemo_ldf_lap_viscosity_cgrid``/``nemo_advective_vertical_momentum_advection``.
``scripts/validate/ocean_fidelity/dino_1226/zu_frc_momentum_row_reconstruction.py``
relies on these two fields being (a) closure-consistent and (b) NOT changing
the model's own numerics under this exact recipe combo -- this test is the
mechanical proof for both, non-vacuously (see the second test, which shows
the closure assertion actually fails if the diagnostic capture is wired to
the wrong term).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)


def _grid():
    # ``ensure_geometry`` (matches test_nemo_ldf_lap_viscosity.py's ``_geo()``)
    # -- nemo_div_curl needs a scalar ``dlon`` (present on the plain grid) and
    # nemo_advective needs ``area_T``/``dx_u``/``dy_u`` etc., only present on
    # the fuller geometry wrapper.
    return ensure_geometry(create_latlon_grid(n_lat=24, n_lon=48))


def _z_coord():
    return create_ocean_z_star(n_levels=8, H_max=4000.0)


def _perturbed_state(grid, z_coord):
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    rng = np.random.default_rng(7)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    import jax.numpy as jnp
    u = 0.05 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.05 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.01 * rng.standard_normal((n_lat, n_lon))
    T = 5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
    T = T + 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u, dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(v, dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(eta, dtype=jnp.float64)),
        T=state.T.replace(data=jnp.asarray(T, dtype=jnp.float64)),
    )


def _dino_ldf_zad_config():
    """The exact operator pair ``nemo_dino_kamm_mlf`` selects for dyn_ldf /
    dyn_adv ZAD (DINOConfig ``lateral_viscosity_operator="nemo_div_curl"``,
    ``vertical_momentum_scheme="nemo_advective"``, ``experiments/dino.py:1141,1102``),
    built directly on ``LatLonCGridOceanConfig`` (not the full DINO recipe
    machinery) to keep this test fast and independent of the fidelity harness."""
    return LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4, A_v=1.0e-4, bottom_drag_r=1.1e-3,
        lateral_viscosity_operator="nemo_div_curl",
        vertical_momentum_scheme="nemo_advective",
    )


def _sum_components(diagnostics, axis_label: str):
    field_names = [f for f in diagnostics._fields
                   if f.endswith(f"_{axis_label}") and not f.startswith("total")]
    s = None
    for name in field_names:
        arr = getattr(diagnostics, name).data
        s = arr if s is None else s + arr
    return s


def test_closure_holds_under_nemo_div_curl_and_nemo_advective():
    """Σ components == total to machine precision under the SAME operator
    selections the DINO nemo_dino_kamm_mlf card uses (dyn_ldf's nemo_div_curl,
    dyn_adv ZAD's nemo_advective) -- the combination the fix_plan.md #1226
    zu_frc reconstruction actually reads Ah_lap_u/v and vertadv_u/v from."""
    grid, z_coord = _grid(), _z_coord()
    cfg = _dino_ldf_zad_config()
    state = _perturbed_state(grid, z_coord)

    tendencies, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=True,
    )
    # Non-vacuity: both terms must actually be nonzero under this config,
    # else the closure would pass trivially without exercising either branch.
    assert float(np.max(np.abs(np.asarray(diag.Ah_lap_u.data)))) > 0.0, (
        "Ah_lap_u is identically zero -- the nemo_div_curl branch did not "
        "fire; this test would prove nothing")
    assert float(np.max(np.abs(np.asarray(diag.vertadv_u.data)))) > 0.0, (
        "vertadv_u is identically zero -- the nemo_advective branch did not "
        "fire; this test would prove nothing")

    sum_u = _sum_components(diag, "u")
    sum_v = _sum_components(diag, "v")
    np.testing.assert_allclose(np.asarray(sum_u), np.asarray(diag.total_u.data),
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(sum_v), np.asarray(diag.total_v.data),
                               atol=1e-12, rtol=1e-12)


def test_closure_is_non_vacuous_catches_a_broken_capture():
    """Prove the closure assertion actually FAILS when a diagnostic capture
    is wired wrong -- e.g. Ah_lap_u double-counted into the sum but not into
    du_dt (simulating a capture bug). Guards against a closure test that
    would pass no matter what (Rule: a source-inspecting/contract test must
    be shown to fail when the property it checks is violated)."""
    grid, z_coord = _grid(), _z_coord()
    cfg = _dino_ldf_zad_config()
    state = _perturbed_state(grid, z_coord)

    _tendencies, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=True,
    )
    sum_u_broken = _sum_components(diag, "u") + np.asarray(diag.Ah_lap_u.data)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(sum_u_broken),
                                   np.asarray(diag.total_u.data),
                                   atol=1e-12, rtol=1e-12)


def test_diagnostics_do_not_change_production_du_dt_dv_dt():
    """BIT-IDENTITY (task requirement): diagnose_momentum=True must return
    the exact same du_dt/dv_dt as the production diagnose_momentum=False
    call -- the diagnostic breakdown is read-only bookkeeping on top of the
    same computation, never an alternate code path."""
    grid, z_coord = _grid(), _z_coord()
    cfg = _dino_ldf_zad_config()
    state = _perturbed_state(grid, z_coord)

    plain = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=False,
    )
    with_diag, _diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=True,
    )
    diff_u = np.max(np.abs(np.asarray(plain.du_dt.data) - np.asarray(with_diag.du_dt.data)))
    diff_v = np.max(np.abs(np.asarray(plain.dv_dt.data) - np.asarray(with_diag.dv_dt.data)))
    assert diff_u == 0.0, f"du_dt changed by diagnose_momentum flag: max|diff|={diff_u!r}"
    assert diff_v == 0.0, f"dv_dt changed by diagnose_momentum flag: max|diff|={diff_v!r}"
