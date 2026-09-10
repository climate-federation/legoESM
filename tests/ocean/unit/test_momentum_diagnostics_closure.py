"""Closure test for the momentum-tendency diagnostic breakdown.

When ``latlon_cgrid_ocean_baroclinic_tendencies(..., diagnose_momentum=True)``
is called, it returns a per-term breakdown of the momentum tendency.  By
construction, the sum of all components must equal the total to machine
precision.  This test enforces that contract — analogous to MOM6's
``MOM_diagnostics`` and NEMO's ``trd_*`` closure assertions.

If this test ever fails, it means either:
- A new term was added to ``compute_tendencies`` without updating the
  diagnostic capture path.
- A diagnostic capture was wired to the wrong sign or shape.
"""

from __future__ import annotations

import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.core.field import Field


def _perturbed_state(grid, z_coord):
    """Build a state with non-trivial u, v, T, eta so all diagnostic terms
    are exercised (rest state would zero out vortcor, vertadv, etc.)."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    rng = np.random.default_rng(42)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
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


def _sum_components(diagnostics, axis_label: str):
    """Sum all u-side or v-side diagnostic components except total_*."""
    field_names = [f for f in diagnostics._fields
                   if f.endswith(f"_{axis_label}") and not f.startswith("total")]
    s = None
    for name in field_names:
        arr = getattr(diagnostics, name).data
        s = arr if s is None else s + arr
    return s


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=36, n_lon=72)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=10, H_max=4000.0)


def test_diagnostic_closure_no_drag_no_physics(grid, z_coord):
    """With minimal config (only A_h active), Σ components == total to
    machine precision."""
    cfg = LatLonCGridOceanConfig.from_flat(A_h=1.0e4, A_v=0.0, bottom_drag_r=0.0)
    state = _perturbed_state(grid, z_coord)

    tendencies, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=True,
    )

    sum_u = _sum_components(diag, "u")
    sum_v = _sum_components(diag, "v")
    total_u = diag.total_u.data
    total_v = diag.total_v.data

    np.testing.assert_allclose(np.asarray(sum_u), np.asarray(total_u),
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(sum_v), np.asarray(total_v),
                               atol=1e-12, rtol=1e-12)
    # And total must equal what the un-diagnosed call returns.
    plain_tend = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=False,
    )
    np.testing.assert_allclose(
        np.asarray(plain_tend.du_dt.data), np.asarray(total_u),
        atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(plain_tend.dv_dt.data), np.asarray(total_v),
        atol=1e-12, rtol=1e-12)


def test_diagnostic_closure_full_config(grid, z_coord):
    """With every momentum term active that can be (A_h, A_v,
    bottom_drag), closure to machine precision."""
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e5, A_v=1.0e-3, bottom_drag_r=1.1e-3,
    )
    state = _perturbed_state(grid, z_coord)

    tendencies, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=True,
    )
    sum_u = _sum_components(diag, "u")
    sum_v = _sum_components(diag, "v")
    np.testing.assert_allclose(np.asarray(sum_u),
                               np.asarray(diag.total_u.data),
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(sum_v),
                               np.asarray(diag.total_v.data),
                               atol=1e-12, rtol=1e-12)


def test_diagnostic_closure_external_surface_stress(grid, z_coord):
    """External stress has its own diagnostic and closes on both faces.

    This is deliberately zonal and meridional: omitting the diagnostic makes
    both closures red, while a zonal-only forcing would hide the V-side miss.
    """
    cfg = LatLonCGridOceanConfig.from_flat(A_h=0.0, A_v=0.0,
                                           bottom_drag_r=0.0)
    state = _perturbed_state(grid, z_coord)
    shape = (grid.n_lat, grid.n_lon)
    forcing = OceanSurfaceForcing(
        tau_x=jnp.full(shape, 0.12, dtype=jnp.float64),
        tau_y=jnp.full(shape, -0.07, dtype=jnp.float64),
    )

    tendencies, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, surface_forcing=forcing,
        diagnose_momentum=True,
    )
    assert float(jnp.max(jnp.abs(diag.surface_stress_u.data))) > 0.0
    assert float(jnp.max(jnp.abs(diag.surface_stress_v.data))) > 0.0
    np.testing.assert_allclose(np.asarray(_sum_components(diag, "u")),
                               np.asarray(diag.total_u.data),
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(_sum_components(diag, "v")),
                               np.asarray(diag.total_v.data),
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(tendencies.du_dt.data),
                                  np.asarray(diag.total_u.data))
    np.testing.assert_array_equal(np.asarray(tendencies.dv_dt.data),
                                  np.asarray(diag.total_v.data))
    plain = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, surface_forcing=forcing,
        diagnose_momentum=False,
    )
    np.testing.assert_array_equal(np.asarray(plain.du_dt.data),
                                  np.asarray(tendencies.du_dt.data))
    np.testing.assert_array_equal(np.asarray(plain.dv_dt.data),
                                  np.asarray(tendencies.dv_dt.data))


def test_default_signature_unchanged(grid, z_coord):
    """Default call (diagnose_momentum=False) must return a single
    LatLonCGridOceanTendencies, not a tuple."""
    from legoesm.ocean.state import LatLonCGridOceanTendencies
    cfg = LatLonCGridOceanConfig.from_flat()
    state = _perturbed_state(grid, z_coord)
    out = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg,
    )
    assert isinstance(out, LatLonCGridOceanTendencies)


def test_diagnose_returns_tuple(grid, z_coord):
    """With diagnose_momentum=True, return a tuple of length 2."""
    from legoesm.ocean.state import (
        LatLonCGridOceanTendencies, MomentumTendencyDiagnostics,
    )
    cfg = LatLonCGridOceanConfig.from_flat()
    state = _perturbed_state(grid, z_coord)
    out = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=True,
    )
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], LatLonCGridOceanTendencies)
    assert isinstance(out[1], MomentumTendencyDiagnostics)
