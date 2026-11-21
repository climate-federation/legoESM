"""MPAS turbulence bridge exports the surface stress it applied (CMOR tauu/tauv).

The stress ON THE SURFACE is minus the column integral of the turbulence
momentum tendency; for a drag closure it points along the near-surface wind,
so ``tau . V_low > 0`` in every column with wind.  The slots ride the CMOR
interval-mean accumulator like ``pr``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import integration as turb_int  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.grids.vertical import make_cam6_l32_levels  # noqa: E402
from legoesm.grids.voronoi import reconstruct_cell_velocity  # noqa: E402


@pytest.fixture(scope="module")
def col_state():
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    coord = make_cam6_l32_levels()
    state = held_suarez_init_mpas(mesh, coord, T_init=290.0, perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    rng = np.random.default_rng(0)
    u_edge = jnp.asarray(rng.standard_normal(state.u.data.shape) * 8.0)
    T = jnp.asarray(290.0 - 40.0 * np.linspace(1.0, 0.0, nlev)[None, :]
                    + rng.standard_normal((ncell, nlev)))
    state = state._replace(
        u=state.u.replace(data=u_edge), T=state.T.replace(data=T),
        tracers={nm: state.T.replace(data=val, name=nm, units="kg/kg")
                 for nm, val in (("q_v", jnp.full((ncell, nlev), 0.004)),
                                 ("q_c", jnp.zeros((ncell, nlev))),
                                 ("q_r", jnp.zeros((ncell, nlev))))})
    u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)
    col = state._replace(u=state.u.replace(data=u_cell, name="u"),
                         v=state.u.replace(data=v_cell, name="v"))
    return mesh, coord, col


def test_stress_points_along_the_surface_wind(col_state):
    mesh, coord, col = col_state
    fn = turb_int._make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0)
    t = fn(col, mesh, coord, forcing={"T_sfc": jnp.full((col.T.data.shape[0],), 292.0)})
    t = t[0] if isinstance(t, tuple) else t
    taux, tauy = np.asarray(t.taux_sfc.data), np.asarray(t.tauy_sfc.data)
    u, v = np.asarray(col.u.data)[:, -1], np.asarray(col.v.data)[:, -1]
    work = taux * u + tauy * v
    assert np.all(np.isfinite(taux)) and np.abs(taux).max() > 0.0
    assert work.min() > 0.0, "surface stress must point along the lowest-level wind"
    assert np.abs(taux).max() < 5.0   # Pa: a bulk drag at <~25 m/s winds


def test_stress_and_prc_ride_the_interval_mean_accumulator():
    from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS
    from legoesm.driver.model_driver import _MPASSfcFluxAccum, _sfc_diag_slot
    for k in ("taux_sfc", "tauy_sfc", "precip_conv_sfc"):
        assert k in MPAS_SFC_DIAG_EXTRA_KEYS
        assert _sfc_diag_slot(k) in _MPASSfcFluxAccum.SLOTS
