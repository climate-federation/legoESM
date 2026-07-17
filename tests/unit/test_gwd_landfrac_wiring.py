"""E3SM GWD landfrac wiring: the coupled pipeline threads its ``f_land``
(set by the model driver next to ``subgrid_topo_stddev``) into the GWD call
for exactly the ``e3sm_cam`` scheme (the only kernel accepting
``land_frac_col``; E3SM gw_drag.F90:904-906 oro landfrac scaling).

Guards the codex-flagged phantom: an extractor reading an attribute nothing
sets.  The recorder tests below EXECUTE ``physics_step_no_rad`` with the
kernel replaced by a capturing stub, so deleting the forwarding at the GWD
call site goes red (codex round-2 MEDIUM: flag-only assertions were
vacuous).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.gravity_wave_drag import GWDOutput
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

NLEV = 5
NLAT, NLON = 8, 16


def _config(gwd):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=NLAT, nlev=NLEV),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray",
        gravity_wave_drag=gwd,
    )


def _pipe(gwd):
    grid = create_latlon_grid(NLAT, NLON, dtype=jnp.float64)
    sigma = create_sigma_coordinate(NLEV)
    pipe = build_physics_pipeline(grid, sigma, _config(gwd))
    return grid, pipe


def _run_step(grid, pipe):
    """Minimal physics_step_no_rad invocation (radiation held at zero)."""
    s2 = grid.grid_lat.shape
    s3 = (*s2, NLEV)
    Tprof = 290.0 - 50.0 * np.linspace(0, 1, NLEV)[::-1]
    T = jnp.asarray(np.broadcast_to(Tprof, s3).copy())
    q_v = jnp.full(s3, 5e-3)
    u = jnp.full(s3, 12.0)
    v = jnp.zeros(s3)
    z2 = jnp.zeros(s2)
    z3 = jnp.zeros(s3)
    return pipe.physics_step_no_rad(
        T, jnp.full(s2, 1.0e5), q_v, jnp.zeros(s3), jnp.zeros(s3), None,
        u, v, jnp.full(s2, 300.0), jnp.zeros(s2),
        jnp.asarray(grid.grid_lat), 600.0,
        z3, z2, z2, z2, z2, z2,
    )


class _Recorder:
    """Stands in for the GWD kernel; captures the call kwargs."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kw):
        self.kwargs = kw
        shape = kw["u"].shape
        zero = jnp.zeros(shape)
        return GWDOutput(du_dt=zero, dv_dt=zero, dT_dt=zero,
                         eps_gwd=jnp.zeros(shape[0]))


def test_e3sm_cam_pipeline_threads_f_land_as_land_frac_col():
    """END-TO-END: with f_land set, physics_step_no_rad delivers it to the
    e3sm_cam kernel as (ncol,) land_frac_col with the right values."""
    grid, pipe = _pipe("e3sm_cam")
    assert pipe._gwd_takes_land_frac is True
    f_land = jnp.asarray(
        np.linspace(0.0, 1.0, NLAT * NLON).reshape(grid.grid_lat.shape)
    )
    pipe.f_land = f_land
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None, "GWD kernel was never called"
    assert "land_frac_col" in rec.kwargs, (
        "pipeline did not forward f_land as land_frac_col")
    got = rec.kwargs["land_frac_col"]
    assert got.shape == (NLAT * NLON,)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(f_land).reshape(-1))


def test_e3sm_cam_f_land_none_sends_no_kwarg():
    """f_land=None (driver never populated it) -> no land_frac_col kwarg ->
    the kernel default (no scaling) applies — legacy bit-identity."""
    grid, pipe = _pipe("e3sm_cam")
    assert pipe.f_land is None
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None
    assert "land_frac_col" not in rec.kwargs


@pytest.mark.parametrize("scheme", ["mcfarlane", "lindzen", "hines"])
def test_other_gwd_schemes_never_send_land_frac(scheme):
    """mcfarlane/lindzen/hines kernels do NOT accept ``land_frac_col`` —
    sending it would TypeError in production.  Even with f_land set, the
    pipeline must not forward it (builder flag False + recorder proof)."""
    grid, pipe = _pipe(scheme)
    assert pipe._gwd_takes_land_frac is False
    pipe.f_land = jnp.full(grid.grid_lat.shape, 0.5)
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None
    assert "land_frac_col" not in rec.kwargs


def test_none_scheme_flag_false():
    """gravity_wave_drag='none' -> no GWD at all; the flag stays False."""
    _, pipe = _pipe("none")
    assert pipe._gwd_takes_land_frac is False
    assert pipe.gwd_fn is None
