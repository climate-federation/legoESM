"""Parity + gating tests for the halo-aware SI-horizontal acoustic substep.

At ``layout.n_ranks == 1`` the packed halo exchange devolves to
``jnp.pad(mode='wrap')`` and the halo-aware operators reduce to the serial
``jnp.roll`` stencils, so :func:`plane_acoustic_substeps_si_horizontal_halo`
must reproduce :func:`plane_acoustic_substeps_si_horizontal` bit-for-bit —
the same contract the halo slow tendency already satisfies
(``test_plane_slow_tend_halo.py``). Multi-rank correctness is covered by
``tests/distributed/test_plane_acoustic_halo_mpi.py`` under ``mpirun``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_acoustic_substeps_si_horizontal,
    plane_acoustic_substeps_si_horizontal_halo,
    plane_compressible_euler_slow_tendencies,
    validate_plane_config,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane_halo import (
    plane_compressible_euler_slow_tendencies_halo,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout
from legoesm.timestepping.split_explicit import SplitExplicitConfig

jax.config.update("jax_enable_x64", True)

_NY, _NX, _NLEV = 6, 8, 10


def _setup(moist=False, beta_oc=0.0):
    grid = create_plane_grid(
        nx=_NX, ny=_NY, nlev=_NLEV, dx=1000.0, dy=2000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(_NLEV, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        semi_implicit_acoustic=True,
        substep_horizontal_acoustic=True,
        acoustic_off_centering=beta_oc,
        n_acoustic_substeps=4,
        smagorinsky_cs=0.0,
        moist_buoyancy=moist,
    )
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    keys = jax.random.split(jax.random.PRNGKey(7), 6)
    state = rest._replace(
        u=rest.u.replace(
            data=0.1 * jax.random.normal(keys[0], rest.u.data.shape)),
        v=rest.v.replace(
            data=0.1 * jax.random.normal(keys[1], rest.v.data.shape)),
        w=rest.w.replace(
            data=0.01 * jax.random.normal(keys[2], rest.w.data.shape)),
        theta_prime=rest.theta_prime.replace(
            data=0.5 * jax.random.normal(
                keys[3], rest.theta_prime.data.shape)),
        rho_prime=rest.rho_prime.replace(
            data=0.001 * jax.random.normal(
                keys[4], rest.rho_prime.data.shape)),
    )
    if moist:
        tr = 1e-3 * jax.random.uniform(keys[5], (_NY, _NX, _NLEV, 3))
        state = state._replace(tracers=state.tracers.replace(data=tr))
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=_NY, nx_global=_NX,
    )
    return grid, hc, tm, cfg, state, layout


def _run_both(grid, hc, tm, cfg, state, layout, n_sub=4, dt_s=0.5):
    se_cfg = SplitExplicitConfig(n_substeps=n_sub)
    serial = plane_acoustic_substeps_si_horizontal(
        state, None, dt_s, n_sub, se_cfg, hc, tm, cfg, grid,
    )
    halo = plane_acoustic_substeps_si_horizontal_halo(
        state, None, dt_s, n_sub, se_cfg, hc, tm, cfg, grid, layout,
    )
    return serial, halo


@pytest.mark.parametrize("beta_oc", [0.0, 0.1])
def test_si_horizontal_halo_matches_serial_single_rank(beta_oc):
    grid, hc, tm, cfg, state, layout = _setup(beta_oc=beta_oc)
    serial, halo = _run_both(grid, hc, tm, cfg, state, layout)
    for name in ("u", "v", "w", "theta_prime", "rho_prime"):
        np.testing.assert_allclose(
            np.asarray(getattr(halo, name).data),
            np.asarray(getattr(serial, name).data),
            rtol=0.0, atol=1e-13, err_msg=name,
        )


def test_si_horizontal_halo_matches_serial_moist():
    """Moist buoyancy leg: frozen SAM b_moist applied each substep."""
    grid, hc, tm, cfg, state, layout = _setup(moist=True)
    serial, halo = _run_both(grid, hc, tm, cfg, state, layout)
    for name in ("u", "v", "w", "theta_prime", "rho_prime"):
        np.testing.assert_allclose(
            np.asarray(getattr(halo, name).data),
            np.asarray(getattr(serial, name).data),
            rtol=0.0, atol=1e-13, err_msg=name,
        )
    # tracers unchanged by the acoustic loop
    np.testing.assert_array_equal(
        np.asarray(halo.tracers.data), np.asarray(state.tracers.data),
    )


def test_slow_tendency_halo_gates_pg_and_continuity():
    """With substep_horizontal_acoustic=True the halo slow tendency must
    drop the horizontal PG + continuity exactly like the serial one —
    anything else double-counts once the halo acoustic substep runs."""
    grid, hc, tm, cfg, state, layout = _setup()
    serial = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    halo = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    for name in ("du_dt", "dv_dt", "dw_dt", "dtheta_prime_dt",
                 "drho_prime_dt"):
        np.testing.assert_allclose(
            np.asarray(getattr(halo, name).data),
            np.asarray(getattr(serial, name).data),
            rtol=0.0, atol=1e-13, err_msg=name,
        )
    # continuity fully substepped ⇒ with sponge + hyperdiff off, the
    # slow rho' tendency is exactly zero
    cfg_nosponge = cfg._replace(sponge_coeff=0.0, hyperdiff_rho_coeff=0.0)
    halo_ns = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg_nosponge, layout,
    )
    assert float(jnp.max(jnp.abs(halo_ns.drho_prime_dt.data))) == 0.0


def test_validate_rejects_substep_horiz_without_si():
    cfg = CompressibleEulerConfig(
        semi_implicit_acoustic=False,
        substep_horizontal_acoustic=True,
    )
    with pytest.raises(ValueError, match="substep_horizontal_acoustic"):
        validate_plane_config(cfg)
