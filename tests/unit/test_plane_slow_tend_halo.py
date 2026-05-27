"""Bit-equivalence: halo-aware slow tendency == original on single rank.

When ``layout.n_ranks == 1`` the halo exchange devolves to
``jnp.pad(mode='wrap')`` + the halo-aware operators reduce to the
original ``jnp.roll`` stencils via slice arithmetic. Output must
match bit-for-bit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane_halo import (
    plane_compressible_euler_slow_tendencies_halo,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


def _setup(use_coriolis=False, hyperdiff=0.0, with_tracers=False):
    grid = create_plane_grid(
        nx=8, ny=6, nlev=10, dx=1000.0, dy=2000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(10, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        hyperdiff_coeff=hyperdiff,
        hyperdiff_rho_coeff=hyperdiff,
        hyperdiff_w_coeff=hyperdiff,
        smagorinsky_cs=0.0,
        use_coriolis=use_coriolis,
        n_acoustic_substeps=12,
    )
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 6)
    u_kick = 0.1 * jax.random.normal(keys[0], rest.u.data.shape)
    v_kick = 0.1 * jax.random.normal(keys[1], rest.v.data.shape)
    w_kick = 0.01 * jax.random.normal(keys[2], rest.w.data.shape)
    th_kick = 0.5 * jax.random.normal(keys[3], rest.theta_prime.data.shape)
    rho_kick = 0.001 * jax.random.normal(
        keys[4], rest.rho_prime.data.shape,
    )
    state = rest._replace(
        u=rest.u.replace(data=u_kick),
        v=rest.v.replace(data=v_kick),
        w=rest.w.replace(data=w_kick),
        theta_prime=rest.theta_prime.replace(data=th_kick),
        rho_prime=rest.rho_prime.replace(data=rho_kick),
    )
    if with_tracers:
        tr = 0.01 * jax.random.normal(
            keys[5], (6, 8, 10, 3),
        )
        state = state._replace(
            tracers=state.tracers.replace(data=tr),
        )
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=6, nx_global=8,
    )
    return grid, hc, tm, cfg, state, layout


def _eq_tendencies(a, b, rtol=1e-12, atol=1e-12):
    for fld in (
        "du_dt", "dv_dt", "dw_dt", "dtheta_prime_dt",
        "drho_prime_dt", "dtracers_dt",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(a, fld).data),
            np.asarray(getattr(b, fld).data),
            rtol=rtol, atol=atol,
            err_msg=f"{fld} mismatch",
        )


def test_halo_equiv_basic_no_coriolis_no_hyperdiff():
    grid, hc, tm, cfg, state, layout = _setup()
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(actual, expected)


def test_halo_equiv_with_coriolis():
    grid, hc, tm, cfg, state, layout = _setup(use_coriolis=True)
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(actual, expected)


def test_halo_equiv_with_hyperdiff():
    grid, hc, tm, cfg, state, layout = _setup(hyperdiff=1.0e6)
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(actual, expected)


def test_halo_equiv_with_tracers():
    grid, hc, tm, cfg, state, layout = _setup(with_tracers=True)
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(actual, expected)


def test_halo_equiv_full():
    """Coriolis + hyperdiff + tracers all on (Smag stays off)."""
    grid, hc, tm, cfg, state, layout = _setup(
        use_coriolis=True, hyperdiff=1.0e6, with_tracers=True,
    )
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(actual, expected)


def test_halo_equiv_with_smagorinsky():
    """R4: Smag LES ported to halo path; single-rank must match serial
    bit-for-bit. Halo K_m uses padded slice stencils equivalent to the
    serial jnp.roll stencils when n_ranks=1 wraps the padded slab."""
    grid, hc, tm, cfg, state, layout = _setup()
    cfg_smag = cfg._replace(smagorinsky_cs=0.2, smagorinsky_prandtl=1.0)
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_smag,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg_smag, layout,
    )
    _eq_tendencies(actual, expected, rtol=1e-11, atol=1e-11)


def test_halo_equiv_with_vertical_theta_diffusion():
    """R5: vertical θ Laplacian is column-local — bit-equivalent on
    single rank."""
    grid, hc, tm, cfg, state, layout = _setup()
    cfg_vtd = cfg._replace(vertical_theta_diffusion=1.0e4)
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_vtd,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg_vtd, layout,
    )
    _eq_tendencies(actual, expected, rtol=1e-11, atol=1e-11)


def test_halo_equiv_with_weno5_advection():
    """R6: WENO5 horizontal advection ported to halo path. Single-rank
    must match the serial WENO5 implementation bit-for-bit. Layout
    halo must be ≥ 3 for the 6-point stencil — verify the gate."""
    grid = create_plane_grid(
        nx=8, ny=6, nlev=10, dx=1000.0, dy=2000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(10, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        smagorinsky_cs=0.0,
        use_coriolis=False,
        n_acoustic_substeps=12,
        horizontal_advection_scheme="weno5",
    )
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 5)
    state = rest._replace(
        u=rest.u.replace(data=0.1 * jax.random.normal(keys[0], rest.u.data.shape)),
        v=rest.v.replace(data=0.1 * jax.random.normal(keys[1], rest.v.data.shape)),
        w=rest.w.replace(data=0.01 * jax.random.normal(keys[2], rest.w.data.shape)),
        theta_prime=rest.theta_prime.replace(
            data=0.5 * jax.random.normal(keys[3], rest.theta_prime.data.shape),
        ),
        rho_prime=rest.rho_prime.replace(
            data=0.001 * jax.random.normal(keys[4], rest.rho_prime.data.shape),
        ),
    )
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=6, nx_global=8, halo=3,
    )
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(actual, expected, rtol=1e-11, atol=1e-11)


def test_halo_weno5_rejects_halo_lt_3():
    """WENO5 needs layout.halo ≥ 3; default layout (halo=1) must
    fail-fast with a clear error message."""
    grid, hc, tm, cfg, state, layout = _setup()
    cfg_weno = cfg._replace(horizontal_advection_scheme="weno5")
    with pytest.raises(ValueError, match="halo >= 3"):
        plane_compressible_euler_slow_tendencies_halo(
            state, grid, hc, tm, cfg_weno, layout,
        )


def test_halo_equiv_smag_plus_vertical_theta_diff_plus_hyperdiff():
    """R4 + R5 + hyperdiff all on at once; still bit-equivalent."""
    grid, hc, tm, cfg, state, layout = _setup(hyperdiff=1.0e6)
    cfg_full = cfg._replace(
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
        vertical_theta_diffusion=1.0e4,
    )
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_full,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg_full, layout,
    )
    _eq_tendencies(actual, expected, rtol=1e-11, atol=1e-11)


def test_halo_uses_cached_f_pad():
    """Codex iter-3 perf: caller can pre-compute f_pad to skip per-step
    MPI exchange of static f_y. Bit-identical to non-cached path."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane_halo import (
        precompute_coriolis_halo,
    )
    grid, hc, tm, cfg, state, layout = _setup(use_coriolis=True)
    f_pad = precompute_coriolis_halo(grid, layout)
    out_cached = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout, f_pad_cached=f_pad,
    )
    out_eager = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(out_cached, out_eager, rtol=1e-14, atol=1e-14)


def test_halo_raises_on_jit_multirank():
    """Codex iter-3: hard guard against JIT under multi-rank."""
    grid, hc, tm, cfg, state, _ = _setup()
    # Fake multi-rank layout (n_ranks=2 metadata; no actual MPI here).
    layout_mr = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=1, n_ranks_x=2,
        ny_global=6, nx_global=8,
    )

    @jax.jit
    def fn(s):
        return plane_compressible_euler_slow_tendencies_halo(
            s, grid, hc, tm, cfg, layout_mr,
        )

    with pytest.raises(RuntimeError, match="JIT-compiled"):
        fn(state)


def test_halo_jit_compilable():
    grid, hc, tm, cfg, state, layout = _setup(
        use_coriolis=True, hyperdiff=1.0e6, with_tracers=True,
    )

    @jax.jit
    def fn(s):
        return plane_compressible_euler_slow_tendencies_halo(
            s, grid, hc, tm, cfg, layout,
        )

    out = fn(state)
    assert bool(jnp.all(jnp.isfinite(out.du_dt.data)))
