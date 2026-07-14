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

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane_halo import (
    plane_compressible_euler_slow_tendencies_halo,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


def _setup(use_coriolis=False, hyperdiff=0.0, with_tracers=False,
           scheme="upwind1", halo=1):
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
        horizontal_advection_scheme=scheme,
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
        ny_global=6, nx_global=8, halo=halo,
    )
    return grid, hc, tm, cfg, state, layout


def test_momentum_advection_split_raises_in_mpi_halo_path():
    """ADV-SPLIT #86 (codex iter-68 [S2]): the per-field momentum/scalar
    advection split is wired ONLY in the serial slow-tendency path. The MPI
    halo path must FAIL LOUD (NotImplementedError) — not silently apply the
    scalar scheme to momentum, which would diverge serial≠MPI — until the
    split is wired here too. Guards the explicit experimental limitation."""
    grid, hc, tm, _, state, layout = _setup()
    cfg_split = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        smagorinsky_cs=0.0, use_coriolis=False, n_acoustic_substeps=12,
        horizontal_advection_scheme="van_leer",
        horizontal_momentum_advection_scheme="centered",   # split ≠ scalar
    )
    with pytest.raises(NotImplementedError, match="ADV-SPLIT"):
        plane_compressible_euler_slow_tendencies_halo(
            state, grid, hc, tm, cfg_split, layout,
        )


def test_vertical_sgs_raises_in_mpi_halo_path():
    """SGS-VERT #81 (iter-68): the vertical SGS flux ∂_z(K ∂_z φ) is wired only
    in the serial path. The MPI halo path must FAIL LOUD — not silently apply
    horizontal-only SGS (serial≠MPI divergence) — until the vertical leg is
    wired here too. Parallels the #86 momentum-split guard."""
    grid, hc, tm, _, state, layout = _setup()
    cfg_vsgs = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        smagorinsky_cs=0.2, use_coriolis=False, n_acoustic_substeps=12,
        sgs_vertical_diffusion=True,
    )
    with pytest.raises(NotImplementedError, match="SGS-VERT"):
        plane_compressible_euler_slow_tendencies_halo(
            state, grid, hc, tm, cfg_vsgs, layout,
        )


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


def test_halo_equiv_weno5_tracers():
    """WENO5 + tracers: the positivity guard routes the TRACER legs onto
    van_leer in BOTH the serial and halo paths (θ′ stays on weno5), so
    serial==MPI stays bit-identical on the new ``tadv_*`` path. The
    van_leer/upwind1 tracer equiv tests above cannot lock this (the guard is
    inactive for a monotone scheme); this halo=3 case is the one that does."""
    grid, hc, tm, cfg, state, layout = _setup(
        with_tracers=True, scheme="weno5", halo=3,
    )
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg, layout,
    )
    _eq_tendencies(actual, expected)


def test_halo_equiv_van_leer_vertical_tracers():
    """VERTICAL van_leer tracer advection is honored on the MPI halo path
    (codex CRM-dycore review): previously the halo tracer leg was hardcoded to
    centered, silently diverging from serial whenever
    ``vertical_tracer_advection="van_leer"`` (the RCE runner default). With
    nonzero w + tracers, serial and halo slow-tendencies must stay
    bit-identical now that the halo mirrors the serial vertical dispatch."""
    grid, hc, tm, cfg, state, layout = _setup(with_tracers=True)
    cfg_vl = cfg._replace(vertical_tracer_advection="van_leer")
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_vl,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg_vl, layout,
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


def test_halo_equiv_with_smagorinsky_and_tracers():
    """SAM-faithfulness parity (Codex iter-1 HIGH): the dosmagor
    stratification term ``−Pr·N²`` builds N² from a MOIST virtual
    potential temperature θ_v (vapor + condensate loading). Serial and
    halo must compute the SAME θ_v / K_m at single rank, so the
    moisture-laden Smag tendency stays bit-consistent across the MPI
    decomposition. Exercises the n_tr>0 θ_v branch in BOTH paths."""
    grid, hc, tm, cfg, state, layout = _setup(with_tracers=True)
    cfg_smag = cfg._replace(smagorinsky_cs=0.2, smagorinsky_prandtl=1.0)
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_smag,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg_smag, layout,
    )
    _eq_tendencies(actual, expected, rtol=1e-11, atol=1e-11)


def test_halo_equiv_with_molecular_closure():
    """DNS-LES (iter-177): the molecular closure (constant ν, no eddy model)
    must give the SAME serial vs single-rank-halo slow tendency as Smagorinsky
    does. A constant K_m is trivially halo-consistent; this pins the new
    closure branch in BOTH paths, with tracers so the K_h = ν/Pr scalar leg is
    exercised too."""
    grid, hc, tm, cfg, state, layout = _setup(with_tracers=True)
    cfg_mol = cfg._replace(
        turbulence_closure="molecular", molecular_viscosity=5.0,
        molecular_prandtl=0.71, smagorinsky_cs=0.0,
    )
    expected = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_mol,
    )
    actual = plane_compressible_euler_slow_tendencies_halo(
        state, grid, hc, tm, cfg_mol, layout,
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane_halo import (
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


def test_halo_multirank_jit_traceable():
    """Multi-rank slow-tendency now JIT-TRACES (the macOS-era guard is
    removed — the ConcretizationTypeError came from
    ``int(jnp.prod(jnp.asarray(...)))`` on static shapes, now ``math.prod``;
    mpi4jax ``sendrecv`` is jit-safe on Linux/MPICH, as ``voronoi_mpi``
    proves).  Eager multi-rank was ~100x slower than np=1; ``step_halo`` now
    JITs the split-explicit core.  Trace only via ``make_jaxpr`` — executing
    needs a real 2-rank MPI comm (covered by the distributed test)."""
    # Skip where MPI can't load — ``mpi4py.MPI`` raises RuntimeError (not
    # ImportError) when libmpi is absent (e.g. the GPU-only .venv), which
    # importorskip would not catch.
    try:
        import mpi4jax  # noqa: F401
        from mpi4py import MPI  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"mpi4jax/MPI unavailable: {exc}")
    grid, hc, tm, cfg, state, _ = _setup()
    # Multi-rank layout whose LOCAL block matches the _setup state shape
    # (ny_local=6, nx_local=8): split y across 2 ranks (ny_global=12), keep
    # x whole. n_ranks_y>1 exercises the mpi4jax sendrecv (NS) branch.
    layout_mr = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=2, n_ranks_x=1,
        ny_global=12, nx_global=8,
    )
    # No longer raises the old RuntimeError guard; produces a valid jaxpr.
    jaxpr = jax.make_jaxpr(
        lambda s: plane_compressible_euler_slow_tendencies_halo(
            s, grid, hc, tm, cfg, layout_mr,
        )
    )(state)
    assert jaxpr is not None


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
