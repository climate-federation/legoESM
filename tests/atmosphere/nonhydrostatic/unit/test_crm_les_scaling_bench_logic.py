"""Decomposition + tile-local initial-condition logic of the scaling benches."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "scripts" / "bench"))
sys.path.insert(0, str(_REPO / "scripts" / "run"))

import bench_plane_crm_dd_scaling as crm  # noqa: E402


@pytest.mark.parametrize("n", range(1, 513))
def test_most_square_factors_is_the_squarest_exact_split(n):
    n_y, n_x = crm.most_square_factors(n)
    assert n_y * n_x == n and n_y <= n_x
    assert not any(n % d == 0 for d in range(n_y + 1, int(n ** 0.5) + 1))


def test_most_square_factors_ladder_values():
    got = {n: crm.most_square_factors(n) for n in (1, 2, 8, 128, 256, 512)}
    assert got == {1: (1, 1), 2: (1, 2), 8: (2, 4), 128: (8, 16),
                   256: (16, 16), 512: (16, 32)}


def test_global_grid_shape():
    assert crm.global_grid_shape("strong", 512, 512, 8, 16) == (512, 512)
    assert crm.global_grid_shape("weak", 64, 64, 8, 16) == (512, 1024)
    with pytest.raises(ValueError):
        crm.global_grid_shape("bogus", 1, 1, 1, 1)


def test_crm_tile_ic_equals_slice_of_driver_global_ic():
    import jax
    import jax.numpy as jnp
    import run_rcemip_plane as rcp
    from legoesm.grids.plane import create_plane_grid
    from legoesm.parallel.plane_mpi import (
        make_plane_pencil_grid, make_plane_pencil_layout,
    )

    jax.config.update("jax_enable_x64", True)
    ny, nx, nlev = 8, 12, 6
    rargs = rcp.parse_args(["--nx", str(nx), "--ny", str(ny), "--nlev",
                            str(nlev), "--theta-noise-amp", "0.1"])
    hc, _, q_sfc = rcp.build_rcemip_height_coord(rargs, rcp.P_SFC_RCEMIP,
                                                 jnp.float64)
    g_glob = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=rargs.dx,
                               dy=rargs.dx, dtype=jnp.float64)
    ref = rcp._build_rcemip_initial_state(
        g_glob, hc, dtype=jnp.float64, theta_noise_amp=0.1, n_tracers=3,
        seed_kind="smooth_k1", q_sfc=q_sfc)
    assert float(jnp.max(jnp.abs(ref.theta_prime.data))) > 0.05  # non-vacuous
    for r in range(6):
        lay = make_plane_pencil_layout(rank=r, n_ranks=6, n_ranks_y=2,
                                       n_ranks_x=3, ny_global=ny,
                                       nx_global=nx)
        g_loc = make_plane_pencil_grid(lay, dx=rargs.dx, dy=rargs.dx,
                                       nlev=nlev, dtype=jnp.float64)
        loc = crm.build_local_ic(rcp, rargs, g_loc, hc, lay, q_sfc,
                                 jnp.float64)
        sl = (slice(lay.iy_start, lay.iy_end), slice(lay.ix_start, lay.ix_end))
        for name in ("u", "v", "w", "theta_prime", "rho_prime", "tracers"):
            a = getattr(loc, name).data
            b = getattr(ref, name).data[sl]
            assert a.shape == b.shape, name
            assert float(jnp.max(jnp.abs(a - b))) < 1e-12, (name, r)


def test_les_scaled_grid_keeps_cell_size_and_grows_domain():
    import bench_spectral_les_dd_scaling as les_b
    from legoesm.atmosphere.les_suite import (
        get_case, list_cases, register_default_catalog,
    )
    if not list_cases():
        register_default_catalog()
    cg = get_case("cbl_nieuwstadt").grid
    ny_g, nx_g = crm.global_grid_shape("weak", 64, 64, 8, 16)
    g = les_b.scaled_grid(cg, nx_g, ny_g, 128)
    assert g.Lx_m / g.nx == pytest.approx(cg.Lx_m / cg.nx)
    assert g.Ly_m / g.ny == pytest.approx(cg.Ly_m / cg.ny)
    assert g.Lz_m / g.nz == pytest.approx(cg.Lz_m / cg.nz)
    assert g.Ly_m == pytest.approx(512 * cg.Ly_m / cg.ny)


def test_les_slab_ic_independent_of_rank_count():
    import jax
    import jax.numpy as jnp
    import bench_spectral_les_dd_scaling as les_b
    import run_les_suite as les
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
    from legoesm.atmosphere.les_suite import (
        get_case, list_cases, register_default_catalog,
    )

    jax.config.update("jax_enable_x64", True)
    if not list_cases():
        register_default_catalog()
    case = get_case("cbl_nieuwstadt")
    args = les.build_arg_parser().parse_args(["--case", case.name])
    args.theta0 = les._CBL_THETA0_K
    grid = les_b.scaled_grid(case.grid, 8, 8, 12)
    g = sl.make_grid(les.cbl_les_config(grid, args, "lasd"), dtype=jnp.float64)
    whole = les_b.local_cbl_state(sl, g, args, 0, 8, 0.0, jnp.float64)
    parts = [les_b.local_cbl_state(sl, g, args, r * 2, 2, 0.0, jnp.float64)
             for r in range(4)]
    stacked = jnp.concatenate([p.theta for p in parts], axis=0)
    assert float(jnp.std(whole.theta[..., 0])) > 0.01  # noise present
    assert float(jnp.max(jnp.abs(stacked - whole.theta))) == 0.0


def test_driver_builders_resolve_production_defaults():
    import run_les_suite as les
    import run_rcemip_plane as rcp

    a = rcp.parse_args([])
    a.hyperdiff = rcp.dx_aware_hyperdiff(a.dx)
    cfg = rcp.build_dycore_config(a)
    assert cfg.substep_horizontal_acoustic is True
    assert cfg.sgs_vertical_diffusion is True
    assert cfg.hyperdiff_coeff == pytest.approx(1.0e8 * (a.dx / 1000.0) ** 4)
    assert [rcp.microphysics_n_tracers(m) for m in
            ("kessler", "morrison", "p3")] == [3, 11, 9]
    la = les.build_arg_parser().parse_args(["--case", "cbl_nieuwstadt"])
    la.theta0 = 300.0
    from legoesm.atmosphere.les_suite import (
        get_case, list_cases, register_default_catalog,
    )
    if not list_cases():
        register_default_catalog()
    lc = les.cbl_les_config(get_case("cbl_nieuwstadt").grid, la, "lasd")
    assert lc.dealias and lc.time_scheme == "rk3" and not lc.sgs_buoyancy
