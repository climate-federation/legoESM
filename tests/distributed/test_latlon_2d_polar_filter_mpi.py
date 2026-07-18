"""Polar filter under a 2-D longitude split (proc_lon>1) — C7.

The polar filter is a per-latitude-row longitude rFFT, so under a longitude
split each rank owns only a lon slice. It is now WIRED via the AD-safe
lat-pencil transpose (``lon_gather_full`` / ``lon_scatter_full``): the filter
gathers the full lon circle, rFFT/mask/irFFTs on it (mask rebuilt at the global
n_lon), then scatters this rank's block back.

Correctness gate: with ``use_polar_filter=True``, a proc_lon=1 layout (4x1 —
full lon per rank, the filter runs LOCALLY on the whole circle, ground truth)
and a proc_lon=2 layout (2x2 — the new gather path) of the SAME global state
must produce the same gathered stepped state. Any bug in the gather/scatter,
the global-n_lon mask, or the transpose would make them disagree.

Run: ``mpirun -np 4 python -m pytest \
    tests/distributed/test_latlon_2d_polar_filter_mpi.py`` (needs exactly 4
ranks — 4x1 and 2x2 both span the world). The dry C-grid step JIT-compiles per
block shape; run under a long-walltime job, not inline.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

mpi4jax = pytest.importorskip("mpi4jax")  # noqa: F841
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (  # noqa: E402
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.grids.halo import set_halo_backend  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.polar_filter import set_polar_filter_lon_gather  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.parallel.latlon_mpi import (  # noqa: E402
    gather_field_latlon_2d,
    make_latlon_2d_layout,
    make_latlon_2d_mpi_step,
    scatter_state_latlon_2d,
    slice_latlon_grid_to_block_2d,
)


def _make_global_state(grid, nlev, seed=31337):
    rng = np.random.default_rng(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    eps = 1.0e-3
    u = eps * rng.standard_normal((n_lat, n_lon + 1, nlev))
    u[:, n_lon, :] = u[:, 0, :]
    v = eps * rng.standard_normal((n_lat + 1, n_lon, nlev))
    T = 300.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))
    p_s = 1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))
    phis = np.zeros((n_lat, n_lon))
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(u), v=jnp.asarray(v), T=jnp.asarray(T),
        p_s=jnp.asarray(p_s), phis=jnp.asarray(phis), tracers={})


def _run_layout(global_state, grid, sigma, config, pr, pc, dt):
    rank = MPI.COMM_WORLD.Get_rank()
    layout = make_latlon_2d_layout(rank, pr, pc, grid.n_lat, grid.n_lon)
    g = slice_latlon_grid_to_block_2d(grid, layout)
    model = CGridLatLonPrimitiveEquationModel(g, sigma, config)
    local = scatter_state_latlon_2d(global_state, layout)
    step = make_latlon_2d_mpi_step(model, layout)      # sets the lon-gather ctx for pc>1
    out = step(local, dt)
    gathered = {
        "u": gather_field_latlon_2d(out.u, layout, is_u_face=True),
        "v": gather_field_latlon_2d(out.v, layout, is_v_face=True),
        "T": gather_field_latlon_2d(out.T, layout),
        "p_s": gather_field_latlon_2d(out.p_s, layout),
    }
    # Reset BOTH process-global contexts before the next leg — else the
    # proc_lon=1 leg would inherit the proc_lon=2 leg's stale lon-gather.
    set_halo_backend("local")
    set_polar_filter_lon_gather(None, None)
    return gathered


def _max_field_diff(a, b):
    return {f: float(np.max(np.abs(np.asarray(a[f]) - np.asarray(b[f]))))
            for f in a}


def test_2d_polar_filter_lon_split_matches_full_lon():
    """The polar filter under a longitude split (proc_lon>1, the new
    lon-gather path) must not break decomposition invariance beyond the
    floor the DYNAMICS already carry.

    A 4x1 (proc_lon=1, filter local on full lon) vs 2x2 (proc_lon=2, lon-gather
    filter) comparison of the SAME global state carries a small floating-point
    reassociation floor from the different halo/reduction patterns — present
    even with the filter OFF.  This test measures that floor (filter off) and
    asserts the filter-ON diff does not exceed it by more than 5x.  Before the
    u-face closure fix the filter-ON p_s diff was ~9e-2 (5 orders above the
    ~5e-7 floor), so this is non-vacuous."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n != 4:
        pytest.skip("needs exactly 4 ranks (4x1 and 2x2 both span the world)")

    grid = create_latlon_grid(8)                       # n_lat=8, n_lon=16
    sigma = create_sigma_coordinate(n_levels=4)

    def _cfg(flt):
        return CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, use_polar_filter=flt,
            polar_filter_cutoff_deg=30.0,              # filter more rows (small grid)
            use_ppm_transport=True, time_integrator="ssp_rk3")

    gstate = _make_global_state(grid, sigma.n_levels)
    dt = 100.0

    # Floor: 4x1 vs 2x2 with the filter OFF (pure dynamics reassociation).
    off_floor = _max_field_diff(
        _run_layout(gstate, grid, sigma, _cfg(False), 4, 1, dt),
        _run_layout(gstate, grid, sigma, _cfg(False), 2, 2, dt))
    # Filter ON: 4x1 (full-lon local) vs 2x2 (lon-gather path).
    on_diff = _max_field_diff(
        _run_layout(gstate, grid, sigma, _cfg(True), 4, 1, dt),
        _run_layout(gstate, grid, sigma, _cfg(True), 2, 2, dt))

    if rank == 0:
        for fld in ("u", "v", "T", "p_s"):
            print(f"POLAR_DIFF {fld}: off_floor={off_floor[fld]:.3e} "
                  f"on={on_diff[fld]:.3e}", flush=True)
        for fld in ("u", "v", "T", "p_s"):
            tol = max(5.0 * off_floor[fld], 1e-9)
            assert on_diff[fld] <= tol, (
                f"polar-filter lon-split ``{fld}`` diff {on_diff[fld]:.3e} "
                f"exceeds 5x the filter-off decomposition floor "
                f"{off_floor[fld]:.3e} — the lon-gather filter or the u-face "
                f"closure is decomposition-dependent.")
        print("LATLON2D_POLAR_OK lon-split filter within the dynamics floor",
              flush=True)


def test_build_order_does_not_leak_lon_gather_context():
    """A 2-D (proc_lon=2) step built FIRST then a proc_lon=1 step built AFTER,
    with the 2-D step invoked LAST, must each filter with its OWN context — the
    lon-gather context is set per-call inside step_fn, not once at build (codex
    C7 P1: ``_step_cgrid`` is jitted and reads the process-global at first-call
    trace time).  With 2 ranks: pc=2 (lon split, gather path) vs pc=1 (full lon,
    local).  Correctness is that BOTH complete without a stale-context shape
    mismatch / hang and the pc=2 result is finite."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n != 2:
        pytest.skip("build-order probe runs under exactly 2 ranks")
    grid = create_latlon_grid(8)
    sigma = create_sigma_coordinate(n_levels=4)
    config = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=True,
        polar_filter_cutoff_deg=30.0, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    gstate = _make_global_state(grid, sigma.n_levels)

    # Build BOTH steps first (2-D then 1-D band), invoke in REVERSE order.
    L2 = make_latlon_2d_layout(rank, 1, 2, grid.n_lat, grid.n_lon)   # lon split
    g2 = slice_latlon_grid_to_block_2d(grid, L2)
    m2 = CGridLatLonPrimitiveEquationModel(g2, sigma, config)
    step2 = make_latlon_2d_mpi_step(m2, L2)
    s2 = scatter_state_latlon_2d(gstate, L2)

    set_halo_backend("local")
    L1 = make_latlon_2d_layout(rank, 2, 1, grid.n_lat, grid.n_lon)   # full lon
    g1 = slice_latlon_grid_to_block_2d(grid, L1)
    m1 = CGridLatLonPrimitiveEquationModel(g1, sigma, config)
    step1 = make_latlon_2d_mpi_step(m1, L1)
    s1 = scatter_state_latlon_2d(gstate, L1)

    # Invoke the 1-D band step (must see None) then the 2-D step (must see its
    # gather) — the interleaving that a build-time global would break.
    set_halo_backend("mpi", L1)
    o1 = step1(s1, 100.0)
    set_halo_backend("mpi", L2)
    o2 = step2(s2, 100.0)

    ok = bool(jnp.all(jnp.isfinite(o1.T)) and jnp.all(jnp.isfinite(o2.T)))
    set_halo_backend("local")
    set_polar_filter_lon_gather(None, None)
    all_ok = comm.allreduce(ok, op=MPI.LAND)
    assert all_ok, "build-order interleaving produced non-finite state"


def test_distributed_polar_filter_is_ad_safe():
    """jax.grad through the lon-split polar filter is finite + non-zero — the
    lon gather/scatter is the custom-VJP ``lon_gather_full`` (forward
    allgather, backward allreduce), so the differentiated tendency stays
    AD-safe under a longitude split."""
    import jax
    from legoesm.grids.polar_filter import (
        compute_polar_filter_mask,
        fourier_filter,
    )
    from legoesm.parallel.latlon_mpi import (
        lon_gather_full,
        lon_scatter_full,
        make_lon_row_comm,
    )
    from legoesm.parallel.reductions import global_sum_mpi

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n != 2:
        pytest.skip("AD probe runs under exactly 2 ranks (proc_lon=2)")
    grid = create_latlon_grid(8)
    layout = make_latlon_2d_layout(rank, 1, 2, grid.n_lat, grid.n_lon)
    bg = slice_latlon_grid_to_block_2d(grid, layout)
    rc = make_lon_row_comm(layout)
    mask = compute_polar_filter_mask(bg, cutoff_lat_deg=30.0, n_lon_override=16)
    set_polar_filter_lon_gather(
        lambda f: lon_gather_full(f, layout, rc),
        lambda f: lon_scatter_full(f, layout))
    try:
        x = jnp.asarray(np.random.default_rng(rank).standard_normal((8, 8)))

        def loss(f):
            return jnp.sum(fourier_filter(f, bg, mask) ** 2)

        g = jax.grad(loss)(x)
        gnorm = float(global_sum_mpi(jnp.sum(g ** 2)))
        assert bool(jnp.all(jnp.isfinite(g))), "non-finite filter gradient"
        assert gnorm > 0.0, "zero gradient — filter VJP is inert"
    finally:
        set_polar_filter_lon_gather(None, None)
