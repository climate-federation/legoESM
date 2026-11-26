"""Lat-lon pole-row treatment under a 2-D (lat x lon) MPI decomposition.

Serial-equivalence of the pole-row operators on a 2x2 (lat x lon) split with
a rotated (cross-polar) planet and a random state whose zonal partial sums
differ per longitude block:

* ``absolute_vorticity_coriolis`` u-component -- the Stokes polar-cap
  vorticity completes its zonal sum across the longitude split
  (``global_sum_mpi`` slot reduction) and the pole-row v estimate;
* ``latlon_pole_cell_v`` -- pole-row v estimate (KE).

The v-component of the Coriolis term is NOT asserted: it already differs from
serial on base dc2bbccd93 (12/80 elements, 5.7e-5, identical with and
without this fix) -- a pre-existing 2-D MPI issue outside the pole rows.
One-row latitude tiles are not exercised: the base 2-D MPI curl already
rejects them ("Cannot apply 'edge' padding to empty axis").

Run (JAX_ENABLE_X64=1; each test needs its own rank count, the other skips):
``mpirun -np 4 python -m pytest tests/distributed/test_latlon_pole_mpi.py``  (2x2)
``mpirun -np 2 python -m pytest tests/distributed/test_latlon_pole_mpi.py``  (mixed width)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
)
from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
    absolute_vorticity_coriolis,
    latlon_pole_cell_v,
)
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid
from legoesm.parallel.latlon_mpi import (
    gather_field_latlon,
    gather_field_latlon_2d,
    make_latlon_band_layout,
    scatter_state_latlon,
    slice_latlon_grid_to_band,
    make_latlon_2d_layout,
    scatter_state_latlon_2d,
    slice_latlon_grid_to_block_2d,
)

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="fp64: run with JAX_ENABLE_X64=1")

N_LAT, N_LON, NLEV = 4, 8, 2


def _all_ranks_ok(comm, ok: bool) -> bool:
    """Rank-uniform verdict: every rank exits together (no rank left waiting
    in a later exchange after another rank's assertion fired)."""
    return bool(comm.allreduce(int(bool(ok)), op=MPI.MIN))


def test_pole_rows_match_serial_2x2():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 4:
        pytest.skip("needs exactly 4 ranks (proc_lat=2 x proc_lon=2)")
    grid = create_latlon_grid(N_LAT, N_LON, coriolis_alpha=np.pi / 4)
    rng = np.random.default_rng(7)
    u = rng.standard_normal((N_LAT, N_LON + 1, NLEV))
    u[:, N_LON] = u[:, 0]
    v = rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    v[0] = v[-1] = 0.0
    st = CGridLatLonHydrostaticState(
        u=jnp.asarray(u), v=jnp.asarray(v),
        T=jnp.full((N_LAT, N_LON, NLEV), 300.0),
        p_s=jnp.full((N_LAT, N_LON), 1e5),
        phis=jnp.zeros((N_LAT, N_LON)), tracers={})

    set_halo_backend("local")
    s_cu = np.asarray(absolute_vorticity_coriolis(st.u, st.v, grid)[0])
    s_vk = np.asarray(latlon_pole_cell_v(0.5 * (st.v[:-1] + st.v[1:]), st.v, grid))
    # serial VJP of one south-pole-row Coriolis output (lon block 0, col 1)
    s_gu = np.asarray(jax.grad(
        lambda a: absolute_vorticity_coriolis(a, st.v, grid)[0][0, 1, 1])(st.u))

    L = make_latlon_2d_layout(comm.Get_rank(), 2, 2, N_LAT, N_LON)
    g = slice_latlon_grid_to_block_2d(grid, L)
    flag_ok = g.global_poles is True
    loc = scatter_state_latlon_2d(st, L)
    seed = L.lat_start == 0 and L.lon_start == 0
    set_halo_backend("mpi", L)
    try:
        # jit, like the production step (eager mpi4jax sendrecvs can stall)
        cu = jax.jit(lambda a, b: absolute_vorticity_coriolis(a, b, g)[0])(loc.u, loc.v)
        vk = jax.jit(lambda b: latlon_pole_cell_v(0.5 * (b[:-1] + b[1:]), b, g))(loc.v)
        # reverse mode through the lon-ring reduction: only block (0,0) seeds
        gu_loc = jax.jit(jax.grad(lambda a: jnp.where(
            seed, absolute_vorticity_coriolis(a, loc.v, g)[0][0, 1, 1], 0.0)))(loc.u)
        gu = gather_field_latlon_2d(cu, L, is_u_face=True)
        gk = gather_field_latlon_2d(vk, L)
        ggu = gather_field_latlon_2d(gu_loc, L, is_u_face=True)
    finally:
        set_halo_backend("local")
    checks = {"global_poles": flag_ok}
    if comm.Get_rank() == 0:
        checks["vk"] = np.allclose(np.asarray(gk), s_vk, rtol=1e-12, atol=1e-12)
        checks["cu"] = np.allclose(np.asarray(gu)[:, :N_LON], s_cu[:, :N_LON],
                                   rtol=1e-12, atol=1e-12)
        # cotangent on the OTHER lon block (interior faces 5..7) of the pole row
        # reaches it only through the polar-cap zonal sum
        other = np.asarray(ggu)[0, 5:N_LON]
        checks["vjp_other_block"] = (np.allclose(other, s_gu[0, 5:N_LON],
                                                 rtol=1e-10, atol=1e-14)
                                     and np.abs(s_gu[0, 5:N_LON]).max() > 0)
    assert _all_ranks_ok(comm, all(checks.values())), (comm.Get_rank(), checks)


def test_pole_cell_v_mixed_width_bands():
    """Lat-band MPI with a 1-row and a 2-row band (n_lat=3, 2 ranks): the
    one-row pole band fetches its second real face; the wider band must join
    the same exchange (forward AND reverse, rank-matched) and discard it."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        pytest.skip("needs exactly 2 ranks")
    grid = create_latlon_grid(3, N_LON, coriolis_alpha=np.pi / 4)
    rng = np.random.default_rng(11)
    v = rng.standard_normal((4, N_LON, NLEV))
    v[0] = v[-1] = 0.0
    u = rng.standard_normal((3, N_LON + 1, NLEV))
    u[:, N_LON] = u[:, 0]
    st = CGridLatLonHydrostaticState(
        u=jnp.asarray(u), v=jnp.asarray(v), T=jnp.full((3, N_LON, NLEV), 300.0),
        p_s=jnp.full((3, N_LON), 1e5), phis=jnp.zeros((3, N_LON)), tracers={})
    set_halo_backend("local")
    s_vk = np.asarray(latlon_pole_cell_v(0.5 * (st.v[:-1] + st.v[1:]), st.v, grid))
    L = make_latlon_band_layout(comm.Get_rank(), 2, 3, N_LON)
    g = slice_latlon_grid_to_band(grid, L)
    loc = scatter_state_latlon(st, L)
    thin = L.n_lat_local == 1
    checks = {}

    set_halo_backend("mpi", L)
    try:
        cell_v = lambda b, gg: latlon_pole_cell_v(0.5 * (b[:-1] + b[1:]), b, gg)
        gk = gather_field_latlon(jax.jit(lambda b: cell_v(b, g))(loc.v), L)

        # Reverse mode: seed ONLY the one-row north pole band's pole-row output.
        def loss(b):
            return jnp.where(thin, cell_v(b, g)[-1, 3, 1], 0.0)
        gr = np.asarray(jax.jit(jax.grad(loss))(loc.v))

        # ... and w.r.t. the exchanged latitudes.
        def loss_lat(lv):
            return jnp.where(thin, cell_v(loc.v, g._replace(lat_v=lv))[-1, 3, 1], 0.0)
        gl_ = np.asarray(jax.jit(jax.grad(loss_lat))(g.lat_v))

        # NaN pole-wall samples on both bands must not leak into any stencil.
        vn = loc.v.at[-1 if thin else 0].set(jnp.nan)
        gkn = gather_field_latlon(jax.jit(lambda b: latlon_pole_cell_v(
            0.5 * (jnp.nan_to_num(b[:-1]) + jnp.nan_to_num(b[1:])), b, g))(vn), L)
    finally:
        set_halo_backend("local")

    if comm.Get_rank() == 0:
        plain = np.asarray(0.5 * (st.v[:-1] + st.v[1:]))
        checks["forward"] = np.allclose(np.asarray(gk), s_vk, rtol=1e-12, atol=1e-12)
        checks["nontrivial"] = np.abs(s_vk[-1] - plain[-1]).max() > 1e-3
        checks["nan_wall"] = np.allclose(np.asarray(gkn), s_vk, rtol=1e-12, atol=1e-12)
    checks["finite"] = bool(np.all(np.isfinite(gr)) and np.all(np.isfinite(gl_)))
    if not thin:
        # North thin band: v_n = v1 + (v1 - v2) w, w = (l1 - lc)/(l2 - l1),
        # v1/l1 = global face 2, v2/l2 = global face 1 (= this band's row 1).
        l1, l2, lc = (float(x) for x in (grid.lat_v[2], grid.lat_v[1], grid.lat[2]))
        v1, v2 = float(v[2, 3, 1]), float(v[1, 3, 1])
        w = (l1 - lc) / (l2 - l1)
        checks["dv"] = np.isclose(gr[1, 3, 1], -w, rtol=1e-6)
        checks["dlat"] = np.isclose(gl_[1], -(v1 - v2) * (l1 - lc) / (l2 - l1) ** 2,
                                    rtol=1e-5)
    ok = _all_ranks_ok(comm, all(checks.values()))
    assert ok, f"rank {comm.Get_rank()} checks: {checks}"


@pytest.mark.parametrize("n_lat", [4, 5])
def test_coriolis_pole_rows_bands_match_serial(n_lat):
    """Lat-band MPI (2 ranks; 2+2 and 2+3 rows): the pole-row
    energy-conserving vorticity term rewrites global u rows 0 / n_lat-1 and
    v-faces 1 / n_lat-1, selected by the physical-pole masks.  Each rank's
    LOCAL cor_u and cor_v, including its copy of the cut face, must equal the
    serial operator (interior cuts untouched, pole tiles rewritten).
    One-row tiles are out of scope: the base operator already raises or
    hangs on them."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        pytest.skip("needs exactly 2 ranks")
    grid = create_latlon_grid(n_lat, N_LON, coriolis_alpha=np.pi / 4)
    rng = np.random.default_rng(5)
    u = rng.standard_normal((n_lat, N_LON + 1, NLEV))
    u[:, N_LON] = u[:, 0]
    v = rng.standard_normal((n_lat + 1, N_LON, NLEV))
    v[0] = v[-1] = 0.0
    st = CGridLatLonHydrostaticState(
        u=jnp.asarray(u), v=jnp.asarray(v), T=jnp.full((n_lat, N_LON, NLEV), 300.0),
        p_s=jnp.full((n_lat, N_LON), 1e5), phis=jnp.zeros((n_lat, N_LON)), tracers={})
    set_halo_backend("local")
    s_cu, s_cv = (np.asarray(a) for a in absolute_vorticity_coriolis(st.u, st.v, grid))
    L = make_latlon_band_layout(comm.Get_rank(), 2, n_lat, N_LON)
    g = slice_latlon_grid_to_band(grid, L)
    loc = scatter_state_latlon(st, L)
    set_halo_backend("mpi", L)
    try:
        cu, cv = jax.jit(lambda a, b: absolute_vorticity_coriolis(a, b, g))(loc.u, loc.v)
    finally:
        set_halo_backend("local")
    j0, n = L.lat_start, L.n_lat_local
    checks = {
        "cu": np.allclose(np.asarray(cu), s_cu[j0:j0 + n], rtol=1e-12, atol=1e-12),
        "cv": np.allclose(np.asarray(cv), s_cv[j0:j0 + n + 1], rtol=1e-12, atol=1e-12),
    }
    assert _all_ranks_ok(comm, all(checks.values())), (comm.Get_rank(), checks)
