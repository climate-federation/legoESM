"""2-D pencil DYCORE step gate for the lat-lon C-grid (task #14 capstone).

The step-level validation of the full 2-D longitude-split path: with the
operator lon ops routed through ``pad_lon_cgrid`` and the
``make_latlon_2d_mpi_step`` ``proc_lon>1`` guard lifted for regular / wall-pole
grids, run a real dry C-grid step on a 2-D process grid and check:

1. **Mass conservation** (truth tier, no reference): global dry p_s mass before
   vs after one step drifts < 1e-12 (the ``fix_mass`` fixer's allreduce sees
   the global total area).  Catches ANY missed longitude op (a lon stencil that
   wrapped the rank-local block would break flux-form mass closure or NaN).
2. **Longitude-decomposition invariance**: a 2x2 split and a 1x4 split of the
   SAME global state — BOTH wall-pole at ``proc_lon>1`` — give the same
   gathered stepped state.  Independent of any serial reference, this proves
   the lon halo + the u-face shared-column convention are correct (a wrong lon
   exchange would make 2x2 and 1x4 disagree).  Faces gathered with
   ``is_u_face`` / ``is_v_face``.

Wall poles only (NOT the atmosphere 180-deg fold — that needs the lat-pencil
transpose).  ``use_polar_filter=False`` (the lon-FFT filter needs full
longitude per rank; the throughput benchmark + this gate run without it).

Run: ``mpirun -np 4 python -m pytest tests/distributed/test_latlon_2d_mpi_step.py``
(needs exactly 4 ranks: 2x2 and 1x4 both use the full world).  The dry C-grid
step JIT-compiles per block shape (~minutes-to-tens-of-minutes on Ginsburg, x2
for the two layouts) — run under a long-walltime sbatch, not inline.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.parallel.latlon_mpi import (
    gather_field_latlon_2d,
    make_latlon_2d_layout,
    make_latlon_2d_mpi_step,
    scatter_state_latlon_2d,
    slice_latlon_grid_to_block_2d,
)
from legoesm.parallel.reductions import global_sum_mpi


def _make_global_state(grid, nlev, seed=31337):
    """Perturbed rest state on the GLOBAL grid; identical on every rank (same
    seed, no comm) so each scatters its own 2-D block.  u carries the n_lon+1
    periodic-closure face (u[:, n_lon] == u[:, 0])."""
    rng = np.random.default_rng(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    eps = 1.0e-3
    u = eps * rng.standard_normal((n_lat, n_lon + 1, nlev))
    u[:, n_lon, :] = u[:, 0, :]                      # periodic closure invariant
    v = eps * rng.standard_normal((n_lat + 1, n_lon, nlev))
    T = 300.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))
    p_s = 1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))
    phis = np.zeros((n_lat, n_lon))
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(u), v=jnp.asarray(v), T=jnp.asarray(T),
        p_s=jnp.asarray(p_s), phis=jnp.asarray(phis), tracers={},
    )


def _local_model(global_grid, sigma, config, layout):
    # slice_latlon_grid_to_block_2d allreduces the block area -> GLOBAL sphere
    # total_area (the mass-fixer denominator), exactly like the band path.
    g = slice_latlon_grid_to_block_2d(global_grid, layout)
    return CGridLatLonPrimitiveEquationModel(g, sigma, config)


def _global_mass(state, model):
    return float(global_sum_mpi(jnp.sum(
        state.p_s.astype(jnp.float64) * model.grid.area.astype(jnp.float64))))


def _run_layout(global_state, grid, sigma, config, pr, pc, dt):
    rank = MPI.COMM_WORLD.Get_rank()
    L = make_latlon_2d_layout(rank, pr, pc, grid.n_lat, grid.n_lon)
    model = _local_model(grid, sigma, config, L)
    local = scatter_state_latlon_2d(global_state, L)
    m_before = _global_mass(local, model)
    step = make_latlon_2d_mpi_step(model, L)
    out = step(local, dt)
    m_after = _global_mass(out, model)
    gathered = {
        "u": gather_field_latlon_2d(out.u, L, is_u_face=True),
        "v": gather_field_latlon_2d(out.v, L, is_v_face=True),
        "T": gather_field_latlon_2d(out.T, L),
        "p_s": gather_field_latlon_2d(out.p_s, L),
    }
    set_halo_backend("local")        # reset before the next layout / leg
    return m_before, m_after, gathered


def test_2d_step_mass_conserved_and_lon_decomposition_invariant():
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n != 4:
        pytest.skip("needs exactly 4 ranks (2x2 and 1x4 both span the world)")

    grid = create_latlon_grid(8)                 # n_lat=8, n_lon=16
    sigma = create_sigma_coordinate(n_levels=4)
    config = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,
        use_polar_filter=False,                  # lon-FFT filter needs full lon
        use_ppm_transport=True,
        time_integrator="ssp_rk3",
    )
    gstate = _make_global_state(grid, sigma.n_levels)
    dt = 100.0

    # Both legs are 2-D MPI (no serial leg), every rank in lockstep — the
    # mass allreduce + halo sendrecvs are SPMD-symmetric, no deadlock.
    mb22, ma22, g22 = _run_layout(gstate, grid, sigma, config, 2, 2, dt)
    mb14, ma14, g14 = _run_layout(gstate, grid, sigma, config, 1, 4, dt)

    # (1) Mass conservation — each layout, global drift < 1e-12.
    for tag, mb, ma in (("2x2", mb22, ma22), ("1x4", mb14, ma14)):
        drift = abs(ma - mb) / abs(mb)
        assert drift < 1e-12, (
            f"{tag} 2-D step dry mass drift {drift:.3e} (before={mb:.6e} "
            f"after={ma:.6e}) — the fixer's global-area allreduce or a lon "
            f"flux is wrong.")

    # (2) Longitude-decomposition invariance (both wall-pole) on rank 0.
    if rank == 0:
        for fld in ("u", "v", "T", "p_s"):
            np.testing.assert_allclose(
                np.asarray(g22[fld]), np.asarray(g14[fld]),
                rtol=1e-8, atol=1e-9,
                err_msg=(
                    f"2x2 vs 1x4 stepped ``{fld}`` differ — the longitude "
                    f"halo / u-face shared-column handling is decomposition-"
                    f"dependent (a lon exchange bug)."))
        print("LATLON2D_STEP_OK mass-conserved + lon-decomp-invariant (2x2==1x4)",
              flush=True)
