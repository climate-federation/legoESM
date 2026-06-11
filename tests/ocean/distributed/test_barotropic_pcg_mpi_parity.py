"""np=2 gathered-field parity for the distributed implicit-CN barotropic
solver (the ocean weak-scaling fix).

Mirrors ``test_ocean_mpi_tvd_parity.py`` but exercises
``barotropic_solver="implicit_cn"``, which under MPI dispatches to the
hand-rolled distributed fixed-iteration PCG (static ``fori_loop`` + two
batched ``allreduce``/iter), UNROLLED and differentiated straight
through.  This is the decisive regression: a fixed-schedule PCG that
DOESN'T deadlock AND whose gathered ``eta``/``u``/``v`` reproduce the
single-rank decomposition to ``<= 1e-10`` (f64).

Acceptance covered:
  * (b) np=2 parity: gathered state after the implicit barotropic solve
        == serial reference, ``<= 1e-10`` (f64), NO deadlock under JIT.
  * (c) AD structural: ``jax.grad`` of a scalar of the band step runs to
        completion under MPI and is finite.  The backward pass
        differentiates straight through the fixed-length scan, using the
        halo ``_sendrecv_vjp`` BACKWARD rule and the ``allreduce(SUM)``
        VJP — both fixed-schedule collectives, so the backward pass
        cannot deadlock either.  (``custom_linear_solve`` is NOT used: it
        would transpose the halo ``custom_vjp``, which has no transpose
        rule and crashes under MPI.)

Run (np=1 degenerate single-band; np=2 puts the cut at the equator):

    mpirun -np 1 python -m pytest \\
        tests/ocean/distributed/test_barotropic_pcg_mpi_parity.py -v
    mpirun -np 2 python -m pytest \\
        tests/ocean/distributed/test_barotropic_pcg_mpi_parity.py -v

Deadlock discipline (the pattern that bit P1): the serial reference is
computed on EVERY rank BEFORE the MPI halo backend is armed.  A
rank-0-only serial step traced after arming embeds collectives no other
rank matches.
"""

from __future__ import annotations

import jax

# f64 BEFORE any array is built: the 1e-10 parity tolerance is
# meaningless in float32.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

# Small symmetric parity grid (np=2 cut lands on the equator), land caps
# poleward of the default threshold so masks are exercised.
N_LAT, N_LON, N_LEV = 16, 32, 4
DT = 1800.0
N_STEPS = 2
RTOL = 1e-10
ATOL = 1e-10
PARITY_FIELDS = ("T", "S", "eta", "u", "v")

# The implicit solver's A_op halo is width-1, but the surrounding ocean
# step uses the width-2 TVD cell pad; keep the same >=2 rows/rank guard.
_MIN_ROWS_PER_RANK = 2
_n_procs = MPI.COMM_WORLD.Get_size()
if _n_procs > 1 and (N_LAT // _n_procs) < _MIN_ROWS_PER_RANK:
    pytest.skip(
        f"MPI size {_n_procs} gives {N_LAT // _n_procs} lat rows/rank on "
        f"n_lat={N_LAT} (need >={_MIN_ROWS_PER_RANK}).",
        allow_module_level=True,
    )

# If the installed jax/mpi4jax stack is outside the tested range the
# allreduce path can drift ~1e-9 vs serial (incompatible custom-call
# ABI) — xfail the tight pin but still REQUIRE a pass on a tested stack.
from legoesm.parallel.reductions import mpi_stack_outside_tested_range

from legoesm.grids.halo import set_halo_backend
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _fp64_policy():
    """Run the FULL model step in float64 — the 1e-10 parity tolerance is
    meaningless if ``model.step`` casts intermediates to the default fp32
    storage policy (the unrolled PCG and stock CG would then only agree at
    the ~1e-7 fp32 round-off floor, not 1e-10)."""
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


@pytest.fixture(autouse=True)
def _reset_halo_backend():
    yield
    set_halo_backend("local")


def _config():
    # implicit_cn => distributed fixed-iteration PCG under MPI.  Use a
    # large fixed-iter count so the solve is fully converged and the
    # parity to serial (also fully-converged stock CG) is tight.
    return LatLonCGridOceanConfig(
        tracer_advection="tvd",
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_fixed_iters=150,
        barotropic_implicit_pcg_residual_tol=1.0e-10,
    )


def _perturbed_global_state(grid, z_coord):
    """Deterministic, mask-aware perturbation — built identically on every
    rank (no comm) BEFORE the MPI backend is armed."""
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    # Force f64 on the floating fields so the 1e-10 parity tolerance is
    # meaningful (the storage policy may default to f32 even under x64).
    state = jax.tree.map(
        lambda x: x.astype(jnp.float64)
        if getattr(x, "dtype", None) is not None
        and jnp.issubdtype(x.dtype, jnp.floating) else x,
        state,
    )
    mask = state.land_mask.data
    eta_pert = state.eta.data + 0.05 * mask * (
        jnp.sin(3.0 * grid.lon2d) * jnp.cos(2.0 * grid.lat2d)
    )
    T_pert = state.T.data + 0.5 * mask[..., jnp.newaxis] * (
        jnp.cos(grid.lon2d) * jnp.sin(2.0 * grid.lat2d)
    )[..., jnp.newaxis]
    v_data = state.v.data
    n_faces, n_lon = v_data.shape[0], v_data.shape[1]
    i_face = jnp.arange(n_faces, dtype=v_data.dtype)[:, None, None]
    j_cell = jnp.arange(n_lon, dtype=v_data.dtype)[None, :, None]
    v_pert = v_data + 0.05 * state.v_mask.data[..., jnp.newaxis] * (
        jnp.sin(jnp.pi * i_face / (n_faces - 1))
        * jnp.cos(2.0 * jnp.pi * j_cell / n_lon)
    )
    return state._replace(
        eta=state.eta.replace(data=eta_pert),
        T=state.T.replace(data=T_pert),
        v=state.v.replace(data=v_pert),
    )


class TestImplicitBarotropicMPIParity:

    def test_step_parity_gathered_vs_serial(self):
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()

        grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
        z_coord = create_ocean_z_star(n_levels=N_LEV)
        config = _config()
        state_global = _perturbed_global_state(grid, z_coord)

        # --- Serial reference: EVERY rank, local backend, BEFORE arming
        # the MPI halo backend (single-rank uses stock CG). ---
        serial_model = LatLonCGridOceanModel(grid, z_coord, config)
        ref = state_global
        for _ in range(N_STEPS):
            ref = serial_model.step(ref, DT)
        ref_np = {
            name: np.asarray(getattr(ref, name).data)
            for name in PARITY_FIELDS
        }

        # --- Arm band layout + MPI halo backend, build model ON the band
        # geometry (multi-rank => distributed fixed-iteration PCG). ---
        from legoesm.parallel.distributed import initialize_distributed_latlon
        from legoesm.parallel.latlon_mpi import (
            gather_state_latlon_cgrid_ocean,
            scatter_state_latlon_cgrid_ocean,
            slice_cgrid_geometry_to_band,
            slice_zcoord_to_band,
        )

        layout = initialize_distributed_latlon(
            global_n_lat=N_LAT, global_n_lon=N_LON,
        )
        band_geom = slice_cgrid_geometry_to_band(ensure_geometry(grid), layout)
        z_band = slice_zcoord_to_band(z_coord, layout)
        band_model = LatLonCGridOceanModel(band_geom, z_band, config)

        local = scatter_state_latlon_cgrid_ocean(state_global, layout)
        for _ in range(N_STEPS):
            local = band_model.step(local, DT)
        jax.block_until_ready(jax.tree.leaves(local))

        gathered = gather_state_latlon_cgrid_ocean(local, layout)

        if rank != 0:
            assert gathered is None
            return

        assert gathered is not None
        if mpi_stack_outside_tested_range() and n_ranks > 1:
            pytest.xfail(
                "jax/mpi4jax outside tested MPI range: allreduce path "
                "may drift ~1e-9 vs serial (incompatible custom-call ABI)."
            )
        for name in PARITY_FIELDS:
            got = np.asarray(getattr(gathered, name).data)
            np.testing.assert_allclose(
                got, ref_np[name], rtol=RTOL, atol=ATOL,
                err_msg=(
                    f"np={n_ranks} gathered '{name}' diverged from the "
                    f"serial reference after {N_STEPS} implicit-CN step(s): "
                    f"distributed PCG parity failure."
                ),
            )

    def test_grad_through_band_step_no_deadlock(self):
        """AD structural check: ``jax.grad`` of a scalar of one band step
        runs to completion under MPI and is finite.  The backward pass
        differentiates straight through the fixed-length PCG scan, using
        the halo ``_sendrecv_vjp`` BACKWARD rule and the ``allreduce(SUM)``
        VJP — both fixed-schedule collectives — so it is deadlock-free on
        every rank (no transpose of the halo custom_vjp is taken)."""
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()

        grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
        z_coord = create_ocean_z_star(n_levels=N_LEV)
        config = _config()
        state_global = _perturbed_global_state(grid, z_coord)

        from legoesm.parallel.distributed import initialize_distributed_latlon
        from legoesm.parallel.latlon_mpi import (
            scatter_state_latlon_cgrid_ocean,
            slice_cgrid_geometry_to_band,
            slice_zcoord_to_band,
        )
        from legoesm.parallel.reductions import global_sum_mpi

        layout = initialize_distributed_latlon(
            global_n_lat=N_LAT, global_n_lon=N_LON,
        )
        band_geom = slice_cgrid_geometry_to_band(ensure_geometry(grid), layout)
        z_band = slice_zcoord_to_band(z_coord, layout)
        band_model = LatLonCGridOceanModel(band_geom, z_band, config)
        local = scatter_state_latlon_cgrid_ocean(state_global, layout)

        def loss(eta0):
            st = local._replace(eta=local.eta.replace(data=eta0))
            st = band_model.step(st, DT)
            # Global (AD-safe SUM) scalar so the gradient couples ranks —
            # forces the backward allreduce/sendrecv schedule to run on
            # every rank in lockstep (deadlock tripwire).
            local_sum = jnp.sum(st.eta.data * st.land_mask.data)
            if n_ranks > 1:
                return global_sum_mpi(local_sum)
            return local_sum

        g = jax.grad(loss)(local.eta.data)
        jax.block_until_ready(g)
        finite = bool(jnp.all(jnp.isfinite(g)))
        nonzero = float(jnp.linalg.norm(g)) > 0.0
        # Reduce the boolean verdict so the test fails on ALL ranks
        # together (avoids a rank passing while another fails silently).
        if n_ranks > 1:
            ok = float(jnp.asarray(1.0 if (finite and nonzero) else 0.0))
            ok = float(global_sum_mpi(jnp.asarray(ok)))
            all_ok = ok >= float(n_ranks) - 0.5
        else:
            all_ok = finite and nonzero
        if rank == 0:
            assert all_ok, (
                "grad through the distributed implicit-CN band step was "
                "non-finite or zero on some rank (or deadlocked)."
            )
