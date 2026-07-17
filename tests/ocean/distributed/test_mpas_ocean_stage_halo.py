"""Stage-correct distributed MPAS-ocean step — the in-step halo-refresh gates.

The stage audit (docs/performance/scaling/mpas_ocean_distributed_stage_audit.md)
proved the multi-rank step consumes more stencil hops than ``halo_depth=2``
between per-step refreshes, silently corrupting owned cells at partition
boundaries.  ``MPASOceanModel.step(halo_refresh=make_mpas_ocean_halo_refresh(
layout))`` re-arms the halo at the audited frontiers (R1-R3 in the step,
T1/T2 in the tendencies, B0-B2 in the explicit substeps, I0-I1 in the
implicit predictor).

Gates here:

*  SERIAL IDENTITY (any np, runs in plain CI): an identity refresh object
   threaded through every site is BIT-identical to ``halo_refresh=None`` —
   the plumbing is value-neutral where the exchange is the identity
   (np=1 ``batched_halo_exchange`` IS the identity), for both barotropic
   solvers with every refresh-bearing branch enabled (B_h del4, C_smag,
   C_leith, K_bih, div-damp, baro-visc, baro-diffusion, TVD).
*  AD: ``jax.grad`` through the identity-refreshed step equals the
   ``None``-path gradient (the refresh sites must not break reverse mode).
*  np=2 PARITY + NON-VACUITY (mpirun): the in-step-refreshed local run
   matches the serial global run on owned cells to near round-off, and the
   refresh MECHANISM is proven load-bearing — the same run WITHOUT it is
   orders of magnitude worse (a synthetic-violation tripwire: if a refactor
   made the refreshes no-ops, the ratio assertion goes red).

Run the MPI part:  mpirun -np 2 python -m pytest \
    tests/ocean/distributed/test_mpas_ocean_stage_halo.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.parallel.voronoi_mpi import MPASOceanHaloRefresh

try:
    from mpi4py import MPI

    _SIZE = MPI.COMM_WORLD.Get_size()
except Exception:                                    # no MPI stack
    _SIZE = 1

DT = 300.0
NLEV = 4

def _layout_for(mesh):
    """Arm (or RE-arm) the Voronoi MPI layout for this test.

    The distributed conftest's autouse ``_isolate_distributed_state``
    fixture calls ``reset_distributed_topology()`` after EVERY test, which
    clears the active-voronoi-layout singleton the implicit-CN entry
    refusal keys on — so a cached layout object from an earlier
    parametrize case is NOT enough (that was the round-3 failure mode:
    ``active_layout=False`` on both ranks).  Re-initialize whenever the
    singleton is cleared; ``initialize_voronoi_mpi`` is deterministic for
    the same mesh and collective across ranks."""
    from legoesm.parallel.voronoi_mpi import (
        get_active_voronoi_layout,
        initialize_voronoi_mpi,
    )

    layout = get_active_voronoi_layout()
    if layout is None:
        _r, _n, layout = initialize_voronoi_mpi(mesh)
    return layout


def _identity_refresh() -> MPASOceanHaloRefresh:
    """Refresh object whose exchanges are identities (np=1 semantics)."""
    return MPASOceanHaloRefresh(
        edges=lambda *fs: tuple(fs),
        cells=lambda *fs: tuple(fs),
        both=lambda es, cs: (tuple(es), tuple(cs)),
        vertices=lambda *fs: tuple(fs),
    )


def _perturbed_state(mesh, z, seed=7):
    state = rest_state_mpas_ocean(mesh, z)
    rng = np.random.default_rng(seed)
    return state._replace(
        u=state.u.replace(data=jnp.asarray(
            1.0e-2 * rng.standard_normal(state.u.data.shape))),
        T=state.T.replace(data=state.T.data + jnp.asarray(
            0.5 * rng.standard_normal(state.T.data.shape))),
        S=state.S.replace(data=state.S.data + jnp.asarray(
            0.05 * rng.standard_normal(state.S.data.shape))),
        eta=state.eta.replace(data=jnp.asarray(
            1.0e-2 * rng.standard_normal(state.eta.data.shape))),
    )


def _all_frontier_config(base, barotropic_solver):
    """Enable every refresh-bearing branch so the identity gate executes
    ALL insertion sites (T1 fused+del4, Smag, Leith, T2 K_bih, B0-B2 with
    div-damp/visc/diffusion, TVD advection, semi-implicit Coriolis)."""
    return base._replace(
        barotropic_solver=barotropic_solver,
        A_h=1.0e3, B_h=1.0e9, C_smag=0.05, C_leith=1.0, K_bih=1.0e9,
        K_h=100.0,
        K_zeta_bih=1.0e9,          # T3 vertex-channel site (codex r1 #2)
        barotropic_div_damp=1.0e-3,
        barotropic_u_viscosity=10.0,
        semi_implicit_coriolis=True,
        tracer_advection="tvd",
    )


def _fields(state):
    return {f: np.asarray(getattr(state, f).data)
            for f in ("u", "T", "S", "eta", "w")}


@pytest.mark.skipif(_SIZE != 1, reason="serial gate; run without mpirun "
                    "(the np>1 world trips world-size-keyed refusals)")
@pytest.mark.parametrize("solver", ["explicit_substep", "implicit_cn"])
def test_identity_refresh_bit_identical_serial(solver):
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

    mesh = create_voronoi_mesh(3)
    z = create_ocean_z_star(n_levels=NLEV)
    cfg = _all_frontier_config(MPASOceanConfig(), solver)
    model = MPASOceanModel(mesh, z, cfg)
    state = _perturbed_state(mesh, z)

    s_none = state
    s_ident = state
    ident = _identity_refresh()
    for _ in range(2):
        s_none = model._step_impl(s_none, DT)
        s_ident = model._step_impl(s_ident, DT, halo_refresh=ident)
    for name in ("u", "T", "S", "eta", "w"):
        np.testing.assert_array_equal(
            np.asarray(getattr(s_ident, name).data),
            np.asarray(getattr(s_none, name).data),
            err_msg=(f"identity halo_refresh changed {name} "
                     f"({solver}) — a refresh site is not value-neutral"))


@pytest.mark.skipif(_SIZE != 1, reason="serial gate; run without mpirun")
def test_identity_refresh_grad_matches_serial():
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

    mesh = create_voronoi_mesh(2)
    z = create_ocean_z_star(n_levels=3)
    cfg = _all_frontier_config(MPASOceanConfig(), "explicit_substep")
    model = MPASOceanModel(mesh, z, cfg)
    state = _perturbed_state(mesh, z, seed=3)
    ident = _identity_refresh()

    def loss(u0, refresh):
        s = state._replace(u=state.u.replace(data=u0))
        out = model._step_impl(s, DT, halo_refresh=refresh)
        return jnp.sum(out.T.data ** 2) + jnp.sum(out.u.data ** 2)

    g_none = jax.grad(lambda u: loss(u, None))(state.u.data)
    g_ident = jax.grad(lambda u: loss(u, ident))(state.u.data)
    assert np.all(np.isfinite(np.asarray(g_ident)))
    np.testing.assert_allclose(np.asarray(g_ident), np.asarray(g_none),
                               rtol=0.0, atol=0.0)


N_STEPS_NP2 = 3


def _write_reference(solver: str, path: str) -> None:
    """Compute the SERIAL global-mesh reference and save it.

    Runs in a SUBPROCESS OUTSIDE the MPI world (mpi4py world = 1), so the
    mesh-matched implicit-CN refusal — which correctly rejects a
    global-mesh solve at world > 1 (it would run rank-local) — does not
    fire, and the guard stays exactly as hardened."""
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

    mesh = create_voronoi_mesh(3)
    z = create_ocean_z_star(n_levels=NLEV)
    cfg = _all_frontier_config(MPASOceanConfig(), solver)
    m_ser = MPASOceanModel(mesh, z, cfg)
    s = _perturbed_state(mesh, z)
    for _ in range(N_STEPS_NP2):
        s = m_ser._step_impl(s, DT)
    np.savez(path, **_fields(s))


if __name__ == "__main__":                    # subprocess entry (rank 0)
    import sys as _sys

    _write_reference(_sys.argv[1], _sys.argv[2])
    raise SystemExit(0)


@pytest.mark.skipif(_SIZE < 2, reason="needs mpirun -np 2")
@pytest.mark.parametrize("solver", ["explicit_substep", "implicit_cn"])
def test_np2_stage_correct_parity_and_tripwire(solver, tmp_path_factory):
    """Owned-cell parity vs serial with the in-step refresh armed, plus the
    non-vacuity tripwire (entry-refresh-only must be measurably worse).

    The serial reference is computed by rank 0 in a subprocess with the
    MPI launcher environment scrubbed (OMPI_/PMIX_/PMI_) so it runs at
    world=1 under the SAME interpreter — no guard weakening, no
    cross-environment numerics skew."""
    import os
    import subprocess
    import sys

    from mpi4py import MPI as _MPI

    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.parallel.voronoi_mpi import (
        exchange_state_mpas_ocean,
        gather_state_mpas_ocean,
        make_mpas_ocean_halo_refresh,
        scatter_state_mpas_ocean,
    )

    comm = _MPI.COMM_WORLD
    n_steps = N_STEPS_NP2
    mesh = create_voronoi_mesh(3)
    z = create_ocean_z_star(n_levels=NLEV)
    cfg = _all_frontier_config(MPASOceanConfig(), solver)
    state0 = _perturbed_state(mesh, z)

    ref_dir = (tmp_path_factory.getbasetemp() if comm.Get_rank() == 0
               else None)
    ref_dir = comm.bcast(str(ref_dir) if ref_dir is not None else None,
                         root=0)
    ref_path = os.path.join(ref_dir, f"stage_halo_ref_{solver}.npz")
    # Failure discipline (codex r2 #1): a raise/timeout on rank 0 BEFORE
    # a bare Barrier would leave rank 1 blocked forever — deadlock, not a
    # test failure.  The success flag rides the (collective) bcast that
    # replaces the Barrier, so every rank fails loudly together.
    ok, err = True, ""
    if comm.Get_rank() == 0:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
        try:
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), solver,
                 ref_path],
                check=True, env=env, timeout=1200)
        except Exception as e:  # surfaced on every rank via the bcast
            ok, err = False, repr(e)
    ok, err = comm.bcast((ok, err), root=0)
    if not ok:
        pytest.fail(f"serial-reference subprocess failed on rank 0: {err}")
    with np.load(ref_path) as f:
        ref = {k: f[k] for k in f.files}

    layout = _layout_for(mesh)
    m_loc = MPASOceanModel(layout.local_mesh, z, cfg)
    refresh = make_mpas_ocean_halo_refresh(layout)

    def run_local(halo_refresh):
        s = scatter_state_mpas_ocean(state0, layout.partition)
        for _ in range(n_steps):
            s = m_loc._step_impl(s, DT, halo_refresh=halo_refresh)
            s = exchange_state_mpas_ocean(s, layout)
        return _fields(gather_state_mpas_ocean(s, layout.partition))

    got_fixed = run_local(refresh)
    got_stale = run_local(None)

    def max_err(got):
        return max(float(np.max(np.abs(got[f] - ref[f])))
                   for f in ("u", "T", "S", "eta"))

    err_fixed = max_err(got_fixed)
    err_stale = max_err(got_stale)
    # Near round-off: every exchange is a bit-copy; the only reassociation
    # is the owned-masked allreduce family.
    assert err_fixed < 5.0e-11, (
        f"in-step refreshed np2 step diverges from serial "
        f"({solver}): max|Δ|={err_fixed:.3e}")
    # Non-vacuity tripwire: the refreshes must be load-bearing.  If a
    # refactor silently disabled them, err_fixed would rise to err_stale
    # and/or this ratio would collapse.
    assert err_stale > 10.0 * max(err_fixed, 1.0e-14), (
        f"stale-halo run unexpectedly as accurate as refreshed "
        f"(err_stale={err_stale:.3e}, err_fixed={err_fixed:.3e}) — "
        f"tripwire: are the in-step refreshes actually wired?")
