"""Production RCEMIP1 CRM dycore on the multi-rank ``step_halo`` path.

The dycore configuration is the one the scaling bench runs: the production
deck (``run_rcemip1_crm.sh``: centred momentum advection, van Leer scalars,
Smagorinsky with vertical SGS, horizontal acoustics on the substeps,
semi-implicit) built by the RCEMIP driver itself, with Morrison's 11 tracer
slots. Multi-rank gathered states, per-step budget changes and the gradient
must match the single-rank run.

    mpiexec -n 2|4|8 python -m pytest tests/distributed/test_plane_crm_production_halo_mpi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

MPI = pytest.importorskip("mpi4py.MPI")
pytest.importorskip("mpi4jax")

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "bench"))
sys.path.insert(0, str(_REPO / "scripts" / "run"))

import bench_plane_crm_dd_scaling as bench  # noqa: E402
import run_rcemip_plane as rcp  # noqa: E402

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    PlaneCompressibleEulerModel, horizontal_advection_halo_width,
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402
from legoesm.parallel.plane_mpi import (  # noqa: E402
    gather_plane_field, make_plane_pencil_layout,
)
from legoesm.parallel.reductions import global_sum_mpi  # noqa: E402

NY, NX, NLEV, N_STEPS = 16, 16, 8, 3
N_TRACERS = rcp.microphysics_n_tracers("morrison")
FIELDS = ("u", "v", "w", "theta_prime", "rho_prime", "tracers")


def _production_setup(fix_mass):
    rargs = rcp.parse_args([*bench.RCEMIP1_DECK_ARGV, "--nx", str(NX),
                            "--ny", str(NY), "--nlev", str(NLEV)])
    hc, _, _ = rcp.build_rcemip_height_coord(rargs, rcp.P_SFC_RCEMIP,
                                             jnp.float64)
    # Exact serial parity of the acoustic moist buoyancy needs the global
    # horizontal mean (the production default uses the rank-local mean).
    cfg = rcp.build_dycore_config(rargs)._replace(
        acoustic_moist_global_mean=True, fix_mass=fix_mass)
    assert cfg.horizontal_momentum_advection_scheme == "centered"
    assert cfg.sgs_vertical_diffusion and cfg.substep_horizontal_acoustic
    return rargs, hc, cfg


def _global_state(grid, hc):
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    k = jax.random.split(jax.random.PRNGKey(7), 6)
    tr = jnp.abs(1.0e-3 * jax.random.normal(k[5], (NY, NX, NLEV, N_TRACERS)))
    return rest._replace(
        u=rest.u.replace(data=0.5 * jax.random.normal(k[0], rest.u.data.shape)),
        v=rest.v.replace(data=0.5 * jax.random.normal(k[1], rest.v.data.shape)),
        w=rest.w.replace(data=(0.05 * jax.random.normal(k[2], rest.w.data.shape))
                         .at[..., 0].set(0.0).at[..., -1].set(0.0)),
        theta_prime=rest.theta_prime.replace(
            data=0.5 * jax.random.normal(k[3], rest.theta_prime.data.shape)),
        rho_prime=rest.rho_prime.replace(
            data=1.0e-3 * jax.random.normal(k[4], rest.rho_prime.data.shape)),
        tracers=rest.tracers.replace(data=tr),
    )


def _slice(state, lay):
    def s(a):
        return a[lay.iy_start:lay.iy_end, lay.ix_start:lay.ix_end]
    return state._replace(**{f: getattr(state, f).replace(
        data=s(getattr(state, f).data)) for f in (*FIELDS, "phis")})


def _budgets(state, hc):
    """Domain integrals per unit column area (fp64): dry mass, tracer mass,
    x/y momentum, kinetic energy, θ' content."""
    dz = np.asarray(hc.dz)
    rho = np.asarray(hc.rho_ref) + np.asarray(state.rho_prime.data)
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    q = np.asarray(state.tracers.data)
    terms = {
        "mass": rho * dz,
        "tracer_mass": (rho[..., None] * q * dz[:, None]).sum(-1),
        "momentum_x": rho * u * dz,
        "momentum_y": rho * v * dz,
        "kinetic_energy": 0.5 * rho * (u ** 2 + v ** 2) * dz,
        "theta_content": rho * np.asarray(state.theta_prime.data) * dz,
    }
    return {k: (t.sum(), np.abs(t).sum()) for k, t in terms.items()}


def _layout_or_skip(nry, nrx, halo):
    comm = MPI.COMM_WORLD
    if comm.Get_size() != nry * nrx:
        pytest.skip(f"needs {nry * nrx} ranks")
    return make_plane_pencil_layout(
        rank=comm.Get_rank(), n_ranks=comm.Get_size(), n_ranks_y=nry,
        n_ranks_x=nrx, ny_global=NY, nx_global=NX, halo=halo)


DECOMPS = [(2, 1), (1, 2), (2, 2), (4, 2), (2, 4)]


def _run(model, state, dt, lay, owned):
    out = [state]
    for _ in range(N_STEPS):
        out.append(model.step_halo(out[-1], dt=dt, layout=lay,
                                   owned_mask=owned))
    return out


@pytest.mark.parametrize("fix_mass", [False, True])
@pytest.mark.parametrize("nry,nrx", DECOMPS)
def test_production_step_halo_matches_single_rank(nry, nrx, fix_mass):
    rargs, hc, cfg = _production_setup(fix_mass)
    halo = horizontal_advection_halo_width(cfg)
    lay = _layout_or_skip(nry, nrx, halo)
    g_glob = create_plane_grid(nx=NX, ny=NY, nlev=NLEV, dx=rargs.dx,
                               dy=rargs.dx, dtype=jnp.float64)
    s_glob = _global_state(g_glob, hc)

    lay1 = make_plane_pencil_layout(rank=0, n_ranks=1, n_ranks_y=1,
                                    n_ranks_x=1, ny_global=NY, nx_global=NX,
                                    halo=halo)
    m_ref = PlaneCompressibleEulerModel(
        g_glob, hc, make_flat_plane_terrain_metric(g_glob, hc), cfg)
    ref = _run(m_ref, s_glob, rargs.dt, lay1, None)

    g_loc = create_plane_grid(nx=lay.nx_local, ny=lay.ny_local, nlev=NLEV,
                              dx=rargs.dx, dy=rargs.dx, dtype=jnp.float64)
    m_loc = PlaneCompressibleEulerModel(
        g_loc, hc, make_flat_plane_terrain_metric(g_loc, hc), cfg)
    owned = jnp.ones((lay.ny_local, lay.nx_local), jnp.float64)
    loc = _run(m_loc, _slice(s_glob, lay), rargs.dt, lay, owned)

    gathered = [{f: gather_plane_field(getattr(s, f).data, lay)
                 for f in FIELDS} for s in loc]
    if lay.rank != 0:
        return
    worst = 0.0
    for n in range(1, N_STEPS + 1):
        for f in FIELDS:
            a, b = np.asarray(gathered[n][f]), np.asarray(getattr(ref[n], f).data)
            worst = max(worst, float(np.max(np.abs(a - b))))
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10,
                                       err_msg=f"{f} step {n}")
    print(f"max|multi-serial| over {N_STEPS} steps = {worst:.3e}")
    # RULE 0: per-step change of every budget equals the serial change.
    for n in range(1, N_STEPS + 1):
        g_prev = ref[n - 1]._replace(**{
            f: getattr(ref[n - 1], f).replace(data=gathered[n - 1][f])
            for f in FIELDS})
        g_now = ref[n]._replace(**{
            f: getattr(ref[n], f).replace(data=gathered[n][f]) for f in FIELDS})
        bm0, bm1 = _budgets(g_prev, hc), _budgets(g_now, hc)
        bs0, bs1 = _budgets(ref[n - 1], hc), _budgets(ref[n], hc)
        for k in bm0:
            d_multi = bm1[k][0] - bm0[k][0]
            d_serial = bs1[k][0] - bs0[k][0]
            scale = max(bs1[k][1], bs0[k][1])
            assert abs(d_multi - d_serial) <= 1e-12 * scale, (
                k, n, d_multi, d_serial, scale)


@pytest.mark.parametrize("nry,nrx", [(2, 1), (2, 2)])
def test_production_step_halo_gradient_matches_single_rank(nry, nrx):
    """d(global KE after 2 steps)/d(θ' perturbation amplitude): multi-rank
    reverse-mode == single-rank reverse-mode == central finite difference."""
    rargs, hc, cfg = _production_setup(fix_mass=False)
    halo = horizontal_advection_halo_width(cfg)
    lay = _layout_or_skip(nry, nrx, halo)
    comm = MPI.COMM_WORLD
    g_glob = create_plane_grid(nx=NX, ny=NY, nlev=NLEV, dx=rargs.dx,
                               dy=rargs.dx, dtype=jnp.float64)
    s_glob = _global_state(g_glob, hc)
    dtheta = jax.random.normal(jax.random.PRNGKey(3), s_glob.theta_prime.data.shape)

    def make_loss(model, state, layout, distributed):
        d = (dtheta[layout.iy_start:layout.iy_end, layout.ix_start:layout.ix_end]
             if distributed else dtheta)

        def loss(eps):
            s = state._replace(theta_prime=state.theta_prime.replace(
                data=state.theta_prime.data + eps * d))
            for _ in range(2):
                s = model.step_halo(s, dt=rargs.dt, layout=layout)
            ke = jnp.sum(s.u.data ** 2 + s.v.data ** 2)
            return global_sum_mpi(ke, final_loss=True) if distributed else ke
        return loss

    lay1 = make_plane_pencil_layout(rank=0, n_ranks=1, n_ranks_y=1,
                                    n_ranks_x=1, ny_global=NY, nx_global=NX,
                                    halo=halo)
    m_ref = PlaneCompressibleEulerModel(
        g_glob, hc, make_flat_plane_terrain_metric(g_glob, hc), cfg)
    g_ref = float(jax.grad(make_loss(m_ref, s_glob, lay1, False))(0.0))

    g_loc = create_plane_grid(nx=lay.nx_local, ny=lay.ny_local, nlev=NLEV,
                              dx=rargs.dx, dy=rargs.dx, dtype=jnp.float64)
    m_loc = PlaneCompressibleEulerModel(
        g_loc, hc, make_flat_plane_terrain_metric(g_loc, hc), cfg)
    loss_loc = make_loss(m_loc, _slice(s_glob, lay), lay, True)
    # eps is replicated, so the total derivative sums every rank's share.
    g_multi = comm.allreduce(float(jax.grad(loss_loc)(0.0)), op=MPI.SUM)
    eps = 1.0e-4
    g_fd = (float(loss_loc(eps)) - float(loss_loc(-eps))) / (2 * eps)

    assert np.isfinite(g_multi) and g_multi != 0.0
    np.testing.assert_allclose(g_multi, g_ref, rtol=1e-9)
    np.testing.assert_allclose(g_multi, g_fd, rtol=1e-5)
