"""Communication-free polynomial preconditioner for the distributed MPAS
barotropic PCG (``MPASOceanConfig.barotropic_implicit_pcg_precond="poly"``).

The preconditioner is K sweeps of damped Jacobi on the device-LOCAL block of
the Helmholtz operator (true diagonal, off-diagonals restricted to
owned-owned couplings by zeroing the halo before the operator).  Built here
on a single process with a synthetic ``owned`` mask, so the tests need no
MPI; the MPI parity of the surrounding solve is covered by
tests/ocean/distributed/test_barotropic_pcg_mpas_mpi.py.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.barotropic_common import _fixed_iteration_pcg
from legoesm.ocean.dynamics.barotropic_implicit_mpas import (
    _helmholtz_inv_diag_mpas,
    _make_helmholtz,
    barotropic_implicit_mpas,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

COEFF = 1.0e8          # level-5 cells: diagonal ~27, like production (~20)
LEVEL = 5
N_DEV = 8
ITERS = 12
SWEEPS = 4
OMEGA = 2.0 / 3.0


def _coastal_problem(seed=7):
    """Global Helmholtz system on a mesh reordered into N_DEV contiguous
    device blocks (the production ownership), plus the device-LOCAL
    operator every device would apply inside its preconditioner: true
    diagonal, cross-device couplings dropped.  Dots run over ALL cells,
    which is what the owned-masked partial sums add up to under MPI."""
    mesh = reorder_voronoi_for_sharding(create_voronoi_mesh(LEVEL), N_DEV,
                                        method="sfc", edge_order="owner")
    n_cells = int(mesh.nCells)
    n_edges = int(mesh.cellsOnEdge.shape[1])
    rng = np.random.default_rng(seed)
    mask = jnp.asarray((rng.random(n_cells) >= 0.15).astype(np.float64))
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    H_e = jnp.asarray(2000.0 + 1000.0 * rng.random(n_edges))
    coeff = jnp.asarray(COEFF)
    cells_per = n_cells // N_DEV
    owner = np.minimum(np.arange(n_cells) // cells_per, N_DEV - 1)
    same = jnp.asarray((owner[np.asarray(c1)] == owner[np.asarray(c2)])
                       .astype(np.float64))
    A_op = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    inv_diag = _helmholtz_inv_diag_mpas(H_e, coeff, mesh, mask, edge_mask)
    A_cut = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask * same)
    inv_diag_cut = _helmholtz_inv_diag_mpas(H_e, coeff, mesh, mask,
                                            edge_mask * same)
    gap = (1.0 / jnp.where(inv_diag > 0, inv_diag, 1.0)
           - 1.0 / jnp.where(inv_diag_cut > 0, inv_diag_cut, 1.0)) * mask

    def A_local(z):
        return A_cut(z) + gap * z

    w = mesh.areaCell.astype(jnp.float64) * mask
    rhs = jnp.asarray(rng.standard_normal(n_cells)) * mask
    return A_op, A_local, inv_diag, mask, w, rhs


def _poly(A_local, inv_diag, sweeps=SWEEPS):
    def M(r):
        z = OMEGA * inv_diag * r
        for _ in range(sweeps - 1):
            z = z + OMEGA * inv_diag * (r - A_local(z))
        return z
    return M


def _wres(A_op, x, rhs, w):
    r = rhs - A_op(x)
    return float(jnp.sqrt(jnp.sum(w * r * r)) / jnp.sqrt(jnp.sum(w * rhs * rhs)))


class TestValidation:
    def test_unknown_precond_raises_at_entry(self):
        # full bundle pinned, so the SOLVER's literal check is what fires
        cfg = MPASOceanConfig(barotropic_implicit_pcg_precond="chebyshev",
                              barotropic_implicit_pcg_variant="standard",
                              barotropic_implicit_pcg_fixed_iters=20)
        with pytest.raises(ValueError, match="barotropic_implicit_pcg_precond"):
            barotropic_implicit_mpas(None, None, None, cfg, 1.0)

    def test_zero_sweeps_raises_at_entry(self):
        cfg = MPASOceanConfig(barotropic_implicit_pcg_precond="poly",
                              barotropic_implicit_pcg_variant="standard",
                              barotropic_implicit_pcg_fixed_iters=20,
                              barotropic_implicit_pcg_poly_sweeps=0)
        with pytest.raises(ValueError, match="poly_sweeps"):
            barotropic_implicit_mpas(None, None, None, cfg, 1.0)

    def test_default_gpoly_refused_on_the_mpi_voronoi_lane(self, monkeypatch):
        """The MPI-per-rank lane has no deep halo: the default "gpoly" is a
        loud refusal there (decks name their solver), never a fallback."""
        import legoesm.parallel.voronoi_mpi as vm
        monkeypatch.setattr(vm, "get_matching_voronoi_layout",
                            lambda mesh: object())
        with pytest.raises(ValueError, match="select 'poly' there"):
            barotropic_implicit_mpas(
                None, None, None,
                MPASOceanConfig(barotropic_implicit_pcg_precond="gpoly",
                                barotropic_implicit_pcg_variant="standard",
                                barotropic_implicit_pcg_fixed_iters=15), 1.0)

    def test_defaults(self):
        cfg = MPASOceanConfig()
        # resolved per backend (owner decision 2026-10-04)
        assert cfg.barotropic_implicit_pcg_precond is None
        assert cfg.barotropic_implicit_pcg_fixed_iters is None
        assert cfg.barotropic_implicit_pcg_variant is None
        assert cfg.barotropic_implicit_pcg_poly_sweeps == 4


class TestPolynomial:
    def test_self_adjoint_in_area_weighted_product(self):
        A_op, A_local, inv_diag, mask, w, _ = _coastal_problem()
        M = _poly(A_local, inv_diag)
        rng = np.random.default_rng(3)
        n = inv_diag.shape[0]
        u = jnp.asarray(rng.standard_normal(n)) * mask
        v = jnp.asarray(rng.standard_normal(n)) * mask
        lhs = float(jnp.sum(w * u * M(v)))
        rhs = float(jnp.sum(w * M(u) * v))
        assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs))

    def test_fewer_iterations_than_jacobi_for_same_residual(self):
        # Non-vacuous: at the SAME fixed iteration count the polynomial
        # preconditioner must beat Jacobi by a margin, on a coastal grid
        # cut into 8 device blocks.
        A_op, A_local, inv_diag, mask, w, rhs = _coastal_problem()
        x0 = jnp.zeros_like(rhs)
        x_j, _ = _fixed_iteration_pcg(A_op, rhs, lambda r: r * inv_diag, x0,
                                      max_iter=ITERS, dot_weight=w)
        x_p, _ = _fixed_iteration_pcg(A_op, rhs, _poly(A_local, inv_diag),
                                      x0, max_iter=ITERS, dot_weight=w)
        res_j, res_p = _wres(A_op, x_j, rhs, w), _wres(A_op, x_p, rhs, w)
        assert np.isfinite(res_j) and np.isfinite(res_p)
        assert res_p < 0.25 * res_j, (res_j, res_p)

    def test_one_sweep_is_scaled_jacobi(self):
        # K=1 is (2/3)*Jacobi; CG is invariant to that scale, so the two
        # solves must agree to rounding — a check that the polynomial
        # path reduces to the existing one.
        A_op, A_local, inv_diag, mask, w, rhs = _coastal_problem()
        x0 = jnp.zeros_like(rhs)
        x_j, _ = _fixed_iteration_pcg(A_op, rhs, lambda r: r * inv_diag, x0,
                                      max_iter=ITERS, dot_weight=w)
        x_1, _ = _fixed_iteration_pcg(A_op, rhs,
                                      _poly(A_local, inv_diag, sweeps=1),
                                      x0, max_iter=ITERS, dot_weight=w)
        assert float(jnp.max(jnp.abs(x_1 - x_j))) <= 1e-10 * float(
            jnp.max(jnp.abs(x_j)))

    def test_grad_through_poly_solve_is_finite(self):
        A_op, A_local, inv_diag, mask, w, rhs = _coastal_problem()
        M = _poly(A_local, inv_diag)

        def f(b):
            x, _ = _fixed_iteration_pcg(A_op, b, M, jnp.zeros_like(b),
                                        max_iter=ITERS, dot_weight=w)
            return jnp.sum(w * x)

        g = jax.grad(f)(rhs)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0


# --- the PRODUCTION distributed branch, driven on one process ---------------
# ``barotropic_implicit_mpas`` takes its distributed leg whenever the refresh
# object carries ``owned_mask_cells`` (the SPMD lane's contract); on one
# process the exchange is the identity and the global sum is the local sum.

class _OneProcessRefresh:
    def __init__(self, n_cells):
        self.owned_mask_cells = jnp.ones((n_cells,))

    def cells(self, *fields):
        return tuple(fields)

    def edges(self, *fields):
        return tuple(fields)

    def global_sum(self, values):
        return list(values)


# A long implicit step on the coarse test mesh makes the Helmholtz operator
# as stiff as production's (diagonal ~15 instead of ~1.001), so the solve
# is not trivially converged after a couple of iterations.
DT_S = 3.0e4


@pytest.fixture(scope="module")
def _ocean():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.vertical import create_ocean_z_star
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        mesh = create_voronoi_mesh(subdivision_level=3)
        z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
        state = rest_state_mpas_ocean(
            mesh, z_coord, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=85.0)
        rng = np.random.default_rng(0)
        F_eta = jnp.asarray(1.0e-4 * rng.standard_normal(state.eta.data.shape),
                            dtype=jnp.float64)
        yield mesh, z_coord, state, F_eta
    finally:
        set_policy(prev)


def _solve(ocean, **cfg_kw):
    mesh, z_coord, state, F_eta = ocean
    # Jacobi vs poly on the SAME recurrence (explicit; the opt-in deep-halo
    # variant refuses poly).
    cfg_kw.setdefault("barotropic_implicit_pcg_variant", "single_reduce")
    # No floor clamp: its global reduction needs an MPI stack even on one
    # process, and the clamp is a no-op on this rest-state problem anyway.
    cfg = MPASOceanConfig(barotropic_solver="implicit_cn",
                          eta_floor_clamp_iters=0, **cfg_kw)
    eta, _u, _hu, rel = barotropic_implicit_mpas(
        state, mesh, z_coord, cfg, dt=DT_S, F_slow_eta=F_eta,
        halo_refresh=_OneProcessRefresh(int(mesh.nCells)),
        return_residual=True)
    assert eta.dtype == jnp.float64
    return eta, float(rel)


class TestProductionPath:
    ITERS = 4   # few enough that neither solve is at its floor

    def test_poly_beats_jacobi_through_the_solver(self, _ocean):
        eta_j, rel_j = _solve(_ocean, barotropic_implicit_pcg_fixed_iters=self.ITERS,
                              barotropic_implicit_pcg_precond="jacobi")
        eta_p, rel_p = _solve(_ocean, barotropic_implicit_pcg_fixed_iters=self.ITERS,
                              barotropic_implicit_pcg_precond="poly",
                              barotropic_implicit_pcg_poly_sweeps=4)
        assert bool(jnp.all(jnp.isfinite(eta_j))) and bool(jnp.all(jnp.isfinite(eta_p)))
        assert rel_j > 0.0
        assert rel_p < 0.5 * rel_j, (rel_j, rel_p)

    def test_one_sweep_reproduces_jacobi_through_the_solver(self, _ocean):
        # Breaking the production sweep loop, its diagonal or its sign
        # separates these two; K=1 is (2/3)*Jacobi and CG is scale-invariant.
        eta_j, _ = _solve(_ocean, barotropic_implicit_pcg_fixed_iters=self.ITERS,
                          barotropic_implicit_pcg_precond="jacobi")
        eta_1, _ = _solve(_ocean, barotropic_implicit_pcg_fixed_iters=self.ITERS,
                          barotropic_implicit_pcg_precond="poly",
                          barotropic_implicit_pcg_poly_sweeps=1)
        scale = float(jnp.max(jnp.abs(eta_j)))
        assert scale > 0.0
        assert float(jnp.max(jnp.abs(eta_1 - eta_j))) <= 1e-10 * scale

    def test_more_sweeps_help_through_the_solver(self, _ocean):
        _, rel_2 = _solve(_ocean, barotropic_implicit_pcg_fixed_iters=self.ITERS,
                          barotropic_implicit_pcg_precond="poly",
                          barotropic_implicit_pcg_poly_sweeps=2)
        _, rel_8 = _solve(_ocean, barotropic_implicit_pcg_fixed_iters=self.ITERS,
                          barotropic_implicit_pcg_precond="poly",
                          barotropic_implicit_pcg_poly_sweeps=8)
        assert rel_8 < rel_2
