"""Single-process unit tests for the distributed fixed-iteration PCG.

The shared implicit-Helmholtz solver
(``ocean.dynamics.barotropic_common.solve_helmholtz_implicit``) is THE
ocean weak-scaling fix for ``barotropic_solver="implicit_cn"``: under MPI
it runs a hand-rolled fixed-iteration PCG (static ``fori_loop`` + two
batched ``allreduce``/iter — ``p·Ap`` then ``r·z``+``r·r``) UNROLLED and
differentiated straight through (the halo ``_sendrecv_vjp`` + the
``allreduce(SUM)`` dots are AD-safe), instead of the deadlock-prone stock
``jax.scipy`` CG.  (``custom_linear_solve`` is NOT used on the MPI path:
it would linear-transpose the halo ``custom_vjp``, which has no transpose
rule — verified to crash under MPI.)  Single-rank keeps stock CG.

These tests run in ONE process — the cross-rank reduction inside the PCG
is gated on ``is_multi_process()`` so the fixed-iteration *algorithm* is
exercised identically with or without MPI.  The np=2 GATHERED parity and
the no-deadlock check live in
``tests/ocean/distributed/test_barotropic_pcg_mpi_parity.py``.

Coverage (acceptance criteria a, c, d + direct helper test):
  * (a) distributed fixed-M PCG vs stock CG: numerical equivalence
        ``<= 1e-10`` (f64) on a small lat-lon grid.
  * (d) global mass conservation after the post-solve projection.
  * (c) AD: ``jax.grad`` through one ``solve_helmholtz_implicit`` is
        finite and matches a finite difference on a tiny grid (both the
        single-rank stock-CG path and the distributed unrolled path).
  * Helmholtz area-weighted self-adjointness ``A^T = W A W^-1`` (an
        operator property — documents the FV adjoint structure; the
        min-rule face-depth requirement is the load-bearing detail).

Run with:
    JAX_ENABLE_X64=1 python -m pytest \\
        tests/ocean/unit/test_distributed_barotropic_pcg.py -v
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.barotropic_common import (
    HelmholtzSolveDiagnostics,
    solve_helmholtz_implicit,
    _fixed_iteration_pcg,
)
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _make_helmholtz,
    _make_diag_preconditioner,
)


# ===========================================================================
# Helpers: build a small SPD area-weighted Helmholtz on a lat-lon grid.
# ===========================================================================

N_LAT, N_LON = 8, 12
COEFF = 1.0e7   # theta^2 dt^2 g H ~ 0.55^2 * 3600^2 * 9.81 * 3000 type scale


def _build_helmholtz(seed: int = 0, all_ocean: bool = True):
    """Return ``(A_op, M_inv, area, inv_area, mask, grid)`` for a small grid.

    ``all_ocean=False`` carves a deterministic land mask so the operator's
    masked rows (and the inv-area land convention) are exercised.
    """
    grid = ensure_geometry(create_latlon_grid(n_lat=N_LAT, n_lon=N_LON))
    rng = np.random.default_rng(seed)
    if all_ocean:
        mask = jnp.ones((N_LAT, N_LON))
    else:
        m = np.ones((N_LAT, N_LON))
        # Land cap at the two pole rows + a couple of interior coast cells.
        m[0, :] = 0.0
        m[-1, :] = 0.0
        m[3, 5] = 0.0
        m[4, 6] = 0.0
        mask = jnp.asarray(m)
    u_mask = jnp.ones((N_LAT, N_LON + 1))
    v_mask = jnp.ones((N_LAT + 1, N_LON))
    # Heterogeneous (but positive) face depths so the operator is not a
    # trivial constant-coefficient Laplacian.  CRITICAL: derive the face
    # depths from a CELL-centred field via the same min-rule that
    # ``_h_total_at_faces`` uses in production (``min(H_left, H_right)``,
    # periodic in lon, pole v-faces zeroed).  Independent random face
    # values would break the FV adjoint identity and make ``W A``
    # spuriously non-symmetric — the operator is exactly W-self-adjoint
    # ONLY for cell-derived (shared-face) depths, which is the only case
    # the production solver ever sees.
    H_cell = jnp.asarray(2000.0 + 1500.0 * rng.random((N_LAT, N_LON)))
    _Hu_inner = jnp.minimum(jnp.roll(H_cell, 1, axis=1), H_cell)
    H_u = jnp.concatenate([_Hu_inner, _Hu_inner[:, 0:1]], axis=1)
    _Hv_inner = jnp.minimum(H_cell[:-1], H_cell[1:])      # (N_LAT-1, N_LON)
    H_v = jnp.concatenate(
        [jnp.zeros((1, N_LON)), _Hv_inner, jnp.zeros((1, N_LON))], axis=0,
    )
    coeff = jnp.asarray(COEFF)
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask)
    M_inv = _make_diag_preconditioner(H_u, H_v, coeff, grid, mask)
    area = grid.area.astype(jnp.float64)
    inv_area = jnp.where(mask > 0.5, 1.0 / area, 0.0)
    return A_op, M_inv, area, inv_area, mask, grid


def _random_rhs(mask, seed: int = 1):
    rng = np.random.default_rng(seed)
    r = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    return r * mask


# ===========================================================================
# (a) Single-process distributed-PCG vs stock CG numerical equivalence.
# ===========================================================================


class TestPCGvsStockCG:
    """``distributed=True`` fixed-M PCG matches stock CG to <= 1e-10 (f64)."""

    @pytest.mark.parametrize("all_ocean", [True, False])
    def test_pcg_matches_stock_cg(self, all_ocean):
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(
            seed=2, all_ocean=all_ocean,
        )
        rhs = _random_rhs(mask, seed=3)
        x0 = jnp.zeros_like(rhs)

        eta_cg, diag_cg = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0,
            distributed=False,
            fixed_iters=60,
            residual_tol=1.0e-10,
            stock_cg_tol=1.0e-12,
            stock_cg_maxiter=400,        )
        eta_pcg, diag_pcg = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0,
            distributed=True,
            fixed_iters=120,        # generous M => fully converged
            residual_tol=1.0e-10,
            stock_cg_tol=1.0e-12,
            stock_cg_maxiter=400,        )
        assert isinstance(diag_pcg, HelmholtzSolveDiagnostics)
        max_err = float(jnp.max(jnp.abs(eta_pcg - eta_cg)))
        assert max_err <= 1.0e-10, (
            f"distributed PCG vs stock CG max|Δη|={max_err:.3e} > 1e-10 "
            f"(all_ocean={all_ocean})"
        )
        # Both must actually have solved the system well.
        assert float(diag_pcg.rel_residual) <= 1.0e-10
        assert bool(diag_pcg.converged)

    def test_residual_diagnostic_detects_underconvergence(self):
        """Tiny M => large residual; the diagnostic must FLAG it (and the
        loop must still run all M iterations — no early exit)."""
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(seed=4)
        rhs = _random_rhs(mask, seed=5)
        _eta, diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, jnp.zeros_like(rhs),
            distributed=True,
            fixed_iters=1,          # deliberately under-converged
            residual_tol=1.0e-10,
            stock_cg_tol=1.0e-12,
            stock_cg_maxiter=400,        )
        assert float(diag.rel_residual) > 1.0e-10
        assert not bool(diag.converged)


# ===========================================================================
# Helmholtz area-weighted self-adjointness (background — A^T = W A W^-1;
# documents why a naive Euclidean custom_linear_solve transpose would be
# wrong; the shipped unrolled path does not form the transpose).
# ===========================================================================


class TestHelmholtzWeightedSymmetry:
    """``A`` is symmetric in the area-weighted inner product but NOT in the
    Euclidean one (``A^T = W A W^-1``).  Background operator identity: it
    is why a naive Euclidean ``custom_linear_solve`` transpose would be
    wrong.  The shipped distributed path is UNROLLED and forms no
    transpose, so this is documentation/regression of the operator
    structure, not of the solver's AD path.
    """

    def _dense(self, A_op):
        N = N_LAT * N_LON

        def apply(vec):
            return A_op(vec.reshape(N_LAT, N_LON)).reshape(-1)

        cols = []
        for k in range(N):
            e = jnp.zeros(N).at[k].set(1.0)
            cols.append(np.asarray(apply(e)))
        return np.stack(cols, axis=1)   # column k = A @ e_k

    def test_area_weighted_symmetry(self):
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(seed=6)
        A = self._dense(A_op)
        a = np.asarray(area).reshape(-1)
        euclid = np.abs(A - A.T).max()
        weighted = np.abs(a[:, None] * A - (a[:, None] * A).T).max()
        # Euclidean asymmetry is real (the 1/area in divergence); the
        # area-weighted operator W A IS symmetric to round-off.
        assert euclid > 1.0e-6, (
            "expected Euclidean asymmetry from the 1/area divergence "
            f"factor, got {euclid:.3e}"
        )
        assert weighted <= 1.0e-6 * np.abs(a[:, None] * A).max(), (
            f"area-weighted operator not symmetric: {weighted:.3e}"
        )

    def test_transpose_operator_is_W_A_Winv(self):
        """The Helmholtz adjoint is ``A^T = W A W^-1`` (W = diag(area)).

        Validated as an OPERATOR identity against the dense ``A^T``:
        ``A^T y == area * A_op(y / area)`` for arbitrary ``y``.  This
        documents the FV adjoint structure (``A`` is self-adjoint only in
        the area-weighted inner product, not the Euclidean one) — the
        reason a naive ``custom_linear_solve`` Euclidean transpose would
        be wrong.  The shipped distributed path is UNROLLED (differentiated
        straight through), so it doesn't need this transpose explicitly,
        but the identity is the load-bearing fact behind why the operator
        is well-posed for CG."""
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(seed=7)
        A = self._dense(A_op)
        rng = np.random.default_rng(8)
        y = jnp.asarray(rng.standard_normal((N_LAT, N_LON))) * mask
        inv_w = inv_area
        eps_w = jnp.asarray(1.0e-30)
        area_w = jnp.where(
            jnp.abs(inv_w) > eps_w, 1.0 / jnp.maximum(inv_w, eps_w), 0.0,
        )
        AT_y_op = np.asarray((area_w * A_op(y * inv_w)).reshape(-1))
        AT_y_dense = A.T @ np.asarray(y).reshape(-1)
        max_err = np.max(np.abs(AT_y_op - AT_y_dense))
        scale = max(np.max(np.abs(AT_y_dense)), 1.0)
        assert max_err / scale <= 1.0e-10, (
            f"A^T = W A W^-1 identity broken: rel err {max_err / scale:.3e}"
        )


# ===========================================================================
# (c) AD: grad through one solve finite + matches finite difference.
# ===========================================================================


class TestSolveDifferentiability:
    """``jax.grad`` of a scalar of the solve w.r.t. rhs is finite and
    matches a central finite difference — both solver paths."""

    @pytest.mark.parametrize("distributed", [False, True])
    def test_grad_matches_finite_difference(self, distributed):
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(seed=9)
        rhs0 = _random_rhs(mask, seed=10)
        # Differentiable weights for the scalar loss (mask out land).
        w = jnp.asarray(
            np.random.default_rng(11).standard_normal((N_LAT, N_LON)),
        ) * mask

        def loss(rhs):
            eta, _diag = solve_helmholtz_implicit(
                A_op, rhs, M_inv, jnp.zeros_like(rhs),
                distributed=distributed,
                fixed_iters=120,
                residual_tol=1.0e-10,
                stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400,            )
            return jnp.sum(w * eta)

        g = jax.grad(loss)(rhs0)
        assert jnp.all(jnp.isfinite(g)), "grad has non-finite entries"
        gnorm = float(jnp.linalg.norm(g))
        assert gnorm > 0.0, "grad is identically zero"

        # Central finite difference along a deterministic wet direction.
        # The loss is LINEAR in rhs (eta = A^-1 rhs), so FD is exact up to
        # the solve accuracy: the distributed UNROLLED-PCG adjoint matches
        # FD to ~1e-6 for a well-converged M; the single-rank STOCK
        # jax.scipy.cg gradient is only ~1e-3 accurate here (its
        # implicit-VJP adjoint solve under-converges on this stiff
        # operator even though the forward residual is ~1e-16 — a known
        # jax.scipy.cg limitation).
        rng = np.random.default_rng(12)
        d = jnp.asarray(rng.standard_normal((N_LAT, N_LON))) * mask
        eps = 1.0e-4
        fd = (loss(rhs0 + eps * d) - loss(rhs0 - eps * d)) / (2.0 * eps)
        ad = float(jnp.sum(g * d))
        rel = abs(ad - float(fd)) / max(abs(float(fd)), 1.0)
        tol = 1.0e-5 if distributed else 5.0e-3
        assert rel <= tol, (
            f"grad·d ({ad:.6e}) disagrees with FD ({float(fd):.6e}), "
            f"rel {rel:.3e} > {tol:.0e} (distributed={distributed})"
        )

    def test_distributed_grad_is_finite_f32(self):
        """f32 reverse-mode AD through the UNROLLED distributed PCG is
        finite (the unrolled loop differentiates straight through
        ``A_op`` + the dot-product reductions; no transpose, no
        ``custom_linear_solve`` — so no f32 floor pitfalls)."""
        grid = ensure_geometry(create_latlon_grid(n_lat=N_LAT, n_lon=N_LON))
        rng = np.random.default_rng(21)
        mask = jnp.ones((N_LAT, N_LON), dtype=jnp.float32)
        u_mask = jnp.ones((N_LAT, N_LON + 1), dtype=jnp.float32)
        v_mask = jnp.ones((N_LAT + 1, N_LON), dtype=jnp.float32)
        H_cell = jnp.asarray(
            2000.0 + 1500.0 * rng.random((N_LAT, N_LON)), dtype=jnp.float32)
        _Hu_inner = jnp.minimum(jnp.roll(H_cell, 1, axis=1), H_cell)
        H_u = jnp.concatenate([_Hu_inner, _Hu_inner[:, 0:1]], axis=1)
        _Hv_inner = jnp.minimum(H_cell[:-1], H_cell[1:])
        H_v = jnp.concatenate(
            [jnp.zeros((1, N_LON), jnp.float32), _Hv_inner,
             jnp.zeros((1, N_LON), jnp.float32)], axis=0)
        coeff = jnp.asarray(COEFF, dtype=jnp.float32)
        A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask)
        M_inv = _make_diag_preconditioner(H_u, H_v, coeff, grid, mask)
        rhs0 = (jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
                * mask).astype(jnp.float32)
        w = (jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
             * mask).astype(jnp.float32)

        def loss(rhs):
            eta, _ = solve_helmholtz_implicit(
                A_op, rhs, M_inv, jnp.zeros_like(rhs),
                distributed=True, fixed_iters=30, residual_tol=1.0e-4,
                stock_cg_tol=1.0e-6, stock_cg_maxiter=200,
            )
            return jnp.sum(w * eta)

        g = jax.grad(loss)(rhs0)
        assert g.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(g)), "f32 grad has NaN/Inf"
        assert float(jnp.linalg.norm(g)) > 0.0, "f32 grad is zero"

    def test_distributed_adjoint_matches_fd_tightly(self):
        """The UNROLLED distributed adjoint matches a central finite
        difference to high precision: for a well-converged fixed-M PCG the
        unrolled gradient equals the implicit-solve gradient to within the
        residual.  The decisive AD-correctness statement for the
        distributed path."""
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(seed=13)
        rhs0 = _random_rhs(mask, seed=14)
        w = jnp.asarray(
            np.random.default_rng(15).standard_normal((N_LAT, N_LON)),
        ) * mask

        def loss(rhs):
            eta, _ = solve_helmholtz_implicit(
                A_op, rhs, M_inv, jnp.zeros_like(rhs),
                distributed=True,
                fixed_iters=120,
                residual_tol=1.0e-10,
                stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400,
            )
            return jnp.sum(w * eta)

        g_dist = jax.grad(loss)(rhs0)
        rng = np.random.default_rng(16)
        # Average over a few directions for a robust check.
        max_rel = 0.0
        for _ in range(4):
            d = jnp.asarray(rng.standard_normal((N_LAT, N_LON))) * mask
            eps = 1.0e-4
            fd = (loss(rhs0 + eps * d) - loss(rhs0 - eps * d)) / (2.0 * eps)
            ad = float(jnp.sum(g_dist * d))
            max_rel = max(max_rel, abs(ad - float(fd)) / max(abs(float(fd)), 1.0))
        assert max_rel <= 1.0e-6, (
            f"distributed adjoint vs FD: worst rel {max_rel:.3e} > 1e-6"
        )

        # Consistency with the stock-CG adjoint, to stock CG's (looser)
        # accuracy — both target the same gradient of A^{-1} rhs.
        def loss_stock(rhs):
            eta, _ = solve_helmholtz_implicit(
                A_op, rhs, M_inv, jnp.zeros_like(rhs),
                distributed=False, fixed_iters=120, residual_tol=1.0e-10,
                stock_cg_tol=1.0e-12, stock_cg_maxiter=400,
            )
            return jnp.sum(w * eta)

        g_stock = jax.grad(loss_stock)(rhs0)
        rel = float(jnp.max(jnp.abs(g_stock - g_dist))) / max(
            float(jnp.max(jnp.abs(g_dist))), 1.0)
        assert rel <= 1.0e-2, (
            f"distributed vs stock-CG adjoint differ by {rel:.3e} > 1e-2 "
            "(stock CG's adjoint is the less-accurate one)"
        )


# ===========================================================================
# (d) Global mass conservation through the full lat-lon solver.
# ===========================================================================


class TestLatLonSolverMassAndResidual:
    """The full ``barotropic_implicit_latlon_cgrid`` distributed path
    (single process) conserves global mass and reports its residual."""

    def _setup(self, barotropic_solver="implicit_cn"):
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.state import LatLonCGridOceanConfig
        grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
        z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
        config = LatLonCGridOceanConfig.from_flat(
            barotropic_solver=barotropic_solver,
            barotropic_implicit_pcg_fixed_iters=120,
        )
        state = rest_state_latlon_cgrid_ocean(grid, z_coord)
        # Force f64 on the prognostic + geometry fields so the stock-CG
        # residual can actually reach the 1e-10 tolerance (the storage
        # policy may default to f32 even under JAX_ENABLE_X64).
        state = jax.tree.map(
            lambda x: x.astype(jnp.float64)
            if getattr(x, "dtype", None) is not None
            and jnp.issubdtype(x.dtype, jnp.floating) else x,
            state,
        )
        # Excite a gravity wave so the solve is non-trivial.
        mask = state.land_mask.data
        eta_pert = state.eta.data + 0.05 * mask * (
            jnp.sin(3.0 * grid.lon2d) * jnp.cos(2.0 * grid.lat2d)
        ).astype(state.eta.data.dtype)
        u_pert = state.u.data + 0.02 * state.u_mask.data[..., None]
        state = state._replace(
            eta=state.eta.replace(data=eta_pert),
            u=state.u.replace(data=u_pert),
        )
        return state, grid, z_coord, config

    def test_global_mass_conserved_and_residual_returned(self):
        from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
            barotropic_implicit_latlon_cgrid,
        )
        state, grid, z_coord, config = self._setup()
        dt = 1800.0
        out = barotropic_implicit_latlon_cgrid(
            state, dt, grid, z_coord, config, return_residual=True,
        )
        state_new, (_Hu, _Hv), rel = out
        assert jnp.isfinite(rel)
        assert float(rel) <= config.barotropic.barotropic_implicit_pcg_residual_tol, (
            f"global rel residual {float(rel):.3e} exceeds tol"
        )

        # Mass target: the in-solver projection enforces
        # sum(eta_new*area) == sum(rhs*area).  Here we check the looser
        # but physical statement that the post-solve volume tracks the
        # pre-solve volume to round-off (no freshwater forcing => closed).
        area = grid.area
        mask = state.land_mask.data
        m_old = float(jnp.sum(state.eta.data * area * mask))
        m_new = float(jnp.sum(state_new.eta.data * area * mask))
        rel_mass = abs(m_new - m_old) / max(abs(m_old), 1.0)
        assert rel_mass <= 1.0e-8, (
            f"global eta mass drifted {rel_mass:.3e} over one implicit step"
        )

    def test_no_while_loop_in_distributed_pcg(self):
        """Static guard: the distributed PCG must compile with NO ``while``
        primitive (a residual-dependent ``while_loop`` is the deadlock-
        prone control flow we replaced with a static ``fori_loop``).

        Walk the jaxpr equations recursively (incl. nested
        ``jaxpr``/``call_jaxpr`` params) and assert no primitive is named
        ``while``; also assert the fixed loop IS present (``scan`` —
        ``fori_loop`` with a static trip count lowers to ``scan``)."""
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(seed=16)
        rhs = _random_rhs(mask, seed=17)

        def run(rhs):
            eta, _ = solve_helmholtz_implicit(
                A_op, rhs, M_inv, jnp.zeros_like(rhs),
                distributed=True,
                fixed_iters=30,
                residual_tol=1.0e-10,
                stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400,            )
            return eta

        closed = jax.make_jaxpr(run)(rhs)

        prim_names: set[str] = set()

        def _walk(jaxpr):
            for eqn in jaxpr.eqns:
                prim_names.add(eqn.primitive.name)
                for param in eqn.params.values():
                    sub = getattr(param, "jaxpr", None)
                    if sub is not None:
                        # ClosedJaxpr.jaxpr is the inner Jaxpr.
                        _walk(getattr(sub, "jaxpr", sub))
                    if isinstance(param, (tuple, list)):
                        for p in param:
                            sub = getattr(p, "jaxpr", None)
                            if sub is not None:
                                _walk(getattr(sub, "jaxpr", sub))

        _walk(closed.jaxpr)
        assert "while" not in prim_names, (
            "distributed PCG jaxpr contains a `while` primitive — the "
            "fixed-iteration schedule must lower to scan/fori_loop only "
            f"(residual-dependent while_loop => MPI deadlock). "
            f"Primitives found: {sorted(prim_names)}"
        )
        assert "scan" in prim_names, (
            "expected the static fixed-iteration loop to lower to `scan`; "
            f"primitives found: {sorted(prim_names)}"
        )


# ===========================================================================
# (e) ``barotropic_implicit_force_pcg`` config plumb (solver-matched parity
#     references + the faster-single-rank option; bisect job 8459362).
# ===========================================================================


class TestForcePCGConfigPlumb:
    def test_default_off(self):
        from legoesm.ocean.state import LatLonCGridOceanConfig

        assert LatLonCGridOceanConfig.from_flat().barotropic.barotropic_implicit_force_pcg is False

    def test_force_pcg_step_matches_stock_cg_step(self):
        """One implicit_cn model step with force_pcg=True equals the
        stock-CG step to solver tolerance (both fully converged)."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.vertical import create_ocean_z_star

        grid = create_latlon_grid(n_lat=16, n_lon=32)
        z = create_ocean_z_star(n_levels=4)
        state0 = rest_state_latlon_cgrid_ocean(grid, z)
        # Nudge eta so the Helmholtz solve is nontrivial.
        rng = np.random.default_rng(7)
        state0 = state0._replace(
            eta=state0.eta.replace(
                data=jnp.asarray(
                    1.0e-2 * rng.standard_normal(state0.eta.data.shape),
                ),
            ),
        )

        etas = {}
        for force in (False, True):
            cfg = LatLonCGridOceanConfig.from_flat(
                barotropic_solver="implicit_cn",
                barotropic_implicit_force_pcg=force,
            )
            model = LatLonCGridOceanModel(grid, z, cfg)
            s1 = model.step(state0, 600.0)
            etas[force] = np.asarray(s1.eta.data)

        max_err = float(np.max(np.abs(etas[True] - etas[False])))
        assert max_err <= 1.0e-8, (
            f"force_pcg step diverged from stock-CG step: "
            f"max|Δη|={max_err:.3e} > 1e-8"
        )


# ===========================================================================
# (f) single_reduce PCG variant (Chronopoulos-Gear; one reduction/iter).
# ===========================================================================


class TestSingleReducePCG:
    def test_matches_stock_cg(self):
        """Fully-converged single_reduce == stock CG to <= 1e-9 (f64)."""
        A_op, M_inv, area, inv_area, mask, grid = _build_helmholtz(
            seed=11, all_ocean=False,
        )
        rhs = _random_rhs(mask, seed=12)
        x0 = jnp.zeros_like(rhs)
        eta_cg, _ = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0,
            distributed=False, fixed_iters=60,
            residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
            stock_cg_maxiter=400,
        )
        eta_sr, diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0,
            distributed=True, fixed_iters=120,
            residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
            stock_cg_maxiter=400, pcg_variant="single_reduce",
            dot_weight=area * mask,
        )
        max_err = float(jnp.max(jnp.abs(eta_sr - eta_cg)))
        assert max_err <= 1.0e-9, f"single_reduce vs stock max|dη|={max_err:.3e}"
        assert float(diag.rel_residual) <= 1.0e-10
        assert bool(diag.converged)

    def test_matches_standard_pcg_at_solver_tol(self):
        A_op, M_inv, area, _inv, mask, _grid = _build_helmholtz(
            seed=13, all_ocean=True,
        )
        rhs = _random_rhs(mask, seed=14)
        x0 = jnp.zeros_like(rhs)
        common = dict(distributed=True, fixed_iters=80,
                      residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
                      stock_cg_maxiter=400)
        eta_std, _ = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0, pcg_variant="standard", **common)
        eta_sr, _ = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0, pcg_variant="single_reduce",
            dot_weight=area * mask, **common)
        assert float(jnp.max(jnp.abs(eta_sr - eta_std))) <= 1.0e-9

    def test_single_reduce_refuses_missing_weight(self):
        """Codex CRITICAL tripwire: no silent Euclidean fallback — the
        recurrence is only valid in the W-self-adjoint inner product."""
        A_op, M_inv, *_rest = _build_helmholtz(seed=23)
        mask = _rest[2]
        rhs = _random_rhs(mask, seed=24)
        with pytest.raises(ValueError, match="dot_weight"):
            solve_helmholtz_implicit(
                A_op, rhs, M_inv, jnp.zeros_like(rhs),
                distributed=True, fixed_iters=10,
                residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400, pcg_variant="single_reduce",
            )

    def test_equal_M_residual_comparable_on_coastal_grid(self):
        """Stability stress (codex MINOR): at the PRODUCTION M=60 on
        the land-carved (coastal, varying-area) operator, the
        single-reduce residual must stay within 10x of standard —
        guards against the known mild CG-CG stability loss silently
        eating the fixed-iteration budget."""
        A_op, M_inv, area, _inv, mask, _grid = _build_helmholtz(
            seed=25, all_ocean=False,
        )
        rhs = _random_rhs(mask, seed=26)
        x0 = jnp.zeros_like(rhs)
        common = dict(distributed=True, fixed_iters=60,
                      residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
                      stock_cg_maxiter=400)
        _eta_std, diag_std = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0, pcg_variant="standard", **common)
        _eta_sr, diag_sr = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0, pcg_variant="single_reduce",
            dot_weight=area * mask, **common)
        r_std = float(diag_std.rel_residual)
        r_sr = float(diag_sr.rel_residual)
        assert r_sr <= max(10.0 * r_std, 1.0e-12), (
            f"single_reduce residual {r_sr:.3e} vs standard {r_std:.3e}"
        )

    def test_unknown_variant_raises(self):
        A_op, M_inv, *_rest = _build_helmholtz(seed=15)
        mask = _rest[2]
        rhs = _random_rhs(mask, seed=16)
        with pytest.raises(ValueError, match="pcg_variant"):
            solve_helmholtz_implicit(
                A_op, rhs, M_inv, jnp.zeros_like(rhs),
                distributed=True, fixed_iters=10,
                residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400, pcg_variant="chebyshev",
            )

    def test_reduction_count_halved(self, monkeypatch):
        """Trace-SITE tripwire for the per-iteration reduction count.

        The loop body is traced ONCE by ``fori_loop``, so the counter
        sees call SITES, not runtime executions: standard = init(1) +
        TWO body sites + diagnostic tail(1) = 4; single_reduce =
        init(1) + ONE body site + final-update residual(1) + tail(1) = 4.
        At runtime the body site executes M-1 times plus the final
        update once — one reduction per iteration, vs two for standard."""
        import legoesm.ocean.dynamics.barotropic_common as bc

        counts = {"n": 0}
        orig = bc._global_dot_batch

        def counting(pairs):
            counts["n"] += 1
            return orig(pairs)

        monkeypatch.setattr(bc, "_global_dot_batch", counting)
        A_op, M_inv, area, _inv, mask, _grid = _build_helmholtz(seed=17)
        rhs = _random_rhs(mask, seed=18)
        x0 = jnp.zeros_like(rhs)
        for variant, expected_sites in (("standard", 4),
                                        ("single_reduce", 4)):
            counts["n"] = 0
            solve_helmholtz_implicit(
                A_op, rhs, M_inv, x0,
                distributed=True, fixed_iters=7,
                residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400, pcg_variant=variant,
                dot_weight=area * mask,
            )
            assert counts["n"] == expected_sites, (
                f"{variant}: {counts['n']} reduction sites, expected "
                f"{expected_sites}"
            )

    def test_single_reduce_skips_discarded_last_matvec(self):
        """M iterations apply the operator M+1 times (r0, w0, then one per
        iteration except the last, whose next direction would be thrown
        away) and the preconditioner M times; 0 iterations return x0."""
        import legoesm.ocean.dynamics.barotropic_common as bc

        A_op, M_inv, area, _inv, mask, _grid = _build_helmholtz(seed=21)
        rhs = _random_rhs(mask, seed=22)
        x0 = jnp.zeros_like(rhs)
        calls = {"A": 0, "M": 0}

        def A_count(v):
            calls["A"] += 1
            return A_op(v)

        def M_count(v):
            calls["M"] += 1
            return M_inv(v)

        with jax.disable_jit():
            x5, _ = bc._fixed_iteration_pcg_single_reduce(
                A_count, rhs, M_count, x0, max_iter=5, dot_weight=area * mask)
        assert calls == {"A": 5 + 1, "M": 5}
        x0_out, _ = bc._fixed_iteration_pcg_single_reduce(
            A_op, rhs, M_inv, x0, max_iter=0, dot_weight=area * mask)
        np.testing.assert_array_equal(np.asarray(x0_out), np.asarray(x0))
        x5_std, _ = bc._fixed_iteration_pcg(
            A_op, rhs, M_inv, x0, max_iter=5, dot_weight=area * mask)
        np.testing.assert_allclose(np.asarray(x5), np.asarray(x5_std),
                                   rtol=1e-8, atol=1e-12)
        x1, rr1 = bc._fixed_iteration_pcg_single_reduce(
            A_op, rhs, M_inv, x0, max_iter=1, dot_weight=area * mask)
        x1_std, rr1_std = bc._fixed_iteration_pcg(
            A_op, rhs, M_inv, x0, max_iter=1, dot_weight=area * mask)
        np.testing.assert_allclose(np.asarray(x1), np.asarray(x1_std),
                                   rtol=1e-10, atol=1e-14)
        assert rr1.dtype == rhs.dtype and float(rr1) >= 0.0
        np.testing.assert_allclose(float(rr1), float(rr1_std), rtol=1e-8)

    def test_grad_finite_through_single_reduce(self):
        A_op, M_inv, area, _inv, mask, _grid = _build_helmholtz(seed=19)
        rhs = _random_rhs(mask, seed=20)
        W = area * mask

        def loss(b):
            eta, _ = solve_helmholtz_implicit(
                A_op, b, M_inv, jnp.zeros_like(b),
                distributed=True, fixed_iters=30,
                residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400, pcg_variant="single_reduce",
                dot_weight=W,
            )
            return jnp.sum(eta**2)

        g = jax.grad(loss)(rhs)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_no_while_primitive(self):
        """Static schedule: jaxpr must contain scan/fori, never while."""
        A_op, M_inv, area, _inv, mask, _grid = _build_helmholtz(seed=21)
        rhs = _random_rhs(mask, seed=22)

        def run(b):
            eta, _ = solve_helmholtz_implicit(
                A_op, b, M_inv, jnp.zeros_like(b),
                distributed=True, fixed_iters=10,
                residual_tol=1.0e-10, stock_cg_tol=1.0e-12,
                stock_cg_maxiter=400, pcg_variant="single_reduce",
                dot_weight=area * mask,
            )
            return eta

        closed = jax.make_jaxpr(run)(rhs)
        prims = set()

        def _walk(jaxpr):
            for eqn in jaxpr.eqns:
                prims.add(eqn.primitive.name)
                for param in eqn.params.values():
                    sub = getattr(param, "jaxpr", None)
                    if sub is not None:
                        _walk(getattr(sub, "jaxpr", sub))
                    if isinstance(param, (tuple, list)):
                        for p in param:
                            sub = getattr(p, "jaxpr", None)
                            if sub is not None:
                                _walk(getattr(sub, "jaxpr", sub))

        _walk(closed.jaxpr)
        assert "while" not in prims, sorted(prims)

    def test_config_field_plumb(self):
        from legoesm.ocean.state import LatLonCGridOceanConfig

        assert (LatLonCGridOceanConfig.from_flat().barotropic.barotropic_implicit_pcg_variant
                == "standard")

    def test_mpas_config_field_plumb(self):
        """MPASOceanConfig MUST carry every distributed implicit-CN knob the
        solver's distributed branch dereferences on ``config`` —
        ``barotropic_implicit_pcg_variant`` was MISSING from the MPAS config
        while barotropic_implicit_mpas.py read it, so the distributed branch
        raised AttributeError (codex scaling-audit finding, 2026-06-12)."""
        from legoesm.ocean.mpas_config import MPASOceanConfig
        from legoesm.ocean.state import LatLonCGridOceanConfig

        cfg = MPASOceanConfig()
        # The exact attribute chain the distributed branch dereferences.
        # standard + gpoly x 15 (owner decision 2026-10-02: the recurrence
        # gpoly x 15 was measured with; deep-halo Jacobi is opt-in).
        assert cfg.barotropic_implicit_pcg_variant == "standard"
        # gpoly at 20 since 2026-10-05 (owner decision; A/B in mpas_config).
        assert cfg.barotropic_implicit_pcg_fixed_iters == 20
        assert cfg.barotropic_implicit_pcg_precond == "gpoly"
        assert cfg.barotropic_implicit_pcg_poly_sweeps == 4
        assert cfg.barotropic_implicit_pcg_residual_tol == 1.0e-10
        assert cfg.barotropic_implicit_pcg_tol == 1.0e-10
        assert cfg.barotropic_implicit_pcg_maxiter == 200
        # Same variant literal set as the lat-lon contract (validated at
        # solver entry by solve_helmholtz_implicit); the DEFAULTS differ since
        # 2026-09-26 -- MPAS single_reduce (measured), lat-lon standard (not).
        # The lat-lon side is NESTED post-#501 (config.barotropic.*).
        assert (LatLonCGridOceanConfig().barotropic.barotropic_implicit_pcg_variant
                == "standard")

