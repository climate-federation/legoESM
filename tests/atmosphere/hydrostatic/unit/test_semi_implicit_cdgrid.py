"""Phase 2 scaffolding tests for the Hoskins–Simmons FV3 D-grid port.

Locks in the current behaviour of
``src/legoesm/atmosphere/dynamics/semi_implicit_cdgrid.py``.  Phase 2
work (see module docstring there) lowers the adjoint-residual
threshold once the cubed-sphere FV gradient + divergence are made
into a proper adjoint pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.semi_implicit_cdgrid import (
    adjoint_residual_norm,
    cdgrid_scalar_laplacian,
    make_helmholtz_op,
    richardson_helmholtz_solve,
)
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.fixture(autouse=True)
def _x64_fp64():
    orig_x64 = jax.config.jax_enable_x64
    orig_pol = get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_pol)
    jax.config.update("jax_enable_x64", orig_x64)


@pytest.fixture(scope="module")
def cdgrid():
    grid = create_cubed_sphere(8)
    return create_cubed_sphere_cdgrid(grid)


# ----------------------------------------------------------------------
# cdgrid_scalar_laplacian — basic sanity
# ----------------------------------------------------------------------


class TestScalarLaplacian:
    def test_constant_field_has_zero_laplacian(self, cdgrid):
        n = cdgrid.base.n
        p = jnp.full((6, n, n), 1.0e5)
        lap = cdgrid_scalar_laplacian(p, cdgrid)
        # Allow O(roundoff) residual from area-weighted divergence.
        assert float(jnp.max(jnp.abs(lap))) < 1.0e-8

    def test_shape_preserved(self, cdgrid):
        n = cdgrid.base.n
        np.random.seed(0)
        p = jnp.asarray(np.random.randn(6, n, n))
        lap = cdgrid_scalar_laplacian(p, cdgrid)
        assert lap.shape == p.shape


# ----------------------------------------------------------------------
# adjoint_residual_norm — locks current Phase 2 status
# ----------------------------------------------------------------------


class TestAdjointResidual:
    def test_current_operator_is_non_symmetric(self, cdgrid):
        """Phase 2 baseline: the composed ``div(grad)`` is **not**
        a discrete FV adjoint pair on the cubed-sphere.  Locks the
        finding so a future Phase-2 PR that drives this below
        ``1e-10`` triggers the assertion update and proves the fix."""
        L = lambda p: cdgrid_scalar_laplacian(p, cdgrid)
        residual = adjoint_residual_norm(L, cdgrid, n_samples=4)
        # Empirically ~0.13 on n=8.  Assert the gap is present
        # (residual > 1e-6) so the test stays meaningful.
        assert residual > 1.0e-6, (
            f"adjoint residual {residual:.3e} unexpectedly small — "
            f"either the operator was made symmetric (great! update "
            f"this assertion to assert symmetry) or the metric weight "
            f"is now masking the true asymmetry"
        )
        assert residual < 1.0, (
            f"adjoint residual {residual:.3e} unreasonably large — "
            f"check that area weighting and operator are wired correctly"
        )


# ----------------------------------------------------------------------
# make_helmholtz_op — identity when coeff = 0
# ----------------------------------------------------------------------


class TestHelmholtzOp:
    def test_fails_closed_by_default(self, cdgrid):
        """Production callers must NOT be able to obtain a non-SPD
        operator and pass it to ``cg`` silently."""
        with pytest.raises(NotImplementedError, match="Phase 2"):
            make_helmholtz_op(coeff=0.5, cdgrid=cdgrid)

    def test_identity_at_zero_coeff(self, cdgrid):
        A = make_helmholtz_op(
            coeff=0.0, cdgrid=cdgrid,
            allow_nonsymmetric_for_testing=True,
        )
        n = cdgrid.base.n
        np.random.seed(1)
        p = jnp.asarray(np.random.randn(6, n, n))
        Ap = A(p)
        # A(p) should equal p exactly when coeff=0.
        assert float(jnp.max(jnp.abs(Ap - p))) < 1.0e-14

    def test_linearity(self, cdgrid):
        A = make_helmholtz_op(
            coeff=0.5, cdgrid=cdgrid,
            allow_nonsymmetric_for_testing=True,
        )
        n = cdgrid.base.n
        np.random.seed(2)
        x = jnp.asarray(np.random.randn(6, n, n))
        y = jnp.asarray(np.random.randn(6, n, n))
        Ax = A(x)
        Ay = A(y)
        Axy = A(x + 3.0 * y)
        residual = float(jnp.max(jnp.abs(Axy - (Ax + 3.0 * Ay))))
        assert residual < 1.0e-10


# ----------------------------------------------------------------------
# Richardson solver — Phase 2 production backend
# ----------------------------------------------------------------------


def _cdgrid_grid():
    return create_cubed_sphere(8)


def _rel_residual(sol, rhs, coeff, grid):
    """``‖rhs − (I − coeff · ∇²) sol‖ / ‖rhs‖`` for verification."""
    from legoesm.core.operators import laplacian_compact
    Asol = sol - coeff * laplacian_compact(sol, grid)
    return float(
        jnp.linalg.norm((rhs - Asol).ravel())
        / jnp.linalg.norm(rhs.ravel())
    )


class TestRichardsonSolve:
    def test_identity_at_zero_coeff(self):
        """``A = I − 0 · ∇² = I``; ``return_residual`` reports
        machine-zero residual on the warm start."""
        grid = _cdgrid_grid()
        np.random.seed(0)
        rhs = jnp.asarray(np.random.randn(6, 8, 8))
        sol, rel_res = richardson_helmholtz_solve(
            rhs, coeff=0.0, grid=grid,
            n_iter_max=10, tol=1.0e-10, return_residual=True,
        )
        assert float(jnp.max(jnp.abs(sol - rhs))) < 1.0e-10
        assert float(rel_res) < 1.0e-10

    # Supported production range: α dt / dx² ≤ 5 reaches tol=1e-6
    # within ~200 iters; beyond that the solver returns un-
    # converged and the production wrapper falls back to *no
    # damping* for that step (``p_s`` left unchanged) and
    # surfaces a ``RuntimeWarning``.  Tests below assert tol is
    # reached in the supported range and reported-vs-verified
    # residual agreement.
    @pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 5.0])
    def test_residual_meets_production_tolerance(self, ratio):
        """For every supported ``α dt / dx²`` ratio the
        Richardson-with-residual-stop backend returns a solution
        whose *externally verified* relative residual is below the
        configured production tolerance."""
        grid = _cdgrid_grid()
        np.random.seed(int(100 * ratio))
        rhs = jnp.asarray(np.random.randn(6, 8, 8))
        dx_min = 6.371e6 * (np.pi / 2) / 8 / np.sqrt(3)
        coeff = ratio * dx_min ** 2
        tol = 1.0e-6
        sol, rel_res_traced = richardson_helmholtz_solve(
            rhs, coeff=coeff, grid=grid,
            n_iter_max=200, tol=tol, return_residual=True,
        )
        reported = float(rel_res_traced)
        verified = _rel_residual(sol, rhs, coeff, grid)
        # The externally computed residual MUST be below tol — no
        # silent under-convergence.
        assert verified < tol, (
            f"α dt / dx²={ratio}: external residual {verified:.3e} > "
            f"tol {tol:.0e} — solver under-converged"
        )
        # Reported ≈ external within numeric noise (no
        # best-iterate drift).
        assert reported <= 2.0 * verified + 1.0e-12, (
            f"α dt / dx²={ratio}: reported={reported:.3e} disagrees "
            f"with verified={verified:.3e}"
        )

    def test_returns_unconverged_beyond_supported_range(self):
        """At α dt / dx² ≫ 5 the solver hits ``n_iter_max`` and
        returns an un-converged ``rel_res > tol``; production
        wrapper falls back, so the solver itself must surface
        the un-converged status faithfully (not lie about
        convergence)."""
        grid = _cdgrid_grid()
        np.random.seed(123)
        rhs = jnp.asarray(np.random.randn(6, 8, 8))
        dx_min = 6.371e6 * (np.pi / 2) / 8 / np.sqrt(3)
        coeff = 50.0 * dx_min ** 2
        tol = 1.0e-6
        sol, rel_res = richardson_helmholtz_solve(
            rhs, coeff=coeff, grid=grid,
            n_iter_max=60, tol=tol, return_residual=True,
        )
        # Verified residual exceeds tol — solver correctly reports
        # un-converged status.
        verified = _rel_residual(sol, rhs, coeff, grid)
        assert float(rel_res) > tol
        assert verified > tol
        # Reported is consistent with verified (no silent drift).
        assert abs(float(rel_res) - verified) < 2.0 * verified + 1.0e-12

    def test_step_fv3_falls_back_safely_when_unconverged(self):
        """Production wrapper contract: when Richardson can't reach
        ``tol`` at the configured ``n_iter_max``, ``_step_fv3``
        leaves ``p_s`` unchanged for that step (no-damping
        fallback) and emits a ``RuntimeWarning``.  Covers the
        end-to-end production-path behaviour Codex flagged."""
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig,
            CDGridPrimitiveEquationModel,
            FV3HydrostaticState,
        )
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        nlev = 6
        sigma = create_sigma_coordinate(nlev)
        n = 8
        np.random.seed(0)
        p_s = 1.0e5 + 100.0 * np.random.randn(6, n, n)
        state = FV3HydrostaticState(
            u_d=Field(data=jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d", dims=()),
            v_d=Field(data=jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d", dims=()),
            T=Field(data=jnp.full((6, n, n, nlev), 250.0), name="T", dims=()),
            p_s=Field(data=jnp.asarray(p_s), name="p_s", dims=()),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=()),
        )

        # α dt / dx² = 50 → Richardson reports un-converged at
        # tol=1e-6 within 200 iters; ``_step_fv3`` must take the
        # safe-fallback branch (leave p_s unchanged) and warn.
        dt = 100.0
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        alpha = 50.0 * dx_min ** 2 / dt
        cfg_pcg = CDGridPrimitiveEquationConfig(
            implicit_grav_wave_damping=alpha,
            implicit_grav_wave_use_pcg=True,
            sponge_tau_sec=-1.0, damp_v=0.0, hyperdiff_coeff=0.0,
            time_integrator="ssp_rk3",
        )
        # Reference run with no damping at all.  Production no-
        # damping fallback should produce a state matching this
        # one (within numeric tolerance) on the p_s field.
        cfg_nodamp = cfg_pcg._replace(implicit_grav_wave_damping=0.0)

        model_pcg = CDGridPrimitiveEquationModel(grid, sigma, cfg_pcg)
        model_nodamp = CDGridPrimitiveEquationModel(grid, sigma, cfg_nodamp)

        import warnings
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            s_pcg = model_pcg._step_fv3(state, dt)
            # Force the host callback queue to drain so any
            # ``jax.debug.callback`` from the un-converged branch
            # is visible to the recorder.
            jax.block_until_ready(s_pcg.p_s.data)
        s_nodamp = model_nodamp._step_fv3(state, dt)

        # Production no-damping fallback path must match the
        # baseline "no damping at all" reference run bit-for-bit
        # (both leave ``p_s_damped == state_new.p_s.data`` for
        # the implicit branch; the explicit branch is skipped
        # entirely when alpha=0).
        assert float(
            jnp.max(jnp.abs(s_pcg.p_s.data - s_nodamp.p_s.data))
        ) < 1.0e-10, (
            "Production no-damping fallback produced p_s that does "
            "not match the reference no-damping run"
        )

        # ``RuntimeWarning`` from the un-converged branch must
        # actually fire — otherwise the host-visibility contract
        # silently regressed.
        warn_msgs = [
            str(w.message)
            for w in captured
            if issubclass(w.category, RuntimeWarning)
            and "implicit_grav_wave" in str(w.message)
        ]
        assert len(warn_msgs) >= 1, (
            f"expected a RuntimeWarning from the Richardson un-"
            f"converged branch; captured warnings: {captured}"
        )

    def test_jit_friendly_coeff_change(self):
        """JIT'd wrapper with traced ``coeff`` (a) deterministic
        outputs on identical inputs, (b) two different coeffs
        produce different solutions, (c) both verified residuals
        below tol."""
        grid = _cdgrid_grid()
        dx_min = 6.371e6 * (np.pi / 2) / 8 / np.sqrt(3)
        np.random.seed(7)
        rhs = jnp.asarray(np.random.randn(6, 8, 8))
        tol = 1.0e-6

        @jax.jit
        def solve(rhs, coeff):
            sol, rel = richardson_helmholtz_solve(
                rhs, coeff=coeff, grid=grid,
                n_iter_max=200, tol=tol, return_residual=True,
            )
            return sol, rel

        coeff_a = jnp.asarray(1.0 * dx_min ** 2)
        coeff_b = jnp.asarray(5.0 * dx_min ** 2)
        sol_a, _rel_a = solve(rhs, coeff_a)
        sol_a2, _rel_a2 = solve(rhs, coeff_a)
        sol_b, _rel_b = solve(rhs, coeff_b)

        # Determinism: identical inputs → identical outputs.
        assert float(jnp.max(jnp.abs(sol_a - sol_a2))) < 1.0e-12
        # Different coeff produces different solution.
        assert float(jnp.max(jnp.abs(sol_a - sol_b))) > 1.0e-6
        # External residuals must meet tol (not just reported).
        verified_a = _rel_residual(sol_a, rhs, float(coeff_a), grid)
        verified_b = _rel_residual(sol_b, rhs, float(coeff_b), grid)
        assert verified_a < tol, (
            f"JIT path α dt / dx²=1: verified residual "
            f"{verified_a:.3e} > tol {tol:.0e}"
        )
        assert verified_b < tol, (
            f"JIT path α dt / dx²=5: verified residual "
            f"{verified_b:.3e} > tol {tol:.0e}"
        )
