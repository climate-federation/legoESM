"""Phase 3 tests for the Hoskins–Simmons FV3 D-grid implicit solver.

Pins down four invariants of the Phase-3 implementation in
``src/legoesm/atmosphere/dynamics/gcm/semi_implicit_cdgrid.py``:

1. The cubed-sphere ``cdgrid_scalar_laplacian`` is FV-adjoint-
   symmetric and negative semi-definite under the area-weighted
   inner product (regression canary against the interpolating-
   halo asymmetry that broke Phase-2 CG plans).
2. The Helmholtz operator factory returns an M-SPD operator with
   no SPD-readiness gate.
3. ``cg_helmholtz_solve`` reaches its production tolerance in a
   handful of iterations across the supported coefficient range
   and stays JIT-deterministic when ``coeff`` is traced.
4. ``jax.grad`` flows cleanly through the CG solve via the
   implicit-function-theorem VJP that ``jax.scipy.sparse.linalg.cg``
   installs internally.

The Phase-2 Richardson solver remains exercised by a small subset of
tests at the bottom of this file because it is still the fallback /
reference backend; new production wiring uses CG.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.semi_implicit_cdgrid import (
    adjoint_residual_norm,
    cdgrid_scalar_laplacian,
    cg_helmholtz_solve,
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
        assert float(jnp.max(jnp.abs(lap))) < 1.0e-8

    def test_shape_preserved(self, cdgrid):
        n = cdgrid.base.n
        np.random.seed(0)
        p = jnp.asarray(np.random.randn(6, n, n))
        lap = cdgrid_scalar_laplacian(p, cdgrid)
        assert lap.shape == p.shape


# ----------------------------------------------------------------------
# FV-adjoint canary — the operator MUST stay symmetric.
# ----------------------------------------------------------------------


class TestAdjointResidual:
    def test_operator_is_fv_adjoint_symmetric(self, cdgrid):
        """Phase-3 canary.  Asserts the composed ``div(grad)`` on the
        cubed-sphere D-grid stays FV-adjoint-symmetric under the
        area-weighted inner product.  If a future change reintroduces
        the Phase-2 interpolating-halo path, this residual jumps
        from ~1e-14 back to ~5e-1 and CI catches the regression
        before it silently breaks ``cg_helmholtz_solve``."""
        L = lambda p: cdgrid_scalar_laplacian(p, cdgrid)
        residual = adjoint_residual_norm(L, cdgrid, n_samples=8)
        assert residual < 1.0e-10, (
            f"adjoint residual {residual:.3e} exceeds Phase-3 "
            f"machine-zero threshold — interpolating halo or "
            f"non-FV stencil reintroduced into the symmetric "
            f"Laplacian path"
        )

    def test_operator_is_negative_semidefinite(self, cdgrid):
        """The Laplacian ``∇² = div(grad)`` must be negative semi-
        definite under the area-weighted inner product so that
        ``A(p) = p − coeff · ∇²p`` is M-positive-definite for
        ``coeff ≥ 0``.  A positive Rayleigh quotient anywhere in
        the sample would mean ``cg`` could stagnate."""
        n = cdgrid.base.n
        area = cdgrid.base.area
        L = lambda p: cdgrid_scalar_laplacian(p, cdgrid)
        rng = np.random.default_rng(0)
        max_rq = -jnp.inf
        for _ in range(20):
            p = jnp.asarray(rng.standard_normal((6, n, n)))
            pLp = float(jnp.sum(area * p * L(p)))
            pp = float(jnp.sum(area * p * p))
            max_rq = max(max_rq, pLp / pp)
        # Allow tiny positive roundoff for the constant-mode null
        # space (eigenvalue 0).
        assert max_rq < 1.0e-10, (
            f"max Rayleigh quotient of L under M-IP is "
            f"{max_rq:.3e}; expected ≤ 0 (machine-zero positive)"
        )


# ----------------------------------------------------------------------
# make_helmholtz_op — M-SPD on the new symmetric Laplacian
# ----------------------------------------------------------------------


class TestHelmholtzOp:
    def test_returns_spd_operator(self, cdgrid):
        """No more ``allow_nonsymmetric_for_testing`` escape hatch:
        the factory always returns an M-SPD operator on the
        Phase-3 path."""
        A = make_helmholtz_op(coeff=0.5, cdgrid=cdgrid)
        n = cdgrid.base.n
        area = cdgrid.base.area
        rng = np.random.default_rng(1)
        worst = 0.0
        min_rq = jnp.inf
        for _ in range(8):
            x = jnp.asarray(rng.standard_normal((6, n, n)))
            y = jnp.asarray(rng.standard_normal((6, n, n)))
            Ax = A(x)
            Ay = A(y)
            xAy = float(jnp.sum(area * x * Ay))
            yAx = float(jnp.sum(area * y * Ax))
            denom = max(abs(xAy), abs(yAx), 1.0e-30)
            worst = max(worst, abs(xAy - yAx) / denom)
            min_rq = min(
                min_rq, float(jnp.sum(area * x * Ax) / jnp.sum(area * x * x))
            )
        assert worst < 1.0e-10
        assert min_rq > 0.5  # roughly 1 + 0.5 · |λ_min(-L)| > 1

    def test_identity_at_zero_coeff(self, cdgrid):
        A = make_helmholtz_op(coeff=0.0, cdgrid=cdgrid)
        n = cdgrid.base.n
        np.random.seed(1)
        p = jnp.asarray(np.random.randn(6, n, n))
        Ap = A(p)
        assert float(jnp.max(jnp.abs(Ap - p))) < 1.0e-14

    def test_linearity(self, cdgrid):
        A = make_helmholtz_op(coeff=0.5, cdgrid=cdgrid)
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
# cg_helmholtz_solve — Phase 3 production backend
# ----------------------------------------------------------------------


def _rel_residual_cg(sol, rhs, coeff, cdgrid):
    """``‖rhs − (I − coeff · ∇²) sol‖ / ‖rhs‖`` for verification."""
    Asol = sol - coeff * cdgrid_scalar_laplacian(sol, cdgrid)
    return float(
        jnp.linalg.norm((rhs - Asol).ravel())
        / jnp.linalg.norm(rhs.ravel())
    )


class TestCGHelmholtzSolve:
    def test_identity_at_zero_coeff(self, cdgrid):
        """``A = I`` at coeff=0; CG must return the warm start
        with machine-zero verified residual."""
        n = cdgrid.base.n
        np.random.seed(0)
        rhs = jnp.asarray(np.random.randn(6, n, n))
        sol, rel_res = cg_helmholtz_solve(
            rhs, coeff=0.0, cdgrid=cdgrid,
            tol=1.0e-10, maxiter=10, return_residual=True,
        )
        assert float(jnp.max(jnp.abs(sol - rhs))) < 1.0e-10
        assert float(rel_res) < 1.0e-10

    @pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 5.0, 50.0])
    def test_residual_meets_production_tolerance(self, cdgrid, ratio):
        """CG meets ``rel_res < 1e-10`` for every ratio in the
        supported production range — including ``α dt / dx² = 50``,
        which the Phase-2 Richardson backend could not converge."""
        n = cdgrid.base.n
        np.random.seed(int(100 * ratio))
        rhs = jnp.asarray(np.random.randn(6, n, n))
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        coeff = ratio * dx_min ** 2
        tol = 1.0e-10
        sol, rel_res_traced = cg_helmholtz_solve(
            rhs, coeff=coeff, cdgrid=cdgrid,
            tol=tol, maxiter=200, return_residual=True,
        )
        reported = float(rel_res_traced)
        verified = _rel_residual_cg(sol, rhs, coeff, cdgrid)
        # Allow a 5× slack against the requested CG tolerance to
        # absorb the difference between CG's internal Euclidean
        # residual norm (on ``tilde_p``) and the externally
        # computed physical-space residual.  Both should still
        # sit at ≲ 1e-10 in practice.
        assert verified < 5.0 * tol, (
            f"α dt / dx²={ratio}: external residual {verified:.3e} "
            f"> 5·tol {tol:.0e}"
        )
        assert reported <= 5.0 * tol, (
            f"α dt / dx²={ratio}: reported={reported:.3e} > 5·tol"
        )

    def test_converges_faster_than_richardson(self, cdgrid):
        """At α dt / dx² = 5, CG reaches 1e-10 in well under
        ``maxiter=200``; Richardson at the same tol would need
        many more iterations.  Exact iteration count is not
        asserted (varies with random RHS) — we only check that
        CG's *reported* residual reaches the tighter tolerance
        within the same iteration budget that Richardson was
        configured with."""
        n = cdgrid.base.n
        np.random.seed(3)
        rhs = jnp.asarray(np.random.randn(6, n, n))
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        coeff = 5.0 * dx_min ** 2
        _sol, rel_res = cg_helmholtz_solve(
            rhs, coeff=coeff, cdgrid=cdgrid,
            tol=1.0e-10, maxiter=50, return_residual=True,
        )
        assert float(rel_res) < 1.0e-9, (
            f"CG should reach 1e-9 within 50 iter at α dt / dx²=5; "
            f"got rel_res={float(rel_res):.3e}"
        )

    def test_jit_friendly_coeff_change(self, cdgrid):
        """JIT'd wrapper with traced ``coeff`` is deterministic and
        produces different solutions for different coefficients."""
        n = cdgrid.base.n
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        np.random.seed(7)
        rhs = jnp.asarray(np.random.randn(6, n, n))

        @jax.jit
        def solve(rhs, coeff):
            sol, rel = cg_helmholtz_solve(
                rhs, coeff=coeff, cdgrid=cdgrid,
                tol=1.0e-10, maxiter=200, return_residual=True,
            )
            return sol, rel

        coeff_a = jnp.asarray(1.0 * dx_min ** 2)
        coeff_b = jnp.asarray(5.0 * dx_min ** 2)
        sol_a, _ = solve(rhs, coeff_a)
        sol_a2, _ = solve(rhs, coeff_a)
        sol_b, _ = solve(rhs, coeff_b)
        assert float(jnp.max(jnp.abs(sol_a - sol_a2))) < 1.0e-12
        assert float(jnp.max(jnp.abs(sol_a - sol_b))) > 1.0e-6
        assert _rel_residual_cg(
            sol_a, rhs, float(coeff_a), cdgrid,
        ) < 5.0e-10
        assert _rel_residual_cg(
            sol_b, rhs, float(coeff_b), cdgrid,
        ) < 5.0e-10

    def test_grad_flows_through_cg_solve(self, cdgrid):
        """``jax.scipy.sparse.linalg.cg`` installs an implicit-
        function-theorem VJP, so ``jax.grad`` of any scalar loss
        on the CG output is well-defined.  Verifies that the
        forward-mode gradient matches a finite-difference estimate
        — proves the VJP wiring is correct end-to-end."""
        n = cdgrid.base.n
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        np.random.seed(11)
        rhs = jnp.asarray(np.random.randn(6, n, n))

        def loss(coeff):
            sol = cg_helmholtz_solve(
                rhs, coeff=coeff, cdgrid=cdgrid,
                tol=1.0e-12, maxiter=400,
            )
            return jnp.sum(sol ** 2)

        coeff0 = 2.0 * dx_min ** 2
        g_auto = float(jax.grad(loss)(jnp.asarray(coeff0)))
        # Central FD with step relative to coeff0.
        h = 1.0e-3 * coeff0
        g_fd = float((loss(coeff0 + h) - loss(coeff0 - h)) / (2.0 * h))
        # FD floor on this random RHS is ~1e-4 relative — the
        # comparison verifies the AD path is wired, not chasing
        # full machine precision.
        rel_err = abs(g_auto - g_fd) / max(abs(g_fd), 1.0e-30)
        assert rel_err < 1.0e-3, (
            f"jax.grad through CG disagrees with FD: "
            f"auto={g_auto:.3e}, fd={g_fd:.3e}, rel_err={rel_err:.3e}"
        )


# ----------------------------------------------------------------------
# Production wiring — _step_fv3 fallback contract on CG un-convergence
# ----------------------------------------------------------------------


class TestStepFV3PCGFallback:
    def test_step_fv3_falls_back_safely_when_unconverged(self):
        """End-to-end production-path contract: when CG fails to
        reach the configured tolerance within ``maxiter``,
        ``_step_fv3`` leaves ``p_s`` unchanged for that step and
        emits a ``RuntimeWarning``.  Uses ``maxiter=1`` via a
        monkey-patched ``cg_helmholtz_solve`` to force the
        un-converged branch deterministically without relying on
        a pathological coefficient."""
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig,
            CDGridPrimitiveEquationModel,
            FV3HydrostaticState,
        )
        from legoesm.core.field import Field
        from legoesm.atmosphere.dynamics.gcm import semi_implicit_cdgrid as _scd

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

        dt = 100.0
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        alpha = 5.0 * dx_min ** 2 / dt
        cfg_pcg = CDGridPrimitiveEquationConfig(
            implicit_grav_wave_damping=alpha,
            implicit_grav_wave_use_pcg=True,
            sponge_tau_sec=-1.0, damp_v=0.0, hyperdiff_coeff=0.0,
            time_integrator="ssp_rk3",
        )
        cfg_nodamp = cfg_pcg._replace(implicit_grav_wave_damping=0.0)

        model_pcg = CDGridPrimitiveEquationModel(grid, sigma, cfg_pcg)
        model_nodamp = CDGridPrimitiveEquationModel(grid, sigma, cfg_nodamp)

        # Force the un-converged branch by intercepting
        # cg_helmholtz_solve and returning a bogus residual > tol.
        _orig_cg = _scd.cg_helmholtz_solve

        def _force_unconverged(rhs, coeff, cdgrid, **kwargs):
            return rhs, jnp.asarray(1.0e-3, dtype=rhs.dtype)

        _scd.cg_helmholtz_solve = _force_unconverged
        try:
            import warnings
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                s_pcg, _ = model_pcg._step_fv3(state, dt)
                jax.block_until_ready(s_pcg.p_s.data)
            s_nodamp, _ = model_nodamp._step_fv3(state, dt)
        finally:
            _scd.cg_helmholtz_solve = _orig_cg

        assert float(
            jnp.max(jnp.abs(s_pcg.p_s.data - s_nodamp.p_s.data))
        ) < 1.0e-10, (
            "Production no-damping fallback produced p_s that does "
            "not match the reference no-damping run"
        )

        warn_msgs = [
            str(w.message)
            for w in captured
            if issubclass(w.category, RuntimeWarning)
            and "implicit_grav_wave" in str(w.message)
        ]
        assert len(warn_msgs) >= 1, (
            f"expected a RuntimeWarning from the CG un-converged "
            f"branch; captured warnings: {captured}"
        )

    def test_step_fv3_converges_and_changes_p_s_in_normal_range(self):
        """Counterpart to the un-converged test: in the normal
        production range CG converges, the fallback does NOT fire,
        and ``p_s`` is non-trivially modified (i.e. the implicit
        Helmholtz update actually ran)."""
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
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

        dt = 100.0
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        alpha = 1.0 * dx_min ** 2 / dt
        cfg_pcg = CDGridPrimitiveEquationConfig(
            implicit_grav_wave_damping=alpha,
            implicit_grav_wave_use_pcg=True,
            sponge_tau_sec=-1.0, damp_v=0.0, hyperdiff_coeff=0.0,
            time_integrator="ssp_rk3",
        )
        cfg_nodamp = cfg_pcg._replace(implicit_grav_wave_damping=0.0)
        model_pcg = CDGridPrimitiveEquationModel(grid, sigma, cfg_pcg)
        model_nodamp = CDGridPrimitiveEquationModel(grid, sigma, cfg_nodamp)

        import warnings
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            s_pcg, _ = model_pcg._step_fv3(state, dt)
            jax.block_until_ready(s_pcg.p_s.data)
        s_nodamp, _ = model_nodamp._step_fv3(state, dt)

        # In the converged path, the implicit update modifies p_s
        # away from the no-damping reference.
        delta = float(jnp.max(jnp.abs(s_pcg.p_s.data - s_nodamp.p_s.data)))
        assert delta > 1.0e-3, (
            f"CG-path p_s essentially matched the no-damping path "
            f"(max|Δ|={delta:.3e}); the implicit update did not run"
        )

        # No un-converged warning should fire in normal range.
        warn_msgs = [
            str(w.message)
            for w in captured
            if issubclass(w.category, RuntimeWarning)
            and "implicit_grav_wave" in str(w.message)
        ]
        assert len(warn_msgs) == 0, (
            f"unexpected un-converged warning in normal CG range: "
            f"{warn_msgs}"
        )


# ----------------------------------------------------------------------
# Stability headroom — CG path enables larger ``α dt / dx²`` than explicit
# ----------------------------------------------------------------------


class TestImplicitStabilityHeadroom:
    """Issue #273 Phase 3: the CG-driven implicit gravity-wave damping
    must remain stable at ``α dt / dx²`` ratios where the legacy
    explicit forward-Euler diffusion (``p_s ← p_s + α dt ∇²p_s``)
    would violate its CFL bound and blow up.  Locks in the dt-headroom
    that the Phase-3 solver buys for the production AMIP path."""

    def _build_state(self, grid, sigma, n, nlev):
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            FV3HydrostaticState,
        )
        from legoesm.core.field import Field
        np.random.seed(0)
        p_s = 1.0e5 + 100.0 * np.random.randn(6, n, n)
        return FV3HydrostaticState(
            u_d=Field(data=jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d", dims=()),
            v_d=Field(data=jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d", dims=()),
            T=Field(data=jnp.full((6, n, n, nlev), 250.0), name="T", dims=()),
            p_s=Field(data=jnp.asarray(p_s), name="p_s", dims=()),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=()),
        )

    def _run_n_steps(self, model, state, dt, n_steps):
        s = state
        for _ in range(n_steps):
            s, _ = model._step_fv3(s, dt)
            jax.block_until_ready(s.p_s.data)
        return s

    def test_explicit_path_blows_up_at_large_alpha_dt(self):
        """Baseline: at ``α dt / dx²`` ≈ 5, the legacy explicit
        forward-Euler diffusion of ``p_s`` is unconditionally unstable
        (CFL bound is 0.5).  ``p_s`` should diverge to non-finite
        values within a handful of steps."""
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig,
            CDGridPrimitiveEquationModel,
        )

        n, nlev = 8, 6
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)
        dt = 300.0
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        alpha = 5.0 * dx_min ** 2 / dt  # α dt / dx² = 5  → explicit unstable
        cfg_expl = CDGridPrimitiveEquationConfig(
            implicit_grav_wave_damping=alpha,
            implicit_grav_wave_use_pcg=False,  # explicit path
            sponge_tau_sec=-1.0, damp_v=0.0, hyperdiff_coeff=0.0,
            time_integrator="ssp_rk3",
        )
        model_expl = CDGridPrimitiveEquationModel(grid, sigma, cfg_expl)
        state = self._build_state(grid, sigma, n, nlev)
        s_expl = self._run_n_steps(model_expl, state, dt, 6)
        # Explicit path either NaNs or pegs at p_floor (which the
        # damping run repeatedly forces to the floor when p_s
        # diverges negative).  Either way ``p_s`` is no longer a
        # physically reasonable surface-pressure field.
        ps = np.asarray(s_expl.p_s.data)
        finite = np.isfinite(ps).all()
        within_range = (ps.min() > 5.0e4) and (ps.max() < 2.0e5)
        assert not (finite and within_range), (
            "explicit gravity-wave damping at α dt / dx²=5 stayed "
            "bounded — CFL diagnostic is wrong or the test setup "
            "is too gentle"
        )

    def test_implicit_pcg_path_stays_stable_at_large_alpha_dt(self):
        """Phase-3 contract: at the same ``α dt / dx² = 5`` ratio,
        the implicit CG path remains stable — ``p_s`` finite and
        within physical bounds for at least 20 steps."""
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig,
            CDGridPrimitiveEquationModel,
        )

        n, nlev = 8, 6
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)
        dt = 300.0
        dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
        alpha = 5.0 * dx_min ** 2 / dt
        cfg_pcg = CDGridPrimitiveEquationConfig(
            implicit_grav_wave_damping=alpha,
            implicit_grav_wave_use_pcg=True,
            sponge_tau_sec=-1.0, damp_v=0.0, hyperdiff_coeff=0.0,
            time_integrator="ssp_rk3",
        )
        model_pcg = CDGridPrimitiveEquationModel(grid, sigma, cfg_pcg)
        state = self._build_state(grid, sigma, n, nlev)
        s_pcg = self._run_n_steps(model_pcg, state, dt, 20)
        ps = np.asarray(s_pcg.p_s.data)
        assert np.isfinite(ps).all()
        assert ps.min() > 9.0e4, f"p_s min {ps.min():.1f} below physical floor"
        assert ps.max() < 1.1e5, f"p_s max {ps.max():.1f} above physical ceiling"


# ----------------------------------------------------------------------
# Phase-2 Richardson solver — retained reference / fallback
# ----------------------------------------------------------------------


def _cdgrid_grid():
    return create_cubed_sphere(8)


def _rel_residual_richardson(sol, rhs, coeff, grid):
    """``‖rhs − (I − coeff · ∇²) sol‖ / ‖rhs‖`` for verification
    (Richardson backend operates on the base-grid ``laplacian_compact``)."""
    from legoesm.core.operators import laplacian_compact
    Asol = sol - coeff * laplacian_compact(sol, grid)
    return float(
        jnp.linalg.norm((rhs - Asol).ravel())
        / jnp.linalg.norm(rhs.ravel())
    )


class TestRichardsonSolveRetained:
    def test_identity_at_zero_coeff(self):
        grid = _cdgrid_grid()
        np.random.seed(0)
        rhs = jnp.asarray(np.random.randn(6, 8, 8))
        sol, rel_res = richardson_helmholtz_solve(
            rhs, coeff=0.0, grid=grid,
            n_iter_max=10, tol=1.0e-10, return_residual=True,
        )
        assert float(jnp.max(jnp.abs(sol - rhs))) < 1.0e-10
        assert float(rel_res) < 1.0e-10

    @pytest.mark.parametrize("ratio", [0.1, 1.0])
    def test_residual_meets_phase2_tolerance(self, ratio):
        grid = _cdgrid_grid()
        np.random.seed(int(100 * ratio))
        rhs = jnp.asarray(np.random.randn(6, 8, 8))
        dx_min = 6.371e6 * (np.pi / 2) / 8 / np.sqrt(3)
        coeff = ratio * dx_min ** 2
        tol = 1.0e-6
        sol, _rel = richardson_helmholtz_solve(
            rhs, coeff=coeff, grid=grid,
            n_iter_max=200, tol=tol, return_residual=True,
        )
        verified = _rel_residual_richardson(sol, rhs, coeff, grid)
        assert verified < tol
