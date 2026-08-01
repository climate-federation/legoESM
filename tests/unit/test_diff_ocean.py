"""Differentiability tests for ocean dynamics and EOS.

Categories:
  3a) Ocean dynamics — cubed-sphere
  3b) Ocean dynamics — lat-lon (incl. the tracer-advection scheme family)
  3c) Ocean dynamics — MPAS
  3d) Equation of state
  3e) Barotropic solver
  3f) Ocean + physics (vertical mixing, bottom drag)

All tests run under ``JAX_ENABLE_X64=1``.  The float32-specific
limiter/FCT gradient-underflow bug class is a SEPARATE axis covered by
``tests/ocean/unit/test_advection_grad_underflow.py`` (issues #1387 /
#1388); the scheme sweep here is the float64 complement — it pins that
each advection option is AD-connected at all, independent of the
float32 NaN-poisoning question.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


# ============================================================================
# 3a  Ocean dynamics — cubed-sphere
# ============================================================================

class TestCubedSphereOcean:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
        )
        self.model = OceanModel(grid, z_coord, config)
        self.state = rest_state_ocean(grid, z_coord)
        self.dt = 300.0

    def test_grad_wrt_T(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, dt)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "CS Ocean single step w.r.t. T")

    def test_grad_wrt_S(self):
        """Salinity is the second prognostic tracer and enters the EOS.

        T alone can look healthy while the S branch of the density /
        pressure-gradient chain is dead, so it needs its own probe.
        """
        model, state, dt = self.model, self.state, self.dt

        def loss(S_data):
            s = state._replace(S=state.S.replace(data=S_data))
            out = model.step(s, dt)
            return jnp.sum(out.S.data ** 2)

        grad = jax.grad(loss)(state.S.data)
        assert_gradient_ok(grad, "CS Ocean single step w.r.t. S")

    def test_grad_S_reaches_T_through_eos_multistep(self):
        """Cross-tracer coupling: a salinity perturbation must reach the
        temperature field through density -> pressure gradient -> flow ->
        advection.  A zero here means the baroclinic EOS chain is
        AD-severed even though each tracer looks fine on its own.

        This needs at least TWO steps, and the reason is the
        time-stepping order, not a defect: the tracer update advects with
        the START-of-step velocities, so within a single step a salinity
        perturbation can only reach the momentum field, never ``T``.
        ``d(T)/dS`` after one step is therefore EXACTLY zero by
        construction (measured: all-zero fp32 gradient) — asserting
        non-zero at one step would be a false alarm.  From two steps on,
        the S-induced velocity increment has had a chance to advect T,
        which is the coupling this test is actually about.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,  # lax.scan barotropic for AD
        )
        model = OceanModel(grid, z_coord, config)
        state = rest_state_ocean(grid, z_coord)
        dt = 300.0

        def loss(S_data):
            s = state._replace(S=state.S.replace(data=S_data))

            def body(carry, _):
                return model.step(carry, dt), None

            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T.data ** 2)

        grad = jax.grad(loss)(state.S.data)
        assert jnp.all(jnp.isfinite(grad)), "dT/dS gradient has NaN/Inf"
        assert jnp.any(grad != 0.0), (
            "d(sum T^2)/dS is identically zero after 3 steps — the salinity "
            "-> EOS -> pressure-gradient -> advection chain is not "
            "AD-connected"
        )


# ============================================================================
# 3b  Ocean dynamics — lat-lon C-grid
# ============================================================================

class TestLatLonOcean:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )

        nlev = 3
        grid = create_latlon_grid(n_lat=8, n_lon=16)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = LatLonCGridOceanConfig.from_flat(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,  # use lax.scan for AD
        )
        self.model = LatLonCGridOceanModel(grid, z_coord, config)
        # land_lat_threshold beyond the grid's max |lat| keeps the whole
        # domain wet (no land cells) so the gradient is dense.
        self.state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, land_lat_threshold=90.0
        )
        self.dt = 300.0

    def test_grad_wrt_T(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, dt)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "LatLon Ocean single step w.r.t. T")

    def test_grad_3_steps(self):
        """Multi-step gradient accumulation through the lat-lon C-grid model."""
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))

            def body(carry, _):
                # Call _step_impl inside scan to avoid nested-JIT boundary.
                return model._step_impl(carry, dt), None

            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "LatLon Ocean 3 steps w.r.t. T")

    def test_grad_wrt_u_momentum(self):
        """Momentum is the other half of the prognostic state; the tracer
        probes above never touch the u-face branch of the adjoint."""
        model, state, dt = self.model, self.state, self.dt
        key = jax.random.PRNGKey(0)
        u0 = 0.05 * jax.random.normal(key, state.u.data.shape)

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            out = model.step(s, dt)
            return jnp.sum(out.u.data ** 2)

        grad = jax.grad(loss)(u0)
        assert jnp.all(jnp.isfinite(grad)), "LatLon Ocean du/du gradient NaN/Inf"
        assert jnp.any(grad != 0.0), "LatLon Ocean du/du gradient identically zero"


# ============================================================================
# 3b-bis  Tracer-advection scheme family — every option must be AD-connected
# ============================================================================

# The full dispatch list from ``LatLonCGridOceanConfig.tracer_advection``.
# "som" is excluded: it carries extra prognostic moment fields (T_som /
# S_som) that ``rest_state_latlon_cgrid_ocean`` only populates when the
# scheme is selected at construction, so it needs a different fixture and
# is not a like-for-like member of this sweep.
_TRACER_ADVECTION_SCHEMES = [
    "upwind", "centered", "tvd", "superbee", "ppm_fct", "fct2",
    "ppm", "dst3", "dst3_multidim", "weno5", "weno7",
]


@pytest.mark.parametrize("scheme", _TRACER_ADVECTION_SCHEMES)
def test_tracer_advection_scheme_grad_x64(scheme):
    """Each tracer-advection option must give finite, non-zero gradients
    in float64 on a NEAR-UNIFORM tracer field.

    Near-uniform is the adversarial case on purpose: it is the ordinary
    ocean state (uniform initial salinity, a mixed layer) and it is what
    drives every limiter/FCT ratio denominator toward its epsilon floor.
    Issues #1387 / #1388 showed that family NaN-poisoning *float32* AD;
    this is the float64 complement, so a regression that widened the
    defect to double precision — where the model's own conservation
    tests and the DA/4D-Var stack live — would be caught here.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    nlev = 3
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(nlev, H_max=500.0)
    config = LatLonCGridOceanConfig.from_flat(
        use_conservation_fixer=False,
        enable_runtime_checks=False,
        n_barotropic_substeps=2,
        differentiable_barotropic=True,
        tracer_advection=scheme,
    )
    model = LatLonCGridOceanModel(grid, z_coord, config)
    state = rest_state_latlon_cgrid_ocean(grid, z_coord, land_lat_threshold=90.0)

    # Near-uniform salinity (the limiter-denominator stressor) plus a
    # non-zero flow so the advective fluxes are actually exercised.
    key = jax.random.PRNGKey(5)
    S0 = 35.0 + 1e-9 * jax.random.normal(key, state.S.data.shape)
    u0 = 0.05 * jnp.ones_like(state.u.data) * state.u_mask.data[..., None]
    state = state._replace(u=state.u.replace(data=u0))

    def loss(S_data):
        s = state._replace(S=state.S.replace(data=S_data))
        out = model.step(s, 300.0)
        return jnp.sum(out.S.data ** 2)

    try:
        grad = jax.grad(loss)(S0)
        assert jnp.all(jnp.isfinite(grad)), (
            f"tracer_advection={scheme}: gradient has NaN/Inf on a "
            f"near-uniform float64 tracer (limiter/FCT ratio-denominator "
            f"underflow)"
        )
        assert jnp.any(grad != 0.0), (
            f"tracer_advection={scheme}: gradient is identically zero — the "
            f"scheme is not AD-connected to the tracer state"
        )
    finally:
        # Each scheme compiles its OWN full ocean-model graph.  Eleven of
        # them accumulate enough live XLA executables that a LATER test in
        # the same process aborts inside
        # ``jax._src.compiler.backend_compile_and_load`` (SIGABRT, core
        # dumped) — reproduced identically at 48 GB and 96 GB, so it is a
        # compiled-executable accumulation limit in the backend, NOT memory
        # pressure and NOT a gradient defect (every one of these tests
        # passes when its class is run alone).  Releasing the executables
        # after each scheme keeps the whole file runnable in one process.
        jax.clear_caches()


# ============================================================================
# 3c  Ocean dynamics — MPAS
# ============================================================================

class TestMPASOcean:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

        nlev = 3
        mesh = create_voronoi_mesh(2)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        self.model = MPASOceanModel(mesh, z_coord)
        self.state = rest_state_mpas_ocean(mesh, z_coord)
        self.dt = 300.0

    def test_grad_wrt_T(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, dt)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "MPAS Ocean single step w.r.t. T")

    def test_grad_3_steps(self):
        """Multi-step accumulation on the unstructured mesh.

        The MPAS edge/cell gather-scatter indexing is the exact op class
        that silently drops adjoint contributions, and a single step can
        hide that (a one-step gradient is dominated by the local term).
        """
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))

            def body(carry, _):
                return model.step(carry, dt), None

            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "MPAS Ocean 3 steps w.r.t. T")


# ============================================================================
# 3d  Equation of state
# ============================================================================

class TestEOS:

    def test_wright_eos_grad_T(self):
        from legoesm.ocean.eos import wright_eos

        T = jnp.array(20.0)
        S = jnp.array(35.0)
        p = jnp.array(0.0)

        drho_dT = jax.grad(lambda T: wright_eos(T, S, p))(T)
        assert jnp.isfinite(drho_dT), "drho/dT is not finite"
        # Warmer water is lighter (thermal expansion)
        assert drho_dT < 0, f"drho/dT should be negative, got {drho_dT}"

    def test_wright_eos_grad_S(self):
        from legoesm.ocean.eos import wright_eos

        T = jnp.array(20.0)
        S = jnp.array(35.0)
        p = jnp.array(0.0)

        drho_dS = jax.grad(lambda S: wright_eos(T, S, p))(S)
        assert jnp.isfinite(drho_dS), "drho/dS is not finite"
        # Saltier water is heavier (haline contraction)
        assert drho_dS > 0, f"drho/dS should be positive, got {drho_dS}"

    def test_wright_eos_grad_p(self):
        """Compressibility: deeper (higher pressure) water is denser."""
        from legoesm.ocean.eos import wright_eos

        T, S, p = jnp.array(20.0), jnp.array(35.0), jnp.array(2.0e7)
        drho_dp = jax.grad(lambda p: wright_eos(T, S, p))(p)
        assert jnp.isfinite(drho_dp), "drho/dp is not finite"
        assert drho_dp > 0, f"drho/dp should be positive, got {drho_dp}"

    # ``has_salinity=False`` marks an EOS that is salinity-INDEPENDENT by
    # construction (documented, not a defect): Veros ``eq_of_state_type=4``
    # (nonlin3) is quadratic in T with no S and no p term, so a zero
    # d(rho)/dS is the CORRECT answer there and asserting otherwise would
    # be a false alarm.
    @pytest.mark.parametrize(
        "eos_name,has_salinity",
        [
            ("wright", True),
            ("linear", True),
            ("nemo_seos", True),
            ("nemo_roquet", True),
            ("unesco80", True),
            ("veros_nonlin2", True),
            ("veros_nonlin3", False),
        ],
    )
    def test_eos_family_grad_signs(self, eos_name, has_salinity):
        """EVERY selectable EOS must be differentiable with the right
        thermodynamic signs, not just the default.

        A recipe / oracle-fidelity run selects a different EOS than the
        default, and a broken adjoint there would only surface as a
        silently-wrong 4D-Var increment.  Signs are the physical
        acceptance criterion: warm -> lighter, salty -> heavier.
        """
        from legoesm.ocean import eos as eos_mod

        fn = {
            "wright": eos_mod.wright_eos,
            "linear": eos_mod.linear_eos,
            "nemo_seos": eos_mod.nemo_seos_eos,
            "nemo_roquet": eos_mod.nemo_roquet_eos,
            "unesco80": eos_mod.unesco80_eos,
            "veros_nonlin2": eos_mod.veros_nonlin2_eos,
            "veros_nonlin3": eos_mod.veros_nonlin3_eos,
        }[eos_name]

        T, S, p = jnp.array(15.0), jnp.array(35.0), jnp.array(1.0e6)
        drho_dT, drho_dS = jax.grad(lambda T, S: fn(T, S, p),
                                    argnums=(0, 1))(T, S)
        assert jnp.isfinite(drho_dT) and jnp.isfinite(drho_dS), (
            f"{eos_name}: EOS gradient not finite "
            f"(drho/dT={drho_dT}, drho/dS={drho_dS})"
        )
        assert drho_dT < 0, (
            f"{eos_name}: drho/dT must be negative (thermal expansion), "
            f"got {drho_dT}"
        )
        if has_salinity:
            assert drho_dS > 0, (
                f"{eos_name}: drho/dS must be positive (haline contraction), "
                f"got {drho_dS}"
            )
        else:
            # NON-DIFFERENTIABLE (by construction, documented): nonlin3 has
            # no salinity term at all.
            assert drho_dS == 0.0, (
                f"{eos_name} is documented as salinity-independent but "
                f"d(rho)/dS = {drho_dS} != 0 — the parametrization table "
                f"in this test is stale relative to the EOS"
            )

    def test_eos_second_derivative_finite(self):
        """The adjoint of the pressure-gradient force differentiates the
        EOS twice (density -> pressure -> tendency), so d2rho/dT2 must be
        finite for the model adjoint to be usable, not just d(rho)/dT."""
        from legoesm.ocean.eos import wright_eos

        T, S, p = jnp.array(20.0), jnp.array(35.0), jnp.array(1.0e6)
        d2rho_dT2 = jax.grad(jax.grad(lambda T: wright_eos(T, S, p)))(T)
        assert jnp.isfinite(d2rho_dT2), (
            f"d2rho/dT2 not finite ({d2rho_dT2}) — second-order adjoint "
            f"of the pressure-gradient force is unusable"
        )
        assert d2rho_dT2 != 0.0, (
            "d2rho/dT2 is exactly zero — the EOS has been linearized in T "
            "(cabbeling / thermobaricity are absent from the adjoint)"
        )


# ============================================================================
# 3e  Barotropic solver differentiability
# ============================================================================

class TestBarotropicSolver:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,  # use lax.scan for AD
        )
        self.model = OceanModel(grid, z_coord, config)
        self.state = rest_state_ocean(grid, z_coord)
        self.dt = 300.0

    def test_grad_wrt_eta(self):
        """Gradient w.r.t. sea surface height through barotropic solver."""
        model, state, dt = self.model, self.state, self.dt

        def loss(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            out = model.step(s, dt)
            return jnp.sum(out.eta.data ** 2)

        grad = jax.grad(loss)(state.eta.data)
        assert_gradient_ok(grad, "Barotropic solver w.r.t. eta")

    def test_grad_3_steps(self):
        """Multi-step gradient through ocean model with differentiable barotropic.

        Regression for the ``cast_pytree(..., "storage")`` round-trip:
        ``OceanModel.step`` upcasts to compute precision on entry and
        used to leave the output at compute precision because the
        storage cast defaulted to ``allow_downcast=False``.  Under
        ``JAX_ENABLE_X64`` that turned fp32 inputs into fp64 outputs and
        ``jax.lax.scan`` rejected the carry-dtype mismatch.  The fix
        passes ``allow_downcast=True`` for the storage cast so the
        output dtype matches the input dtype.
        """
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Ocean 3 steps (diff barotropic) w.r.t. T")


class TestImplicitFreeSurfaceSolveGrad:
    """Gradient through the implicit free-surface Helmholtz solve itself.

    ``solve_helmholtz_implicit`` is the shared entry point for the
    implicit Crank-Nicolson barotropic step and it dispatches on a
    STATIC ``distributed`` flag to two structurally different solvers:

    * ``distributed=False`` -> ``jax.scipy.sparse.linalg.cg``, whose own
      ``custom_linear_solve`` supplies the exact implicit-function
      adjoint;
    * ``distributed=True``  -> an UNROLLED fixed-iteration PCG that
      reverse-mode AD differentiates straight through (deliberately NOT
      ``custom_linear_solve``, because the MPI halo ``_sendrecv_vjp``
      has no transpose rule).

    The module docstring PROMISES ``jax.grad`` works on both.  Because
    they solve the same SPD system, their gradients must also AGREE — an
    adjoint bug in either path shows up as a mismatch, which a
    finiteness check alone would miss.  Exercised here on a synthetic
    SPD operator (1-D Laplacian + diagonal shift) so no grid or MPI
    machinery is needed and the test runs in milliseconds.
    """

    @staticmethod
    def _make_problem(n=32):
        # A = shift*I - Laplacian (symmetric positive definite, well
        # conditioned), Jacobi preconditioner = 1/diag(A).
        shift = 4.0

        def A_op(x):
            lap = (jnp.roll(x, 1) - 2.0 * x + jnp.roll(x, -1))
            return shift * x - lap

        diag = shift + 2.0
        return A_op, (lambda r: r / diag), n

    def _solve(self, rhs, *, distributed):
        from legoesm.ocean.dynamics.barotropic_common import (
            solve_helmholtz_implicit,
        )
        A_op, M_inv, n = self._make_problem(rhs.shape[0])
        x, _diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, jnp.zeros_like(rhs),
            distributed=distributed,
            fixed_iters=60,
            residual_tol=1e-10,
            stock_cg_tol=1e-12,
            stock_cg_maxiter=200,
        )
        return x

    @pytest.mark.parametrize("distributed", [False, True])
    def test_grad_finite_both_solver_paths(self, distributed):
        key = jax.random.PRNGKey(4)
        rhs = jax.random.normal(key, (32,))

        def loss(r):
            return jnp.sum(self._solve(r, distributed=distributed) ** 2)

        grad = jax.grad(loss)(rhs)
        assert jnp.all(jnp.isfinite(grad)), (
            f"implicit free-surface solve (distributed={distributed}): "
            f"gradient has NaN/Inf"
        )
        assert jnp.any(grad != 0.0), (
            f"implicit free-surface solve (distributed={distributed}): "
            f"gradient identically zero"
        )

    def test_grad_agrees_across_solver_paths(self):
        """Both dispatch branches solve the SAME system, so their
        adjoints must agree; a divergence means one of the two is
        wrong (and only one of them is used under MPI)."""
        key = jax.random.PRNGKey(4)
        rhs = jax.random.normal(key, (32,))

        g_stock = jax.grad(
            lambda r: jnp.sum(self._solve(r, distributed=False) ** 2))(rhs)
        g_pcg = jax.grad(
            lambda r: jnp.sum(self._solve(r, distributed=True) ** 2))(rhs)

        denom = jnp.maximum(jnp.linalg.norm(g_stock), 1e-30)
        rel = jnp.linalg.norm(g_pcg - g_stock) / denom
        assert float(rel) < 1e-6, (
            f"implicit free-surface adjoints disagree between the stock-CG "
            f"(custom_linear_solve) and unrolled fixed-iteration PCG paths: "
            f"relative difference {float(rel):.3e}"
        )


# ============================================================================
# 3f  Ocean with physics (vertical mixing)
# ============================================================================

class TestOceanPhysics:

    def test_grad_with_vertical_mixing(self):
        """Gradient through ocean step with vertical mixing enabled."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            K_v=1e-4,  # Vertical diffusivity enabled
            A_v=1e-3,  # Vertical viscosity enabled
        )
        model = OceanModel(grid, z_coord, config)
        state = rest_state_ocean(grid, z_coord)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Ocean + vertical mixing w.r.t. T")

    def test_grad_with_bottom_drag(self):
        """Bottom drag is an extra momentum sink in the tendency; enabling
        it must not sever the tracer adjoint.

        ``bottom_drag_r`` itself is a STATIC feature gate
        (``if config.bottom_drag_r > 0`` — the sanctioned CLAUDE.md
        Python-branch-on-static-bool exception), so what is checked here
        is that the drag BRANCH, once compiled in, is differentiable —
        not that the flag itself is a trainable parameter.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,
            bottom_drag_r=2.5e-3,
        )
        model = OceanModel(grid, z_coord, config)
        state = rest_state_ocean(grid, z_coord)
        key = jax.random.PRNGKey(2)
        u0 = 0.05 * jax.random.normal(key, state.u.data.shape)
        state = state._replace(u=state.u.replace(data=u0))

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.u.data ** 2)

        grad = jax.grad(loss)(u0)
        assert jnp.all(jnp.isfinite(grad)), "Ocean + bottom drag: grad NaN/Inf"
        assert jnp.any(grad != 0.0), "Ocean + bottom drag: grad identically zero"

    def test_grad_with_kpp_vertical_mixing(self):
        """KPP is the production vertical-mixing closure.

        Its boundary-layer-depth diagnostic uses a smooth (sigmoid)
        crossing-depth selector precisely so it stays differentiable; a
        NaN here would mean that smoothing has regressed to a hard
        argmax/searchsorted and every ocean adjoint through the mixed
        layer is dead.
        """
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig, KPPConfig,
        )

        nlev = 4
        grid = create_latlon_grid(n_lat=8, n_lon=16)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="kpp", kpp=KPPConfig(),
            ),
            # The physics lateral-mixing factory is cubed-sphere-only; the
            # lat-lon C-grid supplies viscosity through its own
            # ``lateral_viscosity`` block and REJECTS a non-"none" scheme
            # here (a real dispatch guard, not a gradient issue).
            lateral_mixing=LateralMixingConfig(scheme="none"),
        )
        config = LatLonCGridOceanConfig.from_flat(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,
            implicit_vertical_mixing=True,
            physics=physics,
        )
        model = LatLonCGridOceanModel(grid, z_coord, config)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, land_lat_threshold=90.0
        )

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Ocean + KPP vertical mixing w.r.t. T")
