"""Differentiability tests for atmosphere dynamical cores.

Tests that jax.grad produces finite, non-zero, spatially structured gradients
through single and multi-step integration of every dynamical core.

Categories:
  1a) Shallow water — all grids (CD-grid, lat-lon, spectral, MPAS)
  1b) Hydrostatic PE — cubed-sphere and lat-lon
  1c) Nonhydrostatic compressible Euler — cubed-sphere
  1d) Spectral PE
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    """Check that a gradient array is finite, non-trivially non-zero, and has
    spatial structure (not all identical)."""
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )
    assert not jnp.all(grad_array == grad_array.ravel()[0]), (
        f"{name}: gradient is spatially uniform (no structure)"
    )


def assert_ad_correct(loss, xs, name="", jit=True):
    """Gradient of ``loss`` w.r.t. the tuple ``xs`` (temperature and wind of
    the initial state) matches finite differences and is eager == jit.

    * FD: ``jax.test_util.check_grads`` reverse mode, order 1, eps=1e-4 in
      input units, atol=rtol=1e-5 (JAX's float64 defaults, stated).  Needs
      ``JAX_ENABLE_X64=1``; skipped in float32.
    * jit parity: ``jit(grad)`` vs eager ``grad``, rtol 1e-9 plus an
      absolute floor of 1e-12 x max|g| (XLA may reassociate sums; anything
      above round-off is a tracer bug).
    Side decisions voted (Claude + codex + GLM) 2026-11-06.
    """
    from jax.test_util import check_grads

    if not jax.config.jax_enable_x64:
        pytest.skip("FD gradient check needs JAX_ENABLE_X64=1")
    g_eager = jax.grad(loss)(xs)
    for i, g in enumerate(g_eager):
        assert_gradient_ok(g, f"{name} [input {i}]")
    if jit:
        g_jit = jax.jit(jax.grad(loss))(xs)
        for i, (gj, ge) in enumerate(zip(g_jit, g_eager)):
            scale = float(jnp.max(jnp.abs(ge)))
            assert jnp.allclose(gj, ge, rtol=1e-9, atol=1e-12 * scale), (
                f"{name} [input {i}]: jit vs eager gradient max rel diff "
                f"{float(jnp.max(jnp.abs(gj - ge))) / scale:.3e}")
    check_grads(loss, (xs,), order=1, modes=("rev",), eps=1e-4,
                atol=1e-5, rtol=1e-5)


# ============================================================================
# 1a  Shallow Water — CD-grid (cubed-sphere C4)
# ============================================================================

class TestCDGridShallowWater:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel, CDGridShallowWaterConfig,
            CDGridShallowWaterState,
        )
        n = 4
        base = create_cubed_sphere(n)
        config = CDGridShallowWaterConfig()
        self.model = CDGridShallowWaterModel(base, config)
        # Initial state: rest + Gaussian perturbation in h
        h0 = 1000.0 * jnp.ones((6, n, n))
        key = jax.random.PRNGKey(0)
        h0 = h0 + 10.0 * jax.random.normal(key, (6, n, n))
        u_d = jnp.zeros((6, n + 1, n + 1))
        v_d = jnp.zeros((6, n + 1, n + 1))
        h_s = jnp.zeros((6, n, n))
        self.state = CDGridShallowWaterState(h=h0, u_d=u_d, v_d=v_d, h_s=h_s)
        self.dt = 60.0
        self.n = n

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_init):
            s = state._replace(h=h_init)
            out = model.step(s, dt)
            return jnp.sum(out.h ** 2)

        grad = jax.grad(loss)(state.h)
        assert_gradient_ok(grad, "CDGrid SW single step")

    def test_grad_5_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_init):
            s = state._replace(h=h_init)
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=5)
            return jnp.sum(s_final.h ** 2)

        grad = jax.grad(loss)(state.h)
        assert_gradient_ok(grad, "CDGrid SW 5 steps")


# ============================================================================
# 1a  Shallow Water — lat-lon C-grid (8x16)
# ============================================================================

class TestLatLonShallowWater:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
            CGridLatLonShallowWaterModel, CGridLatLonShallowWaterConfig,
            williamson_test2_cgrid,
        )
        n_lat, n_lon = 8, 16
        grid = create_latlon_grid(n_lat, n_lon)
        # Disable mass fixer: the anchored fixer snapshots a *static* fp64
        # target on the first (un-traced) step, captured as a constant
        # inside jax.grad — keep the loss an honest function of h.
        config = CGridLatLonShallowWaterConfig(fix_mass=False)
        self.model = CGridLatLonShallowWaterModel(grid, config)
        self.dt = 60.0
        # Williamson-2 balanced state + Gaussian perturbation in h.
        # The grid/initial state are built at the policy storage dtype
        # (fp32 under the default policy); ``step`` casts to compute
        # (fp64 with x64) and back, so for a multi-step ``lax.scan`` the
        # initial carry must already carry the dtype ``step`` returns or
        # the scan carry-type check fails.  Promote the whole state.
        state = williamson_test2_cgrid(grid)
        key = jax.random.PRNGKey(1)
        h0 = state.h + 5.0 * jax.random.normal(key, state.h.shape)
        state = state._replace(h=h0)
        self.state = jax.tree_util.tree_map(
            lambda a: a.astype(jnp.float64), state,
        )

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_data):
            s = state._replace(h=h_data)
            out = model.step(s, dt)
            return jnp.sum(out.h ** 2)

        grad = jax.grad(loss)(state.h)
        assert_gradient_ok(grad, "LatLon SW single step")

    def test_grad_5_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_data):
            s = state._replace(h=h_data)
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=5)
            return jnp.sum(s_final.h ** 2)

        grad = jax.grad(loss)(state.h)
        assert_gradient_ok(grad, "LatLon SW 5 steps")


# ============================================================================
# 1a  Shallow Water — spectral (T5)
# ============================================================================

class TestSpectralShallowWater:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
            SpectralShallowWaterModel, SpectralSWConfig, SpectralSWState,
        )

        grid = create_gaussian_grid(5, allow_unsupported_backend=True)
        config = SpectralSWConfig()
        self.model = SpectralShallowWaterModel(grid, config, allow_unsupported_backend=True)
        self.dt = 120.0

        # Build a spectral state: phi_hat = g * H0 (constant mean-depth field)
        n_sh = grid.n_sh
        g = constants.g
        H0 = 5960.0
        phi_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
        phi_hat = phi_hat.at[0].set(g * H0 * jnp.sqrt(4 * jnp.pi))  # n=0,m=0
        key = jax.random.PRNGKey(2)
        phi_hat = phi_hat + 1.0 * jax.random.normal(key, (n_sh,))
        self.state = SpectralSWState(
            vor_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="vor_hat"),
            div_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="div_hat"),
            phi_hat=Field(phi_hat, name="phi_hat"),
            phis_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="phis_hat"),
        )
        self.n_sh = n_sh

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(phi_real):
            phi_hat_new = state.phi_hat.data.real * 0 + phi_real + 1j * state.phi_hat.data.imag
            s = state._replace(phi_hat=state.phi_hat.replace(data=phi_hat_new))
            out = model.step(s, dt)
            return jnp.sum(jnp.abs(out.phi_hat.data) ** 2)

        grad = jax.grad(loss)(state.phi_hat.data.real)
        assert_gradient_ok(grad, "Spectral SW single step")

    def test_grad_5_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(phi_real):
            phi_hat_new = phi_real + 1j * state.phi_hat.data.imag
            s = state._replace(phi_hat=state.phi_hat.replace(data=phi_hat_new))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=5)
            return jnp.sum(jnp.abs(s_final.phi_hat.data) ** 2)

        grad = jax.grad(loss)(state.phi_hat.data.real)
        assert_gradient_ok(grad, "Spectral SW 5 steps")


# ============================================================================
# 1a  Shallow Water — MPAS (level-2 Voronoi)
# ============================================================================

class TestMPASShallowWater:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
            MPASShallowWaterModel, MPASShallowWaterConfig,
        )
        from legoesm.core.state import MPASShallowWaterState

        mesh = create_voronoi_mesh(2)
        config = MPASShallowWaterConfig()
        self.model = MPASShallowWaterModel(mesh, config)
        self.dt = 120.0

        key = jax.random.PRNGKey(3)
        h_data = 1000.0 * jnp.ones(mesh.nCells) + 10.0 * jax.random.normal(key, (mesh.nCells,))
        self.state = MPASShallowWaterState(
            h=Field(h_data, name="h"),
            u=Field(jnp.zeros(mesh.nEdges), name="u"),
            h_s=Field(jnp.zeros(mesh.nCells), name="h_s"),
        )

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            out = model.step(s, dt)
            return jnp.sum(out.h.data ** 2)

        grad = jax.grad(loss)(state.h.data)
        assert_gradient_ok(grad, "MPAS SW single step")

    def test_grad_5_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=5)
            return jnp.sum(s_final.h.data ** 2)

        grad = jax.grad(loss)(state.h.data)
        assert_gradient_ok(grad, "MPAS SW 5 steps")


# ============================================================================
# 1b  Hydrostatic PE — CD-grid (C4, 5 levels)
# ============================================================================

class TestCDGridPE:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
        )
        from legoesm.core.state import FV3HydrostaticState

        n = 4
        nlev = 5
        base = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)
        self.model = CDGridPrimitiveEquationModel(base, sigma)
        self.dt = 60.0

        # Isothermal at-rest state
        T_data = 250.0 * jnp.ones((6, n, n, nlev))
        key = jax.random.PRNGKey(10)
        T_data = T_data + 1.0 * jax.random.normal(key, T_data.shape)
        self.state = FV3HydrostaticState(
            u_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d"),
            v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
        )
        self.n = n
        self.nlev = nlev

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, dt)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "CDGrid PE single step")

    def test_grad_3_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "CDGrid PE 3 steps")

    def test_ad_fd_and_jit_3_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(xs):
            T_data, u_data = xs
            s = state._replace(T=state.T.replace(data=T_data),
                               u_d=state.u_d.replace(data=u_data))
            for _ in range(3):
                s = model.step(s, dt)
            return (jnp.mean((s.T.data - 250.0) ** 2)
                    + jnp.mean(s.u_d.data ** 2))

        assert_ad_correct(loss, (state.T.data, state.u_d.data),
                          "CDGrid PE 3 steps")


# ============================================================================
# 1b  Hydrostatic PE — lat-lon C-grid (8x16, 5 levels)
# ============================================================================

class TestLatLonPE:

    @pytest.fixture(autouse=True)
    def setup(self):
        self._build()

    def _build(self):
        import math
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel,
            CGridLatLonPrimitiveEquationConfig,
            hydrostatic_to_cgrid,
        )
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon

        n_lat, n_lon, nlev = 8, 16, 5
        grid = create_latlon_grid(n_lat, n_lon)
        sigma = create_sigma_coordinate(nlev)
        dx_pole = float(grid.radius) * grid.dlon * math.cos(
            math.pi / 2 - grid.dlat / 2
        )
        self.dt = min(120.0, 0.5 * dx_pole / 300.0)
        # fix_mass off for AD honesty (anchored fixer captures a static
        # target that would otherwise distort gradients).
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=False)
        self.model = CGridLatLonPrimitiveEquationModel(
            grid, sigma, config, dt=self.dt,
        )
        # Held-Suarez initial condition on the native C-grid layout
        # (T has horizontal + vertical structure) — raw arrays.
        self.state = hydrostatic_to_cgrid(
            held_suarez_init_latlon(grid, sigma), grid,
        )

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=T_data)
            out = model.step(s, dt)
            return jnp.sum(out.T ** 2)

        grad = jax.grad(loss)(state.T)
        assert_gradient_ok(grad, "LatLon PE single step")

    def test_grad_3_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=T_data)
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T ** 2)

        grad = jax.grad(loss)(state.T)
        assert_gradient_ok(grad, "LatLon PE 3 steps")

    def test_ad_fd_and_jit_3_steps(self):
        # The lat-lon lane stores in the precision policy's storage dtype
        # (float32 by default) and re-casts at every step boundary, so the
        # FD check needs an fp64 policy for the whole build + rollout.
        from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

        old = get_policy()
        set_policy(PrecisionPolicy.fp64())
        try:
            self._build()
            self._ad_check()
        finally:
            set_policy(old)

    def _ad_check(self):
        model, state, dt = self.model, self.state, self.dt
        assert state.T.dtype == jnp.float64

        def loss(xs):
            s = state._replace(T=xs[0], u=xs[1])
            for _ in range(3):
                s = model.step(s, dt)
            return (jnp.mean((s.T - jnp.mean(state.T)) ** 2)
                    + jnp.mean(s.u ** 2))

        assert_ad_correct(loss, (state.T, state.u), "LatLon PE 3 steps")


# ============================================================================
# 1c  Nonhydrostatic Compressible Euler — CD-grid (C4, 5 levels)
# ============================================================================

class TestCompressibleEuler:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric
        from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
            CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
        )
        from legoesm.core.state import NonHydrostaticState

        n = 4
        nlev = 5
        H = 30000.0  # 30 km model top
        base = create_cubed_sphere(n)
        height = create_height_coordinate(nlev, H)
        z_s = jnp.zeros((6, n, n))
        terrain = compute_terrain_metric(z_s, height)
        config = CDGridCompressibleEulerConfig(use_coriolis=False)
        self.model = CDGridCompressibleEulerModel(base, height, terrain, config)
        self.dt = 1.0  # small dt for compressible

        key = jax.random.PRNGKey(20)
        theta_prime = 0.5 * jax.random.normal(key, (6, n, n, nlev))
        self.state = NonHydrostaticState(
            u=Field(jnp.zeros((6, n, n, nlev)), name="u"),
            v=Field(jnp.zeros((6, n, n, nlev)), name="v"),
            w=Field(jnp.zeros((6, n, n, nlev + 1)), name="w"),
            theta_prime=Field(theta_prime, name="theta_prime"),
            rho_prime=Field(jnp.zeros((6, n, n, nlev)), name="rho_prime"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
            tracers=Field(jnp.zeros((6, n, n, nlev, 0)), name="tracers"),
        )

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(theta_data):
            s = state._replace(theta_prime=state.theta_prime.replace(data=theta_data))
            out = model.step(s, dt)
            return jnp.sum(out.theta_prime.data ** 2)

        grad = jax.grad(loss)(state.theta_prime.data)
        assert_gradient_ok(grad, "CompEuler single step")

    def test_grad_3_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(theta_data):
            s = state._replace(
                theta_prime=state.theta_prime.replace(data=theta_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.theta_prime.data ** 2)

        grad = jax.grad(loss)(state.theta_prime.data)
        assert_gradient_ok(grad, "CompEuler 3 steps")


# ============================================================================
# 1d  Spectral PE (T5, 5 levels)
# ============================================================================

class TestSpectralPE:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
            SpectralHydrostaticState,
        )

        nlev = 5
        grid = create_gaussian_grid(5, allow_unsupported_backend=True)
        sigma = create_sigma_coordinate(nlev)
        config = SpectralPEConfig()
        self.model = SpectralPrimitiveEquationModel(
            grid, sigma, config, allow_unsupported_backend=True,
        )
        self.dt = 120.0
        n_sh = grid.n_sh

        key = jax.random.PRNGKey(30)
        T_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
        # Set mean temperature ~ 250 K
        T_hat = T_hat.at[0, :].set(250.0 * jnp.sqrt(4 * jnp.pi))
        T_hat = T_hat + 0.1 * jax.random.normal(key, (n_sh, nlev))
        # log(p_s) ~ log(1e5) ≈ 11.51
        lnps_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
        lnps_hat = lnps_hat.at[0].set(jnp.log(1e5) * jnp.sqrt(4 * jnp.pi))

        self.state = SpectralHydrostaticState(
            vor_hat=Field(jnp.zeros((n_sh, nlev), dtype=jnp.complex128), name="vor_hat"),
            div_hat=Field(jnp.zeros((n_sh, nlev), dtype=jnp.complex128), name="div_hat"),
            T_hat=Field(T_hat, name="T_hat"),
            lnps_hat=Field(lnps_hat, name="lnps_hat"),
            phis_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="phis_hat"),
        )
        self.n_sh = n_sh

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_real):
            T_hat_new = T_real + 1j * state.T_hat.data.imag
            s = state._replace(T_hat=state.T_hat.replace(data=T_hat_new))
            out = model.step(s, dt)
            return jnp.sum(jnp.abs(out.T_hat.data) ** 2)

        grad = jax.grad(loss)(state.T_hat.data.real)
        assert_gradient_ok(grad, "Spectral PE single step")

    def test_grad_3_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_real):
            T_hat_new = T_real + 1j * state.T_hat.data.imag
            s = state._replace(T_hat=state.T_hat.replace(data=T_hat_new))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(jnp.abs(s_final.T_hat.data) ** 2)

        grad = jax.grad(loss)(state.T_hat.data.real)
        assert_gradient_ok(grad, "Spectral PE 3 steps")

    def test_ad_fd_and_jit_3_steps(self):
        model, state, dt = self.model, self.state, self.dt

        # Vorticity enters in units of 1e-5 (its physical scale), so the FD
        # step check_grads takes (1e-4 in input units) is 1e-9 in vor_hat;
        # unscaled, 1e-4 is ten times the flow and the central difference
        # is truncation-dominated (FD/AD = 1.15 at 1e-4, 1.0015 at 1e-5,
        # 1.000000 at 1e-7: AD right, measured 2026-11-06).
        vor_unit = 1e-5

        def loss(xs):
            T_real, vor_scaled = xs
            s = state._replace(
                T_hat=state.T_hat.replace(
                    data=T_real + 1j * state.T_hat.data.imag),
                vor_hat=state.vor_hat.replace(
                    data=vor_unit * vor_scaled + 1j * state.vor_hat.data.imag))
            for _ in range(3):
                s = model.step(s, dt)
            # anomaly (drop the global-mean mode) + vorticity
            return (jnp.mean(jnp.abs(s.T_hat.data[1:]) ** 2)
                    + jnp.mean(jnp.abs(s.vor_hat.data) ** 2))

        assert_ad_correct(
            loss, (state.T_hat.data.real, state.vor_hat.data.real / vor_unit),
            "Spectral PE 3 steps")


# ============================================================================
# 1e  Hydrostatic PE — MPAS / icosahedral Voronoi (level 2 = 162 cells)
# ============================================================================

class TestMPASPE:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel,
        )
        from legoesm.core.state import MPASHydrostaticState

        nlev = 5
        mesh = create_voronoi_mesh(2, lloyd_iterations=5)
        sigma = create_sigma_coordinate(nlev)
        # Same config / dt as tests/unit/test_mpas_atmosphere.py.
        config = MPASPrimitiveEquationConfig(nu_del2=1e4, K_h=1e3)
        self.model = MPASPrimitiveEquationModel(mesh, sigma, config)
        self.dt = 60.0
        k1, k2 = jax.random.split(jax.random.PRNGKey(40))
        nc, ne = mesh.nCells, mesh.nEdges
        self.state = MPASHydrostaticState(
            u=Field(1.0 * jax.random.normal(k1, (ne, nlev)), name="u"),
            T=Field(250.0 + jax.random.normal(k2, (nc, nlev)), name="T"),
            p_s=Field(1e5 * jnp.ones(nc), name="p_s"),
            phis=Field(jnp.zeros(nc), name="phis"),
        )

    def _loss(self, n_steps):
        model, state, dt = self.model, self.state, self.dt

        def loss(xs):
            s = state._replace(T=state.T.replace(data=xs[0]),
                               u=state.u.replace(data=xs[1]))
            for _ in range(n_steps):
                s = model.step(s, dt)
            return (jnp.mean((s.T.data - 250.0) ** 2)
                    + jnp.mean(s.u.data ** 2))
        return loss

    def _xs(self):
        return (self.state.T.data, self.state.u.data)

    def test_grad_single_step(self):
        for i, g in enumerate(jax.grad(self._loss(1))(self._xs())):
            assert_gradient_ok(g, f"MPAS PE single step [input {i}]")

    def test_ad_fd_and_jit_3_steps(self):
        assert_ad_correct(self._loss(3), self._xs(), "MPAS PE 3 steps")


# ============================================================================
# 1f  Hydrostatic PE — FV3 six-face duo cube (C12, km=5)
# ============================================================================

class TestFV3DuoPE:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoConfig, FV3DuoDynamicsModel)
        from legoesm.grids.factory import create_fv3_duo_grid

        if not jax.config.jax_enable_x64:
            pytest.skip("FV3 duo lane is fp64-only")
        # Same grid / config / dt as tests/unit/test_dycore_energy_fixer.py.
        self.model = FV3DuoDynamicsModel(create_fv3_duo_grid(12, 3),
                                         FV3DuoConfig(km=5, n_split=2))
        self.bundle = self.model.dcmip16_initial_state(n_tracers=1)
        self.dt = 120.0

    def _loss(self, n_steps):
        model, bundle, dt = self.model, self.bundle, self.dt

        def loss(xs):
            b = {**bundle,
                 "state": {**bundle["state"], "pt": xs[0], "u": xs[1]}}
            for _ in range(n_steps):
                b = model.step(b, dt)
            pt0 = jnp.mean(bundle["state"]["pt"])
            return (jnp.mean((b["state"]["pt"] - pt0) ** 2)
                    + jnp.mean(b["state"]["u"] ** 2))
        return loss

    def _xs(self):
        return (self.bundle["state"]["pt"], self.bundle["state"]["u"])

    def test_ad_fd_eager_3_steps(self):
        # Temperature only.  The wind path runs FV3's monotone transport
        # limiters, which are piecewise: FD/AD on u stays within 0.5% but
        # wanders 1e-4..5e-3 without converging as the step shrinks
        # (1e-1..1e-7, 1 and 3 steps), so a 1e-5 check cannot certify it.
        # Recorded as a finding, not loosened here.
        bundle = self.bundle
        u0 = bundle["state"]["u"]
        full = self._loss(3)
        assert_ad_correct(lambda xs: full((xs[0], u0)),
                          (bundle["state"]["pt"],), "FV3 duo 3 steps",
                          jit=False)

    @pytest.mark.xfail(
        strict=True, raises=jax.errors.TracerArrayConversionError,
        reason="FV3DuoDynamicsModel.step reads nsplt on the host every call "
               "(the C5 fail-closed check), so it cannot be traced under jit; "
               "fix design (pure step + explicit flag vs checkify) pending.")
    def test_jit_grad_3_steps(self):
        jax.jit(jax.grad(self._loss(3)))(self._xs())
