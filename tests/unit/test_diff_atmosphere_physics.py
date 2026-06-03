"""Differentiability tests for atmosphere physics parameterizations.

Tests that jax.grad produces finite, non-zero gradients through each
physics scheme independently and through the combined physics pipeline.

Categories:
  2a) Held-Suarez forcing
  2b) Gray radiation
  2c) Convection schemes (sbm, dca, kuo)
  2d) Turbulence schemes (smagorinsky, louis)
  2e) Microphysics (kessler, sundqvist)
  2f) Combined physics (make_physics)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


def make_hydrostatic_state(n, nlev, key=None):
    """Create a minimal C-n hydrostatic state with realistic values."""
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    T_data = 250.0 * jnp.ones((6, n, n, nlev)) + 2.0 * jax.random.normal(k1, (6, n, n, nlev))
    u_data = 10.0 * jax.random.normal(k2, (6, n, n, nlev))
    v_data = 3.0 * jax.random.normal(k3, (6, n, n, nlev))
    q_v_data = 1e-3 * jnp.ones((6, n, n, nlev))
    q_c_data = 1e-5 * jnp.ones((6, n, n, nlev))
    q_r_data = 1e-6 * jnp.ones((6, n, n, nlev))
    return HydrostaticState(
        u=Field(u_data, name="u"),
        v=Field(v_data, name="v"),
        T=Field(T_data, name="T"),
        p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
        phis=Field(jnp.zeros((6, n, n)), name="phis"),
        tracers={
            "q_v": Field(q_v_data, name="q_v"),
            "q_c": Field(q_c_data, name="q_c"),
            "q_r": Field(q_r_data, name="q_r"),
        },
    )


# ============================================================================
# 2a  Held-Suarez forcing
# ============================================================================

class TestHeldSuarezGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

    def test_grad_wrt_T(self):
        from legoesm.atmosphere.held_suarez import held_suarez_forcing
        grid, sigma, state = self.grid, self.sigma, self.state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend = held_suarez_forcing(s, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Held-Suarez dT_dt w.r.t. T")


# ============================================================================
# 2b  Gray radiation
# ============================================================================

class TestGrayRadiationGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        self.gray_radiation = gray_radiation
        self.config = GrayRadiationConfig()
        ncol, nlev = 32, 5
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        self.T = 250.0 + 20.0 * jax.random.normal(k1, (ncol, nlev))
        # Pressure decreasing with height
        p_half = jnp.linspace(1e5, 100.0, nlev + 1)
        self.p_half = jnp.broadcast_to(p_half, (ncol, nlev + 1))
        self.p_full = 0.5 * (self.p_half[:, :-1] + self.p_half[:, 1:])
        self.sfc_temp = 290.0 * jnp.ones(ncol)
        self.lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, ncol)
        self.q_v = 1e-3 * jnp.ones((ncol, nlev))
        self.insolation = 400.0 * jnp.ones(ncol)

    def test_grad_wrt_sfc_temp(self):
        """Gray LW fluxes depend on surface temperature."""
        config = self.config

        def loss(sfc_temp):
            out = self.gray_radiation(self.T, self.p_full, self.p_half,
                                       sfc_temp, self.lat, self.q_v,
                                       self.insolation, config)
            return jnp.sum(out.lw_flux_up ** 2)

        grad = jax.grad(loss)(self.sfc_temp)
        assert_gradient_ok(grad, "Gray radiation w.r.t. sfc_temp")


# ============================================================================
# 2c  Convection schemes
# ============================================================================

class TestConvectionGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.thermo import saturation_mixing_ratio
        n, nlev = 4, 8
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        # Convection schemes (KF, ZM, Tiedtke, Bechtold, Emanuel) require a
        # conditionally unstable, near-saturated column for the trigger and
        # CAPE gates to fire.  The default 250 K / q_v=1e-3 state used by
        # the radiation/turbulence tests has CAPE ≈ 0 → zero gradient.
        sigma_full = jnp.linspace(0.05, 0.95, nlev)
        T_profile = 295.0 + (200.0 - 295.0) * (1.0 - sigma_full)  # ≈ standard lapse
        T_data = jnp.broadcast_to(
            T_profile[None, None, None, :], (6, n, n, nlev),
        )
        # Roughly 80 % RH at every level — provides moisture for plume + CAPE.
        p_full_1d = sigma_full * 1.0e5
        q_sat_1d = saturation_mixing_ratio(T_profile, p_full_1d)
        q_v_data = jnp.broadcast_to(
            (0.8 * q_sat_1d)[None, None, None, :], (6, n, n, nlev),
        )
        u_data = jnp.zeros((6, n, n, nlev))
        v_data = jnp.zeros((6, n, n, nlev))
        self.state = HydrostaticState(
            u=Field(u_data, name="u"),
            v=Field(v_data, name="v"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
            tracers={
                "q_v": Field(q_v_data, name="q_v"),
                "q_c": Field(1e-6 * jnp.ones((6, n, n, nlev)), name="q_c"),
                "q_r": Field(1e-7 * jnp.ones((6, n, n, nlev)), name="q_r"),
            },
        )

    @pytest.mark.parametrize(
        "scheme",
        [
            "sbm",
            "dca",
            "kuo",
            "mass_flux",
            "edmf",
            "zhang_mcfarlane",
            "kain_fritsch",
            "emanuel",
            "tiedtke",
            "bechtold",
        ],
    )
    def test_grad_wrt_T(self, scheme):
        from legoesm.atmosphere.physics.convection.integration import make_convection_physics
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig

        config = ConvectionConfig(scheme=scheme)
        conv_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
        state = self.state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, _ = conv_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, f"Convection({scheme}) w.r.t. T", min_nonzero_frac=0.01)


# ============================================================================
# 2d  Turbulence schemes
# ============================================================================

class TestTurbulenceGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

    @pytest.mark.parametrize(
        "scheme",
        [
            "smagorinsky",
            "louis",
            "tke",
            "clubb_lite",
            "holtslag_boville",
            "ysu",
            "edmf",
        ],
    )
    def test_grad_wrt_T(self, scheme):
        from legoesm.atmosphere.physics.turbulence.integration import make_turbulence_physics
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

        config = TurbulenceConfig(scheme=scheme)
        turb_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        # Use a super-adiabatic (convectively unstable) temperature profile
        # so every turbulence scheme is genuinely active.  The default
        # near-isothermal state is stable in θ almost everywhere, where the
        # faithful shear-driven Smagorinsky-Lilly closure *correctly* gives
        # K≈0 via its hard Lilly stable-cutoff — so ∂(dT/dt)/∂T is sparse
        # there.  An unstable column exercises the active branch of all
        # schemes (the random u/v already supply the vertical shear).
        sigma_full = self.sigma.sigma_full                       # (nlev,)
        T_unstable = 240.0 + 50.0 * sigma_full                   # warm surface
        T_unstable = jnp.broadcast_to(T_unstable, self.state.T.data.shape)
        state = self.state._replace(T=self.state.T.replace(data=T_unstable))

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, _ = turb_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, f"Turbulence({scheme}) w.r.t. T", min_nonzero_frac=0.01)


# ============================================================================
# 2e  Microphysics
# ============================================================================

class TestMicrophysicsGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

    @pytest.mark.parametrize(
        "scheme",
        ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"],
    )
    def test_grad_wrt_qv(self, scheme):
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

        config = MicrophysicsConfig(scheme=scheme)
        micro_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
        state = self.state

        def loss(qv_data):
            tracers_new = {**state.tracers, "q_v": state.tracers["q_v"].replace(data=qv_data)}
            s = state._replace(tracers=tracers_new)
            tend = micro_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.tracers["q_v"].data)
        assert_gradient_ok(grad, f"Microphysics({scheme}) w.r.t. q_v", min_nonzero_frac=0.01)

    @pytest.mark.parametrize(
        "scheme",
        ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"],
    )
    def test_grad_wrt_qr_at_zero(self, scheme):
        """Marshall-Palmer rain processes use fractional powers of q_r whose
        analytic derivative is unbounded at q_r=0.  This test covers BOTH
        the warm-rain path (evaporation/accretion: ``q_r**0.525``,
        ``q_r**0.875``) AND the sedimentation fall-speed path
        (``(q_r * rho/rho_sfc)**b_v_r`` with ``b_v_r < 1``) — the fall
        speeds only feed back through ``dq_r_dt`` (sedimentation), so the
        loss must include tracer tendencies, not just ``dT_dt``.
        """
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

        config = MicrophysicsConfig(scheme=scheme)
        micro_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
        state = self.state
        qr_zero = jnp.zeros_like(state.tracers["q_r"].data)

        def loss(qr_data):
            tracers_new = {
                **state.tracers,
                "q_r": state.tracers["q_r"].replace(data=qr_data),
            }
            s = state._replace(tracers=tracers_new)
            tend = micro_fn(s, self.grid, self.sigma)
            # Sum across ALL output channels so fall-speed bugs in
            # dq_r/dq_i/dq_s/dq_g are not silently masked by a
            # dT_dt-only loss.
            total = jnp.sum(tend.dT_dt.data ** 2)
            if tend.tracer_tendencies is not None:
                for fld in tend.tracer_tendencies.values():
                    total = total + jnp.sum(fld.data ** 2)
            return total

        grad = jax.grad(loss)(qr_zero)
        assert jnp.all(jnp.isfinite(grad)), (
            f"Microphysics({scheme}) w.r.t. q_r at q_r=0: gradient has NaN/Inf — "
            "fractional-power AD guard regressed."
        )

    @pytest.mark.parametrize("scheme", ["morrison", "thompson"])
    def test_grad_wrt_ice_hydrometeors_at_zero(self, scheme):
        """Morrison/Thompson have additional fractional-power sites beyond
        warm rain: ``N_i**(1/3)`` (depositional growth) and ice/snow/graupel
        fall speeds ``(q_x * rho/rho_sfc)**b_v_x``.  These must produce
        finite gradients when ``q_i = q_s = q_g = 0`` (no ice mass) and
        ``N_i = 0`` (no ice number).

        The base hydrostatic state used by the AD-test fixtures only
        carries ``{q_v, q_c, q_r}``; the integration's ``_get_tracer``
        substitutes zeros for any missing ice tracer.  An earlier version
        of this test guarded the override with ``if "q_i" in tracers``,
        which silently skipped (q_i was *not* in the base state) and made
        ``qi_data`` an *unused* argument to ``loss`` — the gradient was
        then trivially zero, which is "finite", and the test passed
        vacuously regardless of whether the AD guard was in place.

        Fix: explicitly add zero ``q_i / q_s / q_g / N_i`` Fields to the
        base state so the override is real, ``qi_data`` is consumed by
        the integration, and the gradient flows through the ice
        fall-speed and ``N_i**(1/3)`` paths.  Two regimes are checked:
          * ``q_i = 0``: gradient must be FINITE (the cold-start case
            this test is named for; without ``safe_pow`` it would be
            ``inf``).
          * ``q_i = small > 0``: gradient must be FINITE *and*
            non-trivial (>0 somewhere) — confirms the override is
            actually wired through and the test isn't ineffective.
        """
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

        config = MicrophysicsConfig(scheme=scheme)
        micro_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
        state = self.state
        zero_field = jnp.zeros_like(state.tracers["q_v"].data)

        # Explicitly add ice-phase tracers to the state so the override
        # below is observed by the integration's _get_tracer (otherwise
        # the dict-only override is silently dropped → ineffective test).
        ice_tracers = {
            **state.tracers,
            "q_i": Field(zero_field, name="q_i"),
            "q_s": Field(zero_field, name="q_s"),
            "q_g": Field(zero_field, name="q_g"),
            "N_i": Field(zero_field, name="N_i"),
        }
        state_with_ice = state._replace(tracers=ice_tracers)
        # Sanity-check the test scaffold itself: q_i must now be a real
        # tracer the integration reads.  This guards future refactors
        # from re-introducing the silent-skip bug.
        assert "q_i" in state_with_ice.tracers

        def loss(qi_data):
            tracers_new = {
                **state_with_ice.tracers,
                "q_i": state_with_ice.tracers["q_i"].replace(data=qi_data),
            }
            s = state_with_ice._replace(tracers=tracers_new)
            tend = micro_fn(s, self.grid, self.sigma)
            total = jnp.sum(tend.dT_dt.data ** 2)
            if tend.tracer_tendencies is not None:
                for fld in tend.tracer_tendencies.values():
                    total = total + jnp.sum(fld.data ** 2)
            return total

        # Cold-start regime: q_i = 0 everywhere.  Without safe_pow this
        # produces inf gradient through the q_i**(1/3) / fall-speed terms.
        grad_zero = jax.grad(loss)(zero_field)
        assert jnp.all(jnp.isfinite(grad_zero)), (
            f"Microphysics({scheme}) w.r.t. q_i at q_i=0: gradient has NaN/Inf — "
            "ice fall-speed / N_i^(1/3) AD guard regressed."
        )

        # Active regime: q_i = small positive ⇒ qi_data must be
        # *consumed* by the integration ⇒ gradient must be non-trivial.
        # If the override were silently dropped (test ineffective), the
        # gradient would be exactly zero — which this assertion catches.
        small_qi = 1e-6 * jnp.ones_like(zero_field)
        grad_small = jax.grad(loss)(small_qi)
        assert jnp.all(jnp.isfinite(grad_small)), (
            f"Microphysics({scheme}) w.r.t. q_i at q_i=1e-6: gradient has NaN/Inf"
        )
        nonzero_frac = float(jnp.mean(jnp.abs(grad_small) > 0.0))
        assert nonzero_frac > 0.0, (
            f"Microphysics({scheme}) w.r.t. q_i at q_i=1e-6: gradient is "
            f"identically zero ({nonzero_frac:.0%} non-zero) — the test is "
            "ineffective; qi_data is not being threaded into the integration. "
            "Check that q_i is in state.tracers and that the override flows "
            "through to micro_fn."
        )


# ============================================================================
# 2f  Combined physics (make_physics)
# ============================================================================

class TestCombinedPhysicsGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.state = make_hydrostatic_state(n, nlev)

        phys_config = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
            microphysics=MicrophysicsConfig(scheme="none"),
        )
        self.physics_fn = make_physics(phys_config, model_type="hydrostatic", dt=300.0)
        self.physics_fn.set_time(80.0, 43200.0)

    def test_grad_wrt_T(self):
        physics_fn, state = self.physics_fn, self.state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, _ = physics_fn(s, self.grid, self.sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Combined physics w.r.t. T")
