"""Differentiability tests for sea ice model.

Categories:
  5a) Slab sea ice thermodynamics
  5b) Dynamic sea ice (EVP)
  5c) Ice strength and rheology
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


def make_ice_forcing(shape):
    from legoesm.core.coupling_fields import AtmToSurface
    ones = jnp.ones(shape)
    return AtmToSurface(
        sw_down=100.0 * ones,
        lw_down=250.0 * ones,
        precip_total=0.0 * ones,
        precip_snow=0.0 * ones,
        T_lowest=260.0 * ones,
        q_lowest=1e-3 * ones,
        u_lowest=5.0 * ones,
        v_lowest=2.0 * ones,
        p_lowest=1e5 * ones,
        p_surface=1.013e5 * ones,
        rho_lowest=1.4 * ones,
        cos_zenith=0.5 * ones,
        co2_ppmv=400.0 * ones,
        has_radiation=1.0 * ones,
        has_precipitation=1.0 * ones,
    )


# ============================================================================
# 5a  Slab sea ice thermodynamics
# ============================================================================

class TestSlabIceGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState

        self.step_fn = step_sea_ice
        self.config = SeaIceConfig(dynamics="none")
        n = 4
        shape = (6, n, n)
        self.state = SeaIceState(
            h_ice=Field(1.0 * jnp.ones(shape), name="h_ice"),
            T_ice=Field(265.0 * jnp.ones(shape), name="T_ice"),
            concentration=Field(0.8 * jnp.ones(shape), name="concentration"),
        )
        self.forcing = make_ice_forcing(shape)
        self.ocean_sst = 271.35 * jnp.ones(shape)
        self.ocean_u = jnp.zeros(shape)
        self.ocean_v = jnp.zeros(shape)
        self.dt = 3600.0

    def test_grad_wrt_T_ice(self):
        state = self.state

        def loss(T_data):
            s = state._replace(T_ice=state.T_ice.replace(data=T_data))
            out, _ = self.step_fn(
                s, self.forcing, self.ocean_sst, self.ocean_u, self.ocean_v,
                self.config, U_min=1.0, dt=self.dt,
            )
            return jnp.sum(out.T_ice.data ** 2)

        grad = jax.grad(loss)(state.T_ice.data)
        assert_gradient_ok(grad, "Slab ice w.r.t. T_ice")

    def test_grad_wrt_ocean_sst(self):
        """SST affects ice thickness via basal melt."""
        state = self.state

        def loss(sst):
            out, _ = self.step_fn(
                state, self.forcing, sst, self.ocean_u, self.ocean_v,
                self.config, U_min=1.0, dt=self.dt,
            )
            return jnp.sum(out.h_ice.data ** 2)

        grad = jax.grad(loss)(self.ocean_sst)
        assert_gradient_ok(grad, "Slab ice h_ice w.r.t. ocean_sst")

    def test_grad_wrt_sw_down(self):
        """Spec 5a: incident shortwave must reach the ice state by AD.

        d(T_ice)/d(sw_down) is the entry point for every surface-energy
        parameter-estimation problem (albedo, conductivity); a zero here
        would mean the SW term never touches the prognostic ice
        temperature.
        """
        state = self.state
        forcing = self.forcing

        def loss(sw_down):
            f = forcing._replace(sw_down=sw_down)
            out, _ = self.step_fn(
                state, f, self.ocean_sst, self.ocean_u, self.ocean_v,
                self.config, U_min=1.0, dt=self.dt,
            )
            return jnp.sum(out.T_ice.data ** 2)

        grad = jax.grad(loss)(forcing.sw_down)
        assert_gradient_ok(grad, "Slab ice T_ice w.r.t. sw_down")
        # Sign: more incident SW warms the ice, and the loss is
        # sum(T_ice**2) with T_ice > 0 K, so the loss must increase.
        assert jnp.all(grad > 0), (
            f"d(sum T_ice^2)/d(sw_down) must be positive (more SW -> warmer "
            f"ice); got min={float(jnp.min(grad)):.3e}"
        )

    def test_partial_cover_lead_freezing_grows_concentration(self):
        """A partially-covered cell (h > 0, A < 1) with destabilizing
        surface flux must refreeze the open-water lead and INCREASE
        concentration via the CICE/Icepack ``add_new_ice`` pathway.

        Why non-vacuous: in iter-48's first attempt I gated dconc_growth
        on ``~ice_mask`` (i.e., suppressed lead refreezing on any
        partially-covered cell).  This test would fail under that
        formulation because ΔA == 0 despite positive lead freezing.
        It also fails the ORIGINAL buggy code's ``max(dh_dt, 0)``
        formulation because the basal growth of existing ice produces
        SPURIOUS ΔA.  This test only passes when lead refreezing is
        the SOLE driver of concentration growth — exactly the CICE
        convention.
        """
        # Partially-covered cell with strong surface cooling so the
        # lead freezing rate is large.  Make ice cold (basal freezing
        # is also active) so that the buggy ``max(dh_dt, 0)`` formula
        # would over-grow concentration.
        ice_state = self.state._replace(
            h_ice=self.state.h_ice.replace(
                data=jnp.full_like(self.state.h_ice.data, 1.0)
            ),
            T_ice=self.state.T_ice.replace(
                data=jnp.full_like(self.state.T_ice.data, 250.0)
            ),
            concentration=self.state.concentration.replace(
                data=jnp.full_like(self.state.concentration.data, 0.5)
            ),
        )
        # Set forcing for very cold air (drives strong surface cooling
        # → Q_sfc < 0 → freeze_flux_open > 0 → lead refreezing).
        cold_forcing = self.forcing._replace(
            T_lowest=jnp.full_like(self.forcing.T_lowest, 230.0),
            sw_down=jnp.zeros_like(self.forcing.sw_down),  # polar night
            lw_down=jnp.full_like(self.forcing.lw_down, 150.0),  # cold sky
        )
        ocean_sst = jnp.full_like(self.ocean_sst, 271.35)
        out, _ = self.step_fn(
            ice_state, cold_forcing, ocean_sst,
            self.ocean_u, self.ocean_v,
            self.config, U_min=1.0, dt=self.dt,
        )

        conc_change = float(jnp.max(
            out.concentration.data - ice_state.concentration.data
        ))
        # Lead refreezing must INCREASE concentration on a partial-
        # cover cell.  Under the over-restrictive iter-48-first-pass
        # gate on ~ice_mask, this would be ΔA == 0 (FAIL).
        assert conc_change > 1e-6, (
            f"Concentration did not grow on partial-cover cell with "
            f"lead refreezing (ΔA = {conc_change:.3e}).  Per CICE / "
            f"Icepack add_new_ice convention, lead refreezing must "
            f"increase A even when ice_mask = True."
        )

    def test_existing_ice_basal_growth_does_not_spread_laterally(self):
        """Basal growth of EXISTING ice (ice_mask=True) should thicken
        the floe (h_new > h) without changing concentration.  CICE
        convention: areal concentration only grows from new-ice
        formation in OPEN-WATER portions of the cell.

        Why non-vacuous: under the prior bug
        ``dconc_growth = max(dh_dt, 0) * (1-A) / h_new_ice``,
        existing ice with positive dh_dt (basal freezing) would spread
        laterally at rate ``(1-A)/h_new_ice`` per second of growth.
        For A=0.8 and h_new_ice=0.05 m, even a small basal-growth rate
        of dh_dt=1e-7 m/s × 3600s = 3.6e-4 m thickening produced a
        spurious dA = 3.6e-4 · 0.2 / 0.05 = 1.4e-3 over 1 hour.
        The test below constructs a column with strong basal growth
        and asserts ΔA < 1e-6 — the buggy code returned ΔA ≈ 1e-3.
        """
        # Ice-covered cell (ice_mask=True), partial concentration.  Make
        # T_ice cold (260 K) and ocean SST = T_freeze (no basal melt) so
        # the conductive-flux-driven F_cond = k * (271.35 - 260) / h
        # produces basal GROWTH (dh_dt_basal > 0).
        cold_state = self.state._replace(
            T_ice=self.state.T_ice.replace(
                data=jnp.full_like(self.state.T_ice.data, 250.0)
            ),
        )
        ocean_sst = jnp.full_like(self.ocean_sst, 271.35)  # ocean at freeze pt
        out, _ = self.step_fn(
            cold_state, self.forcing, ocean_sst,
            self.ocean_u, self.ocean_v,
            self.config, U_min=1.0, dt=self.dt,
        )

        h_change = float(jnp.max(out.h_ice.data - cold_state.h_ice.data))
        conc_change = float(jnp.max(jnp.abs(
            out.concentration.data - cold_state.concentration.data
        )))

        # Sanity: basal growth must be active (h must INCREASE)
        assert h_change > 1e-6, (
            f"Test setup failed: ice did not grow (Δh = {h_change}); "
            f"basal growth path not exercised."
        )

        # Concentration must NOT spread laterally on existing ice growth
        assert conc_change < 1e-6, (
            f"Concentration grew by {conc_change:.3e} on existing ice "
            f"with basal growth.  Per CICE convention, vertical growth "
            f"of existing floes must NOT change areal concentration. "
            f"Under the prior bug ``max(dh_dt, 0) * (1-A) / h_new_ice`` "
            f"this would be O(1e-3) over 1 hour."
        )


# ============================================================================
# 5b  Dynamic sea ice (EVP)
# ============================================================================

class TestDynamicIceGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import DynamicSeaIceState

        n = 4
        self.grid = create_cubed_sphere(n)
        self.step_fn = step_sea_ice
        self.config = SeaIceConfig(
            dynamics="evp",
            N_evp=10,  # small for speed
            differentiable_dynamics=True,
        )
        shape = (6, n, n)
        self.state = DynamicSeaIceState(
            h_ice=Field(1.5 * jnp.ones(shape), name="h_ice"),
            T_ice=Field(263.0 * jnp.ones(shape), name="T_ice"),
            concentration=Field(0.9 * jnp.ones(shape), name="concentration"),
            u_ice=Field(jnp.zeros(shape), name="u_ice"),
            v_ice=Field(jnp.zeros(shape), name="v_ice"),
            sigma_11=Field(jnp.zeros(shape), name="sigma_11"),
            sigma_22=Field(jnp.zeros(shape), name="sigma_22"),
            sigma_12=Field(jnp.zeros(shape), name="sigma_12"),
            h_snow=Field(jnp.zeros(shape), name="h_snow"),
            S_ice=Field(jnp.zeros(shape), name="S_ice"),
            pond_area=Field(jnp.zeros(shape), name="pond_area"),
            pond_depth=Field(jnp.zeros(shape), name="pond_depth"),
        )
        self.forcing = make_ice_forcing(shape)
        self.ocean_sst = 271.35 * jnp.ones(shape)
        self.ocean_u = jnp.zeros(shape)
        self.ocean_v = jnp.zeros(shape)
        self.dt = 3600.0

    def test_grad_wrt_h_ice(self):
        state = self.state

        def loss(h_data):
            s = state._replace(h_ice=state.h_ice.replace(data=h_data))
            out, _ = self.step_fn(
                s, self.forcing, self.ocean_sst, self.ocean_u, self.ocean_v,
                self.config, U_min=1.0, dt=self.dt, grid=self.grid,
            )
            return jnp.sum(out.h_ice.data ** 2)

        grad = jax.grad(loss)(state.h_ice.data)
        assert_gradient_ok(grad, "Dynamic ice w.r.t. h_ice")


# ============================================================================
# 5c  Ice strength and rheology
# ============================================================================

class TestRheologyGrad:

    def test_ice_strength_grad(self):
        from legoesm.ice.rheology import ice_strength

        h = 2.0 * jnp.ones((6, 4, 4))
        A = 0.9 * jnp.ones((6, 4, 4))

        dP_dh = jax.grad(lambda h: jnp.sum(ice_strength(h, A)))(h)
        assert jnp.all(jnp.isfinite(dP_dh)), "dP/dh not finite"
        assert jnp.all(dP_dh > 0), "dP/dh should be positive (thicker = stronger)"

    def test_vp_stress_grad(self):
        """Spec 5c: the VP constitutive law itself (Hibler 1979).

        ``vp_stress`` is the kernel both ``evp_stress_update`` and
        ``mevp_stress_update`` build on, so an AD defect here (the
        ``zeta = P/(2 Delta)`` division, the ``1/e^2`` shear split)
        poisons every rheology.  Checked w.r.t. BOTH the strain rate and
        the ice strength P.
        """
        from legoesm.ice.rheology import vp_stress, delta_deformation

        shape = (6, 4, 4)
        eps_11 = 1e-6 * jnp.ones(shape)
        eps_22 = -0.5e-6 * jnp.ones(shape)
        eps_12 = 0.3e-6 * jnp.ones(shape)
        P = 1e4 * jnp.ones(shape)
        Delta = delta_deformation(eps_11, eps_22, eps_12)

        def loss_eps(e11):
            s11, s22, s12 = vp_stress(e11, eps_22, eps_12, P, Delta)
            return jnp.sum(s11 ** 2 + s22 ** 2 + s12 ** 2)

        def loss_P(p):
            s11, s22, s12 = vp_stress(eps_11, eps_22, eps_12, p, Delta)
            return jnp.sum(s11 ** 2 + s22 ** 2 + s12 ** 2)

        assert_gradient_ok(jax.grad(loss_eps)(eps_11), "VP stress w.r.t. eps_11")
        assert_gradient_ok(jax.grad(loss_P)(P), "VP stress w.r.t. P")

    def test_evp_stress_update_grad(self):
        from legoesm.ice.rheology import evp_stress_update

        shape = (6, 4, 4)
        sigma_11 = jnp.zeros(shape)
        sigma_22 = jnp.zeros(shape)
        sigma_12 = jnp.zeros(shape)
        eps_11 = 1e-6 * jnp.ones(shape)
        eps_22 = -0.5e-6 * jnp.ones(shape)
        eps_12 = 0.3e-6 * jnp.ones(shape)
        P = 1e4 * jnp.ones(shape)

        def loss(eps_11):
            s11, s22, s12 = evp_stress_update(
                sigma_11, sigma_22, sigma_12,
                eps_11, eps_22, eps_12, P,
                e_yield=2.0, T_evp=0.36, dt_s=30.0, N_evp=120,
            )
            return jnp.sum(s11 ** 2 + s22 ** 2 + s12 ** 2)

        grad = jax.grad(loss)(eps_11)
        assert_gradient_ok(grad, "EVP stress update w.r.t. eps_11")

    def test_delta_deformation_zero_strain_grad(self):
        """Delta invariant + VP/EVP stress gradients must be finite at *zero*
        strain (rest state / cold start, eps_ij == 0).

        Regression: ``max(sqrt(max(Delta_sq, 0)), Delta_min)`` returned a NaN
        gradient at zero strain — sqrt'(0) = inf and the outer ``max`` routes a
        zero selector into the sqrt (0*inf = NaN), poisoning every VP/EVP/mEVP
        stress gradient on the first backward pass of a quiescent run.  Flooring
        the sqrt *argument* at Delta_min**2 keeps the forward value (= Delta_min)
        and yields a finite (zero) gradient.  Only finiteness is asserted: zero
        gradient is the *correct* answer in the floored regularization band.
        """
        from legoesm.ice.rheology import delta_deformation, evp_stress_update

        shape = (6, 4, 4)
        zero = jnp.zeros(shape)

        # Direct: Delta at exact zero strain.
        g_delta = jax.grad(lambda e: jnp.sum(delta_deformation(e, zero, zero)))(zero)
        assert jnp.all(jnp.isfinite(g_delta)), "delta_deformation grad NaN at zero strain"

        # End-to-end: zero-strain EVP subcycle (mirrors test_evp_stress_update_grad
        # but at the rest state that triggered the NaN).
        def stress_loss(eps_11):
            s11, s22, s12 = evp_stress_update(
                zero, zero, zero, eps_11, zero, zero, P=1e4 * jnp.ones(shape),
                e_yield=2.0, T_evp=0.36, dt_s=30.0, N_evp=120,
            )
            return jnp.sum(s11 ** 2 + s22 ** 2 + s12 ** 2)

        g_stress = jax.grad(stress_loss)(zero)
        assert jnp.all(jnp.isfinite(g_stress)), "EVP stress grad NaN at zero strain"


# ============================================================================
# 5d  Multi-category ITD — linear_remap differentiability
# ============================================================================

class TestITDRemapGrad:

    def test_linear_remap_grad_h(self):
        """Gradient through ITD linear remapping w.r.t. post-thermo thickness."""
        from legoesm.ice.itd import linear_remap

        n_cat = 5
        shape = (6, 4, 4, n_cat)

        key = jax.random.PRNGKey(50)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Pre-thermo state: uniform across categories
        h_old = jnp.broadcast_to(jnp.array([0.3, 1.0, 1.8, 3.0, 5.0]), shape)
        a_old = 0.15 * jnp.ones(shape)

        # Post-thermo: perturbed thickness
        h_new = h_old + 0.1 * jax.random.normal(k1, shape)
        h_new = jnp.maximum(h_new, 0.0)
        a_new = a_old + 0.01 * jax.random.normal(k2, shape)
        a_new = jnp.clip(a_new, 0.0, 1.0)

        T_new = 265.0 * jnp.ones(shape)

        def loss(h_new_data):
            h_r, a_r, T_r = linear_remap(h_old, a_old, h_new_data, a_new, n_cat, T_new)
            return jnp.sum(h_r ** 2 + a_r ** 2)

        grad = jax.grad(loss)(h_new)
        assert_gradient_ok(grad, "ITD linear_remap w.r.t. h_new", min_nonzero_frac=0.05)

    def test_aggregate_state_grad(self):
        """Gradient through ITD aggregation."""
        from legoesm.ice.itd import aggregate_state

        n_cat = 5
        shape = (6, 4, 4, n_cat)
        h_ice = jnp.broadcast_to(jnp.array([0.3, 1.0, 1.8, 3.0, 5.0]), shape)
        T_ice = 265.0 * jnp.ones(shape)
        conc = 0.15 * jnp.ones(shape)

        def loss(h_data):
            h_agg, T_agg, conc_agg = aggregate_state(h_data, T_ice, conc)
            return jnp.sum(h_agg ** 2)

        grad = jax.grad(loss)(h_ice)
        assert_gradient_ok(grad, "ITD aggregate w.r.t. h_ice", min_nonzero_frac=0.05)


# ============================================================================
# 5e  Ice albedo feedback loop
# ============================================================================

class TestIceAlbedoFeedback:

    def test_albedo_feedback_sign(self):
        """Warmer ice -> lower albedo -> more SW absorption (positive feedback).

        d(absorbed_SW)/d(T_ice) > 0 when temp_dependent_albedo=True.
        """
        from legoesm.surface_albedo import ice_albedo, IceAlbedoConfig

        config = IceAlbedoConfig()
        ncol = 32
        sw_down = 200.0 * jnp.ones(ncol)

        def absorbed_sw(T_ice):
            alpha = ice_albedo(T_ice, config)
            return jnp.sum((1.0 - alpha) * sw_down)

        T_ice = 270.0 * jnp.ones(ncol)
        grad = jax.grad(absorbed_sw)(T_ice)
        assert jnp.all(jnp.isfinite(grad)), "Albedo feedback gradient not finite"
        # Positive feedback: warmer ice -> lower albedo -> more absorption
        assert jnp.mean(grad) > 0, (
            f"Expected positive d(absorbed_SW)/d(T_ice), got mean={jnp.mean(grad):.6e}"
        )

    def test_slab_ice_with_temp_albedo_grad(self):
        """Full slab ice step with temperature-dependent albedo."""
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState

        config = SeaIceConfig(dynamics="none", temp_dependent_albedo=True)
        shape = (6, 4, 4)
        ones = jnp.ones(shape)
        state = SeaIceState(
            h_ice=Field(1.0 * ones, name="h_ice"),
            T_ice=Field(268.0 * ones, name="T_ice"),
            concentration=Field(0.8 * ones, name="concentration"),
        )
        forcing = make_ice_forcing(shape)
        ocean_sst = 271.35 * ones

        def loss(T_data):
            s = state._replace(T_ice=state.T_ice.replace(data=T_data))
            out, response = step_sea_ice(
                s, forcing, ocean_sst, jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0,
            )
            return jnp.sum(out.T_ice.data ** 2)

        grad = jax.grad(loss)(state.T_ice.data)
        assert_gradient_ok(grad, "Slab ice (temp-dependent albedo) w.r.t. T_ice")


# ============================================================================
# 5f  Shortwave thickness-ramp albedo — open-water limit differentiability
# ============================================================================

class TestIceShortwaveGrad:
    """d(albedo)/d(h_ice) must be finite at the open-water limit h_ice == 0.

    Regression: the ``sqrt(max(h_ice, 0))`` thickness ramps in
    ``maykut_untersteiner_albedo`` and the delta-Eddington bare-ice band had an
    *infinite* gradient at h_ice == 0 — the normal state of every ice-edge /
    growth-from-open-water cell — because sqrt'(0) = inf.  The module docstring
    explicitly promises a well-defined gradient for adjoint use, so this guards
    that contract.  Flooring the sqrt argument at a negligible thickness fixes
    it without changing the forward ramp for any physical h_ice.
    """

    @pytest.mark.parametrize("scheme", ["maykut", "delta_eddington"])
    def test_albedo_grad_finite_at_open_water(self, scheme):
        from legoesm.ice import shortwave

        shape = (4, 4)
        zero = jnp.zeros(shape)
        T_sfc = 270.0 * jnp.ones(shape)

        if scheme == "maykut":
            def loss(h_ice):
                return jnp.sum(shortwave.maykut_untersteiner_albedo(T_sfc, h_ice))
        else:
            def loss(h_ice):
                a_vis, a_nir = shortwave.delta_eddington_albedo(
                    T_sfc, h_ice, zero, zero, zero,  # no snow, no pond
                )
                return jnp.sum(a_vis + a_nir)

        grad = jax.grad(loss)(zero)
        assert jnp.all(jnp.isfinite(grad)), (
            f"{scheme} albedo gradient not finite at h_ice=0 (open water)"
        )
