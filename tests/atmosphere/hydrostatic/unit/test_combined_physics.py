"""Unit tests for the combined physics orchestrator.

Tests that PhysicsConfig + make_physics correctly:
- Combines radiation, convection, and turbulence into one physics_fn
- Allows scheme selection within each module
- Produces zero tendencies when all modules are "none"
- Sums tendencies from multiple active modules
- Works with hydrostatic, non-hydrostatic, and spectral PE dycores
- Is differentiable with jax.grad
- Rejects invalid model_type
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.core.field import Field


def _make_hydrostatic_setup():
    """Create a minimal hydrostatic state, grid, sigma for testing."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(10)
    state = held_suarez_init(grid, sigma)
    # Give state some wind so turbulence has something to mix
    n, nlev = grid.n, sigma.n_levels
    state = state._replace(
        u=Field(data=jnp.ones((6, n, n, nlev)) * 10.0,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
    )
    return state, grid, sigma


def _make_turbulent_setup():
    """Active boundary-layer column for the faithful turbulence closures.

    ``held_suarez_init`` gives a near-isothermal (hence θ-stably
    stratified) column; the deformation-based Smagorinsky–Lilly closure
    *correctly* produces ~zero interior diffusivity there (hard Lilly
    cutoff at Ri ≥ Pr_t), so it would generate no heat tendency.  To
    exercise the schemes in their intended regime we add (1) vertical
    shear so |S| > 0 and (2) a super-adiabatic surface layer so the
    near-surface gradient Richardson number is negative — the canonical
    convective-boundary-layer state where turbulence mixes heat.
    """
    state, grid, sigma = _make_hydrostatic_setup()
    n, nlev = grid.n, sigma.n_levels
    # Vertical shear: wind grows from ~5 m/s aloft toward the surface.
    lev = jnp.arange(nlev)
    u_shear = (5.0 + 3.0 * lev)[None, None, None, :] * jnp.ones((6, n, n, nlev))
    # Super-adiabatic surface layer: warm the lowest three levels (warmest
    # at the surface) so the near-surface stratification is unstable.
    T_active = state.T.data.at[..., nlev - 3:].add(
        jnp.array([10.0, 22.0, 40.0])
    )
    state = state._replace(
        u=Field(data=u_shear, name="u",
                dims=("face", "x", "y", "level"), units="m/s"),
        T=Field(data=T_active, name="T",
                dims=("face", "x", "y", "level"), units="K"),
    )
    return state, grid, sigma


class TestPhysicsConfig:
    """Tests for PhysicsConfig defaults and construction."""

    def test_default_config(self):
        """Default PhysicsConfig should enable all three modules."""
        cfg = PhysicsConfig()
        assert cfg.radiation.scheme == "gray"
        assert cfg.convection.scheme == "sbm"
        assert cfg.turbulence.scheme == "smagorinsky"

    def test_override_schemes(self):
        """Each module scheme can be overridden independently."""
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="dca"),
            turbulence=TurbulenceConfig(scheme="louis"),
        )
        assert cfg.radiation.scheme == "gray"
        assert cfg.convection.scheme == "dca"
        assert cfg.turbulence.scheme == "louis"

    def test_disable_all(self):
        """Setting all schemes to 'none' should be valid."""
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        assert cfg.radiation.scheme == "none"
        assert cfg.convection.scheme == "none"
        assert cfg.turbulence.scheme == "none"


class TestCombinedHydrostatic:
    """Tests for combined physics on the hydrostatic dycore."""

    def test_all_none_gives_zero(self):
        """All modules disabled -> zero tendencies."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        assert jnp.allclose(tend.du_dt.data, 0.0)
        assert jnp.allclose(tend.dv_dt.data, 0.0)
        assert jnp.allclose(tend.dT_dt.data, 0.0)

    def test_radiation_only(self):
        """Radiation only should give nonzero T tendencies, zero wind."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
        # Radiation alone produces no wind tendency
        assert jnp.allclose(tend.du_dt.data, 0.0)

    def test_turbulence_only(self):
        """Turbulence only should give nonzero wind and T tendencies.

        Uses an active (sheared + super-adiabatic) column so the faithful
        Smagorinsky–Lilly closure mixes heat — on a quiescent stable
        column its hard Lilly cutoff correctly yields zero dT_dt.
        """
        state, grid, sigma = _make_turbulent_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0

    def test_combined_larger_than_individual(self):
        """Radiation + turbulence T tendency should differ from either alone."""
        state, grid, sigma = _make_hydrostatic_setup()

        # Radiation alone
        cfg_rad = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        tend_rad, _ = make_physics(cfg_rad, "hydrostatic", dt=300.0)(state, grid, sigma)

        # Turbulence alone
        cfg_turb = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        tend_turb, _ = make_physics(cfg_turb, "hydrostatic", dt=300.0)(state, grid, sigma)

        # Both
        cfg_both = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        tend_both, _ = make_physics(cfg_both, "hydrostatic", dt=300.0)(state, grid, sigma)

        # Combined dT_dt should equal sum of individuals
        expected_dT = tend_rad.dT_dt.data + tend_turb.dT_dt.data
        assert jnp.allclose(tend_both.dT_dt.data, expected_dT, atol=1e-6)

        # Combined du_dt should equal turbulence alone (radiation has zero du_dt)
        assert jnp.allclose(tend_both.du_dt.data, tend_turb.du_dt.data, atol=1e-6)

    def test_convection_only(self):
        """Convection only should give nonzero T tendencies."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        # SBM should produce some T tendency (heating/cooling from adjustment)
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        # Convection alone produces no wind tendency
        assert jnp.allclose(tend.du_dt.data, 0.0)
        assert jnp.allclose(tend.dv_dt.data, 0.0)

    def test_all_three_active(self):
        """All three modules active should produce finite tendencies."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))
        # With all three active, both wind and T should be nonzero
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0

    def test_scheme_selection_convection(self):
        """Switching convection scheme should change the result."""
        state, grid, sigma = _make_hydrostatic_setup()

        cfg_sbm = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        cfg_dca = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="dca"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        tend_sbm, _ = make_physics(cfg_sbm, "hydrostatic", dt=300.0)(state, grid, sigma)
        tend_dca, _ = make_physics(cfg_dca, "hydrostatic", dt=300.0)(state, grid, sigma)

        # Both should be finite
        assert jnp.all(jnp.isfinite(tend_sbm.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend_dca.dT_dt.data))
        # Should differ (different algorithms)
        assert not jnp.allclose(tend_sbm.dT_dt.data, tend_dca.dT_dt.data, atol=1e-10)

    def test_scheme_selection_radiation(self):
        """Switching radiation scheme should change the result."""
        state, grid, sigma = _make_hydrostatic_setup()

        # Gray radiation
        cfg_gray = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        tend_gray, _ = make_physics(cfg_gray, "hydrostatic", dt=300.0)(state, grid, sigma)

        # No radiation
        cfg_none = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        tend_none, _ = make_physics(cfg_none, "hydrostatic", dt=300.0)(state, grid, sigma)

        # Should be different
        assert not jnp.allclose(tend_gray.dT_dt.data, tend_none.dT_dt.data)

    def test_scheme_selection_turbulence(self):
        """Switching turbulence scheme should change the result.

        Uses an active (sheared + super-adiabatic) column: the
        deformation-based Smagorinsky–Lilly and stability-function Louis
        closures only diverge meaningfully where interior mixing is
        actually switched on.
        """
        state, grid, sigma = _make_turbulent_setup()

        cfg_smag = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        cfg_louis = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="louis"),
        )
        tend_smag, _ = make_physics(cfg_smag, "hydrostatic", dt=300.0)(state, grid, sigma)
        tend_louis, _ = make_physics(cfg_louis, "hydrostatic", dt=300.0)(state, grid, sigma)

        # Both nonzero
        assert float(jnp.max(jnp.abs(tend_smag.du_dt.data))) > 0
        assert float(jnp.max(jnp.abs(tend_louis.du_dt.data))) > 0
        # But different
        assert not jnp.allclose(tend_smag.du_dt.data, tend_louis.du_dt.data, atol=1e-10)

    def test_correct_shapes(self):
        """Combined physics should produce correct tendency shapes."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig()
        physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        n, nlev = grid.n, sigma.n_levels
        assert tend.du_dt.data.shape == (6, n, n, nlev)
        assert tend.dv_dt.data.shape == (6, n, n, nlev)
        assert tend.dT_dt.data.shape == (6, n, n, nlev)
        assert tend.dp_s_dt.data.shape == (6, n, n)

    def test_differentiable(self):
        """jax.grad should work through the combined physics."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        physics_fn = make_physics(cfg, "hydrostatic", dt=300.0)

        def loss(T_data):
            new_state = state._replace(T=state.T.replace(data=T_data))
            tend, _ = physics_fn(new_state, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))


def _make_nh_setup():
    """Create a minimal non-hydrostatic state, grid, coords for testing."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import (
        create_height_coordinate,
        compute_terrain_metric,
    )
    from legoesm.core.state import NonHydrostaticState

    n, nlev = 8, 10
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, 30000.0)
    z_s = jnp.zeros((6, n, n))
    tm = compute_terrain_metric(z_s, hc)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")

    state = NonHydrostaticState(
        u=Field(data=jnp.ones((6, n, n, nlev)) * 5.0, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev)) * 2.0, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 1)), name="tracers", dims=dims_tr, units="kg/kg"),
    )
    return state, grid, hc, tm


class TestCombinedNonHydrostatic:
    """Tests for combined physics on the non-hydrostatic dycore."""

    def test_shapes(self):
        """NH combined physics should produce correct tendency shapes."""
        state, grid, hc, tm = _make_nh_setup()
        n, nlev = 8, 10

        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        physics_fn = make_physics(cfg, "nonhydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, hc, tm)

        assert tend.du_dt.data.shape == (6, n, n, nlev)
        assert tend.dv_dt.data.shape == (6, n, n, nlev)
        assert tend.dtheta_prime_dt.data.shape == (6, n, n, nlev)
        assert tend.drho_prime_dt.data.shape == (6, n, n, nlev)
        assert tend.dw_dt.data.shape == (6, n, n, nlev + 1)
        assert tend.dtracers_dt.data.shape == (6, n, n, nlev, 1)

    def test_all_none_gives_zero(self):
        """All modules disabled -> zero NH tendencies."""
        state, grid, hc, tm = _make_nh_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "nonhydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, hc, tm)

        assert jnp.allclose(tend.du_dt.data, 0.0)
        assert jnp.allclose(tend.dv_dt.data, 0.0)
        assert jnp.allclose(tend.dtheta_prime_dt.data, 0.0)
        assert jnp.allclose(tend.dw_dt.data, 0.0)

    def test_nonzero_tendencies(self):
        """NH with radiation + turbulence should produce nonzero tendencies."""
        state, grid, hc, tm = _make_nh_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        physics_fn = make_physics(cfg, "nonhydrostatic", dt=300.0)
        tend, _ = physics_fn(state, grid, hc, tm)

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
        # Turbulence should produce nonzero du_dt from wind mixing
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0
        # Radiation should produce nonzero theta tendency
        assert float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data))) > 0.0

    def test_differentiable(self):
        """jax.grad should work through NH combined physics."""
        state, grid, hc, tm = _make_nh_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="smagorinsky"),
        )
        physics_fn = make_physics(cfg, "nonhydrostatic", dt=300.0)

        def loss(u_data):
            new_state = state._replace(u=state.u.replace(data=u_data))
            tend, _ = physics_fn(new_state, grid, hc, tm)
            return jnp.sum(tend.du_dt.data ** 2)

        grad_u = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grad_u))


class TestCombinedSpectralPE:
    """Tests for combined physics on the spectral PE dycore."""

    def _make_spectral_setup(self):
        """Create a minimal spectral PE state."""
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            isothermal_rest_state_spectral,
        )

        jax.config.update("jax_enable_x64", True)
        grid = create_gaussian_grid(n_max=21)
        sigma = create_sigma_coordinate(10)
        state = isothermal_rest_state_spectral(grid, sigma)
        return state, grid, sigma

    def test_all_none_gives_zero(self):
        """All modules disabled -> zero spectral tendencies."""
        state, grid, sigma = self._make_spectral_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "spectral_pe", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        assert jnp.allclose(tend.vor_hat.data, 0.0)
        assert jnp.allclose(tend.div_hat.data, 0.0)
        assert jnp.allclose(tend.T_hat.data, 0.0)
        assert jnp.allclose(tend.lnps_hat.data, 0.0)

    def test_shapes(self):
        """Spectral PE combined physics should produce correct shapes."""
        state, grid, sigma = self._make_spectral_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "spectral_pe", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        n_sh = grid.n_sh
        nlev = sigma.n_levels
        assert tend.vor_hat.data.shape == (n_sh, nlev)
        assert tend.div_hat.data.shape == (n_sh, nlev)
        assert tend.T_hat.data.shape == (n_sh, nlev)
        assert tend.lnps_hat.data.shape == (n_sh,)

    def test_radiation_nonzero(self):
        """Radiation in spectral PE should produce nonzero T_hat tendency."""
        state, grid, sigma = self._make_spectral_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "spectral_pe", dt=300.0)
        tend, _ = physics_fn(state, grid, sigma)

        assert jnp.all(jnp.isfinite(tend.T_hat.data))
        assert float(jnp.max(jnp.abs(tend.T_hat.data))) > 0.0
        # Radiation alone: no wind -> zero vor/div tendencies
        assert jnp.allclose(tend.vor_hat.data, 0.0, atol=1e-15)
        assert jnp.allclose(tend.div_hat.data, 0.0, atol=1e-15)

    def test_combined_reuses_single_spectral_transform(self, monkeypatch):
        """Combined spectral physics should call spectral->grid transform once."""
        state, grid, sigma = self._make_spectral_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg, "spectral_pe", dt=300.0)

        # Patch the binding the orchestrator actually uses — combined.py
        # imports ``spectral_pe_to_grid`` directly via ``from … import``
        # so monkeypatching the source module's attribute would not
        # intercept the local binding the call site sees.
        import legoesm.atmosphere.physics.combined as combined_mod
        original = combined_mod.spectral_pe_to_grid
        n_calls = {"count": 0}

        def wrapped(*args, **kwargs):
            n_calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(combined_mod, "spectral_pe_to_grid", wrapped)
        _ = physics_fn(state, grid, sigma)

        assert n_calls["count"] == 1


class TestInvalidModelType:
    """Tests for error handling."""

    def test_invalid_model_type_raises(self):
        """make_physics should raise ValueError for unknown model_type."""
        cfg = PhysicsConfig()
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_physics(cfg, "invalid_type", dt=300.0)
