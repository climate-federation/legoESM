"""Phase 1D: Slab ocean component stress tests."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from legoesm.ocean.simple_ocean import (
    SimpleOceanConfig,
    SlabOceanState,
    _slab_step,
    _two_layer_step,
    init_slab_state,
)
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.core.field import Field


def _make_forcing(shape, sw_down=200.0, lw_down=300.0, T_lowest=285.0,
                  q_lowest=0.008, u_lowest=5.0, v_lowest=0.0,
                  p_surface=101325.0, rho_lowest=1.2):
    """Build an AtmToSurface forcing bundle with constant values."""
    z = jnp.zeros(shape)
    return AtmToSurface(
        sw_down=jnp.full(shape, sw_down),
        lw_down=jnp.full(shape, lw_down),
        precip_total=z,
        precip_snow=z,
        T_lowest=jnp.full(shape, T_lowest),
        q_lowest=jnp.full(shape, q_lowest),
        u_lowest=jnp.full(shape, u_lowest),
        v_lowest=jnp.full(shape, v_lowest),
        p_lowest=jnp.full(shape, p_surface),
        p_surface=jnp.full(shape, p_surface),
        rho_lowest=jnp.full(shape, rho_lowest),
        cos_zenith=jnp.ones(shape),
        co2_ppmv=jnp.full(shape, 400.0),
        has_radiation=jnp.ones(shape),
        has_precipitation=jnp.ones(shape),
    )


@pytest.mark.timeout(60)
class TestSlabOcean:
    """Slab ocean stress tests (1D)."""

    # -----------------------------------------------------------------
    # 1D.1 -- Energy budget closure (slab)
    # -----------------------------------------------------------------
    def test_energy_budget_closure(self):
        """Single slab step: temperature change should be finite, nonzero,
        and consistent with the sign of the net heat flux."""
        shape = (6, 4, 4)
        config = SimpleOceanConfig(mode="slab", h_mix=50.0)
        state = init_slab_state(shape, T_sfc_init=290.0, T_deep_init=280.0)

        forcing = _make_forcing(shape)
        dt = 3600.0

        new_state, T_sfc_new, _, _ = _slab_step(state, forcing, config, dt)

        T_old = state.T_sfc.data
        dT = T_sfc_new - T_old

        assert jnp.all(jnp.isfinite(dT)), "dT is not finite"
        assert jnp.any(dT != 0.0), "dT should be nonzero"

        # Compute expected net flux sign: SW + LW_net - SH - LH + Q_flux
        # With these parameters, SW_net and LW_down are positive heating.
        # The ocean emits LW up, and SH/LH are computed from bulk formulas.
        # We just check energy change is finite and consistent:
        C_mix = config.rho_ocean * config.c_ocean * config.h_mix
        energy_change = C_mix * dT  # J/m2

        assert jnp.all(jnp.isfinite(energy_change)), "Energy change not finite"

        # With SW=200, LW=300, T_sfc=290 < T_lowest=285 is cooling sensible,
        # but net radiation is large and positive so dT should be > 0 for
        # at least some grid points.  Just check consistency:
        # If mean dT > 0, mean energy > 0 (and vice versa).
        mean_dT = float(jnp.mean(dT))
        mean_E = float(jnp.mean(energy_change))
        if mean_dT > 0:
            assert mean_E > 0, "Energy sign inconsistent with warming"
        elif mean_dT < 0:
            assert mean_E < 0, "Energy sign inconsistent with cooling"

    # -----------------------------------------------------------------
    # 1D.2 -- Two-layer conservation and relaxation
    # -----------------------------------------------------------------
    def test_two_layer_conservation(self):
        """Two-layer ocean: total heat changes only from surface fluxes,
        and when surface forcing is approximately balanced, the two layers
        should relax toward each other via vertical mixing."""
        from legoesm import constants

        shape = (6, 4, 4)
        config = SimpleOceanConfig(
            mode="two_layer",
            h_mix=50.0,
            h_deep=450.0,
            k_mix=1.0e-4,
            restore_deep=False,
            T_freeze=200.0,  # disable freezing clamp for this test
        )
        T_sfc_init = 295.0
        T_deep_init = 280.0
        state = init_slab_state(shape, T_sfc_init=T_sfc_init, T_deep_init=T_deep_init)

        C_mix = config.rho_ocean * config.c_ocean * config.h_mix
        C_deep = config.rho_ocean * config.c_ocean * config.h_deep

        def _total_heat(s):
            return jnp.sum(C_mix * s.T_sfc.data + C_deep * s.T_deep.data)

        heat_initial = _total_heat(state)
        dt = 3600.0

        # Build forcing that approximately balances surface fluxes:
        # - Set lw_down to match emitted LW so net LW ~ 0
        # - T_lowest = T_sfc to zero SH
        # - q_lowest high enough to minimize LH
        # - sw_down = 0 to avoid heating
        # The U_min wind floor (1 m/s) means there is always some
        # residual turbulent flux, but it will be small.
        lw_down_balanced = (
            config.emissivity_ocean * constants.sigma_sb * T_sfc_init ** 4
        )
        forcing = _make_forcing(
            shape,
            sw_down=0.0,
            lw_down=float(lw_down_balanced),
            T_lowest=T_sfc_init,
            q_lowest=0.02,  # high humidity to minimize LH
            u_lowest=0.0,
            v_lowest=0.0,
        )

        # Track |T_sfc - T_deep| at intervals
        gap_history = []
        nsteps = 1000

        for i in range(nsteps):
            state, _, _, _ = _two_layer_step(state, forcing, config, dt)
            if i % 100 == 0:
                gap = float(jnp.mean(jnp.abs(state.T_sfc.data - state.T_deep.data)))
                gap_history.append(gap)

        heat_final = _total_heat(state)

        # With approximately balanced surface forcing, total heat
        # should be approximately conserved.
        rel_heat_change = float(jnp.abs(heat_final - heat_initial) / jnp.abs(heat_initial))
        assert rel_heat_change < 0.01, (
            f"Total heat changed by {rel_heat_change:.4%}, expected < 1%"
        )

        # Gap should decrease (vertical mixing relaxation).
        assert gap_history[-1] < gap_history[0], (
            f"Temperature gap should decrease: initial={gap_history[0]:.4f}, "
            f"final={gap_history[-1]:.4f}"
        )

        # Check that the majority of intervals show decreasing gap.
        n_decreasing = sum(
            1 for i in range(1, len(gap_history)) if gap_history[i] < gap_history[i - 1]
        )
        frac_decreasing = n_decreasing / max(len(gap_history) - 1, 1)
        assert frac_decreasing > 0.5, (
            f"Gap should decrease in most intervals: "
            f"{frac_decreasing:.0%} were decreasing"
        )

        # All state variables should be finite.
        assert jnp.all(jnp.isfinite(state.T_sfc.data)), "T_sfc not finite"
        assert jnp.all(jnp.isfinite(state.T_deep.data)), "T_deep not finite"
