"""Constant-scheme SW transmittance through ice to the ocean (codex L1).

The old OMIP wiring gave the ocean an ``A*tau*sw_down`` under-ice SW dribble
that was never debited from the ice tile's energy balance (the constant
shortwave scheme absorbed ALL non-reflected SW) — a ~tau non-closure of SW
over ice.  The fix moves ownership into the ice model:
``SeaIceConfig.sw_transmittance_const`` debits the ice surface-absorbed SW and
delivers the transmitted part to the ocean through the EXISTING
``sw_penetrated -> ocean_heat_extraction`` channel, so

    reflected + absorbed_by_ice + penetrated_to_ocean == sw_down   (exactly)

and the ocean-side surrogate is retired (the runner passes
``sw_transmittance_ice=0.0`` to the blend).

Also pins codex r5 #1: ``TileResponse.ice_concentration_thermo`` exposes the
post-transport pre-thermo aggregate concentration the atmospheric fluxes were
integrated over, for the forced-ocean open-water partition.

Run under JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ice.shortwave import compute_ice_sw


def test_constant_scheme_transmittance_closes_sw_budget():
    """reflected + absorbed + penetrated == sw_down exactly, for tau in
    {0, 0.03, 0.3}; tau=0 is bit-identical to the legacy no-penetration
    constant scheme."""
    sw = jnp.asarray([0.0, 50.0, 320.0])
    T = jnp.full(3, 260.0)
    h = jnp.full(3, 1.0)
    z = jnp.zeros(3)
    alpha_c = 0.65

    legacy = compute_ice_sw(sw, T, h, z, z, z, scheme="constant",
                            albedo_const=alpha_c)
    np.testing.assert_allclose(np.asarray(legacy.sw_penetrated), 0.0)
    np.testing.assert_allclose(np.asarray(legacy.sw_absorbed_surface),
                               (1.0 - alpha_c) * np.asarray(sw))

    for tau in (0.0, 0.03, 0.3):
        r = compute_ice_sw(sw, T, h, z, z, z, scheme="constant",
                           albedo_const=alpha_c, sw_transmittance_const=tau)
        reflected = alpha_c * np.asarray(sw)
        total = (reflected + np.asarray(r.sw_absorbed_surface)
                 + np.asarray(r.sw_penetrated))
        np.testing.assert_allclose(total, np.asarray(sw), rtol=1e-14)
        assert np.all(np.asarray(r.sw_absorbed_surface) >= 0.0)
        assert np.all(np.asarray(r.sw_penetrated) >= 0.0)
        if tau == 0.0:
            np.testing.assert_allclose(np.asarray(r.sw_penetrated), 0.0)
        else:
            # tau is measured on the INCIDENT sw (the surrogate's semantic).
            np.testing.assert_allclose(
                np.asarray(r.sw_penetrated),
                np.minimum(tau, 1.0 - alpha_c) * np.asarray(sw), rtol=1e-14)


def test_transmittance_bounded_by_non_reflected_input():
    """tau > (1 - alpha): the penetrated flux is capped at the column input so
    the budget cannot go negative (mirrors the delta_eddington bound)."""
    sw = jnp.asarray([200.0])
    z = jnp.zeros(1)
    r = compute_ice_sw(sw, jnp.full(1, 260.0), jnp.full(1, 1.0), z, z, z,
                       scheme="constant", albedo_const=0.9,
                       sw_transmittance_const=0.3)
    np.testing.assert_allclose(np.asarray(r.sw_penetrated), 0.1 * 200.0)
    np.testing.assert_allclose(np.asarray(r.sw_absorbed_surface), 0.0,
                               atol=1e-12)


def _mini_ice_setup(tau):
    """One step_sea_ice on a tiny latlon-like field with seeded ice.

    brine=ON pins the NEW-PHYSICS (v2) path for BOTH tau values (same path the
    OMIP runner drives), so the tau=0 vs tau>0 comparison isolates the
    transmittance — with all v2 gates off, tau=0 would route to the LEGACY
    dynamic path (different response assembly) and the A/B would be
    confounded."""
    from legoesm.ice import SeaIceConfig, init_dynamic_ice_state, step_sea_ice
    from legoesm.ice.config import BrineConfig
    from legoesm.core.coupling_fields import AtmToSurface

    shape = (4,)
    cfg = SeaIceConfig(dynamics="free_drift", transport="none",
                       brine=BrineConfig(enabled=True),
                       sw_transmittance_const=tau)
    st = init_dynamic_ice_state(shape, S_ice_init=0.0)
    st = st._replace(
        h_ice=st.h_ice.replace(data=jnp.full(shape, 1.0)),
        concentration=st.concentration.replace(data=jnp.full(shape, 0.8)),
    )
    o = jnp.ones(shape)
    atm = AtmToSurface(
        sw_down=200.0 * o, lw_down=260.0 * o,
        precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
        T_lowest=262.0 * o, q_lowest=2e-3 * o,
        u_lowest=4.0 * o, v_lowest=jnp.zeros(shape),
        p_lowest=1.0e5 * o, p_surface=1.0e5 * o, rho_lowest=1.3 * o,
        cos_zenith=0.5 * o, co2_ppmv=jnp.asarray(400.0),
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )
    sst = jnp.full(shape, float(constants.T_freeze_ocean) + 0.2)
    new_st, resp = step_sea_ice(st, atm, sst, jnp.zeros(shape),
                                jnp.zeros(shape), cfg, U_min=0.0, dt=1800.0,
                                grid=None)
    return new_st, resp


def test_transmitted_sw_reaches_ocean_via_heat_extraction():
    """With tau > 0 the ocean heat extraction DROPS by exactly the
    concentration-weighted transmitted SW relative to tau = 0 (ocean gains the
    penetrated flux through the existing response channel; the ice pays)."""
    _, r0 = _mini_ice_setup(0.0)
    _, r1 = _mini_ice_setup(0.03)
    conc_thermo = np.asarray(r1.ice_concentration_thermo)
    expected_gain = 0.03 * 200.0 * conc_thermo        # [W/m^2 per grid cell]
    np.testing.assert_allclose(
        np.asarray(r0.ocean_heat_extraction)
        - np.asarray(r1.ocean_heat_extraction),
        expected_gain, rtol=1e-6)


def test_response_exposes_thermo_time_concentration():
    """codex r5 #1: the response carries the aggregate concentration the
    thermodynamics integrated over (== the seeded 0.8 here: transport='none'
    so pre-call == thermo-time), clipped to [0, 1]."""
    _, resp = _mini_ice_setup(0.0)
    a = np.asarray(resp.ice_concentration_thermo)
    np.testing.assert_allclose(a, 0.8, rtol=1e-12)
    assert np.all(a >= 0.0) and np.all(a <= 1.0)


def test_runner_retires_ocean_side_sw_surrogate():
    """Drift guard: the OMIP runner passes sw_transmittance_ice=0.0 to the
    blend (the ice model owns transmission via sw_transmittance_const) and
    threads --ice-thermo-sw-trans into SeaIceConfig; the blend partition uses
    the thermo-time concentration when the response provides it."""
    import inspect
    from scripts.run import run_omip_core2 as R
    src = inspect.getsource(R.main)
    assert "sw_transmittance_ice=0.0" in src
    from tests.ice.unit.test_omip_bulk_snow import _main_config
    assert _main_config(["--ice-thermo-sw-trans", "0.17"]).sw_transmittance_const == 0.17
    assert "ice_resp.ice_concentration_thermo" in src
