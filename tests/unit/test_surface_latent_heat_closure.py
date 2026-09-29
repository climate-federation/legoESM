"""Water and energy cross the surface interface with ONE latent heat.

Every tile charges its latent heat at ``L(T_sfc, phase)`` (legoesm.thermo) and
publishes the water flux it lost; the atmosphere takes that water directly and
books ``lhflx - L_v * E`` (the gap to its constant-L moist enthalpy) into its
heat lower boundary.  These tests pin that contract at the tile and kernel level;
the pipeline-level kick is pinned in test_air_sea_flux_coupling_conservation.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
)
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.thermo import (
    latent_heat_sublimation,
    latent_heat_vaporization,
    surface_latent_heat,
)

SHAPE = (2, 3)


def _forcing(T_lowest=300.0, q_lowest=0.012):
    o = jnp.ones(SHAPE)
    z = jnp.zeros(SHAPE)
    return AtmToSurface(
        sw_down=400.0 * o, lw_down=380.0 * o, precip_total=z, precip_snow=z,
        T_lowest=T_lowest * o, q_lowest=q_lowest * o, u_lowest=6.0 * o, v_lowest=z,
        p_lowest=9.8e4 * o, p_surface=1.0e5 * o, rho_lowest=1.15 * o,
        cos_zenith=0.6 * o, co2_ppmv=400.0 * o, has_radiation=o, has_precipitation=o)


def test_ocean_tile_water_is_its_latent_heat_over_L_v_at_the_sst():
    """Warm ocean: the published mass flux and latent heat are the same flux,
    related by L_v(SST), not by the constant (3 % apart at 30 degC)."""
    from legoesm.coupler.coupler import CouplerConfig, ocean_tile_response
    sst = jnp.full(SHAPE, constants.T_freeze + 30.0)
    z = jnp.zeros(SHAPE)
    r = ocean_tile_response(_forcing(), sst, z, z, CouplerConfig())
    assert float(jnp.min(r.lhflx)) > 10.0   # evaporating, so the test is not vacuous
    np.testing.assert_allclose(np.asarray(r.surface_mass_flux * latent_heat_vaporization(sst)),
                               np.asarray(r.lhflx), rtol=1e-12)
    # ...and NOT the constant: a consumer dividing by L_v would lose ~3 % of the water.
    assert float(jnp.max(jnp.abs(r.surface_mass_flux * constants.L_v / r.lhflx - 1.0))) > 0.02


@pytest.mark.parametrize("T_epi, frozen", [(constants.T_freeze + 12.0, 0.0),
                                           (constants.T_freeze - 8.0, 1.0)])
def test_lake_tile_charges_the_phase_at_its_temperature(T_epi, frozen):
    """Open lake evaporates at L_v(T); frozen lake sublimates at L_s(T); the
    published water flux inverts exactly the latent heat that was charged."""
    from legoesm.core.field import Field
    from legoesm.coupler.lake.two_layer_lake import LakeConfig, LakeState, step_lake
    state = LakeState(T_epi=Field(jnp.full(SHAPE, T_epi), name="T_epi", dims=("y", "x"), units="K"),
                      T_hypo=Field(jnp.full(SHAPE, T_epi - 4.0), name="T_hypo", dims=("y", "x"), units="K"))
    _, r = step_lake(state, _forcing(T_lowest=T_epi - 2.0, q_lowest=0.002), LakeConfig(),
                     U_min=1.0, dt=600.0)
    L = surface_latent_heat(jnp.full(SHAPE, T_epi), frozen)
    assert float(jnp.min(jnp.abs(r.lhflx))) > 1.0
    np.testing.assert_allclose(np.asarray(r.surface_mass_flux * L), np.asarray(r.lhflx), rtol=1e-12)
    want = latent_heat_sublimation if frozen else latent_heat_vaporization
    np.testing.assert_allclose(np.asarray(L), np.asarray(want(jnp.full(SHAPE, T_epi))), rtol=1e-14)


def test_kernel_moisture_bc_is_the_prescribed_water_or_the_exact_inverse():
    """The kernels' moisture BC: the coupler's water when folded in, else the
    inverse of the SAME L_v(T_sfc) the bulk law charged; and the heat BC gains
    exactly what closes the column's energy intake to shflx + lhflx."""
    T_sfc = jnp.array([constants.T_freeze + 28.0, constants.T_freeze - 5.0])
    E = jnp.array([4.0e-5, 5.0e-6])
    lhflx = E * latent_heat_vaporization(T_sfc)          # what a bulk law charges
    cfg = SurfaceLayerConfig()
    np.testing.assert_allclose(np.asarray(surface_moisture_flux(cfg, lhflx, T_sfc)), np.asarray(E), rtol=1e-14)
    cfg_w = cfg._replace(prescribed_evap_kg_m2_s=2.0 * E)
    np.testing.assert_allclose(np.asarray(surface_moisture_flux(cfg_w, lhflx, T_sfc)), np.asarray(2.0 * E), rtol=1e-14)
    # Energy closure: heat BC + reference latent credit == physical shflx + lhflx.
    shflx = jnp.array([20.0, -5.0])
    heat_bc = shflx + latent_enthalpy_correction(lhflx, E)
    np.testing.assert_allclose(np.asarray(heat_bc + constants.L_v * E), np.asarray(shflx + lhflx), rtol=1e-13)
    # ...and the correction is exactly zero when the surface charged the constant.
    assert float(jnp.max(jnp.abs(latent_enthalpy_correction(constants.L_v * E, E)))) == 0.0


def test_bulk_charge_and_kernel_inverse_round_trip_exactly():
    """The atmosphere's OWN bulk law (compute_surface_fluxes) charges L_v(T_sfc)
    and the kernel's fallback inverse recovers the water it moved.  The water
    reference is INDEPENDENT of the temperature-dependent branch: the same
    constant-coefficient law evaluated with an explicit constant L, divided by
    that constant -- so a charge reverted to the constant fails here (the
    inverse would then return E * L_v / L_v(T), 2 % off at 28 degC)."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes
    from legoesm.core.bulk_flux import simple_bulk_fluxes
    n = 4
    T_sfc = jnp.array([301.0, 295.0, 285.0, 271.0])
    u, v = jnp.full((n,), 5.0), jnp.zeros((n,))
    T_air, q_v, q_sfc, rho = T_sfc - 1.5, jnp.full((n,), 0.008), jnp.full((n,), 0.02), jnp.full((n,), 1.15)
    for conv in ("legoesm", "aerobulk"):
        cfg = SurfaceLayerConfig(bulk_scheme="constant", thermo_convention=conv)
        _, _, _, lhflx, _ = compute_surface_fluxes(u, v, T_air, q_v, T_sfc, q_sfc, rho, cfg)
        assert float(jnp.min(jnp.abs(lhflx))) > 1.0
        E = surface_moisture_flux(cfg, lhflx, T_sfc)
        # Independent water: the constant-coefficient law with an EXPLICIT constant L.
        wind = jnp.sqrt(u ** 2 + v ** 2 + 1e-4)
        _, _, _, lh_const = simple_bulk_fluxes(u, v, T_air, q_v, T_sfc, q_sfc, rho, wind,
                                               cfg.Cd_neutral, cfg.Ch_neutral, L_latent=constants.L_v)
        E_ref = lh_const / constants.L_v
        np.testing.assert_allclose(np.asarray(E), np.asarray(E_ref), rtol=1e-12)
        # ...and the charge itself is L_v(T_sfc) * E_ref, not L_v * E_ref.
        np.testing.assert_allclose(np.asarray(lhflx), np.asarray(E_ref * latent_heat_vaporization(T_sfc)), rtol=1e-12)
        assert float(jnp.abs(lhflx[0] / (constants.L_v * E_ref[0]) - 1.0)) > 0.02

    # The iterative MOST/COARE law too: its explicit-L evaluation is the
    # independent water reference (compute_most_fluxes(L_latent=...) wins).
    from legoesm.core.bulk_flux import compute_most_fluxes
    for conv in ("legoesm", "aerobulk"):
        cfg = SurfaceLayerConfig(bulk_scheme="coare3", z0=1e-4, z_ref=10.0, bulk_n_iter=5,
                                 thermo_convention=conv)
        _, _, _, lhflx, _ = compute_surface_fluxes(u, v, T_air, q_v, T_sfc, q_sfc, rho, cfg)
        E = surface_moisture_flux(cfg, lhflx, T_sfc)
        lh_const = compute_most_fluxes(
            u, v, T_air, q_v, T_sfc, q_sfc, rho, z_ref=cfg.z_ref, z0_init=cfg.z0,
            scheme="coare3", n_iter=cfg.bulk_n_iter, thermo_convention=conv,
            L_latent=constants.L_v)[3]
        E_ref = lh_const / constants.L_v
        np.testing.assert_allclose(np.asarray(E), np.asarray(E_ref), rtol=1e-10)
        np.testing.assert_allclose(np.asarray(lhflx), np.asarray(E_ref * latent_heat_vaporization(T_sfc)), rtol=1e-10)


def test_coupled_hook_hands_the_tiles_water_flux():
    """The coupled driver's hook returns (shflx, physical lhflx, mass flux) and
    refuses a response without the water flux."""
    import inspect
    from legoesm.driver import coupled_esm_driver as m
    src = inspect.getsource(m)
    hook = src[src.index("def _coupled_get_sfc_flux_override(day):"):]
    hook = hook[:hook.index("self._atm.get_sfc_flux_override = ")]
    assert "return r.shflx, r.lhflx, r.surface_mass_flux" in hook
    assert "carries no " in hook and "surface_mass_flux;" in hook
    assert "* constants.L_v" not in hook and "/ constants.L_v" not in hook
