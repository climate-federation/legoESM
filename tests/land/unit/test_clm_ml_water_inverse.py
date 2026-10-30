"""The multilayer land step turns CLM-ML's latent heat back into water with
the latent heat CLM-ML charged.  CLM-ML's LatVap charges all its water (leaf
and soil) at its reference AIR temperature: hvap (== constants.L_v) above
freezing, hsub (== constants.L_s) at or below -- not the Kirchhoff L_v of the
skin temperature the two-leaf / SimpleSEB solves charge.  The CLM-ML backend is
stubbed to a known flux so no clm-ml-jax solve runs: this pins the driver's
inverse only.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme.base import SurfaceFluxOutput
from legoesm.thermo import charged_latent_heat, latent_heat_vaporization

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"), reason="float64 water budget; JAX_ENABLE_X64=1")

_LH = 100.0      # [W/m2] CLM-ML latent heat, snow-free column...
_LE_SOIL = 30.0  # ...of which 30 W/m2 below-canopy soil evaporation


def _stub(*, T_soil_top, forcing, canopy_state, **_):
    o = jnp.ones_like(T_soil_top)
    return SurfaceFluxOutput(
        shflx=20.0 * o, lhflx=_LH * o, tau_x=0.0 * o, tau_y=0.0 * o,
        sw_net=300.0 * o, lw_net=-60.0 * o, lw_up=460.0 * o, G_soil=120.0 * o,
        T_surface=T_soil_top, q_surface=0.015 * o, albedo=0.15 * o,
        emissivity=0.97 * o, z0=0.1 * o, T_canopy_air=T_soil_top, stomatal_ratio=o,
        LE_soil=_LE_SOIL * o, LE_canopy=(_LH - _LE_SOIL) * o,
        L_soil=charged_latent_heat("clm_ml", forcing.T_lowest)), canopy_state


@pytest.mark.parametrize("T_soil,T_air,L_charged", [
    (300.0, 298.0, constants.L_v),   # warm air: hvap
    (276.0, 270.0, constants.L_s),   # thawed soil under freezing air: hsub
], ids=["warm_air_hvap", "cold_air_hsub"])
def test_clm_ml_water_is_its_latent_heat_over_what_it_charged(
        monkeypatch, T_soil, T_air, L_charged):
    import legoesm.land.canopy.clm_ml_interface as clm
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state, step_multilayer_land_with_diagnostics)
    monkeypatch.setattr(clm, "compute_clm_ml_canopy_fluxes", _stub)
    cfg = MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig())
    s0 = init_multilayer_land_state(1, cfg, T_init=T_soil, theta_init=0.30)
    assert float(s0.snow_depth[0]) == 0.0
    o = jnp.ones(1)
    f = AtmToSurface(
        sw_down=400.0 * o, lw_down=380.0 * o, precip_total=0.0 * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=0.003 * o,
        u_lowest=3.0 * o, v_lowest=0.0 * o, p_lowest=9.9e4 * o, p_surface=1.0e5 * o,
        rho_lowest=1.25 * o, cos_zenith=0.6 * o, co2_ppmv=412.0 * o,
        has_radiation=o, has_precipitation=o)
    _, resp, _, out = step_multilayer_land_with_diagnostics(
        s0, f, cfg, 1.0, 1800.0, lat=jnp.array([0.3]), carbon_state=None,
        doy=180.0, land_params=None)
    assert float(out.lhflx[0]) == pytest.approx(_LH)
    # Moist, snow-free column: the whole demand is supplied, so the water is
    # exactly the charge over CLM-ML's latent heat, and the reported latent
    # heat is the charge itself.
    np.testing.assert_allclose(float(resp.surface_mass_flux[0]), _LH / L_charged, rtol=1e-9)
    np.testing.assert_allclose(float(resp.lhflx[0]), _LH, rtol=1e-9)
    # Non-vacuous: the Kirchhoff inverse at the skin temperature differs by > 2 %.
    assert abs(L_charged / float(latent_heat_vaporization(T_soil)) - 1.0) > 0.02
