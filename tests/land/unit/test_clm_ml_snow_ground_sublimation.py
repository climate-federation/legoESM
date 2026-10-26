"""CLM-ML's ground over snow is charged the sublimation latent heat inside its
solve (#1875) -- a documented legoESM deviation from CLM-ML, which charges the
ground at LatVap(tref).  The snow weight ``snowfrac_soil`` of the ground latent
flux is charged ``lsub_soil``; weight 0 is the original CLM-ML charge.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.thermo import latent_heat_sublimation

pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")
pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"), reason="float64 budgets; JAX_ENABLE_X64=1")

from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import (  # noqa: E402
    create_mlcanopy)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmh2o  # noqa: E402
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLSoilFluxesMod import (  # noqa: E402
    SoilFluxes)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import (  # noqa: E402
    LatVap)

_TREF, _TG = 275.0, 270.0      # thawing air over a frozen ground: hvap vs L_s


def _soil_patch(w):
    m = create_mlcanopy(1, 1)
    s = lambda a, v: a.at[1].set(v)   # noqa: E731
    return m._replace(
        tref_forcing=s(m.tref_forcing, _TREF), pref_forcing=s(m.pref_forcing, 1.0e5),
        rhomol_forcing=s(m.rhomol_forcing, 43.0), cpair_forcing=s(m.cpair_forcing, 29.2),
        rnsoi_soil=s(m.rnsoi_soil, 80.0), rhg_soil=s(m.rhg_soil, 1.0),
        soilres_soil=s(m.soilres_soil, 50.0), gac0_soil=s(m.gac0_soil, 0.4),
        soil_t_soil=s(m.soil_t_soil, _TG - 1.0), soil_dz_soil=s(m.soil_dz_soil, 0.05),
        soil_tk_soil=s(m.soil_tk_soil, 1.0), tg_bef_soil=s(m.tg_bef_soil, _TG),
        tair_profile=m.tair_profile.at[1, 1].set(_TREF - 1.0),
        eair_profile=m.eair_profile.at[1, 1].set(300.0),
        snowfrac_soil=s(m.snowfrac_soil, w),
        lsub_soil=s(m.lsub_soil, float(latent_heat_sublimation(_TG)) * mmh2o))


@pytest.mark.parametrize("w", [0.0, 0.4, 1.0])
def test_soil_fluxes_charge_the_snow_weighted_latent_heat_and_close(w):
    m = SoilFluxes(1, _soil_patch(w))
    lam = float(m.lhsoi_soil[1] / m.etsoi_soil[1])
    expect = (1.0 - w) * float(LatVap(_TREF)) + w * float(m.lsub_soil[1])
    np.testing.assert_allclose(lam, expect, rtol=1e-13)
    assert abs(float(m.lhsoi_soil[1])) > 1.0
    err = m.rnsoi_soil[1] - m.shsoi_soil[1] - m.lhsoi_soil[1] - m.gsoi_soil[1]
    assert abs(float(err)) < 1e-9


def test_soil_fluxes_weight_zero_is_original_clm_ml_bitwise():
    """Weight 0 is the CLM-ML charge exactly, whatever lsub_soil holds."""
    a = SoilFluxes(1, _soil_patch(0.0))
    b = SoilFluxes(1, _soil_patch(0.0)._replace(
        lsub_soil=jnp.full(2, 9.9e9)))
    for f in ("lhsoi_soil", "shsoi_soil", "gsoi_soil", "etsoi_soil", "tg_soil"):
        np.testing.assert_array_equal(getattr(a, f)[1], getattr(b, f)[1])
    assert float(a.lhsoi_soil[1] / a.etsoi_soil[1]) == float(LatVap(_TREF))


def test_soil_flux_gradient_wrt_snow_weight_is_finite():
    g = jax.grad(lambda w: SoilFluxes(1, _soil_patch(w)).gsoi_soil[1])(0.5)
    assert np.isfinite(float(g)) and float(g) != 0.0


def _forcing():
    from legoesm.core.coupling_fields import AtmToSurface
    o = jnp.ones(1)
    return AtmToSurface(
        sw_down=600.0 * o, lw_down=300.0 * o, precip_total=0.0 * o,
        precip_snow=0.0 * o, T_lowest=_TREF * o, q_lowest=0.001 * o,
        u_lowest=4.0 * o, v_lowest=1.0 * o, p_lowest=95000.0 * o,
        p_surface=1.0e5 * o, rho_lowest=1.2 * o, cos_zenith=0.6 * o,
        co2_ppmv=400.0 * o, has_radiation=o, has_precipitation=o)


def _call(monkeypatch, **kw):
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLFluxProfileSolutionMod as fps
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    monkeypatch.setattr(fps, "DEBUG_FPS_CHECKS", True)   # ground balance check on
    T_soil = jnp.full((1, 8), _TG)
    cfg = CLMMLCanopyConfig()
    return compute_clm_ml_canopy_fluxes(
        T_soil_top=T_soil[:, 0], forcing=_forcing(), canopy_config=cfg,
        land_config=MultiLayerLandConfig(surface_scheme=cfg), land_params=None,
        canopy_state=None, dt=1800.0, T_soil=T_soil,
        psi_soil=jnp.full((1, 8), -0.5), theta_soil=jnp.full((1, 8), 0.25),
        lat=jnp.full(1, 0.8), doy=60.0, **kw)[0]


@pytest.mark.timeout(600)
def test_clm_ml_step_over_snow_charges_sublimation_and_closes(monkeypatch):
    """A real CLM-ML step under thawing air (LatVap = hvap) over snow-covered
    ground (weight 1): the ground is charged L_s(T_ground) -- 13 % above hvap --
    and CLM-ML's own closure checks pass: the implicit solution's ground energy
    balance (ErrorCheck02, enabled here) and the canopy totals' balance
    avail = SH + LE + storage (CanopyFluxesDiagnostics), which only closes if
    the total latent flux carries the ground's sublimation charge.  ``None``
    snow weight is the original CLM-ML step bitwise."""
    L_s = latent_heat_sublimation(jnp.full(1, _TG))
    snow = _call(monkeypatch, ground_snow_weight=jnp.ones(1), ground_sublimation_heat=L_s)
    bare = _call(monkeypatch)
    zero = _call(monkeypatch, ground_snow_weight=jnp.zeros(1), ground_sublimation_heat=L_s)
    np.testing.assert_allclose(snow.L_soil, L_s, rtol=1e-12)
    np.testing.assert_allclose(bare.L_soil, float(LatVap(_TREF)) / mmh2o, rtol=1e-12)
    assert abs(float(snow.LE_soil[0])) > 0.3
    assert abs(float(snow.G_soil[0] - bare.G_soil[0])) > 0.03
    for f in bare._fields:
        a, b = getattr(bare, f), getattr(zero, f)
        if a is None:
            assert b is None, f
            continue
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f)
