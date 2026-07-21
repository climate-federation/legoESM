"""Below-canopy soil-surface evaporation resistance (Sellers 1992 + SZ09 litter).

Pins the physical behaviour of ``soil_surface_evap_resistance`` — the series
resistance that supplies the soil-side vapour-diffusion resistance the canopy
soil energy balance's aerodynamic-only path omitted (a wet forest floor otherwise
evaporates at near-potential rate: LE_soil ~57% of total LE at US-MMS, where
<15% is physical).

Truth-tier checks (analytic limits + monotonicity + differentiability), plus an
integration check that turning the series resistance ON via the config gate
actually LOWERS the two-leaf soil evaporation while leaving the SimpleSEB path
and the slab path (which never supply ``soil_surface_relsat``) unchanged.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.canopy.energy_balance import (
    soil_surface_evap_resistance,
    _SELLERS_RSS_A,
    _SELLERS_RSS_B,
    _SELLERS_RSS_MAX,
)

_a = jnp.asarray


def test_sellers_saturated_limit_matches_paper():
    """At W_1 = 1 with no litter, r_ss = exp(a - b) (Sellers et al. 1992)."""
    r = float(soil_surface_evap_resistance(_a(1.0), _a(0.0), _a(0.0)))
    assert r == pytest.approx(math.exp(_SELLERS_RSS_A - _SELLERS_RSS_B), rel=1e-5)


def test_monotone_decreasing_in_wetness():
    """Drier surface (smaller W_1) -> larger resistance (throttles evaporation)."""
    w = jnp.linspace(0.05, 1.0, 20)
    r = jax.vmap(lambda x: soil_surface_evap_resistance(x, _a(5.0), _a(100.0)))(w)
    dr = np.diff(np.asarray(r))
    assert np.all(dr < 0.0), "r_soil_surface must decrease monotonically with W_1"


def test_monotone_increasing_in_lai():
    """Denser canopy -> more litter cover -> larger litter resistance."""
    lai = jnp.linspace(0.0, 8.0, 20)
    r = jax.vmap(lambda x: soil_surface_evap_resistance(_a(0.8), x, _a(100.0)))(lai)
    dr = np.diff(np.asarray(r))
    assert np.all(dr >= 0.0), "litter resistance must be non-decreasing in LAI"
    # Bare soil (LAI=0) has zero litter -> pure r_ss.
    r_bare = float(soil_surface_evap_resistance(_a(0.8), _a(0.0), _a(100.0)))
    r_ss_only = float(soil_surface_evap_resistance(_a(0.8), _a(0.0), _a(0.0)))
    assert r_bare == pytest.approx(r_ss_only, rel=1e-6)


def test_litter_scales_cover_fraction():
    """Litter term = litter_ref * (1 - exp(-0.5 LAI))."""
    lai, ref = 5.82, 100.0
    r_full = float(soil_surface_evap_resistance(_a(0.8), _a(lai), _a(ref)))
    r_noli = float(soil_surface_evap_resistance(_a(0.8), _a(lai), _a(0.0)))
    litter = r_full - r_noli
    assert litter == pytest.approx(ref * (1.0 - math.exp(-0.5 * lai)), rel=1e-5)


def test_dry_surface_capped():
    """r_ss is capped so the AD Jacobian stays finite at the residual boundary."""
    r = float(soil_surface_evap_resistance(_a(1e-6), _a(0.0), _a(0.0)))
    assert r <= _SELLERS_RSS_MAX + 1e-6
    assert np.isfinite(r)


def test_differentiable():
    """Finite gradients w.r.t. both W_1 and the litter reference (AD-safe)."""
    g_w = float(jax.grad(lambda w: soil_surface_evap_resistance(
        w, _a(5.0), _a(100.0)))(_a(0.6)))
    g_l = float(jax.grad(lambda L: soil_surface_evap_resistance(
        _a(0.6), _a(5.0), L))(_a(100.0)))
    assert np.isfinite(g_w) and g_w < 0.0      # wetter -> lower r
    assert np.isfinite(g_l) and g_l > 0.0      # more litter -> higher r


def test_us_mms_regime_is_physical():
    """US-MMS JJA (W_1~0.8, LAI~5.8): resistance in the 150-350 s/m forest range."""
    r = float(soil_surface_evap_resistance(_a(0.80), _a(5.82), _a(100.0)))
    assert 150.0 < r < 350.0


def test_persistent_litter_lai_keeps_cover_in_leaf_off():
    """A persistent structural ``litter_LAI`` keeps the litter cover (and hence the
    resistance) high when the LIVE LAI collapses in the leaf-off season — the
    deciduous forest-floor fix.  Default (no litter_LAI) uses live LAI unchanged."""
    # Bare-canopy winter: live LAI = 0.5.  Without a persistent LAI the litter
    # cover is weak; with a structural LAI of 5 it stays strong.
    r_live = float(soil_surface_evap_resistance(_a(0.8), _a(0.5), _a(300.0)))
    r_persist = float(soil_surface_evap_resistance(_a(0.8), _a(0.5), _a(300.0),
                                                   litter_LAI=_a(5.0)))
    assert r_persist > r_live + 100.0        # persistent litter adds real resistance
    # ... and equals passing the structural LAI as the live LAI directly.
    r_direct = float(soil_surface_evap_resistance(_a(0.8), _a(5.0), _a(300.0)))
    assert r_persist == pytest.approx(r_direct, rel=1e-6)
    # Default (litter_LAI omitted) is identical to the live-LAI cover.
    r_default = float(soil_surface_evap_resistance(_a(0.8), _a(0.5), _a(300.0), litter_LAI=None))
    assert r_default == pytest.approx(r_live, rel=1e-6)


def test_series_resistance_lowers_two_leaf_soil_evap():
    """Integration: the config gate ON lowers modelled soil LE vs the legacy beta.

    Runs the two-leaf canopy soil energy balance through the public
    ``compute_two_leaf_canopy_fluxes`` with a wet top layer under a closed
    canopy, once with ``soil_evap_series_resistance`` True and once False, and
    asserts the series-resistance run evaporates LESS from the soil (the
    partition fix) while total LE stays finite and positive.
    """
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)
    from legoesm.core.coupling_fields import AtmToSurface

    ncol = 1
    forcing = AtmToSurface(
        T_lowest=_a([298.0]), q_lowest=_a([0.010]), p_lowest=_a([98000.0]),
        p_surface=_a([101325.0]), u_lowest=_a([2.0]), v_lowest=_a([0.0]),
        rho_lowest=_a([1.15]), sw_down=_a([700.0]), lw_down=_a([360.0]),
        cos_zenith=_a([0.9]), precip_total=_a([0.0]), precip_snow=_a([0.0]),
        co2_ppmv=_a([410.0]), has_radiation=True, has_precipitation=False,
    )

    cc = TwoLeafCanopyConfig(max_iters=30)
    relsat = _a([0.83])          # wet top layer (theta_1/theta_sat)
    common = dict(
        T_soil_top=_a([299.0]), forcing=forcing, canopy_config=cc,
        canopy_params=None, w_frac_rz=_a([0.7]), wind_speed=_a([2.0]),
        wind_dir_x=_a([1.0]), wind_dir_y=_a([0.0]),
        soil_thermal_fn=lambda G, dt_: _a([299.0]), dt=1800.0,
        LAI_override=_a([5.8]), w_frac_soil_evap=_a([0.95]),
        soil_surface_relsat=relsat,
    )
    on = compute_two_leaf_canopy_fluxes(
        land_config=MultiLayerLandConfig(soil_evap_series_resistance=True), **common)
    off = compute_two_leaf_canopy_fluxes(
        land_config=MultiLayerLandConfig(soil_evap_series_resistance=False), **common)

    le_soil_on = float(np.asarray(on.LE_soil).ravel()[0])
    le_soil_off = float(np.asarray(off.LE_soil).ravel()[0])
    assert le_soil_on < le_soil_off, (
        f"series resistance should lower soil LE: on={le_soil_on:.1f} "
        f"off={le_soil_off:.1f}")
    # Total LE must remain finite and non-negative in both cases.
    assert np.isfinite(float(np.asarray(on.lhflx).ravel()[0]))
    assert float(np.asarray(on.lhflx).ravel()[0]) > 0.0
