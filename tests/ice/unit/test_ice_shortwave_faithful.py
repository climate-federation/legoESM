"""Faithfulness pins for the sea-ice shortwave albedo + penetration kernel.

Target: ``legoesm.ice.shortwave`` — ``compute_ice_sw`` and its two physics
albedo schemes ``maykut_untersteiner`` (CCSM3 thin-ice α(T,h) ramp) and
``delta_eddington`` (Briegleb & Light 2007 two-band VIS/NIR surrogate with
snow / pond modifications and a Beer-Lambert interior penetration).

Most-trustful source
--------------------
The band albedo CONSTANTS (dry/melting snow, cold/melting bare ice, deep pond,
i0 penetration fractions) are the published Briegleb & Light (2007) NCAR
delta-Eddington values (CICE lineage), carried in ``legoesm.constants``; the
sea-ice extinction ``κ_ice = 1.4 1/m`` follows Grenfell & Maykut (1977).  The
FORMS, by the module's own docstring, are a *tabulated / empirical surrogate*
for the full CICE6 delta-Eddington radiative-transfer lookup — smooth
interpolations chosen for a well-defined adjoint.  So the certification is:

1. TRUTH-TIERS (oracle-independent, strongest) — pinned to round-off:
   * SW energy closure ``reflected + absorbed + penetrated == sw_down`` for
     EVERY scheme, made non-tautological by a case where the raw transmittance
     would over-draw the column so the ``min`` clamp fires.
   * Coverage partition of unity ``f_snow + f_bare + f_pond == 1``.
   * Beer-Lambert attenuation ``exp(-κ_ice·h)`` — pinned by the ratio
     ``T(2h)/T(h) == exp(-κ_ice·h)`` (independent of the f_bare prefactor) and
     strict monotone decrease to 0 with thickness.
2. ISOLATION form pins — a crafted column makes ONE surface type the sole
   contributor so its band construction + the ``f_vis`` band mixing are pinned
   against literal BL07 constants without the coverage-partition ambiguity
   (pure snow / pure bare ice / pure deep pond).
3. DEPARTURE canaries — each surrogate simplification vs full CICE6:
   sqrt thickness ramp, i0 off *incident* (not net-absorbed) SW, pond albedo
   DECREASING with depth (a previously-inverted sign), the deep-pond floor
   clamped to the underlying ice, and melt_fraction reaching 1 *at* T_melt.
4. Published-constant canary + differentiability at the sqrt / clip kinks.

The existing behavioral tests (test_shortwave_pond_albedo.py,
test_sw_transmittance_const.py) check qualitative monotonicity + the constant
scheme's budget; these add the coefficient-faithful magnitude pins.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import legoesm.ice.shortwave as _sw_mod
import numpy as np
import pytest
from legoesm.ice.shortwave import (
    compute_ice_sw,
    delta_eddington_albedo,
    maykut_untersteiner_albedo,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)  # matches the sibling ice faithful tests

# --- BL07 / MU71 published constants, typed as LITERALS (asserted == the
#     module's constants in test_published_constants_canary; used by the
#     independent oracle so a drift in constants.py cannot silently pass). ---
A_SNOW_COLD_VIS, A_SNOW_COLD_NIR = 0.98, 0.70
A_SNOW_MELT_VIS, A_SNOW_MELT_NIR = 0.80, 0.55
A_ICE_COLD_VIS, A_ICE_COLD_NIR = 0.78, 0.36
A_ICE_MELT_VIS, A_ICE_MELT_NIR = 0.68, 0.30
A_POND_VIS, A_POND_NIR = 0.27, 0.07
I0_VIS, I0_NIR = 0.70, 0.0
A_OCEAN = 0.06
F_VIS = 0.52
KAPPA = 1.4
# Surrogate ramp lengths (module-private; mirrored here, asserted via behaviour).
H_SNOW_SAT, H_POND_SAT, H_BARE_SAT = 0.05, 0.30, 0.50
H_RAMP, H_SNOW_MASK, T_WIDTH = 0.50, 0.02, 1.0
T_FREEZE = float(constants.T_freeze)


def _arr(x):
    return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))


def _mf(T):
    """Melt fraction ramp over [T_melt-T_width, T_melt] (reaches 1 AT T_melt)."""
    return np.clip(1.0 + (T - T_FREEZE) / T_WIDTH, 0.0, 1.0)


def _sw(sw_down, T_sfc, h_ice, h_snow, pond_area, pond_depth, scheme,
        albedo_const=0.6, sw_transmittance_const=0.0):
    return compute_ice_sw(
        _arr(sw_down), _arr(T_sfc), _arr(h_ice), _arr(h_snow),
        _arr(pond_area), _arr(pond_depth),
        scheme=scheme, albedo_const=albedo_const,
        sw_transmittance_const=sw_transmittance_const,
    )


# ---------------------------------------------------------------------------
# 1. TRUTH-TIER — SW energy closure: reflected + absorbed + penetrated == sw_down.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["constant", "maykut_untersteiner", "delta_eddington"])
def test_sw_energy_closure_every_scheme(scheme):
    # A general marginal-ice column; reflected = alpha*sw_down.
    out = _sw(300.0, T_FREEZE - 2.0, 0.4, 0.01, 0.3, 0.05, scheme,
              sw_transmittance_const=0.2)
    reflected = float(out.albedo_eff[0]) * 300.0
    total = reflected + float(out.sw_absorbed_surface[0]) + float(out.sw_penetrated[0])
    np.testing.assert_allclose(total, 300.0, rtol=1e-13, atol=1e-10)
    assert float(out.sw_absorbed_surface[0]) >= 0.0
    assert float(out.sw_penetrated[0]) >= 0.0


def test_sw_closure_holds_when_penetration_clamp_fires():
    # Non-tautological: pick a thin bare column whose RAW transmittance*sw would
    # exceed the non-reflected column input, so the min() clamp is ACTIVE.  A
    # scheme that left penetrated unbounded would create energy (total > sw_down).
    # constant scheme with a huge transmittance forces the clamp.
    out = _sw(500.0, T_FREEZE - 5.0, 0.2, 0.0, 0.0, 0.0, "constant",
              albedo_const=0.5, sw_transmittance_const=5.0)  # 5.0 >> (1-alpha)
    net_into_column = (1.0 - 0.5) * 500.0
    # penetrated clamped to the column input; absorbed -> 0.
    np.testing.assert_allclose(float(out.sw_penetrated[0]), net_into_column, rtol=1e-13)
    np.testing.assert_allclose(float(out.sw_absorbed_surface[0]), 0.0, atol=1e-10)
    total = (float(out.albedo_eff[0]) * 500.0
             + float(out.sw_absorbed_surface[0]) + float(out.sw_penetrated[0]))
    np.testing.assert_allclose(total, 500.0, rtol=1e-13, atol=1e-10)


def test_delta_branch_penetration_clamp_fires(monkeypatch):
    # The delta_eddington branch has its OWN min() clamp (line 400); with physical
    # inputs transmittance<=f_vis*i0_vis=0.364 never exceeds (1-alpha), so the
    # clamp is unreachable through the real albedo fit.  Force it by replacing the
    # albedo kernel with a bright tile (alpha=0.9) + large transmittance (0.5):
    # raw penetration 0.5*sw > net (0.1*sw) => clamp bounds penetrated to net,
    # absorbed -> 0, and closure still holds (codex R1: the delta clamp was
    # otherwise uncertified — removing it would survive the physical closure case).
    monkeypatch.setattr(
        _sw_mod, "delta_eddington_albedo",
        lambda *a, **k: (jnp.full_like(a[0], 0.9), jnp.full_like(a[0], 0.5)),
    )
    out = _sw(400.0, 260.0, 1.0, 0.0, 0.0, 0.0, "delta_eddington")
    net_into_column = (1.0 - 0.9) * 400.0
    np.testing.assert_allclose(float(out.sw_penetrated[0]), net_into_column, rtol=1e-13)
    np.testing.assert_allclose(float(out.sw_absorbed_surface[0]), 0.0, atol=1e-10)
    total = (float(out.albedo_eff[0]) * 400.0
             + float(out.sw_absorbed_surface[0]) + float(out.sw_penetrated[0]))
    np.testing.assert_allclose(total, 400.0, rtol=1e-13, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. TRUTH-TIER — coverage partition (f_snow + f_bare + f_pond == 1) OBSERVED
#    directly by replacing the three band helpers with distinct marker albedos,
#    so the public alpha_total reveals the weights without reproducing the band
#    construction (codex R1: the [min,max] bound could not catch a weight bug).
# ---------------------------------------------------------------------------
def _marker_band(vis, nir):
    """A band-helper stand-in returning constant (vis, nir) shaped like input."""
    def _fn(first, *args, **kwargs):
        return (jnp.full_like(first, vis), jnp.full_like(first, nir))
    return _fn


def _patch_marker_bands(monkeypatch, snow, ice, pond):
    monkeypatch.setattr(_sw_mod, "_band_albedo_snow", _marker_band(*snow))
    monkeypatch.setattr(_sw_mod, "_band_albedo_bare_ice", _marker_band(*ice))
    monkeypatch.setattr(_sw_mod, "_band_albedo_pond", _marker_band(*pond))


def test_coverage_partition_of_unity(monkeypatch):
    # All three surface bands == (1, 1): alpha_total == f_snow+f_bare+f_pond must
    # be 1.0 EXACTLY for any coverage state — the partition-of-unity truth-tier.
    # Asserted with '==' (verified bit-exact for every state): a coverage deficit
    # of ANY size fails, closing the tolerance gap codex flagged.
    _patch_marker_bands(monkeypatch, (1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
    for h_snow, pond_area, pond_depth in [
        (0.0, 0.0, 0.0), (0.01, 0.4, 0.1), (0.03, 1.0, 0.5), (1.0, 0.7, 0.2),
    ]:
        alpha, _ = delta_eddington_albedo(
            _arr(T_FREEZE - 0.5), _arr(0.6), _arr(h_snow),
            _arr(pond_area), _arr(pond_depth),
        )
        assert float(alpha[0]) == 1.0


def test_coverage_weights_and_fvis_mixing_via_distinct_markers(monkeypatch):
    # Distinct VIS/NIR markers per band: alpha_total == f_vis*(2 f_snow + 3 f_bare
    # + 5 f_pond) + (1-f_vis)*(20 f_snow + 30 f_bare + 50 f_pond).  Pins BOTH the
    # coverage weights AND the f_vis band mix.  A weight bug (e.g. f_bare missing
    # the pond subtraction) or a wrong f_vis fails.
    _patch_marker_bands(monkeypatch, (2.0, 20.0), (3.0, 30.0), (5.0, 50.0))
    h_snow, pond_area = 0.01, 0.4                    # f_snow = min(0.01/0.02,1)=0.5
    f_snow = min(h_snow / H_SNOW_MASK, 1.0)
    f_bare_after = 1.0 - f_snow
    f_pond = f_bare_after * pond_area
    f_bare = f_bare_after * (1.0 - pond_area)
    assert abs(f_snow + f_bare + f_pond - 1.0) < 1e-15
    alpha, _ = delta_eddington_albedo(
        _arr(T_FREEZE - 0.5), _arr(0.6), _arr(h_snow), _arr(pond_area), _arr(0.1),
    )
    vis = 2.0 * f_snow + 3.0 * f_bare + 5.0 * f_pond
    nir = 20.0 * f_snow + 30.0 * f_bare + 50.0 * f_pond
    expect = F_VIS * vis + (1.0 - F_VIS) * nir
    # np.testing.assert_allclose defaults atol=0 (NOT np.allclose's 1e-8), so this
    # is a pure relative round-off pin; the weighted sum is not bit-exact (0.3 etc.
    # are inexact in binary) so '==' is inappropriate here.
    np.testing.assert_allclose(float(alpha[0]), expect, rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# 3. ISOLATION form pins — one surface type the sole contributor.
# ---------------------------------------------------------------------------
def test_pure_snow_pins_snow_band_and_fvis_mixing():
    # h_snow >= H_SNOW_MASK => f_snow == 1 (bare/pond masked) AND >= H_SNOW_SAT
    # => full snow ramp.  alpha == f_vis*snow_vis + (1-f_vis)*snow_nir exactly.
    T = T_FREEZE - 3.0   # cold: mf == 0
    alpha, trans = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.10),
                                          _arr(0.0), _arr(0.0))
    expect = F_VIS * A_SNOW_COLD_VIS + (1 - F_VIS) * A_SNOW_COLD_NIR
    np.testing.assert_allclose(float(alpha[0]), expect, rtol=1e-13)
    np.testing.assert_allclose(float(trans[0]), 0.0, atol=1e-15)  # snow => no bare penetration


def test_pure_bare_ice_pins_ice_band_and_fvis_mixing():
    # No snow, no pond, thick enough that the sqrt ramp saturates at 1.
    T = T_FREEZE - 3.0
    alpha, _ = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
                                      _arr(0.0), _arr(0.0))
    expect = F_VIS * A_ICE_COLD_VIS + (1 - F_VIS) * A_ICE_COLD_NIR
    np.testing.assert_allclose(float(alpha[0]), expect, rtol=1e-13)


def test_pure_deep_pond_pins_deep_floor():
    # Full pond coverage, depth past saturation => pond band == deep floor
    # min(alpha_pond, alpha_ice).  Thick cold ice_vis(0.78) > 0.27 so floor==0.27;
    # ice_nir(0.36) > 0.07 so floor==0.07.
    T = T_FREEZE - 3.0
    alpha, _ = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
                                      _arr(1.0), _arr(1.0))
    expect = F_VIS * A_POND_VIS + (1 - F_VIS) * A_POND_NIR
    np.testing.assert_allclose(float(alpha[0]), expect, rtol=1e-13)


def test_pure_snow_at_freezing_pins_melt_snow_band():
    # T == T_melt => melt_fraction == 1: pins the MELTING snow coefficients
    # (codex R1: all cold isolation cases left the delta melt constants unpinned;
    # a bug swapping melt->cold constants or freezing mf==0 would survive them).
    alpha, _ = delta_eddington_albedo(_arr(T_FREEZE), _arr(1.0), _arr(0.10),
                                      _arr(0.0), _arr(0.0))
    expect = F_VIS * A_SNOW_MELT_VIS + (1 - F_VIS) * A_SNOW_MELT_NIR
    np.testing.assert_allclose(float(alpha[0]), expect, rtol=1e-13)


def test_pure_bare_ice_at_freezing_pins_melt_ice_band():
    # T == T_melt, thick bare ice => pins the MELTING bare-ice coefficients.
    alpha, _ = delta_eddington_albedo(_arr(T_FREEZE), _arr(1.0), _arr(0.0),
                                      _arr(0.0), _arr(0.0))
    expect = F_VIS * A_ICE_MELT_VIS + (1 - F_VIS) * A_ICE_MELT_NIR
    np.testing.assert_allclose(float(alpha[0]), expect, rtol=1e-13)


# ---------------------------------------------------------------------------
# 4. Maykut-Untersteiner thin-ice ramp.
#    NOTE (codex R1): _mu_oracle mirrors the target's melt-ramp/sqrt/floor form,
#    so this is a REGRESSION-FORM pin (a shared temperature/ramp misinterpretation
#    would pass), not an oracle-independent truth tier.  The genuinely independent
#    certification is the ENDPOINT pins (cold->0.7, melt->0.5, ocean-floor) and
#    the sqrt METAMORPHIC canary below; the parametrized form-pin is a
#    round-off regression lock grounded in MU71 / CCSM3 (α_cold=0.7, α_melt=0.5).
# ---------------------------------------------------------------------------
def _mu_regression_form(T, h, a_cold=0.7, a_melt=0.5, h_ramp=0.5):
    mf = _mf(T)
    reg = (1 - mf) * a_cold + mf * a_melt
    tf = np.clip(np.sqrt(max(h, 1e-12) / max(h_ramp, 1e-6)), 0.0, 1.0)
    return max(reg * tf, A_OCEAN)


@pytest.mark.parametrize("T,h", [
    (250.0, 2.0),                 # cold, thick -> alpha_cold  (independent endpoint)
    (T_FREEZE, 2.0),              # melting, thick -> alpha_melt (independent endpoint)
    (T_FREEZE - 0.5, 2.0),        # mid melt transition (regression form)
    (260.0, 0.1),                 # thin -> sqrt ramp active (regression form)
    (260.0, 1e-4),               # vanishing -> ocean floor active (independent endpoint)
])
def test_maykut_untersteiner_matches_regression_form(T, h):
    got = maykut_untersteiner_albedo(_arr(T), _arr(h))
    np.testing.assert_allclose(float(got[0]), _mu_regression_form(T, h),
                               rtol=1e-12, atol=1e-15)


def test_mu_independent_endpoints():
    # Oracle-INDEPENDENT endpoints (no shared form): cold thick == α_cold=0.7;
    # melting thick == α_melt=0.5; vanishing ice == ocean floor.
    np.testing.assert_allclose(
        float(maykut_untersteiner_albedo(_arr(250.0), _arr(2.0))[0]), 0.7, rtol=1e-12)
    np.testing.assert_allclose(
        float(maykut_untersteiner_albedo(_arr(T_FREEZE), _arr(2.0))[0]), 0.5, rtol=1e-12)
    np.testing.assert_allclose(
        float(maykut_untersteiner_albedo(_arr(260.0), _arr(1e-6))[0]), A_OCEAN, rtol=1e-12)


def test_mu_melt_fraction_reaches_one_at_freezing():
    # Departure/fix canary: at exactly T_melt the melting albedo is FULLY reached
    # (a 0.5-centred ramp would only reach the midpoint).  Thick ice, ramp==1.
    got = float(maykut_untersteiner_albedo(_arr(T_FREEZE), _arr(2.0))[0])
    np.testing.assert_allclose(got, 0.5, rtol=1e-12)   # == alpha_melt_bare, not 0.6


def test_mu_thin_ice_sqrt_ramp_is_the_departure():
    # CANARY: thin-ice albedo scales as sqrt(h) (the surrogate ramp), so 4x
    # thickness gives 2x albedo (below saturation, above the ocean floor).
    a1 = float(maykut_untersteiner_albedo(_arr(250.0), _arr(0.02))[0])
    a4 = float(maykut_untersteiner_albedo(_arr(250.0), _arr(0.08))[0])
    np.testing.assert_allclose(a4 / a1, 2.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# 5. Beer-Lambert interior penetration (delta_eddington).
# ---------------------------------------------------------------------------
def test_transmittance_beer_lambert_ratio_pins_kappa():
    # T(h) = prefactor * exp(-kappa*h).  The RATIO T(2h)/T(h) == exp(-kappa*h)
    # isolates kappa independent of the f_bare * i0 prefactor.
    T = T_FREEZE - 3.0
    _, t1 = delta_eddington_albedo(_arr(T), _arr(0.5), _arr(0.0), _arr(0.0), _arr(0.0))
    _, t2 = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0), _arr(0.0), _arr(0.0))
    ratio = float(t2[0]) / float(t1[0])
    np.testing.assert_allclose(ratio, np.exp(-KAPPA * 0.5), rtol=1e-12)


def test_transmittance_only_visible_band_penetrates():
    # i0_nir == 0, so the whole penetrated fraction is f_vis*f_bare*i0_vis*exp(-kh).
    T = T_FREEZE - 3.0
    h = 0.5
    _, t = delta_eddington_albedo(_arr(T), _arr(h), _arr(0.0), _arr(0.0), _arr(0.0))
    # pure bare ice: f_bare == 1.
    expect = F_VIS * 1.0 * I0_VIS * np.exp(-KAPPA * h)
    np.testing.assert_allclose(float(t[0]), expect, rtol=1e-12)


def test_transmittance_decreases_to_zero_with_thickness():
    T = T_FREEZE - 3.0
    ts = [float(delta_eddington_albedo(_arr(T), _arr(h), _arr(0.0), _arr(0.0),
                                       _arr(0.0))[1][0]) for h in (0.1, 0.5, 2.0, 10.0)]
    assert ts[0] > ts[1] > ts[2] > ts[3] >= 0.0
    assert ts[3] < 1e-5   # thick ice transmits ~nothing


# ---------------------------------------------------------------------------
# 6. Pond darkening (sign + deep-floor clamp) departure canaries.
# ---------------------------------------------------------------------------
def test_pond_albedo_decreases_with_depth():
    # The FIXED sign: deepening a pond LOWERS the tile albedo (was inverted).
    T = T_FREEZE - 0.5
    shallow = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
                                           _arr(1.0), _arr(0.01))[0][0])
    deep = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
                                        _arr(1.0), _arr(1.0))[0][0])
    assert deep < shallow


def test_pond_deep_floor_clamped_to_thin_dark_ice():
    # CANARY: a pond over THIN (dark) ice must NOT brighten as it deepens — the
    # deep floor is min(alpha_pond, alpha_ice_underneath).  With very thin ice
    # (sqrt ramp -> ice bands near the ocean floor 0.06 < 0.27/0.07) the pond
    # band stays pinned at the ice albedo for all depths (monotone non-increasing).
    T = T_FREEZE - 3.0
    a_shallow = float(delta_eddington_albedo(_arr(T), _arr(2e-4), _arr(0.0),
                                             _arr(1.0), _arr(0.001))[0][0])
    a_deep = float(delta_eddington_albedo(_arr(T), _arr(2e-4), _arr(0.0),
                                          _arr(1.0), _arr(1.0))[0][0])
    assert a_deep <= a_shallow + 1e-12   # never brightens


# ---------------------------------------------------------------------------
# 7. Snow masking departure canary.
# ---------------------------------------------------------------------------
def test_snow_masks_bare_ice_above_mask_depth():
    # With h_snow >= H_SNOW_MASK (f_snow == 1) AND >= H_SNOW_SAT (snow ramp
    # saturated), the tile albedo is the pure snow albedo REGARDLESS of the
    # underlying bare-ice albedo — vary h_ice (thin vs thick, hence different
    # ice bands) and the result is unchanged, proving the mask.
    T = T_FREEZE - 3.0
    a_thin_ice = float(delta_eddington_albedo(_arr(T), _arr(0.1), _arr(0.10),
                                              _arr(0.0), _arr(0.0))[0][0])
    a_thick_ice = float(delta_eddington_albedo(_arr(T), _arr(3.0), _arr(0.10),
                                               _arr(0.0), _arr(0.0))[0][0])
    np.testing.assert_allclose(a_thin_ice, a_thick_ice, rtol=1e-13)
    # ...and it equals the pure cold-snow band mix (full ramp).
    expect = F_VIS * A_SNOW_COLD_VIS + (1 - F_VIS) * A_SNOW_COLD_NIR
    np.testing.assert_allclose(a_thin_ice, expect, rtol=1e-13)


# ---------------------------------------------------------------------------
# 7b. Surrogate ramp-length pins (codex R1: the isolation endpoints alone did
#     not fix H_BARE_SAT / H_SNOW_SAT / H_SNOW_MASK / H_POND_SAT).
# ---------------------------------------------------------------------------
def test_bare_ice_sqrt_ramp_saturates_at_h_bare_sat():
    # Pure cold bare ice.  Saturates AT h == H_BARE_SAT=0.5 (albedo(0.5)==albedo
    # (2.0)); below it the sqrt ramp gives albedo(H_BARE_SAT/4)==0.5*albedo(sat).
    T = T_FREEZE - 3.0
    a_sat = float(delta_eddington_albedo(_arr(T), _arr(0.5), _arr(0.0),
                                         _arr(0.0), _arr(0.0))[0][0])
    a_thick = float(delta_eddington_albedo(_arr(T), _arr(2.0), _arr(0.0),
                                           _arr(0.0), _arr(0.0))[0][0])
    a_quarter = float(delta_eddington_albedo(_arr(T), _arr(0.5 / 4), _arr(0.0),
                                             _arr(0.0), _arr(0.0))[0][0])
    np.testing.assert_allclose(a_sat, a_thick, rtol=1e-13)       # saturated at 0.5
    np.testing.assert_allclose(a_quarter, 0.5 * a_sat, rtol=1e-12)  # sqrt(1/4)=1/2


def test_snow_ramp_saturates_at_h_snow_sat():
    # Full snow coverage (h_snow >= H_SNOW_MASK=0.02 so f_snow==1) isolates the
    # SNOW-BAND ramp (h_snow/H_SNOW_SAT): albedo(0.05)==albedo(0.5) (saturated);
    # albedo(H_SNOW_SAT/2=0.025) is midway from bare ice to snow.
    T = T_FREEZE - 3.0
    full = F_VIS * A_SNOW_COLD_VIS + (1 - F_VIS) * A_SNOW_COLD_NIR
    a_sat = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.05),
                                         _arr(0.0), _arr(0.0))[0][0])
    a_deep = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.5),
                                          _arr(0.0), _arr(0.0))[0][0])
    a_half = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.025),
                                          _arr(0.0), _arr(0.0))[0][0])
    np.testing.assert_allclose(a_sat, full, rtol=1e-13)         # saturated at 0.05
    np.testing.assert_allclose(a_deep, full, rtol=1e-13)
    bare = F_VIS * A_ICE_COLD_VIS + (1 - F_VIS) * A_ICE_COLD_NIR
    np.testing.assert_allclose(a_half, 0.5 * (bare + full), rtol=1e-12)


def test_snow_mask_coverage_threshold_at_h_snow_mask(monkeypatch):
    # Isolate f_snow == min(h_snow/H_SNOW_MASK, 1): patch snow band to 1, bare/pond
    # to 0 -> alpha_total == f_snow exactly.  Pins H_SNOW_MASK=0.02.
    _patch_marker_bands(monkeypatch, (1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    # snow==1/bare==pond==0 => alpha_total == f_snow exactly; f_snow at these
    # depths is 0.5 / 1.0 / 1.0 (bit-exact), so assert with '=='.
    for h_snow, expect in [(0.01, 0.5), (0.02, 1.0), (0.05, 1.0)]:
        alpha, _ = delta_eddington_albedo(_arr(T_FREEZE - 3.0), _arr(1.0),
                                          _arr(h_snow), _arr(0.5), _arr(0.1))
        assert float(alpha[0]) == expect


def test_pond_decay_saturates_at_h_pond_sat():
    # Pure pond over thick cold ice.  Pond band decays linearly to the deep floor,
    # saturating AT pond_depth == H_POND_SAT=0.3.  albedo(0.3)==albedo(0.6); at
    # half-depth it is the midpoint between the ice albedo and the deep floor.
    T = T_FREEZE - 3.0
    ice = F_VIS * A_ICE_COLD_VIS + (1 - F_VIS) * A_ICE_COLD_NIR
    deep = F_VIS * A_POND_VIS + (1 - F_VIS) * A_POND_NIR
    a_sat = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
                                         _arr(1.0), _arr(0.3))[0][0])
    a_deeper = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
                                            _arr(1.0), _arr(0.6))[0][0])
    a_half = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
                                          _arr(1.0), _arr(0.15))[0][0])
    np.testing.assert_allclose(a_sat, deep, rtol=1e-13)          # saturated at 0.3
    np.testing.assert_allclose(a_deeper, deep, rtol=1e-13)
    np.testing.assert_allclose(a_half, 0.5 * (ice + deep), rtol=1e-12)  # linear midpoint


# ---------------------------------------------------------------------------
# 8. Published-constant canary (BL07 / MU71 / Grenfell-Maykut).
# ---------------------------------------------------------------------------
def test_published_constants_canary():
    assert constants.alpha_snow_cold_vis == A_SNOW_COLD_VIS
    assert constants.alpha_snow_cold_nir == A_SNOW_COLD_NIR
    assert constants.alpha_snow_melt_vis == A_SNOW_MELT_VIS
    assert constants.alpha_snow_melt_nir == A_SNOW_MELT_NIR
    assert constants.alpha_ice_cold_vis == A_ICE_COLD_VIS
    assert constants.alpha_ice_cold_nir == A_ICE_COLD_NIR
    assert constants.alpha_ice_melt_vis == A_ICE_MELT_VIS
    assert constants.alpha_ice_melt_nir == A_ICE_MELT_NIR
    assert constants.alpha_pond_max_vis == A_POND_VIS
    assert constants.alpha_pond_max_nir == A_POND_NIR
    assert constants.i0_vis == I0_VIS
    assert constants.i0_nir == I0_NIR
    assert constants.alpha_ocean_broadband == A_OCEAN


# ---------------------------------------------------------------------------
# 9. Dispatch hardening.
# ---------------------------------------------------------------------------
def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        _sw(200.0, 260.0, 1.0, 0.0, 0.0, 0.0, "bogus_scheme")


# ---------------------------------------------------------------------------
# 10. Differentiability at the sqrt-floor (open-water edge) and clip kinks.
# ---------------------------------------------------------------------------
def test_gradient_finite_at_open_water_sqrt_floor():
    # h_ice == 0 is the ice edge; sqrt'(0) is infinite but the _H_SQRT_FLOOR
    # guard must keep d(alpha)/d(h) finite (zero via the max).
    def loss(h):
        alpha, trans = delta_eddington_albedo(_arr(260.0), h, _arr(0.0),
                                              _arr(0.0), _arr(0.0))
        return jnp.sum(alpha ** 2 + trans ** 2)
    g = jax.grad(loss)(jnp.asarray([0.0], dtype=jnp.float64))
    assert jnp.all(jnp.isfinite(g))


def test_gradient_finite_at_penetration_clamp_tie():
    # The min() clamp kink is where transmittance*sw == net == (1-alpha)*sw, i.e.
    # tau == 1-alpha.  With alpha=0.5, tau=0.5 sits EXACTLY on the tie for all sw
    # (codex R1: tau=5 was inside the clamped region, not at the kink).  Grad wrt
    # sw_down must stay finite at the tie.
    def loss(sw):
        out = compute_ice_sw(sw, _arr(255.0), _arr(0.2), _arr(0.0),
                             _arr(0.0), _arr(0.0),
                             scheme="constant", albedo_const=0.5,
                             sw_transmittance_const=0.5)
        return jnp.sum(out.sw_penetrated ** 2 + out.sw_absorbed_surface ** 2)
    g = jax.grad(loss)(jnp.asarray([300.0], dtype=jnp.float64))
    assert jnp.all(jnp.isfinite(g))
