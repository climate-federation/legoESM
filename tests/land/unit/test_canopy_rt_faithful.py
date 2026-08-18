"""Canopy SHORTWAVE radiative-transfer ORACLE-FAITHFULNESS tests.

``canopy/radiative_transfer`` implements the two-leaf shortwave two-stream of
Sellers (1985) / Ryu et al. (2011) with an Erbs et al. (1982) direct/diffuse
split and fixed PAR/NIR/UV spectral fractions (visible/NIR broadly after Weiss
& Norman 1985; the 2% UV band is a model choice).  The existing
test_canopy_rt.py is behavioral/conservation (total == sw_down, night -> 0,
0 <= fSun <= 1, absorbed <= incident) and rich on the LONGWAVE side, but it
never pins the shortwave closed FORMS: the Erbs f_d(k_t) branches, the Beer-law
sunlit fraction, the Ryu extinction coefficients, or the beam/diffuse/scattered
sunlit/shaded/soil partition.

These pin the shortwave ALGEBRA to round-off (rel 1e-9) against an independent
scalar reimplementation:
  * split_sw_components -> Erbs (1982) 3-branch f_d(k_t) + fixed spectral split.
  * canopy_shortwave_rt -> the full Sellers/Ryu two-stream: extinction table,
    fSun = (1 - e^{-kb L_CI})/(kb LAI), the PAR/NIR beam+diffuse+scattered
    partition with the k/(k+kb) weights, the UV band, and the exponential-N
    canopy Vcmax25 integral.

Independence: every Erbs / spectral-fraction / Sellers-Ryu SCHEME coefficient is
a local ``_O_*`` literal here, canaried against the module constant
(module == _O_* == value); S_0 is the shared legoesm.constants value, not the
SUT.  The numeric guard thresholds (night 89 deg, cos floor 0.01, sentinel 50,
epsilons) have no named module symbol, so the oracle reproduces them raw purely
to match the guard branches.  Departures (night guards, fSun clip, max(.,0)
floors, profile guards, the tanh dusk ramp) are reproduced by the oracle so the
pin stays exact, and are separately probed by dedicated canaries.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

# The rel-1e-9 oracle pins need float64.  Rather than force x64 at MODULE scope
# (which leaks x64=True into any other test file whenever a single test here is
# selected), an autouse fixture sets it per-test and ALWAYS restores the
# process-entry state in finally — so importing/selecting this module never
# mutates global JAX config past the individual test.
_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)

from legoesm import constants                                        # noqa: E402
from legoesm.land.canopy import radiative_transfer as rt             # noqa: E402
from legoesm.land.canopy.radiative_transfer import (                 # noqa: E402
    split_sw_components,
    canopy_shortwave_rt,
    _APAR_CONVERSION,
    PAR_FRACTION, NIR_FRACTION, UV_FRACTION,
    _ERBS_KT_LOW, _ERBS_KT_HIGH, _ERBS_LOW_SLOPE,
    _ERBS_MID_C0, _ERBS_MID_C1, _ERBS_MID_C2, _ERBS_MID_C3, _ERBS_MID_C4,
    _ERBS_FD_HIGH,
    _SIGMA_PAR, _SIGMA_NIR, _RHO_PAR_SOIL, _RHO_NIR_SOIL,
    _KPB_PAR, _KD_PAR, _KD_NIR_COEF, RHO_UV, _KB_BEAM,
    _NIGHT_RAMP_CENTER_WM2, _NIGHT_RAMP_HALFWIDTH_WM2,
)

# --- Independent oracle literals (canaried against the module constants in
#     test_rt_constants_match_published_literals) --------------------------------
# Weiss & Norman (1985) spectral fractions.
_O_PAR_FRAC, _O_NIR_FRAC, _O_UV_FRAC = 0.48, 0.50, 0.02
# Erbs et al. (1982) diffuse-fraction correlation.
_O_KT_LOW, _O_KT_HIGH, _O_LOW_SLOPE = 0.22, 0.80, 0.09
_O_MID = (0.9511, 0.1604, 4.388, 16.638, 12.336)
_O_FD_HIGH = 0.165
# Sellers (1985) / Ryu (2011) scattering + extinction.
_O_SIGMA_PAR, _O_SIGMA_NIR = 0.175, 0.825
_O_RHO_PAR_SOIL, _O_RHO_NIR_SOIL = 0.15, 0.30
_O_KPB_PAR, _O_KD_PAR, _O_KD_NIR_COEF = 0.46, 0.72, 0.35
_O_RHO_UV, _O_KB_BEAM = 0.05, 0.5
_O_APAR_CONV = 4.56
_O_RAMP_CENTER, _O_RAMP_HALF = 30.0, 20.0
# Numeric GUARD thresholds. These are NOT physical scheme coefficients and the
# module writes them as inline `# coeff-ok` literals (no named symbol to canary
# against), so they are reproduced raw here purely so the oracle matches the
# production guard branches exactly — the independence claim scopes to the
# Erbs / Weiss-Norman / Sellers-Ryu SCHEME coefficients above.
_O_NIGHT_SZA_DEG = 89.0
_O_COS_FLOOR = 0.01
_O_NIGHT_KB_SENTINEL = 50.0
_O_UV_SPLIT_EPS = 1e-5
_O_PROFILE_EPS = 1e-6


def _erbs_fd(k_t):
    """Independent Erbs (1982) diffuse fraction f_d(k_t), clipped to [0, 1]."""
    if k_t <= _O_KT_LOW:
        fd = 1.0 - _O_LOW_SLOPE * k_t
    elif k_t <= _O_KT_HIGH:
        c0, c1, c2, c3, c4 = _O_MID
        fd = c0 - c1 * k_t + c2 * k_t**2 - c3 * k_t**3 + c4 * k_t**4
    else:
        fd = _O_FD_HIGH
    return min(max(fd, 0.0), 1.0)


def _split_oracle(sw_down, cos_z):
    """Independent split_sw_components (Erbs + Weiss-Norman)."""
    is_day = cos_z > _O_COS_FLOOR
    I_ext = float(constants.S_0) * (cos_z if is_day else 1.0)
    k_t = min(max(sw_down / I_ext, 0.0), 1.0)
    fd = _erbs_fd(k_t)
    sw_diff = fd * sw_down
    sw_dir = sw_down - sw_diff
    out = [_O_PAR_FRAC * sw_dir, _O_PAR_FRAC * sw_diff,
           _O_NIR_FRAC * sw_dir, _O_NIR_FRAC * sw_diff,
           _O_UV_FRAC * sw_down]
    if not is_day:
        out = [0.0] * 5
    return tuple(out)


def _sw_rt_oracle(PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV, SZA, LAI, CI,
                  ALB_VIS, ALB_NIR, Vc3, Vc4, kn, FNonVeg):
    """Independent Sellers/Ryu two-stream shortwave RT -> dict of the 10 outputs.

    Transcribed term-for-term from Bonan/Ryu closed forms with independent
    literal coefficients; reproduces every production guard so the pin is exact.
    """
    night = SZA > _O_NIGHT_SZA_DEG
    sigma_P, sigma_N = _O_SIGMA_PAR, _O_SIGMA_NIR
    rho_PSoil = _O_RHO_PAR_SOIL * (1.0 - FNonVeg)
    rho_NSoil = _O_RHO_NIR_SOIL * (1.0 - FNonVeg)
    cos_sza = math.cos(math.radians(SZA))
    cos_sza_safe = _O_COS_FLOOR if night else cos_sza
    kb = _O_KB_BEAM / cos_sza_safe
    kk_Pb = _O_KPB_PAR / cos_sza_safe
    if night:
        kb, kk_Pb = _O_NIGHT_KB_SENTINEL, _O_NIGHT_KB_SENTINEL
    kk_Pd = _O_KD_PAR
    kk_Nb = kb * math.sqrt(1.0 - sigma_N)
    kk_Nd = _O_KD_NIR_COEF * math.sqrt(1.0 - sigma_N)

    L_CI = LAI * CI
    if LAI > 0:
        fSun = min(max((1.0 / kb) * (1.0 - math.exp(-kb * L_CI)) / LAI, 0.0), 1.0)
    else:
        fSun = 0.0
    exp_kk_Pd = math.exp(-kk_Pd * L_CI)
    exp_kk_Nd = math.exp(-kk_Nd * L_CI)

    # PAR
    Q_PDn = ((1.0 - ALB_VIS) * PAR_dir * (1.0 - math.exp(-kk_Pb * L_CI))
             + (1.0 - ALB_VIS) * PAR_diff * (1.0 - exp_kk_Pd))
    Q_PbSunDn = PAR_dir * (1.0 - sigma_P) * (1.0 - math.exp(-kb * L_CI))
    Q_PdSunDn = (PAR_diff * (1.0 - ALB_VIS)
                 * (1.0 - math.exp(-(kk_Pd + kb) * L_CI)) * kk_Pd / (kk_Pd + kb))
    Q_PsSunDn = max(PAR_dir * ((1.0 - ALB_VIS)
                               * (1.0 - math.exp(-(kk_Pb + kb) * L_CI))
                               * kk_Pb / (kk_Pb + kb)
                               - (1.0 - sigma_P) * (1.0 - math.exp(-2.0 * kb * L_CI)) / 2.0),
                    0.0)
    Q_PSunDn = Q_PbSunDn + Q_PdSunDn + Q_PsSunDn
    Q_PShDn = max(Q_PDn - Q_PSunDn, 0.0)
    I_PSoil = ((1.0 - ALB_VIS) * PAR_dir + (1.0 - ALB_VIS) * PAR_diff
               - (Q_PSunDn + Q_PShDn))
    APAR_Soil = (1.0 - rho_PSoil) * I_PSoil
    Q_PSunUp = I_PSoil * rho_PSoil * exp_kk_Pd
    Q_PShUp = I_PSoil * rho_PSoil * (1.0 - exp_kk_Pd)
    APAR_Sun = Q_PSunDn + Q_PSunUp
    APAR_Sh = Q_PShDn + Q_PShUp

    # NIR
    Q_NSunDn = (NIR_dir * (1.0 - sigma_N) * (1.0 - math.exp(-kb * L_CI))
                + NIR_diff * (1.0 - ALB_NIR)
                * (1.0 - math.exp(-(kk_Nd + kb) * L_CI)) * kk_Nd / (kk_Nd + kb)
                + NIR_dir * ((1.0 - ALB_NIR)
                             * (1.0 - math.exp(-(kk_Nb + kb) * L_CI))
                             * kk_Nb / (kk_Nb + kb)
                             - (1.0 - sigma_N) * (1.0 - math.exp(-2.0 * kb * L_CI)) / 2.0))
    Q_NShDn = ((1.0 - ALB_NIR) * NIR_dir * (1.0 - math.exp(-kk_Nb * L_CI))
               + (1.0 - ALB_NIR) * NIR_diff * (1.0 - exp_kk_Nd) - Q_NSunDn)
    I_NSoil = ((1.0 - ALB_NIR) * NIR_dir + (1.0 - ALB_NIR) * NIR_diff
               - (Q_NSunDn + Q_NShDn))
    ANIR_Soil = (1.0 - rho_NSoil) * I_NSoil
    Q_NSunUp = I_NSoil * rho_NSoil * exp_kk_Nd
    Q_NShUp = I_NSoil * rho_NSoil * (1.0 - exp_kk_Nd)
    ANIR_Sun = Q_NSunDn + Q_NSunUp
    ANIR_Sh = Q_NShDn + Q_NShUp

    # UV
    total_PAR = PAR_dir + PAR_diff + _O_UV_SPLIT_EPS
    UV_dir = UV * PAR_dir / total_PAR
    UV_diff = UV - UV_dir
    Q_U = ((1.0 - _O_RHO_UV) * UV_dir * (1.0 - math.exp(-kk_Pb * L_CI))
           + (1.0 - _O_RHO_UV) * UV_diff * (1.0 - exp_kk_Pd))
    AUV_Sun = Q_U * fSun
    AUV_Sh = Q_U * (1.0 - fSun)
    AUV_Soil = (1.0 - _O_RHO_UV) * UV - Q_U

    ASW_Sun = APAR_Sun + ANIR_Sun + AUV_Sun
    ASW_Sh = APAR_Sh + ANIR_Sh + AUV_Sh
    ASW_Soil = APAR_Soil + ANIR_Soil + AUV_Soil

    APAR_Sun *= _O_APAR_CONV
    APAR_Sh *= _O_APAR_CONV

    # Vcmax nitrogen-profile integral
    kn_kb_Lc = kn + kb * LAI
    kn_CI_safe = max(kn * CI, _O_PROFILE_EPS)
    LAI_Vc3 = LAI * Vc3
    Vc3Tot = LAI_Vc3 * (1.0 - math.exp(-kn * CI)) / kn_CI_safe
    Vc3Sun = LAI_Vc3 * (1.0 - math.exp(-CI * kn_kb_Lc)) / max(kn_kb_Lc, _O_PROFILE_EPS)
    Vc3Sh = Vc3Tot - Vc3Sun
    LAI_Vc4 = LAI * Vc4
    Vc4Tot = LAI_Vc4 * (1.0 - math.exp(-kn * CI)) / kn_CI_safe
    Vc4Sun = LAI_Vc4 * (1.0 - math.exp(-CI * kn_kb_Lc)) / max(kn_kb_Lc, _O_PROFILE_EPS)
    Vc4Sh = Vc4Tot - Vc4Sun

    # LAI > 0 guard (NOT applied to ASW_Soil or fSun, matching production)
    if not (LAI > 0):
        APAR_Sun = APAR_Sh = ASW_Sun = ASW_Sh = 0.0
        Vc3Sun = Vc3Sh = Vc4Sun = Vc4Sh = 0.0

    # tanh dusk ramp on fSun
    sw_dir_total = PAR_dir + NIR_dir
    day_weight = 0.5 * (1.0 + math.tanh(
        (sw_dir_total - _O_RAMP_CENTER) / _O_RAMP_HALF))
    fSun = fSun * day_weight

    return dict(
        fSun=fSun, APAR_Sun=APAR_Sun, APAR_Sh=APAR_Sh,
        ASW_Sun=ASW_Sun, ASW_Sh=ASW_Sh, ASW_Soil=ASW_Soil,
        Vcmax25_C3Sun=Vc3Sun, Vcmax25_C3Sh=Vc3Sh,
        Vcmax25_C4Sun=Vc4Sun, Vcmax25_C4Sh=Vc4Sh,
    )


def _arr(x):
    return jnp.array([float(x)])


# --- split_sw_components: Erbs + Weiss-Norman ----------------------------------

@pytest.mark.parametrize("sw_down,cos_z", [
    (1250.0, 0.92),  # very clear, k_t=0.998 -> Erbs HIGH branch (f_d = 0.165)
    (900.0, 0.90),   # clear, k_t=0.735 -> Erbs mid quartic (upper end)
    (500.0, 0.60),   # partly cloudy, k_t=0.612 -> Erbs mid quartic branch
    (120.0, 0.50),   # overcast, k_t=0.176 -> Erbs LOW linear branch
    (40.0, 0.30),    # very low k_t=0.098 -> Erbs low linear branch
    (300.0, 0.005),  # sun below horizon -> night guard (all zero)
])
def test_split_sw_matches_erbs_weiss_norman_oracle(sw_down, cos_z):
    """split_sw_components PAR/NIR/UV direct+diffuse match the independent Erbs
    (1982) f_d + fixed-spectral-fraction reimplementation across all three Erbs
    branches (low/mid/high, pinned on the SUT) and the night guard."""
    got = split_sw_components(_arr(sw_down), _arr(cos_z))
    exp = _split_oracle(sw_down, cos_z)
    for g, e in zip(got, exp):
        assert float(g[0]) == pytest.approx(e, rel=1e-9, abs=0.0)


def test_split_sw_erbs_branches_are_distinct():
    """Non-vacuity: the three Erbs branches give genuinely different f_d at a
    representative k_t in each, so a mis-set branch boundary would move the
    diffuse fraction (guards against a collapsed/constant f_d)."""
    # k_t values chosen to land squarely in low / mid / high branches.
    fd_low = _erbs_fd(0.10)
    fd_mid = _erbs_fd(0.50)
    fd_high = _erbs_fd(0.95)
    assert fd_low > 0.99                      # near-clear-sky-free: mostly diffuse
    assert 0.10 < fd_mid < 0.99               # quartic interior
    assert fd_high == _O_FD_HIGH              # saturated high branch (exact)
    assert fd_low != fd_mid != fd_high


# --- canopy_shortwave_rt: full two-stream --------------------------------------

_SW_CASES = [
    # PAR_dir,PAR_diff,NIR_dir,NIR_diff,UV, SZA, LAI, CI, ALB_VIS,ALB_NIR, Vc3,Vc4, kn, FNonVeg
    (300.0, 90.0, 320.0, 95.0, 12.0, 30.0, 3.0, 0.8, 0.08, 0.25, 60.0, 30.0, 0.30, 0.10),
    (180.0, 140.0, 190.0, 150.0, 9.0, 55.0, 5.0, 0.7, 0.10, 0.30, 55.0, 25.0, 0.50, 0.20),
    (400.0, 40.0, 420.0, 45.0, 16.0, 15.0, 1.5, 0.9, 0.06, 0.22, 70.0, 35.0, 0.20, 0.05),
    # SZA>89 night sentinel (kb=kk_Pb=50) with NON-zero incident flux -> pins the
    # sentinel-extinction branch, not just a zero-in/zero-out identity.
    (50.0, 40.0, 55.0, 45.0, 3.0, 95.0, 3.0, 0.8, 0.08, 0.25, 60.0, 30.0, 0.30, 0.10),
    # kn -> ~0 engages the max(kn*CI, eps) profile guard (kn+kb*LAI ~ 2.3 here).
    (300.0, 90.0, 320.0, 95.0, 12.0, 30.0, 4.0, 0.8, 0.08, 0.25, 60.0, 30.0, 1e-8, 0.10),
    # tiny LAI AND tiny kn -> engages BOTH profile denominators
    # max(kn*CI, eps) and max(kn+kb*LAI, eps).
    (300.0, 90.0, 320.0, 95.0, 12.0, 30.0, 1e-7, 0.8, 0.08, 0.25, 60.0, 30.0, 1e-8, 0.10),
    # tiny PAR (5e-6, comparable to the 1e-5 eps) with UV present -> the
    # total_PAR += 1e-5 UV-split eps is LOAD-BEARING in UV_dir=UV*PAR_dir/total_PAR
    # (UV/3 with the eps vs UV without it), so the pin actually depends on it.
    (5e-6, 0.0, 200.0, 60.0, 10.0, 30.0, 3.0, 0.8, 0.08, 0.25, 60.0, 30.0, 0.30, 0.10),
    # dense snow-bright canopy at oblique sun -> drives the scattered-sunlit
    # max(.,0) PAR floor (high ALB_VIS makes the raw scattered term negative).
    (350.0, 30.0, 360.0, 35.0, 14.0, 75.0, 8.0, 0.95, 0.30, 0.20, 65.0, 32.0, 0.40, 0.05),
]


@pytest.mark.parametrize("case", _SW_CASES)
def test_canopy_shortwave_rt_matches_sellers_ryu_oracle(case):
    """Every CanopySWOutput field matches the independent Sellers/Ryu two-stream
    reimplementation to round-off — pinning fSun, the PAR/NIR/UV
    sunlit/shaded/soil partition, and the N-profile Vcmax integral."""
    (PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV, SZA, LAI, CI,
     ALB_VIS, ALB_NIR, Vc3, Vc4, kn, FNonVeg) = case
    out = canopy_shortwave_rt(
        _arr(PAR_dir), _arr(PAR_diff), _arr(NIR_dir), _arr(NIR_diff), _arr(UV),
        _arr(SZA), _arr(LAI), _arr(CI), _arr(ALB_VIS), _arr(ALB_NIR),
        _arr(Vc3), _arr(Vc4), _arr(kn), _arr(FNonVeg))
    exp = _sw_rt_oracle(PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV, SZA, LAI, CI,
                        ALB_VIS, ALB_NIR, Vc3, Vc4, kn, FNonVeg)
    for field, e in exp.items():
        g = float(getattr(out, field)[0])
        assert g == pytest.approx(e, rel=1e-9, abs=0.0), (field, g, e)


def test_canopy_shortwave_sunlit_fraction_beer_law_and_vcmax_partition():
    """Non-vacuity structural checks on the PRODUCTION output: fSun decreases with
    LAI (Beer-law shading), and canopy-integrated Vcmax sunlit + shaded == total
    for BOTH C3 and C4 (the N-profile integral partitions, never creates, Vcmax)."""
    SZA, CI, kn = 25.0, 0.85, 0.30
    Vc3, Vc4 = 60.0, 30.0
    fsun_prev = None
    for LAI in (1.0, 3.0, 6.0):
        out = canopy_shortwave_rt(
            _arr(300.0), _arr(90.0), _arr(320.0), _arr(95.0), _arr(12.0),
            _arr(SZA), _arr(LAI), _arr(CI), _arr(0.08), _arr(0.25),
            _arr(Vc3), _arr(Vc4), _arr(kn), _arr(0.1))
        # sunlit + shaded == total (partition closes) — total from the N-profile
        # integral with the SAME beam kb the production uses.
        kb = _O_KB_BEAM / math.cos(math.radians(SZA))
        tot3 = LAI * Vc3 * (1.0 - math.exp(-kn * CI)) / max(kn * CI, _O_PROFILE_EPS)
        tot4 = LAI * Vc4 * (1.0 - math.exp(-kn * CI)) / max(kn * CI, _O_PROFILE_EPS)
        assert (float(out.Vcmax25_C3Sun[0]) + float(out.Vcmax25_C3Sh[0])
                == pytest.approx(tot3, rel=1e-9, abs=0.0))
        assert (float(out.Vcmax25_C4Sun[0]) + float(out.Vcmax25_C4Sh[0])
                == pytest.approx(tot4, rel=1e-9, abs=0.0))
        assert float(out.Vcmax25_C3Sun[0]) > 0.0 and float(out.Vcmax25_C3Sh[0]) > 0.0
        if fsun_prev is not None:
            assert float(out.fSun[0]) < fsun_prev   # denser canopy -> smaller sunlit fraction
        fsun_prev = float(out.fSun[0])


def test_canopy_shortwave_scattered_sunlit_floor_engages():
    """Non-vacuity of the Q_PsSunDn max(.,0) floor: for the dense-canopy oblique
    case the UN-floored scattered-sunlit PAR term is genuinely negative, so the
    production floor is load-bearing (not a no-op), and the production output
    still matches the (floored) oracle."""
    PAR_dir, LAI, CI, SZA, ALB_VIS = 350.0, 8.0, 0.95, 75.0, 0.30
    kb = _O_KB_BEAM / math.cos(math.radians(SZA))
    kk_Pb = _O_KPB_PAR / math.cos(math.radians(SZA))
    L_CI = LAI * CI
    unfloored = PAR_dir * ((1.0 - ALB_VIS)
                           * (1.0 - math.exp(-(kk_Pb + kb) * L_CI)) * kk_Pb / (kk_Pb + kb)
                           - (1.0 - _O_SIGMA_PAR) * (1.0 - math.exp(-2.0 * kb * L_CI)) / 2.0)
    assert unfloored < 0.0            # the raw scattered term is negative here -> floor bites


def test_canopy_shortwave_bare_soil_guard():
    """Departure canary: LAI == 0 zeros the leaf/Vcmax outputs (the module's
    `jnp.where(LAI>0, ...)` bare-soil guard) while leaving ASW_Soil untouched.

    (The SZA>89 night EXTINCTION sentinel is a separate branch — it is pinned
    non-trivially by the nonzero-flux night case in _SW_CASES, not here: the
    module has no mskNight zeroing of absorbed flux, only this LAI guard.)"""
    bare = canopy_shortwave_rt(
        _arr(300.0), _arr(90.0), _arr(320.0), _arr(95.0), _arr(12.0),
        _arr(30.0), _arr(0.0), _arr(0.8), _arr(0.08), _arr(0.25),
        _arr(60.0), _arr(30.0), _arr(0.30), _arr(0.1))
    for f in ("APAR_Sun", "APAR_Sh", "ASW_Sun", "ASW_Sh", "fSun",
              "Vcmax25_C3Sun", "Vcmax25_C3Sh", "Vcmax25_C4Sun", "Vcmax25_C4Sh"):
        assert float(getattr(bare, f)[0]) == 0.0, f
    # Soil still absorbs the (unshaded) incident beam+diffuse — not zeroed.
    assert float(bare.ASW_Soil[0]) > 0.0


# --- constant canaries ---------------------------------------------------------

def test_rt_constants_match_published_literals():
    """Every Erbs / spectral-fraction / Sellers-Ryu SCHEME coefficient the oracle
    uses equals both the independent oracle literal and the module constant
    (module == _O_* == value) — no scheme coefficient is sourced from the SUT.
    (The numeric guard thresholds — 89, 0.01, 50, 1e-5, 1e-6 — are module inline
    `# coeff-ok` literals with no named symbol; the oracle reproduces them raw to
    match the guard branches.  Coverage: the SZA>89/cos-floor/kb-sentinel branch
    is pinned by the nonzero-flux night case, the 1e-5 UV-split eps by the
    tiny-PAR(5e-6)/UV case, and BOTH 1e-6 profile denominators by the
    tiny-LAI/tiny-kn case in _SW_CASES.  The fSun clip to [0,1] is an unreachable safety FOR VALID
    physical inputs (0 <= CI <= 1): the Beer fraction (1-e^{-kb L_CI})/(kb LAI) is
    analytically in [0, CI] ⊆ [0,1], so it has no dedicated canary; it only guards
    against out-of-range CI, which is not runtime-constrained.)"""
    assert PAR_FRACTION == _O_PAR_FRAC == 0.48
    assert NIR_FRACTION == _O_NIR_FRAC == 0.50
    assert UV_FRACTION == _O_UV_FRAC == 0.02
    assert _ERBS_KT_LOW == _O_KT_LOW == 0.22
    assert _ERBS_KT_HIGH == _O_KT_HIGH == 0.80
    assert _ERBS_LOW_SLOPE == _O_LOW_SLOPE == 0.09
    assert (_ERBS_MID_C0, _ERBS_MID_C1, _ERBS_MID_C2, _ERBS_MID_C3, _ERBS_MID_C4) == _O_MID
    assert _ERBS_FD_HIGH == _O_FD_HIGH == 0.165
    assert _SIGMA_PAR == _O_SIGMA_PAR == 0.175
    assert _SIGMA_NIR == _O_SIGMA_NIR == 0.825
    assert _RHO_PAR_SOIL == _O_RHO_PAR_SOIL == 0.15
    assert _RHO_NIR_SOIL == _O_RHO_NIR_SOIL == 0.30
    assert _KPB_PAR == _O_KPB_PAR == 0.46
    assert _KD_PAR == _O_KD_PAR == 0.72
    assert _KD_NIR_COEF == _O_KD_NIR_COEF == 0.35
    assert RHO_UV == _O_RHO_UV == 0.05
    assert _KB_BEAM == _O_KB_BEAM == 0.5
    assert _APAR_CONVERSION == _O_APAR_CONV == 4.56
    assert _NIGHT_RAMP_CENTER_WM2 == _O_RAMP_CENTER == 30.0
    assert _NIGHT_RAMP_HALFWIDTH_WM2 == _O_RAMP_HALF == 20.0


# --- AD-safety -----------------------------------------------------------------

def _par_diff(sw, cz):
    # PAR_diff = PAR_FRACTION * f_d(k_t) * sw_down — grad wrt sw_down traverses
    # the Erbs quartic (NOT identically sw_down, unlike the full-sum which is).
    return split_sw_components(sw, cz)[1][0]


def _sw_scalar(*args):
    o = canopy_shortwave_rt(*args)
    # Include fSun so LAI/SZA gradients traverse the tanh dusk ramp + 1/kb.
    return o.fSun[0] + o.APAR_Sun[0] + o.ASW_Sun[0] + o.Vcmax25_C3Sun[0]


def _fsun_of_beam(beam):
    """fSun as a function of total beam SW (split evenly into PAR_dir+NIR_dir so
    PAR_dir + NIR_dir == beam), for differentiating THROUGH the tanh dusk ramp
    whose only argument is PAR_dir + NIR_dir."""
    half = jnp.reshape(beam * 0.5, (1,))
    return canopy_shortwave_rt(
        half, _arr(90.0), half, _arr(95.0), _arr(12.0),
        _arr(30.0), _arr(3.0), _arr(0.8), _arr(0.08), _arr(0.25),
        _arr(60.0), _arr(30.0), _arr(0.30), _arr(0.1)).fSun[0]


def test_rt_grad_finite_x64_and_float32():
    """grad of the split and the two-stream is finite in both x64 and float32.

    Non-vacuity: (a) the split grad differentiates PAR_diff = frac*f_d(k_t)*
    sw_down (a genuine function of the Erbs branch, asserted non-zero, not the
    identity sw_down); (b) the LAI/SZA grads traverse the 1/kb sunlit fraction,
    the exp extinctions, and the max(.,eps) profile denominators; (c) a dedicated
    beam-flux grad of fSun at the tanh centre (30 W m-2) exercises the dusk ramp
    (asserted non-zero), since the ramp depends ONLY on PAR_dir + NIR_dir."""
    def _check():
        # split: grad wrt sw_down in the Erbs quartic branch (k_t=0.61 at cz=0.6).
        gs = jax.grad(lambda s: _par_diff(jnp.reshape(s, (1,)), _arr(0.6)))(jnp.array(500.0))
        assert bool(jnp.isfinite(gs)) and abs(float(gs)) > 0.0   # non-trivial slope
        # two-stream: grad wrt LAI and SZA (through 1/kb, exp, profile guards).
        args = [_arr(300.0), _arr(90.0), _arr(320.0), _arr(95.0), _arr(12.0),
                _arr(30.0), _arr(3.0), _arr(0.8), _arr(0.08), _arr(0.25),
                _arr(60.0), _arr(30.0), _arr(0.30), _arr(0.1)]

        def with_lai(lai):
            a = list(args)
            a[6] = jnp.array([lai])
            return _sw_scalar(*a)

        def with_sza(sza):
            a = list(args)
            a[5] = jnp.array([sza])
            return _sw_scalar(*a)

        gl = jax.grad(with_lai)(jnp.array(3.0))
        gz = jax.grad(with_sza)(jnp.array(30.0))
        assert bool(jnp.isfinite(gl)) and bool(jnp.isfinite(gz)), (gl, gz)
        # dusk ramp: grad of fSun wrt beam SW at the tanh centre (non-vacuous).
        gr = jax.grad(_fsun_of_beam)(jnp.array(30.0))
        assert bool(jnp.isfinite(gr)) and abs(float(gr)) > 0.0

    _check()
    # float32 leg — the autouse _force_x64 fixture restores the entry state after
    # this test, so no local finally is needed.
    jax.config.update("jax_enable_x64", False)
    _check()
