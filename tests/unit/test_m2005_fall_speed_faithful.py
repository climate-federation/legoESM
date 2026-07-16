"""Oracle-faithfulness pins for the SAM M2005 predicted-PSD fall speeds.

Oracle: gSAM ``MICRO_M2005`` (``module_mp_graupel.f90`` / ``micro_params.f90``),
the reference Fortran the ``fall_speed_scheme="m2005_psd"`` branch of
``legoesm.atmosphere.physics.microphysics.morrison`` ports.  The verbatim
mass-/number-weighted terminal-velocity block is (module_mp_graupel.f90
501-510, 569-572, 1854-1857, 1862-1865, 1907; slopes 1640/2549; density
corrections 1537/1542)::

    CONS3=GAMMA(4.+BS)/6.  CONS4=GAMMA(4.+BR)/6.  CONS5=GAMMA(1.+BS)  CONS6=GAMMA(1.+BR)
    ARN(K) = (RHOSU/RHO(K))**0.54 * AR        AIN(K) = (RHOSU/RHO(K))**0.35 * AI
    UMR = ARN*CONS4/LAMR**BR      UNR = ARN*CONS6/LAMR**BR
    UMR = MIN(UMR, 9.1*dum)       UMS = MIN(UMS, 1.2*dum)   UMG = MIN(UMG, 20.*dum)
    LAMR = (pi*RHOW*N_r/q_r)**(1/3),  LAMI = (pi*RHO_CI*N_i/q_i)**(1/3),
    LAMS = (pi*RHOSN*N_s/q_s)**(1/3),  LAMG = (pi*RHOG*N_g/q_g)**(1/3)
      clamped to [LAMMIN*, LAMMAX*]: LAMMINR=1/2800um LAMMAXR=1/20um ;
      LAMMINI=1/(2*DCS+100um)=1/600um LAMMAXI=1/1um ; LAMMINS=1/2000um LAMMAXS=1/10um ;
      LAMMING=1/2000um LAMMAXG=1/20um  (module_mp_graupel.f90:501-510)
    AR=841.99667 BR=0.8 ; AI=clice_fall_a=700 BI=clice_fall_b=0.865(gSAM MK)/1.0(M2005-orig) ;
    AS=11.72 BS=0.41 ; AG=19.3 BG=0.37 ; RHOSU=85000/(R*TMELT) ;
    RHOW=997 RHO_CI=500 RHOSN=100 RHOG=400  (micro_params.f90:43 / :458)

This module ports that block INLINE in ``morrison_microphysics`` (rain/ice/snow
at morrison.py:948-1038, graupel at :1054-1071).  Two isolation techniques pin
the inline speeds against the independent numpy oracle to round-off (rel 1e-12):

- MASS-weighted (rain, cloud ice): the sedimentation SURFACE FLUX.
  ``sedimentation_tendency`` sets ``flux = V_t*q*rho`` from the INITIAL ``q`` and
  returns the bottom flux as ``precipitation`` (output.py:187, 212).  A
  single-species 2-level column (species only in the base cell), a HUGE ``dz`` and
  SMALL ``dt`` (so the positivity/extra-sink cap can never bind), yields
  ``precipitation = V_t*q*rho`` exactly, so ``V_t = precip/(q*rho)``.  (Each
  species' surface flux uses its OWN initial ``q``; redistribution like ice->snow
  autoconversion moves only TENDENCIES, so a single-species column is clean.)
- MASS + NUMBER (all species): a sedimentation_tendency INTERCEPT.  The number
  speeds ``V_n_*`` have no surface-flux observable (the number TENDENCY is
  confounded by self-collection / aggregation), so a spy on
  ``morrison.sedimentation_tendency`` captures the actual ``V_t``/``V_n`` array the
  module passes for each species (the mass call has ``max|q|<1`` = mixing ratio,
  the number call has ``max|q|>1`` = concentration), compared to the oracle.  The
  captured MASS speed is cross-checked against the independent surface-flux
  recovery for rain/ice.

The ``_O_*`` oracle constants are typed from the Fortran (NOT read back from the
module — non-circular), including the slope LIMITS and particle densities, each
also canaried against the shipped config (drift test).  Snow and graupel are pinned
in their DOUBLE-moment (prognostic ``N_s``/``N_g``) mode — the SAM-faithful path
(``LAMS/LAMG = (pi*rho*N/q)**(1/3)``, matching ``(CONS2*N/Q)**(1/DG)``, DG=3);
single-moment graupel is a legoESM closure (departure #5).  Complements
``test_m5_psd_fall_speeds.py`` (behavioral: monotonicity, size-sorting, AD).

DEPARTURES from the SAM oracle (all deliberate, all canaried / documented below):
  1. The LAMR slope uses ``constants.rho_water = 1000`` (legoESM), NOT SAM's
     ``rho_water = 997`` (micro_params.f90:43) — a ~0.1% slope difference
     (a sanctioned ``legoesm.constants`` adaptation).
  2. ``rho_su = 85000/(R_d*T_freeze)`` uses legoESM ``constants.R_d`` where SAM's
     ``RHOSU = 85000/(R*TMELT)`` uses ``rgas`` — a ~0.03% density-correction
     difference (also a ``legoesm.constants`` adaptation; SAM's form is preserved).
  3. **Flavor-resolved cloud-ice fall exponent.**  ``morrison_microphysics`` runs
     ``resolve_morrison_flavor`` FIRST, and the GLOBAL DEFAULT ``morrison_flavor
     = "mg"`` overrides the cloud-ice fall exponent to the M2005-ORIGINAL
     ``BI = 1.0`` (plus ``lami_max = 1/10um``, ``rho_snow = 250``).  The gSAM
     Fortran oracle (``BI = 0.865``, the MK tune) maps to ``morrison_flavor="sam"``.
     The ice pins cover BOTH flavors against their published exponent; rain/graupel
     exponents are NOT flavor-overridden.
  4. **Air-density floor ``_RHO_FLOOR = 0.1 kg/m^3``.**  The slopes and density
     correction clip ``rho`` (and floor ``rho*q``) for AD/zero-division safety
     (morrison.py:958-995), so at ``rho < 0.1`` the result is NOT the Fortran
     formula.  The round-off pins hold for NORMAL atmospheric density
     (``rho > 0.1``); the floor is inert everywhere the pins run.
  5. **Single-moment graupel is a legoESM Marshall-Palmer closure**, NOT SAM:
     ``LAMG = (pi*rho_g*N0G/(rho*q_g))**(1/4)`` with a FIXED ``n0_graupel`` — SAM
     graupel is inherently two-moment (``LAMG=(CONS2*NG3D/QG3D)**(1/DG)``,
     module_mp_graupel.f90:1745).  This audit pins the SAM-faithful DOUBLE-moment
     graupel slope; the single-moment fixed-N0G form is out of the SAM oracle's scope.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.physics.microphysics import morrison as _m
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
    resolve_morrison_flavor,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice

from legoesm import constants

jax.config.update("jax_enable_x64", True)


# --- SAM M2005 fall-speed oracle constants (module_mp_graupel.f90 / micro_params.f90; verbatim) ---
_O_AR = 841.99667          # rain fall a          (module_mp_graupel.f90:429)
_O_BR = 0.8                # rain fall b          (:433)
_O_AI = 700.0             # cloud-ice fall a = clice_fall_a  (micro_params.f90:61; :426)
_O_BI = 0.865             # cloud-ice fall b, gSAM MK tune (:430)
_O_BI_MG = 1.0            # cloud-ice fall b, M2005-ORIGINAL (micro_params.f90:65) — the
#                          resolve_morrison_flavor("mg") override; see departure #3
_O_AS = 11.72             # snow fall a = snow_fall_a        (micro_params.f90:50; :428)
_O_BS = 0.41              # snow fall b = snow_fall_b        (:432)
_O_AG = 19.3              # graupel fall a       (:436)
_O_BG = 0.37              # graupel fall b       (:437)
_O_RHO_WATER_SAM = 997.0  # SAM rho_water (micro_params.f90:43) — legoESM uses 1000 (departure #1)
_O_RHO_CLOUD_ICE = 500.0  # SAM rho_cloud_ice    (micro_params.f90:43)
_O_RHO_SNOW = 100.0       # SAM rho_snow         (micro_params.f90:43)
_O_RHO_GRAUPEL = 400.0    # SAM RHOG             (module_mp_graupel.f90:458)
_O_FALL_RHO_EXP = 0.54    # (RHOSU/RHO)^0.54 rain/snow/graupel (:1537)
_O_FALL_RHO_EXP_ICE = 0.35   # (RHOSU/RHO)^0.35 cloud ice, Ikawa-Saito 1991 (:1542)
_O_VT_CAP_RAIN = 9.1      # UMR/UNR cap factor   (:1864)
_O_VT_CAP_SNOW_ICE = 1.2  # UMS/UMI cap factor   (:1862)
_O_VT_CAP_GRAUPEL = 20.0  # UMG cap factor       (:1907)
_O_RHOSU_NUM = 8.5e4      # RHOSU = 85000/(R*TMELT)   (:452)
# PSD slope clamps LAMMIN*/LAMMAX* (module_mp_graupel.f90:501-510; DCS=250um -> :502).
_O_LAMR_MIN = 1.0 / 2800.0e-6   # LAMMINR (:506)
_O_LAMR_MAX = 1.0 / 20.0e-6     # LAMMAXR (:503)
_O_LAMI_MIN = 1.0 / 600.0e-6    # LAMMINI = 1/(2*DCS+100um) (:502)
_O_LAMI_MAX = 1.0 / 1.0e-6      # LAMMAXI (:501, "sam" flavor)
_O_LAMS_MIN = 1.0 / 2000.0e-6   # LAMMINS (:508)
_O_LAMS_MAX = 1.0 / 10.0e-6     # LAMMAXS (:507)
_O_LAMG_MIN = 1.0 / 2000.0e-6   # LAMMING (:510)
_O_LAMG_MAX = 1.0 / 20.0e-6     # LAMMAXG (:509)

# Reference density the MODULE uses (RHOSU with legoESM constants — departure #2).
_RHO_SU = _O_RHOSU_NUM / (constants.R_d * constants.T_freeze)


def _um_oracle(a, b, lam, rho, dens_exp, cap):
    """Mass-weighted UM = a*Gamma(4+b)/6*LAM^-b*(RHOSU/rho)^dens_exp, capped."""
    dum = (_RHO_SU / rho) ** dens_exp
    um = a * math.gamma(4.0 + b) / 6.0 * lam ** (-b) * dum
    return min(um, cap * dum)


def _un_oracle(a, b, lam, rho, dens_exp, cap):
    """Number-weighted UN = a*Gamma(1+b)*LAM^-b*(RHOSU/rho)^dens_exp, capped."""
    dum = (_RHO_SU / rho) ** dens_exp
    un = a * math.gamma(1.0 + b) * lam ** (-b) * dum
    return min(un, cap * dum)


def _lam_slope(rho_particle, N, q, rho, per_volume, lam_min, lam_max):
    """PSD slope LAM = (pi*rho_particle*N/denom)^(1/3), clamped to [lam_min, lam_max].

    per_volume=True (rain, N in #/m^3): denom = rho*q (converts N to per-mass);
    per_volume=False (ice/snow/graupel, N in #/kg): denom = q.
    """
    denom = rho * q if per_volume else q
    lam = (math.pi * rho_particle * N / denom) ** (1.0 / 3.0)
    return min(max(lam, lam_min), lam_max)


# --- isolation harness: single-species 2-level column, species in the base cell ---
_P_FULL = jnp.asarray([[7.0e4, 9.0e4]])
_P_HALF = jnp.asarray([[6.0e4, 8.0e4, 1.0e5]])
# The surface flux V_t*q*rho is INDEPENDENT of dz AND dt; only the positivity /
# extra-sink cap ceiling (q - extra_sink*dt)*rho*dz/dt binds it. A very large dz
# AND a small dt keep that ceiling far above V_t*q*rho so it never binds — even
# for cloud ice, whose in-column aggregation sink (~2.5e-6/s at N_i=1e5) would over
# a 20 s step consume ~all of q_i and cap the sed flux near zero (deposition offsets
# it in the NET tendency, hiding it, but the cap uses the gross sink). With
# dt=0.1 s the sink removes <1% of q_i, so precip = V_t*q*rho exactly here.
_DZ = jnp.full((1, 2), 1.0e6)
_DT = 0.1


def _rho_col(T):
    return _P_FULL / (constants.R_d * T)


def _rho_bottom(T):
    return float(_rho_col(T)[0, -1])


def _cfg(flavor):
    return MorrisonConfig(morrison_flavor=flavor, N_i0=0.0)


def _run(T, q_v, hm, cfg):
    return morrison_microphysics(
        jnp.full((1, 2), T), q_v, hm, _P_FULL, _P_HALF, _rho_col(T), _DZ, _DT, cfg)


def _base(**species):
    """Build a HydrometeorState with the given fields set in the base cell only."""
    z = jnp.zeros((1, 2))
    fields = dict(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)
    for k, v in species.items():
        fields[k] = z.at[0, -1].set(v)
    return HydrometeorState(**fields)


def _warm_qv():
    return saturation_mixing_ratio(jnp.asarray(290.0), _P_FULL)     # RH=1 => no evap/cond


def _cold_qv(T=250.0):
    return saturation_mixing_ratio_ice(jnp.asarray(T), _P_FULL)     # S_ice=1 => dep=0, nucl off


def _recover_vt_rain(q_r, N_r, flavor="sam"):
    """Mass-weighted rain fall speed via surface precip (rain flavor-invariant)."""
    hm = _base(q_r=q_r, N_r=N_r)
    out = _run(290.0, _warm_qv(), hm, _cfg(flavor))
    return float(out.precipitation[0]) / (q_r * _rho_bottom(290.0))


def _recover_vt_ice(q_i, N_i, flavor, T=250.0):
    """Mass-weighted cloud-ice fall speed via surface precip (flavor-resolved BI)."""
    hm = _base(q_i=q_i, N_i=N_i)
    out = _run(T, _cold_qv(T), hm, _cfg(flavor))
    return float(out.precipitation[0]) / (q_i * _rho_bottom(T))


def _captured_speeds(hm, T, q_v, flavor):
    """Spy on morrison.sedimentation_tendency; return {"mass":V, "number":V} for the
    single active species (mass call: max|q|<1 mixing ratio; number call: >1)."""
    orig = _m.sedimentation_tendency
    caught: dict[str, float] = {}

    def spy(q, rho, V_t, dz, dt=None, return_surface_flux=False, extra_sink=None):
        mq = float(jnp.max(jnp.abs(q)))
        if mq > 0.0:
            caught["mass" if mq < 1.0 else "number"] = float(V_t[0, -1])
        return orig(q, rho, V_t, dz, dt=dt,
                    return_surface_flux=return_surface_flux, extra_sink=extra_sink)

    _m.sedimentation_tendency = spy
    try:
        _run(T, q_v, hm, _cfg(flavor))
    finally:
        _m.sedimentation_tendency = orig
    return caught


# ===================== Part A: config defaults are the SAM Fortran values =====================

def test_config_fall_coefficients_match_sam_fortran():
    # The RAW config defaults (pre-flavor-resolution) are the gSAM values. Note
    # fall_b_i=0.865 here is the flavor-independent config value; the runtime "mg"
    # flavor overrides it to 1.0 (see test_flavor_resolution_*).
    cfg = MorrisonConfig()
    assert cfg.fall_a_r == _O_AR
    assert cfg.fall_b_r == _O_BR
    assert cfg.fall_a_i == _O_AI
    assert cfg.fall_b_i == _O_BI
    assert cfg.fall_a_s == _O_AS
    assert cfg.fall_b_s == _O_BS
    assert cfg.fall_a_g == _O_AG
    assert cfg.fall_b_g == _O_BG


def test_config_particle_densities_match_sam_fortran():
    cfg = MorrisonConfig()
    assert cfg.rho_cloud_ice == _O_RHO_CLOUD_ICE
    assert cfg.rho_snow == _O_RHO_SNOW
    assert cfg.rho_graupel == _O_RHO_GRAUPEL


def test_config_slope_limits_match_sam_fortran():
    # Slope clamps are typed from the Fortran (:501-510) and pinned here so the
    # oracle's use of them is NOT circular (a default drift would fail this).
    cfg = MorrisonConfig()
    assert cfg.lamr_min == pytest.approx(_O_LAMR_MIN, rel=1e-12)
    assert cfg.lamr_max == pytest.approx(_O_LAMR_MAX, rel=1e-12)
    assert cfg.lami_min == pytest.approx(_O_LAMI_MIN, rel=1e-12)
    assert cfg.lami_max == pytest.approx(_O_LAMI_MAX, rel=1e-12)   # raw default = SAM LAMMAXI
    assert cfg.lams_min == pytest.approx(_O_LAMS_MIN, rel=1e-12)
    assert cfg.lams_max == pytest.approx(_O_LAMS_MAX, rel=1e-12)
    assert cfg.lamg_min == pytest.approx(_O_LAMG_MIN, rel=1e-12)
    assert cfg.lamg_max == pytest.approx(_O_LAMG_MAX, rel=1e-12)


def test_config_rho_su_is_sam_form_with_legoesm_constants():
    cfg = MorrisonConfig()
    assert cfg.rho_su == pytest.approx(
        _O_RHOSU_NUM / (constants.R_d * constants.T_freeze), rel=1e-12)


def test_module_fall_speed_constants_match_sam_fortran():
    assert _m._FALL_RHO_EXP == _O_FALL_RHO_EXP
    assert _m._FALL_RHO_EXP_ICE == _O_FALL_RHO_EXP_ICE
    assert _m._VT_CAP_RAIN == _O_VT_CAP_RAIN
    assert _m._VT_CAP_SNOW_ICE == _O_VT_CAP_SNOW_ICE
    assert _m._VT_CAP_GRAUPEL == _O_VT_CAP_GRAUPEL


def test_flavor_resolution_overrides_cloud_ice_fall_exponent():
    # DEPARTURE #3: morrison_microphysics resolves morrison_flavor FIRST; the GLOBAL
    # DEFAULT "mg" overrides the cloud-ice fall exponent to M2005-original 1.0 (and
    # lami_max=1/10um, rho_snow=250) — so the RUNTIME default ice fall speed is NOT
    # the gSAM MK 0.865 that MorrisonConfig() advertises. gSAM maps to flavor="sam".
    assert MorrisonConfig().morrison_flavor == "mg"          # global default
    sam = resolve_morrison_flavor(MorrisonConfig(morrison_flavor="sam"))
    mg = resolve_morrison_flavor(MorrisonConfig(morrison_flavor="mg"))
    assert sam.fall_b_i == _O_BI                             # gSAM MK tune 0.865
    assert mg.fall_b_i == _O_BI_MG                           # M2005-original 1.0
    assert mg.lami_max == pytest.approx(1.0 / 10.0e-6)       # mg override: 1/10um
    assert mg.rho_snow == 250.0                              # mg override
    assert sam.fall_b_r == mg.fall_b_r == _O_BR              # rain flavor-invariant
    assert sam.fall_b_g == mg.fall_b_g == _O_BG              # graupel flavor-invariant


# ===================== Part B: mass-weighted speeds via surface-flux recovery =====================

@pytest.mark.parametrize("q_r,N_r", [
    (2.0e-3, 1.0e4),
    (5.0e-4, 1.0e3),
    (1.0e-3, 1.0e5),
    (3.0e-3, 5.0e4),
])
def test_umr_matches_sam_formula(q_r, N_r):
    rho = _rho_bottom(290.0)
    lam = _lam_slope(constants.rho_water, N_r, q_r, rho, per_volume=True,
                     lam_min=_O_LAMR_MIN, lam_max=_O_LAMR_MAX)
    expected = _um_oracle(_O_AR, _O_BR, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_RAIN)
    assert _recover_vt_rain(q_r, N_r) == pytest.approx(expected, rel=1e-12)


def test_umr_cap_fires_for_few_large_drops():
    # Few, huge drops -> LAMR clamped to lamr_min -> raw UMR > 9.1*dum -> capped.
    rho = _rho_bottom(290.0)
    q_r, N_r = 1.0e-2, 1.0e2
    lam = _lam_slope(constants.rho_water, N_r, q_r, rho, per_volume=True,
                     lam_min=_O_LAMR_MIN, lam_max=_O_LAMR_MAX)
    dum = (_RHO_SU / rho) ** _O_FALL_RHO_EXP
    raw = _O_AR * math.gamma(4.0 + _O_BR) / 6.0 * lam ** (-_O_BR) * dum
    assert raw > _O_VT_CAP_RAIN * dum                     # precondition: cap binds
    assert _recover_vt_rain(q_r, N_r) == pytest.approx(_O_VT_CAP_RAIN * dum, rel=1e-12)


def test_umr_slope_clamp_at_lamr_min():
    assert _lam_slope(constants.rho_water, 1.0e2, 1.0e-2, _rho_bottom(290.0),
                      per_volume=True, lam_min=_O_LAMR_MIN, lam_max=_O_LAMR_MAX) == _O_LAMR_MIN


def test_umr_flavor_invariant():
    # Rain fall params are NOT flavor-overridden (departure #3 touches only ice).
    q_r, N_r = 2.0e-3, 1.0e4
    assert _recover_vt_rain(q_r, N_r, "sam") == pytest.approx(
        _recover_vt_rain(q_r, N_r, "mg"), rel=1e-12)


@pytest.mark.parametrize("flavor,b_i", [("sam", _O_BI), ("mg", _O_BI_MG)])
@pytest.mark.parametrize("q_i,N_i", [
    (1.0e-4, 1.0e3),
    (5.0e-5, 1.0e5),
    (2.0e-4, 1.0e4),
])
def test_umi_matches_sam_formula(flavor, b_i, q_i, N_i):
    # Cloud-ice fall b is FLAVOR-RESOLVED: "sam" -> 0.865 (gSAM MK), "mg" -> 1.0
    # (M2005-original). Pin BOTH against their oracle exponent (departure #3).
    cfg = resolve_morrison_flavor(MorrisonConfig(morrison_flavor=flavor))
    assert cfg.fall_b_i == b_i                              # resolution precondition
    rho = _rho_bottom(250.0)
    lam = _lam_slope(_O_RHO_CLOUD_ICE, N_i, q_i, rho, per_volume=False,
                     lam_min=_O_LAMI_MIN, lam_max=_O_LAMI_MAX)
    expected = _um_oracle(_O_AI, b_i, lam, rho, _O_FALL_RHO_EXP_ICE, _O_VT_CAP_SNOW_ICE)
    assert _recover_vt_ice(q_i, N_i, flavor) == pytest.approx(expected, rel=1e-12)


# ============== Part C: mass + number speeds via the sed-call intercept ==============
# The number speeds V_n_* have no surface-flux observable; the intercept captures
# the actual V_t/V_n the module passes, and cross-checks the mass speeds against
# the Part-B surface-flux recovery (same module V_t reached two independent ways).

def test_intercept_rain_mass_and_number():
    q_r, N_r = 2.0e-3, 1.0e4
    rho = _rho_bottom(290.0)
    lam = _lam_slope(constants.rho_water, N_r, q_r, rho, per_volume=True,
                     lam_min=_O_LAMR_MIN, lam_max=_O_LAMR_MAX)
    caught = _captured_speeds(_base(q_r=q_r, N_r=N_r), 290.0, _warm_qv(), "sam")
    um = _um_oracle(_O_AR, _O_BR, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_RAIN)
    un = _un_oracle(_O_AR, _O_BR, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_RAIN)
    assert caught["mass"] == pytest.approx(um, rel=1e-12)
    assert caught["number"] == pytest.approx(un, rel=1e-12)
    assert caught["number"] < caught["mass"]                        # size-sorting
    # cross-check: intercepted mass == surface-flux-recovered mass
    assert caught["mass"] == pytest.approx(_recover_vt_rain(q_r, N_r), rel=1e-12)


@pytest.mark.parametrize("flavor,b_i", [("sam", _O_BI), ("mg", _O_BI_MG)])
def test_intercept_ice_mass_and_number(flavor, b_i):
    q_i, N_i = 5.0e-5, 1.0e5
    rho = _rho_bottom(250.0)
    lam = _lam_slope(_O_RHO_CLOUD_ICE, N_i, q_i, rho, per_volume=False,
                     lam_min=_O_LAMI_MIN, lam_max=_O_LAMI_MAX)
    caught = _captured_speeds(_base(q_i=q_i, N_i=N_i), 250.0, _cold_qv(), flavor)
    um = _um_oracle(_O_AI, b_i, lam, rho, _O_FALL_RHO_EXP_ICE, _O_VT_CAP_SNOW_ICE)
    un = _un_oracle(_O_AI, b_i, lam, rho, _O_FALL_RHO_EXP_ICE, _O_VT_CAP_SNOW_ICE)
    assert caught["mass"] == pytest.approx(um, rel=1e-12)
    assert caught["number"] == pytest.approx(un, rel=1e-12)


def test_intercept_snow_double_moment_mass_and_number():
    # Snow PSD is the DOUBLE-moment path (prognostic N_s); SAM-faithful
    # LAMS=(pi*rho_sn*N_s/q_s)^(1/3), UMS/UNS with the 1.2 cap and 0.54 density exp.
    q_s, N_s = 1.0e-3, 1.0e5
    rho = _rho_bottom(250.0)
    hm = _base(q_s=q_s)._replace(N_s=jnp.zeros((1, 2)).at[0, -1].set(N_s))
    lam = _lam_slope(_O_RHO_SNOW, N_s, q_s, rho, per_volume=False,
                     lam_min=_O_LAMS_MIN, lam_max=_O_LAMS_MAX)
    caught = _captured_speeds(hm, 250.0, _cold_qv(), "sam")
    um = _um_oracle(_O_AS, _O_BS, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_SNOW_ICE)
    un = _un_oracle(_O_AS, _O_BS, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_SNOW_ICE)
    assert caught["mass"] == pytest.approx(um, rel=1e-12)
    assert caught["number"] == pytest.approx(un, rel=1e-12)


def test_intercept_graupel_double_moment_mass_and_number():
    # SAM graupel is inherently two-moment: LAMG=(pi*rho_g*N_g/q_g)^(1/3). Pin the
    # SAM-faithful double-moment path (departure #5 covers the single-moment closure).
    q_g, N_g = 1.0e-3, 1.0e4
    rho = _rho_bottom(250.0)
    hm = _base(q_g=q_g)._replace(N_g=jnp.zeros((1, 2)).at[0, -1].set(N_g))
    lam = _lam_slope(_O_RHO_GRAUPEL, N_g, q_g, rho, per_volume=False,
                     lam_min=_O_LAMG_MIN, lam_max=_O_LAMG_MAX)
    caught = _captured_speeds(hm, 250.0, _cold_qv(), "sam")
    um = _um_oracle(_O_AG, _O_BG, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_GRAUPEL)
    un = _un_oracle(_O_AG, _O_BG, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_GRAUPEL)
    assert caught["mass"] == pytest.approx(um, rel=1e-12)
    assert caught["number"] == pytest.approx(un, rel=1e-12)


def test_snow_slope_clamp_and_mass_cap():
    # Few, large flakes -> LAMS clamped to lams_min AND raw UMS > 1.2*dum cap.
    # Drives the snow clamp (morrison.py:1009) AND the mass cap (:1014); the number
    # speed at the same slope stays UNDER its cap (pinned uncapped).
    q_s, N_s = 1.0e-2, 1.0e3
    rho = _rho_bottom(250.0)
    lam = _lam_slope(_O_RHO_SNOW, N_s, q_s, rho, per_volume=False,
                     lam_min=_O_LAMS_MIN, lam_max=_O_LAMS_MAX)
    assert lam == _O_LAMS_MIN                                # slope clamped to floor
    dum = (_RHO_SU / rho) ** _O_FALL_RHO_EXP
    raw = _O_AS * math.gamma(4.0 + _O_BS) / 6.0 * lam ** (-_O_BS) * dum
    assert raw > _O_VT_CAP_SNOW_ICE * dum                   # mass cap binds precondition
    hm = _base(q_s=q_s)._replace(N_s=jnp.zeros((1, 2)).at[0, -1].set(N_s))
    caught = _captured_speeds(hm, 250.0, _cold_qv(), "sam")
    assert caught["mass"] == pytest.approx(_O_VT_CAP_SNOW_ICE * dum, rel=1e-12)   # capped
    un = _un_oracle(_O_AS, _O_BS, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_SNOW_ICE)
    assert un < _O_VT_CAP_SNOW_ICE * dum                    # number below cap here
    assert caught["number"] == pytest.approx(un, rel=1e-12)


@pytest.mark.parametrize("N_g,bound", [(10.0, _O_LAMG_MIN), (1.0e8, _O_LAMG_MAX)])
def test_graupel_slope_clamps(N_g, bound):
    # Drive LAMG to BOTH clamp bounds (morrison.py:1063 via graupel_lamg): N_g=10
    # -> lamg_min (max drops), N_g=1e8 -> lamg_max (min drops). The 20*dum cap is
    # loose (unreachable within the clamped slope range), so it is not exercised.
    q_g = 1.0e-3
    rho = _rho_bottom(250.0)
    lam = _lam_slope(_O_RHO_GRAUPEL, N_g, q_g, rho, per_volume=False,
                     lam_min=_O_LAMG_MIN, lam_max=_O_LAMG_MAX)
    assert lam == bound                                     # slope clamped to the bound
    hm = _base(q_g=q_g)._replace(N_g=jnp.zeros((1, 2)).at[0, -1].set(N_g))
    caught = _captured_speeds(hm, 250.0, _cold_qv(), "sam")
    assert caught["mass"] == pytest.approx(
        _um_oracle(_O_AG, _O_BG, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_GRAUPEL), rel=1e-12)
    assert caught["number"] == pytest.approx(
        _un_oracle(_O_AG, _O_BG, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_GRAUPEL), rel=1e-12)


def test_intercept_number_moment_canary():
    # Non-vacuity of the number pins: the module's number speed uses Gamma(1+b) NOT
    # the mass moment Gamma(4+b)/6, so the captured V_n must DIFFER from the mass
    # oracle by the moment ratio (a cons6->cons4 swap in the module would fail this).
    q_r, N_r = 2.0e-3, 1.0e4
    rho = _rho_bottom(290.0)
    lam = _lam_slope(constants.rho_water, N_r, q_r, rho, per_volume=True,
                     lam_min=_O_LAMR_MIN, lam_max=_O_LAMR_MAX)
    caught = _captured_speeds(_base(q_r=q_r, N_r=N_r), 290.0, _warm_qv(), "sam")
    um = _um_oracle(_O_AR, _O_BR, lam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_RAIN)
    ratio = 6.0 * math.gamma(1.0 + _O_BR) / math.gamma(4.0 + _O_BR)
    assert caught["number"] == pytest.approx(um * ratio, rel=1e-12)
    assert abs(caught["number"] - um) / um > 0.5          # clearly not the mass moment


# ===================== Part D: departure canaries =====================

def test_departure_rain_slope_uses_legoesm_rho_water_not_sam_997():
    # LAMR uses constants.rho_water=1000, NOT SAM's rho_water=997.
    q_r, N_r = 2.0e-3, 1.0e4
    rho = _rho_bottom(290.0)
    assert constants.rho_water == 1000.0                  # legoESM value
    assert _O_RHO_WATER_SAM == 997.0                      # SAM value
    lam_lego = _lam_slope(constants.rho_water, N_r, q_r, rho, per_volume=True,
                          lam_min=_O_LAMR_MIN, lam_max=_O_LAMR_MAX)
    lam_sam = _lam_slope(_O_RHO_WATER_SAM, N_r, q_r, rho, per_volume=True,
                         lam_min=_O_LAMR_MIN, lam_max=_O_LAMR_MAX)
    vt_lego = _um_oracle(_O_AR, _O_BR, lam_lego, rho, _O_FALL_RHO_EXP, _O_VT_CAP_RAIN)
    vt_sam = _um_oracle(_O_AR, _O_BR, lam_sam, rho, _O_FALL_RHO_EXP, _O_VT_CAP_RAIN)
    assert _recover_vt_rain(q_r, N_r) == pytest.approx(vt_lego, rel=1e-12)   # matches 1000
    assert abs(vt_lego - vt_sam) / vt_lego > 5.0e-4                          # 997 detectably off


def test_departure_rho_su_uses_legoesm_R_d_not_a_hardcoded_number():
    cfg = MorrisonConfig()
    assert cfg.rho_su == pytest.approx(8.5e4 / (constants.R_d * constants.T_freeze), rel=1e-12)
    rho_su_sam = 8.5e4 / (287.15 * constants.T_freeze)  # const-ok: SAM rgas, canaried oracle value
    assert abs(cfg.rho_su - rho_su_sam) / cfg.rho_su > 1.0e-4


# ===================== differentiability =====================

def test_precip_gradient_finite_through_fall_speed():
    q_v = _warm_qv()

    def precip_of_qr(q_r):
        hm = _base(q_r=q_r, N_r=1.0e4)
        out = _run(290.0, q_v, hm, _cfg("sam"))
        return jnp.sum(out.precipitation)

    g = jax.grad(precip_of_qr)(jnp.asarray(2.0e-3))
    assert bool(jnp.isfinite(g)) and float(g) > 0.0      # more rain -> more precip
