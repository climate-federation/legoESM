"""IFS convection departure search / test ascent (port of OpenIFS cubasen.F90).

This module ports the "departure point search + test ascent" of the IFS
moist-convection closure (OpenIFS main pin 8f6f722, cubasen.F90 lines
296-735) into a pure JAX implementation for legoESM.  For each candidate
departure level (surface level up to ~350 hPa, the source's KLEV..NJKT1
loop) a test parcel is lifted with the source's mixing forms, saturation
adjustment (CUADJTQ), pseudo-microphysics and kinetic-energy equation
until the parcel vertical velocity squared becomes negative; the surface
departure yields shallow convection (unless its depth exceeds RDEPTHS,
in which case it is rejected) and the first elevated departure whose
cloud depth >= RDEPTHS is taken as deep convection and replaces the
profiles (the source's LLRESET fill).

Index convention (surface-last): full-level arrays are (ncol, nlev) with
nlev-1 the surface (IFS full level KLEV <-> our index nlev-1, IFS level 1
<-> our 0); half-level arrays are (ncol, nlev+1) with index nlev the
surface (IFS half level K <-> our half index K-1).  IFS loops
"DO JK=JKK-1,JKT2,-1" run upward and so does our lax.scan.

Departures from the source (each forced by the repo's available
primitives, none changes the physics where the primitive exists):
  * Half-level environment values ZTENH/ZQENH/ZSENH are the arithmetic
    means of the adjacent full levels, as in cuinin (cubasen.F90:296-312).
  * Saturation uses the liquid curve only
    (legoesm.thermo.saturation_mixing_ratio); the IFS liquid/ice blend
    FOEALFCU / FOEALFA-R5LES/R5IES (cubasen.F90:486-489 and 525-540) is
    NOT available, hence the freezing correction ZLGLAC is zero
    (liquid-only condensate) and the cloud-base saturation-deficit
    interpolation uses the liquid curve and its dT derivative.
  * CUADJTQ (two Newton corrections + q = min(q, qsat)) is reproduced as
    _cuadjtq_condense (cubasen.F90:462-464).
  * The land humidity-advection perturbation is weighted by land_frac
    instead of the binary LDLAND mask (cubasen.F90:415-417).
  * The source's double ZTEXC on the initial ZTU of the mixed-layer
    elevated parcel is reproduced literally (cubasen.F90:396-397/413-414).
  * NJKT2 (~60 hPa) and NJKT1 (~350 hPa) / NJKT6 (~700 hPa) level-index
    bounds are approximated by pressure thresholds
    (cfg.test_top_pa / departure_top_pa / qadv_land_min_pa).
  * IFS KINDEX = NJKT1 < KLEV-1 on Earth, so the branches
    "KINDEX==KLEV-1" (cubasen.F90:437, 650-652, 660) are dead code and
    the 1/z (shallow) mixing is used only for the surface departure
    (JKK==KLEV).
  * Hard oracle decisions are piecewise differentiable (jnp.where), not
    sigmoids, matching the discontinuous source.

Cost: O(n_departures x nlev) per column (one lax.scan of length <= nlev
per candidate departure level; the candidate loop is unrolled in Python
over the static nlev).

Surface flux sign convention: shf_w_m2 / lhf_w_m2 follow the IFS PAHFS /
PQHFL convention, NEGATIVE = upward (into the atmosphere); the latent
mass flux is PQHFL [kg m-2 s-1] = -lhf_w_m2 / L_v (cubasen.F90:335).
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.thermo import (
    saturation_specific_humidity, saturation_mixing_ratio, saturation_mixing_ratio_dT,
)
from legoesm.atmosphere.physics._shared import virtual_temperature


# --- IFS cubasen / sucumf defaults (OpenIFS main 8f6f722; sucumf.F90 lines cited) ---
_WS_EXP = 0.3333            # ZWS**.3333, cubasen.F90:352
_QSAT_RATIO_EXP = 3.0       # MIN(1,(PQSEN(JK)/PQSEN(KLEV))**3), cubasen.F90:472
# ZAW = ZBW = 1 in the kinetic-energy recurrence (cubasen.F90:505-507) -> the literal 2.0 factors.

# Derived constants (no literals: RETV = R_v/R_d - 1, RALFDCP = L_f/c_pd, LVDCP = L_v/c_pd)
_RETV = 1.0 / constants.epsilon - 1.0
_RALFDCP = constants.L_f / constants.c_pd
_RCPD_INV = 1.0 / constants.c_pd
_G_INV = 1.0 / constants.g


class IFSTestAscentConfig(NamedTuple):
    """Tunables of the departure search / test ascent (defaults = sucumf)."""
    entr_test_c1: float = 0.8             # ENTSTPC1  (sucumf.F90:147)
    entr_test_c2: float = 2.0e-4          # ENTSTPC2  (sucumf.F90:148) [1/m]
    entr_deep_base: float = 1.75e-3       # ENTRORG   (sucumf.F90) [1/m]
    deep_test_mix_factor: float = 0.4     # 0.4*ENTRORG deep-test mixing (cubasen.F90:472)
    depth_split_pa: float = 2.0e4         # RDEPTHS [Pa]
    parcel_dT_excess_min_K: float = 0.2   # ZTEXC (cubasen.F90:328, 396)
    parcel_dq_excess_min: float = 1.0e-4  # ZQEXC (cubasen.F90:329, 397)
    parcel_dT_excess_max_K: float = 1.0   # MIN(ZTEXC,1.0) (cubasen.F90:362, 408)
    parcel_dq_excess_max: float = 5.0e-4  # MIN(ZQEXC,5.E-4) (cubasen.F90:363, 409)
    elevated_w2: float = 1.0              # ZWU2H=1.0 for elevated departures (cubasen.F90:435) [m2/s2]
    surface_w2_add: float = 0.1           # ZWU2H=ZWS**2+0.1 (cubasen.F90:368) [m2/s2]
    surface_ws_factor: float = 1.2        # 1.2*ZWS**.3333 (cubasen.F90:352)
    surface_flux_factor: float = 1.5      # the two 1.5_JPRB flux factors (cubasen.F90:335, 360-361)
    ustar_min: float = 0.1                # MAX(SQRT(PKMFL),0.1) (cubasen.F90:336) [m/s]
    mixed_layer_depth_pa: float = 6000.0  # 60.E2 mixed-layer trigger (cubasen.F90:419) [Pa]
    mixed_layer_span_pa: float = 5000.0   # 50.E2 accumulation span (cubasen.F90:424) [Pa]
    land_qadv_timescale_s: float = 600.0  # PTENQA*600. (cubasen.F90:416) [s]
    land_qadv_cap: float = 3.0e-4         # MIN(3.E-4, ...) (cubasen.F90:416) [kg/kg]
    land_qadv_rh_max: float = 0.9         # RH < 0.9 gate (cubasen.F90:415)
    test_condensate_retained: float = 0.5 # 0.5*ZLU pseudo-microphysics (cubasen.F90:493)
    departure_top_pa: float = 35000.0     # NJKT1 ~ 350 hPa [Pa]
    test_top_pa: float = 6000.0           # NJKT2 ~ 60 hPa [Pa]
    qadv_land_min_pa: float = 70000.0     # NJKT6 ~ 700 hPa [Pa]: land humidity
                                          # advection applies to departures at
                                          # pressures ABOVE this value, i.e.
                                          # BELOW ~700 hPa (cubasen.F90:392,
                                          # JKK > NJKT6 selects higher pressure)
    mixed_layer_gate: str = "half_above"  # mixed-layer parcel condition
                                          # (cubasen.F90:400): "half_above" =
                                          # the literal source test
                                          # PAPH(KLEV+1)-PAPH(JKK-1) < 60 hPa
                                          # (never true on 33 hPa layers);
                                          # "cell_centre" = the resolution-aware
                                          # reinterpretation
                                          # PAPH(KLEV+1)-PAPH(JKK) < 60 hPa
                                          # using the departure's FULL-level
                                          # (cell-centre) pressure, which
                                          # enables the lowest elevated
                                          # candidate on L137-like grids


__param_spec__ = {
    "IFSTestAscentConfig": {
        "scheme_key": "atm.conv.IFSTestAscentConfig",
        "excluded": {
            "departure_top_pa": "pure pressure/level convention (NJKT1), not tunable",
            "test_top_pa": "pure pressure/level convention (NJKT2), not tunable",
            "qadv_land_min_pa": "pure pressure/level convention (NJKT6), not tunable",
            "mixed_layer_depth_pa": "fixed IFS pressure convention (60 hPa, cubasen.F90:419)",
            "mixed_layer_span_pa": "fixed IFS pressure convention (50 hPa, cubasen.F90:424)",
            "ustar_min": "numerical floor on the friction velocity (cumastrn passes 0.1)",
            "mixed_layer_gate": "str mode selector ('half_above' | 'cell_centre'), not a tunable float",
        },
        "params": {
            "entr_test_c1": {"units": "1", "bounds": (0.2, 2.0), "tunable_tier": 2,
                             "transform": "sigmoid", "category": "entrainment",
                             "reference": "OpenIFS sucumf.F90:147 ENTSTPC1", "shape": None},
            "entr_test_c2": {"units": "1/m", "bounds": (0.0, 6.0e-4), "tunable_tier": 2,
                             "transform": "sigmoid", "category": "entrainment",
                             "reference": "OpenIFS sucumf.F90:148 ENTSTPC2", "shape": None},
            "entr_deep_base": {"units": "1/m", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 1,
                               "transform": "sigmoid", "category": "entrainment",
                               "reference": "OpenIFS sucumf.F90 ENTRORG (Gregory et al. 2000)", "shape": None},
            "deep_test_mix_factor": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 2,
                                     "transform": "sigmoid", "category": "entrainment",
                                     "reference": "OpenIFS cubasen.F90:472 (0.4*ENTRORG)", "shape": None},
            "depth_split_pa": {"units": "Pa", "bounds": (5.0e3, 5.0e4), "tunable_tier": 2,
                               "transform": "sigmoid", "category": "trigger",
                               "reference": "OpenIFS sucumf.F90 RDEPTHS = 2.E4", "shape": None},
            "parcel_dT_excess_min_K": {"units": "K", "bounds": (0.05, 0.5), "tunable_tier": 2,
                                       "transform": "sigmoid", "category": "trigger",
                                       "reference": "OpenIFS cubasen.F90:328 ZTEXC = 0.2", "shape": None},
            "parcel_dq_excess_min": {"units": "kg/kg", "bounds": (2.0e-5, 3.0e-4), "tunable_tier": 2,
                                     "transform": "sigmoid", "category": "trigger",
                                     "reference": "OpenIFS cubasen.F90:329 ZQEXC = 1.E-4", "shape": None},
            "parcel_dT_excess_max_K": {"units": "K", "bounds": (0.5, 2.0), "tunable_tier": 2,
                                       "transform": "sigmoid", "category": "trigger",
                                       "reference": "OpenIFS cubasen.F90:362 MIN(ZTEXC,1.0)", "shape": None},
            "parcel_dq_excess_max": {"units": "kg/kg", "bounds": (1.0e-4, 1.5e-3), "tunable_tier": 2,
                                     "transform": "sigmoid", "category": "trigger",
                                     "reference": "OpenIFS cubasen.F90:363 MIN(ZQEXC,5.E-4)", "shape": None},
            "elevated_w2": {"units": "m2/s2", "bounds": (0.5, 2.0), "tunable_tier": 2,
                            "transform": "sigmoid", "category": "trigger",
                            "reference": "OpenIFS cubasen.F90:435 ZWU2H = 1.0", "shape": None},
            "surface_w2_add": {"units": "m2/s2", "bounds": (0.0, 0.5), "tunable_tier": 2,
                               "transform": "sigmoid", "category": "trigger",
                               "reference": "OpenIFS cubasen.F90:368 ZWS**2+0.1", "shape": None},
            "surface_ws_factor": {"units": "1", "bounds": (1.0, 1.5), "tunable_tier": 2,
                                  "transform": "sigmoid", "category": "trigger",
                                  "reference": "OpenIFS cubasen.F90:352 1.2*ZWS**.3333", "shape": None},
            "surface_flux_factor": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2,
                                    "transform": "sigmoid", "category": "trigger",
                                    "reference": "OpenIFS cubasen.F90:335,360-361 factor 1.5", "shape": None},
            "land_qadv_timescale_s": {"units": "s", "bounds": (300.0, 1200.0), "tunable_tier": 2,
                                      "transform": "sigmoid", "category": "trigger",
                                      "reference": "OpenIFS cubasen.F90:416 PTENQA*600.", "shape": None},
            "land_qadv_cap": {"units": "kg/kg", "bounds": (1.0e-4, 1.0e-3), "tunable_tier": 2,
                              "transform": "sigmoid", "category": "trigger",
                              "reference": "OpenIFS cubasen.F90:416 MIN(3.E-4,...)", "shape": None},
            "land_qadv_rh_max": {"units": "1", "bounds": (0.8, 1.0), "tunable_tier": 2,
                                 "transform": "sigmoid", "category": "trigger",
                                 "reference": "OpenIFS cubasen.F90:415 RH < 0.9 gate", "shape": None},
            "test_condensate_retained": {"units": "1", "bounds": (0.3, 0.7), "tunable_tier": 2,
                                         "transform": "sigmoid", "category": "microphysics",
                                         "reference": "OpenIFS cubasen.F90:493 0.5*ZLU", "shape": None},
        },
    }
}


__physics_contract__ = {
    "summary": "IFS (cubasen) departure-level search and test ascent: finds the "
               "convective departure level, cloud base/top, test-parcel profiles and "
               "base vertical velocity; classifies deep vs shallow vs none.",
    "inputs": {
        "T": "K", "q_v": "kg/kg (mixing ratio)", "p_full": "Pa", "p_half": "Pa",
        "geo_full": "m2/s2 (geopotential, surface-relative; geo_half[:, nlev] = 0)",
        "geo_half": "m2/s2", "shf_w_m2": "W/m2 (PAHFS sign: negative = upward)",
        "lhf_w_m2": "W/m2 (PQHFL sign: negative = upward; PQHFL = -lhf/L_v [kg m-2 s-1])",
        "ustar": "m/s", "land_frac": "1 (0..1)", "dq_dt_adv": "kg/kg/s (PTENQA)",
    },
    "outputs": {
        "ldcum": "1 (bool-valued float)", "ktype": "1 deep, 2 shallow, 0 none",
        "k_dpl/k_cbot/k_ctop": "surface-last full-level indices (-1 when none)",
        "w_base": "m/s (PWUBASE)", "T_u": "K", "q_u/l_u": "kg/kg",
        "klab": "0/1/2", "cape_test": "J/kg (PCAPE = max of ZCAPE over departures)",
        "w2": "m2/s2 (ZWU2H/PWU2H of the selected ascent)",
    },
    "sign_convention": "surface fluxes follow the IFS PAHFS/PQHFL convention: "
                       "NEGATIVE = upward (into the atmosphere); PQHFL = -lhf_w_m2 / L_v.",
    "conserves": [],
    "differentiable": True,
    "reference": "OpenIFS cubasen.F90 (main 8f6f722)",
    "idealized_test": "tests/unit/test_ifs_test_ascent.py",
}


def _cuadjtq_pair(T, q, p):
    """cuadjtq.F90:324-345, KCALL=3 branch: two successive UNCLIPPED Newton
    corrections ZCOND1 = (q - q_s)/(1 + (L_v/c_pd) dq_s/dT); T += (L_v/c_pd)
    ZCOND1; q -= ZCOND1 (second pass uses the updated T).  Specific-humidity
    form: q_s = w_s/(1+w_s) of the shared curve (legoesm.thermo
    .saturation_specific_humidity) with dq_s/dT from _dqsat_dT.
    Declared departures (cuadjtq.F90:162-168; suphec.F90:231-240): the shared
    curve's Tetens coefficients differ slightly from IFS FOEEWM, and saturation
    is liquid-only (no FOEALFA/FOEALFCU ice blend)."""
    lvcp = constants.L_v / constants.c_pd
    for _ in range(2):
        qs = saturation_specific_humidity(T, p)
        zcond1 = (q - qs) / (1.0 + lvcp * _dqsat_dT(T, p))
        T = T + lvcp * zcond1
        q = q - zcond1
    return T, q


_RLMIN = 1.0e-8      # sucldp.F90:253 default (cloud configuration, NOT yoecumf);
                     # applied at the Sc cloud base, cubasen.F90:601
_TINY = 1.0e-12      # AD-safe floor for differentiated denominators (cloud-base
                     # interpolation divides by zdqsdT*zdtdp, which vanishes on
                     # inactive branches)
_Z_FLOOR = 1.0       # [m] floor on the 1/z entrainment height (geo_full - geo_half
                     # at the surface); keeps 1/z finite on inactive surface branches


def _dqsat_dT(T, p):
    """dq_s/dT for the SPECIFIC-humidity saturation curve q_s = w_s/(1+w_s)
    of the shared mixing-ratio helpers (chain rule, no re-derived curve):
    dq_s/dT = dw_s/dT / (1 + w_s)**2."""
    return saturation_mixing_ratio_dT(T, p) / (1.0 + saturation_mixing_ratio(T, p)) ** 2


def _half_level_env(T, q_v, p_full, p_half, geo_full, geo_half, cfg):
    """Half-level environment (ZTENH, ZQENH, ZSENH) ported from cuinin.F90:150-192,
    with ZSENH = RCPD*PTENH + PGEOH (cubasen.F90:306).  The rule actually
    implemented is the source's, NOT arithmetic means:
      * T_h from the MAXIMUM of the adjacent full-level DRY STATIC ENERGIES
        (cuinin.F90:156-157), T_h = (max(s_up, s_dn) - PGEOH)/RCPD;
      * q_h from the FULL LEVEL ABOVE (cuinin.F90:158), PQENH = PQEN(JK-1);
      * followed by the CUADJTQ KCALL=3 pair adjustment (cuinin.F90:163-180)
        restricted to IFS JK = NJKT2..KLEV-2 (CYCLE at :163), with
        PQENH = MIN(PQEN, PQSEN) + (PQSENH - PQSEN), clipped >= 0 (:178-180).
    Specific-humidity convention: q_v/q_h are specific humidities; saturation
    uses saturation_specific_humidity (q_s = w_s/(1+w_s) of the shared curve)
    with the FOEEWM-coefficient and liquid-only departures declared in
    _cuadjtq_pair.
    Index map: IFS half JK (1..KLEV) <-> our half h = JK-1; our h = nlev is the
    surface (IFS KLEV+1, PGEOH/PAPH only). Surface half index nlev is filled
    with the lowest full level's T and q and s = c_pd*T + geo_half[:, nlev];
    cubasen never reads it."""
    c_pd = constants.c_pd
    ncol, nlev = T.shape
    dt = T.dtype
    # dry static energy on full levels (cuinin.F90:156-157)
    s_full = c_pd * T + geo_full  # (:156-157)
    T_h = jnp.zeros((ncol, nlev + 1), dt)
    q_h = jnp.zeros((ncol, nlev + 1), dt)
    # interior half levels h = 1..nlev-1 (IFS JK = 2..KLEV):
    # T from MAX of adjacent full-level s (cuinin.F90:156-157),
    # q and qs from the level ABOVE (:158-159)
    s_max = jnp.maximum(s_full[:, :-1], s_full[:, 1:])
    T_int = (s_max - geo_half[:, 1:nlev]) / c_pd          # T_int[:, m] <-> half h = m+1
    q_int = q_v[:, :-1]                                   # (:158)
    qs_above = saturation_specific_humidity(T[:, :-1], p_full[:, :-1])  # PQSEN(JK-1) (:159)

    # cuadjtq KCALL=3 pair adjustment: IFS JK = NJKT2..KLEV-2 <-> our half
    # indices 1..nlev-3 <-> T_int[:, 0:nlev-3] paired with p_half[:, 1:nlev-2]
    # (cuinin.F90:163 CYCLE skips JK >= KLEV-1 and JK < NJKT2; :164-175).
    p_int = p_half[:, 1:nlev - 2]
    mask = p_int > jnp.asarray(cfg.test_top_pa, dt)       # traced pressure mask (NJKT2 ~ 60 hPa, sucumf.F90:285)
    T_corr, qs_corr = _cuadjtq_pair(T_int[:, 0:nlev - 3],
                                    qs_above[:, 0:nlev - 3], p_int)      # (:165-175)
    T_int = T_int.at[:, 0:nlev - 3].set(jnp.where(mask, T_corr, T_int[:, 0:nlev - 3]))
    # PQENH = MIN(PQEN(JK-1), PQSEN(JK-1)) + (PQSENH - PQSEN(JK-1)), clipped >= 0 (:178-180)
    q_corr = jnp.maximum(
        jnp.minimum(q_v[:, :-1], qs_above)[:, 0:nlev - 3]
        + (qs_corr - qs_above[:, 0:nlev - 3]),
        0.0)
    q_int = q_int.at[:, 0:nlev - 3].set(jnp.where(mask, q_corr, q_int[:, 0:nlev - 3]))

    T_h = T_h.at[:, 1:nlev].set(T_int)
    q_h = q_h.at[:, 1:nlev].set(q_int)

    # h = nlev-1 (IFS KLEV) override (cuinin.F90:186-187)
    T_h = T_h.at[:, nlev - 1].set(
        (c_pd * T[:, nlev - 1] + geo_full[:, nlev - 1] - geo_half[:, nlev - 1]) / c_pd)
    q_h = q_h.at[:, nlev - 1].set(q_v[:, nlev - 1])
    # h = 0 (IFS 1) (cuinin.F90:188-189)
    T_h = T_h.at[:, 0].set(T[:, 0])
    q_h = q_h.at[:, 0].set(q_v[:, 0])
    # h = nlev (surface, IFS KLEV+1): fill with lowest full level; cubasen never reads it
    T_h = T_h.at[:, nlev].set(T[:, nlev - 1])
    q_h = q_h.at[:, nlev].set(q_v[:, nlev - 1])

    s_h = c_pd * T_h + geo_half  # ZSENH (cubasen.F90:306)
    return T_h, q_h, s_h


def _cuadjtq_condense(T, q, p):
    """CUADJTQ (KCALL=1, condensation only), cuadjtq.F90:165-186, specific-humidity
    form: FIRST correction ZCOND = MAX(0,(q - q_s)/(1 + (L_v/c_pd) dq_s/dT))
    (clipped >= 0), then a SECOND correction ZCOND1 that is UNCLIPPED but zeroed
    where the first ZCOND was zero (cuadjtq.F90:182).  There is NO final
    q = min(q, q_s) clipping in this source branch.  q_s = w_s/(1+w_s) of the
    shared curve (saturation_specific_humidity) with dq_s/dT from _dqsat_dT.
    Declared departures (cuadjtq.F90:162-168; suphec.F90:231-240): the shared
    curve's coefficients differ slightly from IFS FOEEWM, and saturation is
    liquid-only (ZLGLAC freezing correction therefore zero)."""
    lvcp = constants.L_v / constants.c_pd
    qs = saturation_specific_humidity(T, p)
    dqs = _dqsat_dT(T, p)
    zcond = jnp.maximum(0.0, (q - qs) / (1.0 + lvcp * dqs))   # cuadjtq.F90:171
    T1 = T + lvcp * zcond
    q1 = q - zcond
    qs1 = saturation_specific_humidity(T1, p)
    dqs1 = _dqsat_dT(T1, p)
    zcond1 = jnp.where(zcond > 0.0,                            # cuadjtq.F90:182
                       (q1 - qs1) / (1.0 + lvcp * dqs1), 0.0)
    T2 = T1 + lvcp * zcond1
    q2 = q1 - zcond1
    return T2, q2


def _test_ascent_from_departure(k_dep, is_surface, T, q_v, p_full, p_half,
                                geo_full, geo_half, env_half, parcel_init, cfg):
    """One test ascent from departure level k_dep upward (IFS DO JK=JKK-1,JKT2,-1,
    cubasen.F90:437-600). lax.scan over the k_dep levels above the departure;
    levels with p_full <= cfg.test_top_pa are masked out (NJKT2 ~ 60 hPa).

    Parcel profiles T_u/q_u/l_u/w2h/ilab are HALF-LEVEL quantities of length
    nlev (IFS half level JK <-> our index JK-1); unvisited levels are
    initialised/filled from the half-level environment T_h[:, :nlev],
    q_h[:, :nlev] (cubasen.F90:209-213, 711-713) -- no full-level values are
    written into the parcel profiles anywhere in this function.  Specific
    humidity throughout (virtual temperature T(1 + RETV q) directly as in the
    source); saturation on the specific-humidity curve with the FOEEWM /
    liquid-only departures declared in _cuadjtq_pair / _cuadjtq_condense."""
    nlev = T.shape[1]
    T_h, q_h, s_h = env_half
    dt = T_h.dtype
    ncol = T_h.shape[0]

    # dtype pinning (x64-safe carries): every array input and every cfg float
    # used inside the scan body is cast once, before the scan.
    T = jnp.asarray(T, dt); q_v = jnp.asarray(q_v, dt)
    p_full = jnp.asarray(p_full, dt); p_half = jnp.asarray(p_half, dt)
    geo_full = jnp.asarray(geo_full, dt); geo_half = jnp.asarray(geo_half, dt)
    T_h = jnp.asarray(T_h, dt); q_h = jnp.asarray(q_h, dt); s_h = jnp.asarray(s_h, dt)
    c1 = jnp.asarray(cfg.entr_test_c1, dt)
    c2 = jnp.asarray(cfg.entr_test_c2, dt)
    deep_fac = jnp.asarray(cfg.deep_test_mix_factor, dt)
    entr_base = jnp.asarray(cfg.entr_deep_base, dt)
    ret_frac = jnp.asarray(cfg.test_condensate_retained, dt)
    top_pa = jnp.asarray(cfg.test_top_pa, dt)
    g_inv = jnp.asarray(_G_INV, dt)
    retv = jnp.asarray(_RETV, dt)
    rcpd_inv = jnp.asarray(_RCPD_INV, dt)
    cg = jnp.asarray(constants.g, dt)
    lvcp = jnp.asarray(constants.L_v / constants.c_pd, dt)
    rlmin = jnp.asarray(_RLMIN, dt)
    tiny = jnp.asarray(_TINY, dt)
    z_floor = jnp.asarray(_Z_FLOOR, dt)

    # candidate ascent levels: our indices k_dep-1 .. 0 (surface-last: upward),
    # int32 so all index carries keep a single dtype under lax.scan
    js = jnp.arange(k_dep - 1, -1, -1, dtype=jnp.int32)

    # deep-test denominator: PQSEN(JL,KLEV) on the lowest FULL level
    # (cubasen.F90:472); the shallow 1/z height is the FULL-level geopotential
    # above the surface (PGEO(JK) - PGEOH(KLEV+1), cubasen.F90:446).
    qsat_sfc = saturation_specific_humidity(T[:, nlev - 1], p_full[:, nlev - 1])

    # half-level parcel profiles: unvisited values from the half-level
    # environment (cubasen.F90:209-213), departure level from parcel_init
    active = parcel_init["active"]
    T_dep = (parcel_init["s"] - geo_half[:, k_dep]) * rcpd_inv
    T_u = T_h[:, :nlev].at[:, k_dep].set(jnp.where(active, T_dep, T_h[:, k_dep]))
    q_u = q_h[:, :nlev].at[:, k_dep].set(
        jnp.where(active, parcel_init["q"].astype(dt), q_h[:, k_dep]))
    l_u = jnp.zeros((ncol, nlev), dt)                    # ZLU(JKK) = 0 (:344/:385)
    w2h = jnp.zeros((ncol, nlev), dt).at[:, k_dep].set(
        jnp.where(active, parcel_init["w2"].astype(dt), 0.0))
    ilab0 = jnp.zeros((ncol, nlev), jnp.int32).at[:, k_dep].set(
        jnp.where(active, 1, 0).astype(jnp.int32))       # ILAB=1, :345/:375

    carry0 = {
        "q": parcel_init["q"].astype(dt), "s": parcel_init["s"].astype(dt),
        "w2": parcel_init["w2"].astype(dt), "buoh": parcel_init["buoh"].astype(dt),
        "active": active,
        "icbot": parcel_init["icbot"].astype(jnp.int32),
        "ictop": parcel_init["ictop"].astype(jnp.int32),
        "lldcum": parcel_init["lldcum"], "cape": jnp.zeros(ncol, dt),
        "T_u": T_u, "q_u": q_u, "l_u": l_u, "w2h": w2h, "ilab": ilab0,
    }

    def body(carry, j):
        j = j.astype(jnp.int32)                            # keep int32 carries
        q, s, w2, buoh, active, icbot, ictop = (
            carry["q"], carry["s"], carry["w2"], carry["buoh"], carry["active"],
            carry["icbot"], carry["ictop"])
        lldcum, cape = carry["lldcum"], carry["cape"]
        T_u, q_u, l_u, w2h, ilab = (carry["T_u"], carry["q_u"], carry["l_u"],
                                    carry["w2h"], carry["ilab"])
        work = active & (p_full[:, j] > top_pa)            # NJKT2 bound

        dz = (geo_half[:, j] - geo_half[:, j + 1]) * g_inv
        qf = 0.5 * (q_h[:, j + 1] + q_h[:, j])
        sf = 0.5 * (s_h[:, j + 1] + s_h[:, j])
        if is_surface:
            # 1/z mixing for the shallow/surface departure (cubasen.F90:437, 446-451);
            # KINDEX==KLEV-1 branches are dead on Earth (NJKT1 < KLEV-1).
            # AD safety: the 1/z height is floored away from zero.
            z_h = jnp.maximum((geo_full[:, j] - geo_half[:, nlev]) * g_inv, z_floor)
            zeps = c1 / z_h + c2
            zmix = jnp.minimum(1.0, 0.5 * dz * zeps)       # cubasen.F90:450-452
            tmp = 1.0 / (1.0 + zmix)
            q_new = (q * (1.0 - zmix) + 2.0 * zmix * qf) * tmp  # cubasen.F90:456-458
            s_new = (s * (1.0 - zmix) + 2.0 * zmix * sf) * tmp  # cubasen.F90:459-461
        else:
            # deep-test mixing 0.4*ENTRORG*dz*min(1,(PQSEN(JK)/PQSEN(KLEV))**3)
            # (cubasen.F90:472-478); PQSEN is a specific humidity
            qsat_j = saturation_specific_humidity(T[:, j], p_full[:, j])
            zmix = jnp.minimum(1.0, deep_fac * entr_base * dz
                               * jnp.minimum(1.0, (qsat_j / qsat_sfc) ** _QSAT_RATIO_EXP))
            q_new = q * (1.0 - zmix) + qf * zmix           # cubasen.F90:483
            s_new = s * (1.0 - zmix) + sf * zmix           # cubasen.F90:484

        # condensation (CUADJTQ), condensate added, half retained (cubasen.F90:466-493)
        q_old = q_new
        T_new0 = (s_new - geo_half[:, j]) * rcpd_inv
        T_adj, q_adj = _cuadjtq_condense(T_new0, q_new, p_half[:, j])
        zdq = jnp.maximum(q_old - q_adj, 0.0)
        l_new = ret_frac * (l_u[:, j + 1] + zdq)
        # freezing correction ZLGLAC = 0: liquid-only saturation (FOEALFCU unavailable)
        T_new = T_adj
        s_new = jnp.asarray(constants.c_pd, dt) * T_new + geo_half[:, j]

        # buoyancy on half levels (cubasen.F90:499-508)
        tvu = (1.0 + retv * q_adj - l_new) * T_new
        tven = (1.0 + retv * q_h[:, j]) * T_h[:, j]
        buoh_new = (tvu - tven) * cg / tven
        buof = 0.5 * (buoh_new + buoh)
        # kinetic-energy recurrence, ZAW = ZBW = 1 (cubasen.F90:511-513)
        w2_new = (w2 * (1.0 - 2.0 * zmix) + 2.0 * buof * dz) / (1.0 + 2.0 * zmix)
        cape_new = cape + jnp.maximum(0.0, buof * dz)      # cubasen.F90:521

        # first layer with liquid water: exact cloud base (cubasen.F90:524-551).
        # Saturation-deficit interpolation on the liquid specific-humidity curve
        # (declared departure: FOEEWM coefficients differ, FOEALFA-R5LES/R5IES
        # blend unavailable, cubasen.F90:527, 532-535).  AD safety: the
        # denominator is floored away from zero on inactive branches.
        cond_first = (l_new > 0.0) & (ilab[:, j + 1] == 1)
        zqsu = saturation_specific_humidity(T_u[:, j + 1], p_half[:, j + 1])
        zdqsdT = _dqsat_dT(T_u[:, j + 1], p_half[:, j + 1])
        zdq_cb = jnp.minimum(0.0, q_u[:, j + 1] - zqsu)
        zdtdp = jnp.asarray(constants.R_d, dt) * T_u[:, j + 1] / (
            jnp.asarray(constants.c_pd, dt) * p_half[:, j + 1])
        zcb = p_half[:, j + 1] + zdq_cb / jnp.maximum(zdqsdT * zdtdp, tiny)
        pdtop = zcb - p_half[:, j]
        pdbot = p_half[:, j + 1] - zcb
        case_top = cond_first & (pdtop > pdbot) & (w2 > 0.0)     # cubasen.F90:548
        case_bot = cond_first & (pdtop <= pdbot) & (w2_new > 0.0)  # cubasen.F90:555
        jkb = jnp.minimum(nlev - 2, j + 1).astype(jnp.int32)     # MIN(KLEV-1,JK+1)
        icbot_new = jnp.where(case_top, jkb, icbot)
        icbot_new = jnp.where(case_bot, j, icbot_new)
        # ZLU(JK+1) = RLMIN is set BEFORE the w2 < 0 test reads it
        # (cubasen.F90:601, 615-621) -> the UPDATED value is used below
        l_below = jnp.where(case_top, rlmin, l_u[:, j + 1])      # cubasen.F90:601
        # scratch label lifecycle (cubasen.F90:265-268): the previous carried
        # ilab(:, j) is left where the level is not accepted (no zeroing)
        labj = ilab[:, j]
        labj = jnp.where(case_top | case_bot, 2, labj)

        # stop at w2 < 0; ICTOP/LLDCUM use the UPDATED l_below
        # (cubasen.F90:615-621, strict '<')
        stop = w2_new < 0.0
        ictop_new = jnp.where(stop & (l_below > 0.0), j, ictop)
        lldcum_new = jnp.where(stop, l_below > 0.0, lldcum)
        # carry state changes ONLY on executed (work) iterations
        active_new = active & ~(work & stop)
        labj = jnp.where(work & ~stop, jnp.where(l_new > 0.0, 2, 1), labj)  # 588-592

        # store this level (masked by 'work'); every field preserved exactly
        # on masked iterations
        def put(arr, val):
            return arr.at[:, j].set(jnp.where(work, val, arr[:, j]))
        T_u = put(T_u, T_new); q_u = put(q_u, q_adj)
        l_u = put(l_u, l_new); w2h = put(w2h, w2_new)
        ilab = put(ilab, labj.astype(ilab.dtype))
        l_u = l_u.at[:, j + 1].set(jnp.where(work, l_below, l_u[:, j + 1]))
        ilab = ilab.at[:, jkb].set(
            jnp.where(work & case_top, jnp.asarray(2, ilab.dtype), ilab[:, jkb]))

        new = {"q": jnp.where(work, q_adj, q), "s": jnp.where(work, s_new, s),
               "w2": jnp.where(work, w2_new, w2),
               "buoh": jnp.where(work, buoh_new, buoh),
               "active": active_new,
               "icbot": jnp.where(work, icbot_new, icbot),
               "ictop": jnp.where(work, ictop_new, ictop),
               "lldcum": jnp.where(work, lldcum_new, lldcum),
               "cape": jnp.where(work, cape_new, cape),
               "T_u": T_u, "q_u": q_u, "l_u": l_u, "w2h": w2h, "ilab": ilab}
        return new, ()

    final, _ = lax.scan(body, carry0, js)
    return (final["ilab"], final["T_u"], final["q_u"], final["l_u"], final["w2h"],
            final["icbot"], final["ictop"], final["lldcum"], final["cape"])


class TestAscent(NamedTuple):
    ldcum: jnp.ndarray       # (ncol,) bool-valued float
    ktype: jnp.ndarray       # (ncol,) int32: 1 deep, 2 shallow, 0 none
    k_dpl: jnp.ndarray       # (ncol,) int32, -1 when none
    k_cbot: jnp.ndarray
    k_ctop: jnp.ndarray
    w_base: jnp.ndarray      # PWUBASE [m/s]
    T_u: jnp.ndarray         # (ncol, nlev) K, half-level parcel values
    q_u: jnp.ndarray         # (ncol, nlev) kg/kg MIXING RATIO per unit dry air
    l_u: jnp.ndarray         # (ncol, nlev) kg/kg MIXING RATIO per unit dry air
    klab: jnp.ndarray        # (ncol, nlev) int32
    cape_test: jnp.ndarray   # (ncol,) J/kg (PCAPE = max of ZCAPE over departures)
    w2: jnp.ndarray          # (ncol, nlev) m2/s2, ZWU2H of the SELECTED ascent
    w2_surface: jnp.ndarray  # (ncol, nlev) m2/s2, PWU2H = the SURFACE test's
                             # velocity profile, kept for ALL columns regardless
                             # of which departure was selected
                             # (cubasen.F90:358, 559-560)
    ldsc: jnp.ndarray        # (ncol,) bool: Stratus/cloud-base (LDSC) flag from
                             # the surface test (cubasen.F90:598-607)
    k_botsc: jnp.ndarray     # (ncol,) int32: Sc cloud-base level (KBOTSC),
                             # -1 when none (cubasen.F90:639-645)


def _init_departure_parcel(k_dep, is_surface, T, q_v, p_full, p_half,
                           geo_full, geo_half, env_half, shf_w_m2, lhf_w_m2,
                           ustar, land_frac, dq_dt_adv, cfg, elig_active=None):
    """Initialise the departure-level parcel and scan carry (cubasen.F90:314-435).

    q_v here is SPECIFIC HUMIDITY (the repo-convention mixing ratio is
    converted once at the ifs_departure_search entry).  Internal defaults
    follow the source: ICBOT = JKK = k_dep, ICTOP = KLEV-1 = nlev-1
    (cubasen.F90:323-325); these are translated to the public -1 sentinel
    only in the returned TestAscent for unsuccessful columns
    (cubasen.F90:664-668).

    elig_active carries the pre-ascent eligibility (fix: only ZKHVFL < 0
    columns are active for the surface departure, cubasen.F90:367; only
    columns with p_full[:, k_dep] > departure_top_pa and not yet
    deep-resolved are active for elevated departures, :655/:735), so that
    inactive columns accumulate no CAPE and MAXVAL(ZCAPE) matches the
    source (:741).

    The elevated loop calls this with k_dep >= 1, so every k_dep - 1 index
    below is inside the source's index domain (JKK = KLEV-1..2, :302)."""
    nlev = T.shape[1]
    T_h, q_h, s_h = env_half
    ncol = T.shape[0]
    dt = T.dtype

    def _surface_fluxes():
        # ZRHO from the surface half level (cubasen.F90:333)
        zrho = p_half[:, nlev] / (constants.R_d * T[:, nlev - 1]
                                  * (1.0 + _RETV * q_v[:, nlev - 1]))
        pahfs = shf_w_m2                                    # PAHFS (neg = up)
        pqhfl = lhf_w_m2 / constants.L_v                    # PQHFL = lhf/L_v (neg = up, no extra minus)
        zkhvfl = (pahfs * _RCPD_INV + _RETV * T[:, nlev - 1] * pqhfl) / zrho  # :335
        zust = jnp.maximum(ustar, cfg.ustar_min)            # :336
        # ZWS = ZUST**3 - 1.5*RKAP*ZKHVFL*(PGEOH(KLEV)-PGEOH(KLEV+1))/PTEN(KLEV)
        # (cubasen.F90:337-338) -- no 1/c_pd factor
        zws = zust ** 3 - cfg.surface_flux_factor * constants.kappa_von_karman \
            * zkhvfl * (geo_half[:, nlev - 1] - geo_half[:, nlev]) \
            / T[:, nlev - 1]                                # :337-338
        return zrho, pahfs, pqhfl, zkhvfl, zws

    zrho, pahfs, pqhfl, zkhvfl, zws = _surface_fluxes()
    # surface excesses (0 unless ZKHVFL < 0; kept for the KLEV-1 inheritance, :396-401).
    # AD safety (cubasen.F90:341-352, 360-363): the fractional power is taken on
    # max(zws, _TINY) and the flux divisions on floored zrho*zws_f, so inactive /
    # zero-wind branches remain finite for autodiff.
    zws_f = cfg.surface_ws_factor * jnp.maximum(zws, _TINY) ** _WS_EXP       # :352
    zws_fd = jnp.maximum(zrho * zws_f, _TINY)
    ztex_s = jnp.clip(-cfg.surface_flux_factor * pahfs
                      / (zws_fd * constants.c_pd),
                      cfg.parcel_dT_excess_min_K, cfg.parcel_dT_excess_max_K)  # :360,362
    zqex_s = jnp.clip(-cfg.surface_flux_factor * pqhfl / zws_fd,
                      cfg.parcel_dq_excess_min, cfg.parcel_dq_excess_max)      # :361,363
    ztex_s = jnp.where(zkhvfl < 0.0, ztex_s, 0.0)
    zqex_s = jnp.where(zkhvfl < 0.0, zqex_s, 0.0)

    if is_surface:
        # surface departure (cubasen.F90:346-370); inactive where ZKHVFL >= 0 (:367,370)
        zws_u = zws_f
        w2 = zws_u ** 2 + cfg.surface_w2_add                # :368
        q_u = q_h[:, nlev - 1] + zqex_s                     # :364
        s_u = s_h[:, nlev - 1] + constants.c_pd * ztex_s    # :365
        T_u = (s_h[:, nlev - 1] - geo_half[:, nlev - 1]) * _RCPD_INV + ztex_s  # :366
        l_u = jnp.zeros(ncol, dt)                           # :367
        active = zkhvfl < 0.0                               # LLGO_ON (:367)
        k_dep = nlev - 1
    else:
        ztexc = cfg.parcel_dT_excess_min_K                  # :373-374
        zqexc = cfg.parcel_dq_excess_min
        inherit = k_dep == nlev - 2
        ztexc = jnp.where(inherit, jnp.maximum(ztexc, ztex_s), ztexc)  # :396-397
        zqexc = jnp.where(inherit, jnp.maximum(zqexc, zqex_s), zqexc)
        ztexc = jnp.where(inherit, jnp.minimum(ztexc, cfg.parcel_dT_excess_max_K), ztexc)  # :408
        zqexc = jnp.where(inherit, jnp.minimum(zqexc, cfg.parcel_dq_excess_max), zqexc)
        # land-only advective moistening (cubasen.F90:415-417), land_frac-weighted.
        # JKK > NJKT6 selects levels at pressures ABOVE ~700 hPa (below it);
        # the RH gate uses the HALF-level humidity of the departure over the
        # FULL-level saturation: ZQENH(JKK)/PQSEN(JKK) (:392).
        qs_dep = saturation_specific_humidity(T[:, k_dep], p_full[:, k_dep])
        adv = jnp.minimum(cfg.land_qadv_cap,
                          jnp.maximum(0.0, dq_dt_adv[:, k_dep] * cfg.land_qadv_timescale_s))
        gate = ((k_dep < nlev - 2) & (p_full[:, k_dep] > cfg.qadv_land_min_pa)
                & (q_h[:, k_dep] / qs_dep < cfg.land_qadv_rh_max))       # :415
        zqexc = zqexc + land_frac * jnp.where(gate, adv, 0.0)            # :416
        q_u = q_h[:, k_dep] + zqexc
        s_u = s_h[:, k_dep] + constants.c_pd * ztexc
        # ZTU = (ZSENH - PGEOH)*ZRCPD + ZTEXC built from the ENVIRONMENT dry
        # static energy (cubasen.F90:397): single excess
        T_u = (s_h[:, k_dep] - geo_half[:, k_dep]) * _RCPD_INV + ztexc   # :397
        # mixed layer for parcels within 60 hPa of the surface (cubasen.F90:400,
        # 424-430).  cfg.mixed_layer_gate selects the condition:
        #   "half_above"  : the literal source test
        #                    PAPH(KLEV+1) - PAPH(JKK-1) < 60 hPa (:400),
        #                    i.e. our p_half[:, nlev] - p_half[:, k_dep-1];
        #                    with uniform ~33 hPa layers the first elevated
        #                    candidate tests 3*dp ~ 100 hPa and is never mixed.
        #   "cell_centre" : the resolution-aware reinterpretation
        #                    PAPH(KLEV+1) - PAPH(JKK) < 60 hPa using the
        #                    departure's FULL-level (cell-centre) pressure
        #                    p_full[:, k_dep], which enables the lowest
        #                    elevated candidate on L137-like grids.
        # The three-sample 50 hPa accumulation (cubasen.F90:424-430) is the
        # source's in BOTH modes.  k_dep >= 1 always here, so k_dep-1 is in
        # the source's index domain.
        if cfg.mixed_layer_gate == "half_above":
            mixed = p_half[:, nlev] - p_half[:, k_dep - 1] < cfg.mixed_layer_depth_pa  # :400
        else:  # "cell_centre"
            mixed = p_half[:, nlev] - p_full[:, k_dep] < cfg.mixed_layer_depth_pa
        # accumulate over exactly the three IFS half levels JK = JKK+1, JKK,
        # JKK-1 (our half indices k_dep+1, k_dep, k_dep-1), each weighted by
        # ZWORK2 = PAPH(JK) - PAPH(JK-1), while the accumulated span ZWORK1
        # < 50 hPa; divide by ZWORK1 (the span actually accumulated).
        # AD safety: the division by the accumulated span is floored and the
        # mixed values are selected only where the span is positive.
        qw = jnp.zeros(ncol, dt); sw = jnp.zeros(ncol, dt)
        span = jnp.zeros(ncol, dt)
        for l in (k_dep + 1, k_dep, k_dep - 1):             # :424-430
            dp = p_half[:, l] - p_half[:, l - 1]            # ZWORK2 (:426)
            inc = span < cfg.mixed_layer_span_pa             # ZWORK1 < 50 hPa (:425)
            w = jnp.where(inc, dp, 0.0)
            qw = qw + q_h[:, l] * w
            sw = sw + s_h[:, l] * w
            span = span + w
        span_pos = span > 0.0
        q_m = qw / jnp.maximum(span, _TINY) + zqexc
        s_m = sw / jnp.maximum(span, _TINY) + constants.c_pd * ztexc
        T_m = (s_m - geo_half[:, k_dep]) * _RCPD_INV + ztexc  # literal double ZTEXC, :412-414
        take_m = mixed & span_pos
        q_u = jnp.where(take_m, q_m, q_u)
        s_u = jnp.where(take_m, s_m, s_u)
        T_u = jnp.where(take_m, T_m, T_u)
        l_u = jnp.zeros(ncol, dt)
        w2 = jnp.full(ncol, cfg.elevated_w2, dt)            # :435
        active = jnp.ones(ncol, bool) if elig_active is None else elig_active
    # buoyancy at the departure half level (cubasen.F90:427-431)
    tven = (1.0 + _RETV * q_h[:, k_dep]) * (s_h[:, k_dep] - geo_half[:, k_dep]) * _RCPD_INV
    tvu = (1.0 + _RETV * q_u) * T_u
    buoh = (tvu - tven) * constants.g / tven

    carry = {
        "q": q_u.astype(dt), "s": s_u.astype(dt), "w2": w2.astype(dt),
        "buoh": buoh.astype(dt), "active": active,
        # internal source defaults: ICBOT = JKK, ICTOP = KLEV-1
        # (cubasen.F90:323-325); translated to -1 on unsuccessful return
        "icbot": jnp.full(ncol, k_dep, jnp.int32),
        "ictop": jnp.full(ncol, nlev - 1, jnp.int32),
        "lldcum": jnp.zeros(ncol, bool),
    }
    return carry


def ifs_departure_search(T, q_v, p_full, p_half, geo_full, geo_half,
                         shf_w_m2, lhf_w_m2, ustar, land_frac, dq_dt_adv, cfg):
    """Full departure search (cubasen.F90:314-735).

    API humidity convention (repo convention): the input ``q_v`` is a MIXING
    RATIO; it is converted to specific humidity ``q = q_v/(1 + q_v)`` at
    entry and everything inside (half-level environment, saturation, ascent)
    works in specific humidity.  The returned ``q_u`` and ``l_u`` profiles are
    converted back to mixing ratios per unit dry air at the boundary:
    ``w = q/(1 - q)`` and ``l_w = l/(1 - q - l)``.

    Surface flux sign convention: shf_w_m2 / lhf_w_m2 follow the IFS PAHFS /
    PQHFL convention, NEGATIVE = upward (into the atmosphere); the latent
    mass flux is PQHFL [kg m-2 s-1] = lhf_w_m2 / L_v with NO extra minus
    (cubasen.F90:335).

    Land perturbation: land_frac weighting is an explicitly chosen
    approximation of the source's binary LDLAND mask (cubasen.F90:415-417);
    exact fidelity requires the caller's binary land mask.

    mixed_layer_gate (validated here): "half_above" reproduces the literal
    source condition PAPH(KLEV+1)-PAPH(JKK-1) < 60 hPa (cubasen.F90:400;
    never satisfied on ~33 hPa layers, so no elevated departure is ever
    mixed-layer initialised there); "cell_centre" reinterprets eligibility
    by the departure cell centre, PAPH(KLEV+1)-PAPH(JKK) < 60 hPa, which
    enables the lowest elevated candidate on L137-like grids.  The
    three-sample 50 hPa accumulation (cubasen.F90:424-430) is unchanged in
    both modes.

    Outputs beyond the selected ascent: ``w2_surface`` is PWU2H, the
    SURFACE test's velocity profile, always retained (cubasen.F90:358,
    559-560); ``ldsc`` / ``k_botsc`` are the LDSC / KBOTSC boundary-layer
    cloud outputs of the surface test (cubasen.F90:598-607, 639-645), with
    the surface profiles copied for ALL columns at levels >= ictop
    (:671-683), not only accepted ones.  ktype is derived from the ACCEPTED
    cloud depth (depth >= depth_split_pa -> deep, cumastrn.F90:517-521),
    not from the departure identity.  Internal defaults ICBOT = JKK /
    ICTOP = KLEV-1 (cubasen.F90:323-325) are translated to the public -1
    sentinel for unsuccessful columns (:664-668)."""
    if cfg.mixed_layer_gate not in ("half_above", "cell_centre"):
        raise ValueError(
            f"IFSTestAscentConfig.mixed_layer_gate must be 'half_above' or "
            f"'cell_centre', got {cfg.mixed_layer_gate!r}")

    ncol, nlev = T.shape
    # API boundary: mixing ratio -> specific humidity for everything inside
    q = q_v / (1.0 + q_v)
    env_half = _half_level_env(T, q, p_full, p_half, geo_full, geo_half, cfg)
    T_h, q_h, s_h = env_half
    idx = jnp.arange(ncol)

    # surface fluxes gate ONLY the surface departure (LLGO_ON=.FALSE. at :367);
    # PQHFL = lhf/L_v with NO extra minus (negative = upward)
    zrho = p_half[:, nlev] / (constants.R_d * T[:, nlev - 1] * (1.0 + _RETV * q[:, nlev - 1]))
    zkhvfl = (shf_w_m2 * _RCPD_INV + _RETV * T[:, nlev - 1] * (lhf_w_m2 / constants.L_v)) / zrho
    col_go = zkhvfl < 0.0                                   # :367

    ldcum = jnp.zeros(ncol, T.dtype)
    ktype = jnp.zeros(ncol, jnp.int32)
    kdpl = jnp.full(ncol, -1, jnp.int32)                    # public sentinel
    kcbot = jnp.full(ncol, -1, jnp.int32)
    kctop = jnp.full(ncol, -1, jnp.int32)
    wbase = jnp.zeros(ncol, T.dtype)
    # unvisited/reset parcel profiles are HALF-level values: initialise from
    # T_h[:, :nlev] / q_h[:, :nlev] (cubasen.F90:265-268), never full-level
    Tu = T_h[:, :nlev].copy()
    qu = q_h[:, :nlev].copy()
    lu = jnp.zeros((ncol, nlev), T.dtype)
    klab = jnp.zeros((ncol, nlev), jnp.int32)
    cape_out = jnp.zeros(ncol, T.dtype)
    w2_out = jnp.zeros((ncol, nlev), T.dtype)
    w2_sfc_out = jnp.zeros((ncol, nlev), T.dtype)
    ldsc = jnp.zeros(ncol, bool)
    kbotsc = jnp.full(ncol, -1, jnp.int32)
    # resolved tracks only DEEP-resolved columns (LLFIRST at :695-717, 733);
    # a shallow surface result does NOT stop the search
    resolved = jnp.zeros(ncol, bool)

    lev = jnp.arange(nlev)[None, :]
    # candidate loop: surface level, then elevated k_dep = nlev-2 .. 1
    # (JKK = KLEV..2 of the source's KLEV..NJKT1 loop, cubasen.F90:314);
    # k_dep = 0 is never a departure, so every k_dep - 1 index below is valid
    for k_dep in range(nlev - 1, 0, -1):
        is_surface = k_dep == nlev - 1
        if is_surface:
            # eligibility BEFORE the ascent: only ZKHVFL < 0 columns are
            # active for the surface departure (cubasen.F90:367)
            elig = col_go
            cand = jnp.ones(ncol, bool)
        else:
            # elevated eligibility: NJKT1 pressure floor (:314/:655) AND not
            # yet deep-resolved (LLGO_ON = .NOT.LLDEEP, :655/:735); passed as
            # the initial `active` carry so inactive columns accumulate no CAPE
            cand = p_full[:, k_dep] > cfg.departure_top_pa   # NJKT1 (:314)
            elig = cand & ~resolved
        init = _init_departure_parcel(k_dep, is_surface, T, q, p_full, p_half,
                                      geo_full, geo_half, env_half, shf_w_m2,
                                      lhf_w_m2, ustar, land_frac, dq_dt_adv,
                                      cfg, elig_active=elig)
        ilab, ztu, zqu, zlu, w2h, icbot, ictop, lldcum, zcape = \
            _test_ascent_from_departure(k_dep, is_surface, T, q, p_full, p_half,
                                        geo_full, geo_half, env_half, init, cfg)
        jkb = jnp.clip(icbot, 0, nlev - 1); jkt = jnp.clip(ictop, 0, nlev - 1)
        depth = p_half[idx, jkb] - p_half[idx, jkt]
        wb = jnp.sqrt(jnp.maximum(w2h[idx, jkb], 0.0))       # :670 / :716
        # PCAPE = MAXVAL(ZCAPE(JL,:)) over ALL departures (cubasen.F90:741)
        cape_out = jnp.maximum(cape_out, zcape)
        if is_surface:
            lldeep = depth > cfg.depth_split_pa              # '>' (:666)
            sel = lldcum & ~lldeep & col_go                  # :666-667, gated by ZKHVFL<0 (:367)
            # LDSC / KBOTSC from the surface test for ALL columns
            # (cubasen.F90:598-607, 639-645): LLDSC/LL_LDBASE are set when the
            # surface ascent found a cloud base, i.e. ICBOT was updated from
            # its internal default JKK = nlev-1 (base levels are <= nlev-2)
            lldsc = icbot < (nlev - 1)
            ldsc = jnp.where(lldsc, jnp.ones(ncol, bool), ldsc)
            kbotsc = jnp.where(lldsc, icbot, jnp.full(ncol, -1, jnp.int32))
            # PWU2H: the SURFACE test's velocity profile, always retained
            # (cubasen.F90:358, 559-560)
            w2_sfc_out = w2h
            # copy surface ascent values for JK >= JKT for ALL columns
            # (cubasen.F90:671-683); ictop carries the internal default
            # nlev-1 (= KLEV-1) when no top was found, so all levels copy
            m = lev >= ictop[:, None]
        else:
            lldeep = depth >= cfg.depth_split_pa             # '>=' (:691)
            # only the FIRST deep elevated departure resets (LLRESET :693-713,
            # LLFIRST=.FALSE. at :733) and REPLACES any shallow surface result;
            # on LLDEEP & LLFIRST the source copies LDCUM = LLDCUM (:722-726),
            # which `sel` (containing lldcum) reproduces without forcing it
            sel = lldcum & lldeep & cand & ~resolved         # :695-717
            resolved = resolved | sel                        # :733
            # LLRESET fill: ascent inside [JKT, KDPL], half-level environment
            # (klab 1) outside, klab 0 above JKT (cubasen.F90:697-713) -- the
            # environment fill uses the HALF-level values T_h/q_h (:711-713)
            inside = (lev >= jkt[:, None]) & (lev <= k_dep)
            m = sel[:, None] & inside
            env_m = sel[:, None] & ~inside
            klab = jnp.where(env_m, 1, klab)
            Tu = jnp.where(env_m, T_h[:, :nlev], Tu)
            qu = jnp.where(env_m, q_h[:, :nlev], qu)
            lu = jnp.where(env_m, 0.0, lu)
            klab = jnp.where(sel[:, None] & (lev < jkt[:, None]), 0, klab)
            # deep acceptance clears the Sc outputs (cubasen.F90:727-728)
            ldsc = jnp.where(sel, jnp.zeros(ncol, bool), ldsc)
            kbotsc = jnp.where(sel, jnp.full(ncol, -1, jnp.int32), kbotsc)
        # ktype from the ACCEPTED cloud depth, not the departure identity:
        # a selected surface parcel with depth exactly == depth_split_pa is
        # deep by cumastrn's '>=' (cumastrn.F90:517-521)
        kt = jnp.where(depth >= cfg.depth_split_pa, 1, 2)
        ldcum = jnp.where(sel, 1.0, ldcum)                   # = LLDCUM via sel
        ktype = jnp.where(sel, kt, ktype)
        kdpl = jnp.where(sel, jnp.full(ncol, k_dep, jnp.int32), kdpl)
        kcbot = jnp.where(sel, icbot, kcbot)
        kctop = jnp.where(sel, ictop, kctop)
        wbase = jnp.where(sel, wb, wbase)
        w2_out = jnp.where(sel[:, None], w2h, w2_out)
        klab = jnp.where(m, ilab, klab)
        Tu = jnp.where(m, ztu, Tu); qu = jnp.where(m, zqu, qu)
        lu = jnp.where(m, zlu, lu)

    # API boundary: specific humidity -> mixing ratio per unit dry air
    qu_w = qu / (1.0 - qu)
    l_w = lu / (1.0 - qu - lu)

    return TestAscent(ldcum, ktype, kdpl, kcbot, kctop, wbase, Tu, qu_w, l_w,
                      klab, cape_out, w2_out, w2_sfc_out, ldsc, kbotsc)


