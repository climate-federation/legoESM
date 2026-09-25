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
    (legoesm.thermo.saturation_specific_humidity); the IFS liquid/ice blend
    FOEALFCU / FOEALFA-R5LES/R5IES (cubasen.F90:486-489 and 525-540) is
    NOT available, hence the freezing correction ZLGLAC is zero
    (liquid-only condensate) and the cloud-base saturation-deficit
    interpolation uses the liquid curve and its dT derivative.
  * CUADJTQ (two Newton corrections + q = min(q, qsat)) is reproduced as
    cuadjtq_condense (cubasen.F90:462-464).
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

import jax
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
    column_refine: int = 1                # REFINED-COLUMN trigger stopgap
                                          # (codex design r14, item 3): r
                                          # sub-layers per parent layer for
                                          # ifs_departure_search_refined.
                                          # STATIC (Python int; must be a
                                          # compile-time constant because the
                                          # refined level count shapes every
                                          # refined array).  1 = no
                                          # refinement (the wrapper calls
                                          # ifs_departure_search directly).


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
        "T": "K", "q_v": "kg/kg (specific humidity, PQEN)", "p_full": "Pa", "p_half": "Pa",
        "geo_full": "m2/s2 (geopotential, surface-relative; geo_half[:, nlev] = 0)",
        "geo_half": "m2/s2", "shf_w_m2": "W/m2 (PAHFS sign: negative = upward)",
        "lhf_w_m2": "W/m2 (PQHFL sign: negative = upward; PQHFL = -lhf/L_v [kg m-2 s-1])",
        "ustar": "m/s", "land_frac": "1 (0..1)", "dq_dt_adv": "kg/kg/s (PTENQA)",
    },
    "outputs": {
        "ldcum": "1 (bool-valued float)", "ktype": "1 deep, 2 shallow, 0 none",
        "k_dpl/k_cbot/k_ctop": "surface-last full-level indices (-1 when none)",
        "w_base": "m/s (PWUBASE)", "T_u": "K",
        "q_u/l_u": "kg/kg (PQU/PLU, same moist-mass basis as PQEN)",
        "klab": "0/1/2", "cape_test": "J/kg (PCAPE = max of ZCAPE over departures)",
        "w2": "m2/s2 (ZWU2H/PWU2H of the selected ascent)",
    },
    "sign_convention": "surface fluxes follow the IFS PAHFS/PQHFL convention: "
                       "NEGATIVE = upward (into the atmosphere); PQHFL = -lhf_w_m2 / L_v.",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "OpenIFS cubasen.F90 (main 8f6f722)",
    "idealized_test": "tests/unit/test_ifs_test_ascent.py",
}


def _cuadjtq_pair(T, q, p):
    """cuadjtq.F90:324-345, KCALL=3 branch: two successive UNCLIPPED Newton
    corrections ZCOND1 = (q - q_s)/(1 + (L_v/c_pd) dq_s/dT); T += (L_v/c_pd)
    ZCOND1; q -= ZCOND1 (second pass uses the updated T).  Specific-humidity
    form: q_s = saturation_specific_humidity, dq_s/dT from dqsat_dT (the
    exact derivative of that same function).
    AD safety (JAX where-NaN rule): temperatures fed to the saturation curve
    and its derivative are clamped to the physical range
    [T_PHYS_MIN, T_PHYS_MAX] K on ALL branches (including inactive ones)
    BEFORE the thermo helpers are evaluated; masking only the results would
    not protect the cotangents.
    Declared departures (cuadjtq.F90:162-168; suphec.F90:231-240): the shared
    curve's Tetens coefficients differ slightly from IFS FOEEWM, and saturation
    is liquid-only (no FOEALFA/FOEALFCU ice blend)."""
    lvcp = constants.L_v / constants.c_pd
    for _ in range(2):
        T_s = jnp.clip(T, T_PHYS_MIN, T_PHYS_MAX)
        qs = saturation_specific_humidity(T_s, p)
        zcond1 = (q - qs) / (1.0 + lvcp * dqsat_dT(T_s, p))
        T = T + lvcp * zcond1
        q = q - zcond1
    return T, q


RLMIN = 1.0e-8      # sucldp.F90:253 default (cloud configuration, NOT yoecumf);
                     # applied at the Sc cloud base, cubasen.F90:601
TINY = 1.0e-12      # AD-safe floor for differentiated denominators (cloud-base
                     # interpolation divides by zdqsdT*zdtdp, which vanishes on
                     # inactive branches)
_Z_FLOOR = 1.0       # [m] floor on the 1/z entrainment height (geo_full - geo_half
                     # at the surface); keeps 1/z finite on inactive surface branches


# Physical temperature range used to sanitize operands fed to the saturation
# helpers on inactive/unphysical branches (JAX where-NaN rule: masking the
# RESULT is not enough, every evaluated branch must be finite).
T_PHYS_MIN = 150.0     # [K] lower physical bound for the Tetens curve
T_PHYS_MAX = 350.0     # [K] upper physical bound for the Tetens curve


def dqsat_dT(T, p):
    """dq_s/dT of the very saturation function used elsewhere in this module
    (legoesm.thermo.saturation_specific_humidity), obtained as its EXACT
    autodifferentiated derivative via jax.vmap(jax.grad(...)) on flattened
    arrays, instead of a hand-re-derived chain rule on the mixing-ratio
    helpers (review finding 5): this guarantees the Newton corrections in
    _cuadjtq_pair / cuadjtq_condense are consistent with the curve actually
    evaluated, including its internal floors/caps, and the derivative is
    itself differentiable.  Declared departures (cuadjtq.F90:162-168;
    suphec.F90:231-240) are unchanged: the shared curve's Tetens coefficients
    differ slightly from IFS FOEEWM and saturation is liquid-only."""
    flat_T = T.reshape(-1)
    flat_p = p.reshape(-1)
    grad = jax.vmap(
        jax.grad(lambda t, pp: saturation_specific_humidity(t, pp)))(flat_T, flat_p)
    return grad.reshape(T.shape)


def half_level_env(T, q_v, p_full, p_half, geo_full, geo_half, cfg):
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


def cuadjtq_condense(T, q, p):
    """CUADJTQ (KCALL=1, condensation only), cuadjtq.F90:165-186, specific-humidity
    form: FIRST correction ZCOND = MAX(0,(q - q_s)/(1 + (L_v/c_pd) dq_s/dT))
    (clipped >= 0), then a SECOND correction ZCOND1 that is UNCLIPPED but zeroed
    where the first ZCOND was zero (cuadjtq.F90:182).  There is NO final
    q = min(q, q_s) clipping in this source branch.  q_s =
    saturation_specific_humidity with dq_s/dT from dqsat_dT (the exact
    derivative of that same function, so the two Newton corrections are
    consistent with the evaluated curve near its internal modifications).
    AD safety (JAX where-NaN rule): temperatures fed to the saturation curve
    and its derivative are clamped to [T_PHYS_MIN, T_PHYS_MAX] K on ALL
    branches (including the inactive branch of the ZCOND1 where) BEFORE the
    thermo helpers are evaluated.
    Declared departures (cuadjtq.F90:162-168; suphec.F90:231-240): the shared
    curve's coefficients differ slightly from IFS FOEEWM, and saturation is
    liquid-only (ZLGLAC freezing correction therefore zero)."""
    lvcp = constants.L_v / constants.c_pd
    T_s = jnp.clip(T, T_PHYS_MIN, T_PHYS_MAX)
    qs = saturation_specific_humidity(T_s, p)
    dqs = dqsat_dT(T_s, p)
    zcond = jnp.maximum(0.0, (q - qs) / (1.0 + lvcp * dqs))   # cuadjtq.F90:171
    T1 = T + lvcp * zcond
    q1 = q - zcond
    T1_s = jnp.clip(T1, T_PHYS_MIN, T_PHYS_MAX)
    qs1 = saturation_specific_humidity(T1_s, p)
    dqs1 = dqsat_dT(T1_s, p)
    zcond1 = jnp.where(zcond > 0.0,                            # cuadjtq.F90:182
                       (q1 - qs1) / (1.0 + lvcp * dqs1), 0.0)
    T2 = T1 + lvcp * zcond1
    q2 = q1 - zcond1
    return T2, q2


def _test_ascent_from_departure(k_dep, is_surface, T, q_v, p_full, p_half,
                                geo_full, geo_half, env_half, parcel_init, cfg):
    """One test ascent from departure level k_dep upward (IFS DO JK=JKK-1,JKT2,-1,
    cubasen.F90:437-600).  FIXED-LENGTH masked lax.scan (codex r14 item 2):
    ``k_dep`` is a TRACED int32 scalar (or (ncol,) array) and ``is_surface``
    a TRACED bool (the surface departure is the first step of the departure
    scan; the 1/z vs deep mixing is selected with jnp.where on it, both
    branches being cheap).  The scan runs the FIXED levels j = N-2 ... 0
    (starting at N-1 would make the profile gather at j+1 out of bounds);
    the departure-sized arange of the previous unrolled implementation is
    replaced by applying work = (j < k_dep) & active & top_mask to every
    carry and write, so levels at or above the departure (and above NJKT2)
    are no-ops that preserve the carry bit-for-bit.  Levels with
    p_full <= cfg.test_top_pa are masked out (NJKT2 ~ 60 hPa).

    The key ``parcel_init["T_dep"]`` IS consumed: when present it is the
    source's ZTU(JKK) computed by _init_departure_parcel (single excess,
    literal double excess in the mixed-layer branch, surface formula) and is
    stored into the scratch T_u at the departure level; the reconstruction
    ``T = (s - geo_half)/c_pd`` is only a fallback when the key is absent.

    Parcel profiles T_u/q_u/l_u/w2h/ilab are HALF-LEVEL quantities of length
    nlev (IFS half level JK <-> our index JK-1).  They are NOT created fresh
    here: the caller passes the persistent scratch arrays of the previous
    departure candidate through ``parcel_init`` (review finding 3; the source
    initialises them once before the outer search, cubasen.F90:263-271), and
    only the departure level is (re)set from the current parcel state via a
    DYNAMIC scatter ``.at[idx, k_dep].set`` (idx = arange(ncol)), keeping the
    exact values of the unrolled version; the label lifecycle at a
    terminating level leaves the previous label untouched (no forced zero),
    matching cubasen.F90:615-629.  Specific humidity throughout (virtual
    temperature T(1 + RETV q) directly as in the source); saturation on the
    specific-humidity curve with the FOEEWM / liquid-only departures declared
    in _cuadjtq_pair / cuadjtq_condense.  The masked-scan version is
    checked bit-for-bit against the previous unrolled implementation.

    AD safety (JAX where-NaN rule, review gradient-hazard table): every
    division on an evaluated branch takes a positive-floored operand --
    the deep-test saturation ratio floors the surface saturation and is
    clipped to [0, 1] BEFORE cubing; the buoyancy division floors the
    environmental virtual temperature; the cloud-base ZDTDP division floors
    the half-level pressure -- and temperatures fed to the saturation
    helpers (qsat_j, qsat_sfc, ZQSU/ZDQSDT) are clamped to the physical
    range [T_PHYS_MIN, T_PHYS_MAX] K before evaluation."""
    nlev = T.shape[1]
    T_h, q_h, s_h = env_half
    dt = T_h.dtype
    ncol = T_h.shape[0]
    idx = jnp.arange(ncol)
    k_dep = jnp.asarray(k_dep, jnp.int32)
    is_surface = jnp.asarray(is_surface, bool)

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
    r_d = jnp.asarray(constants.R_d, dt)
    c_pd_arr = jnp.asarray(constants.c_pd, dt)
    t_min = jnp.asarray(T_PHYS_MIN, dt)
    t_max = jnp.asarray(T_PHYS_MAX, dt)
    rlmin = jnp.asarray(RLMIN, dt)
    tiny = jnp.asarray(TINY, dt)
    z_floor = jnp.asarray(_Z_FLOOR, dt)

    # FIXED-LENGTH candidate ascent levels j = nlev-2 ... 0 (surface-last:
    # upward), int32 so all index carries keep a single dtype under
    # lax.scan; the departure-dependent length of the previous
    # ``jnp.arange(k_dep-1, -1, -1)`` is replaced by the (j < k_dep) work
    # gate below.  j starts at nlev-2 (NOT nlev-1) so every profile gather
    # at j+1 stays inside the arrays on all masked iterations.
    js = jnp.arange(nlev - 2, -1, -1, dtype=jnp.int32)

    # deep-test denominator: PQSEN(JL,KLEV) on the lowest FULL level
    # (cubasen.F90:472); a specific humidity.  AD safety: the temperature is
    # clamped to the physical range and the denominator floored positive
    # before the (pre-cube-clipped) ratio is formed.
    qsat_sfc = saturation_specific_humidity(
        jnp.clip(T[:, nlev - 1], t_min, t_max), p_full[:, nlev - 1])
    qsat_sfc = jnp.maximum(qsat_sfc, tiny)

    # half-level parcel profiles from the PERSISTENT scratch arrays passed by
    # the caller (review finding 3; cubasen.F90:263-271): no fresh labels or
    # profiles are created here; only the departure level is set from the
    # current parcel state (ILAB=1 at :345/:375), via a DYNAMIC scatter at
    # the TRACED k_dep with idx = arange(ncol) -- the exact values of the
    # unrolled version.
    active = parcel_init["active"]
    # the source's ZTU(JKK), preserved by _init_departure_parcel (review
    # finding 1); the (s - geo_half)/c_pd reconstruction is only a fallback
    if "T_dep" in parcel_init:
        T_dep = jnp.asarray(parcel_init["T_dep"], dt)
    else:
        T_dep = (parcel_init["s"] - geo_half[idx, k_dep]) * rcpd_inv
    T_u = jnp.asarray(parcel_init["T_u"], dt)
    q_u = jnp.asarray(parcel_init["q_u"], dt)
    l_u = jnp.asarray(parcel_init["l_u"], dt)
    w2h = jnp.asarray(parcel_init["w2h"], dt)
    ilab0 = jnp.asarray(parcel_init["ilab"], jnp.int32)
    T_u = T_u.at[idx, k_dep].set(jnp.where(active, T_dep, T_u[idx, k_dep]))
    q_u = q_u.at[idx, k_dep].set(
        jnp.where(active, parcel_init["q"].astype(dt), q_u[idx, k_dep]))
    l_u = l_u.at[idx, k_dep].set(jnp.where(active, 0.0, l_u[idx, k_dep]))  # ZLU(JKK)=0 (:344/:385)
    w2h = w2h.at[idx, k_dep].set(
        jnp.where(active, parcel_init["w2"].astype(dt), w2h[idx, k_dep]))
    ilab0 = ilab0.at[idx, k_dep].set(
        jnp.where(active, jnp.asarray(1, jnp.int32), ilab0[idx, k_dep]))  # ILAB=1, :345/:375

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
        # FIXED-LENGTH work gate: NJKT2 bound AND the traced departure
        # (levels j >= k_dep are the no-ops the unrolled loop never ran)
        work = (j < k_dep) & active & (p_full[:, j] > top_pa)  # NJKT2 bound

        dz = (geo_half[:, j] - geo_half[:, j + 1]) * g_inv
        qf = 0.5 * (q_h[:, j + 1] + q_h[:, j])
        sf = 0.5 * (s_h[:, j + 1] + s_h[:, j])
        # BOTH mixing forms are evaluated and selected with jnp.where on the
        # TRACED is_surface (both are cheap; KINDEX==KLEV-1 branches are dead
        # on Earth, NJKT1 < KLEV-1).
        # 1/z mixing for the shallow/surface departure (cubasen.F90:437, 446-451).
        # AD safety: the 1/z height is floored away from zero.
        z_h = jnp.maximum((geo_full[:, j] - geo_half[:, nlev]) * g_inv, z_floor)
        zeps = c1 / z_h + c2
        zmix_s = jnp.minimum(1.0, 0.5 * dz * zeps)          # cubasen.F90:450-452
        tmp = 1.0 / (1.0 + zmix_s)
        q_new_s = (q * (1.0 - zmix_s) + 2.0 * zmix_s * qf) * tmp  # cubasen.F90:456-458
        s_new_s = (s * (1.0 - zmix_s) + 2.0 * zmix_s * sf) * tmp  # cubasen.F90:459-461
        # deep-test mixing 0.4*ENTRORG*dz*min(1,(PQSEN(JK)/PQSEN(KLEV))**3)
        # (cubasen.F90:472-478); PQSEN is a specific humidity.
        # AD safety: T clamped to the physical range before the saturation
        # call; the ratio is divided by a POSITIVE-floored qsat_sfc and
        # clipped to [0, 1] BEFORE the cube (no overflow, no zero divide).
        qsat_j = saturation_specific_humidity(
            jnp.clip(T[:, j], t_min, t_max), p_full[:, j])
        ratio = jnp.clip(qsat_j / qsat_sfc, 0.0, 1.0)
        zmix_d = jnp.minimum(1.0, deep_fac * entr_base * dz
                             * ratio ** _QSAT_RATIO_EXP)
        q_new_d = q * (1.0 - zmix_d) + qf * zmix_d         # cubasen.F90:483
        s_new_d = s * (1.0 - zmix_d) + sf * zmix_d         # cubasen.F90:484
        zmix = jnp.where(is_surface, zmix_s, zmix_d)
        q_new = jnp.where(is_surface, q_new_s, q_new_d)
        s_new = jnp.where(is_surface, s_new_s, s_new_d)

        # condensation (CUADJTQ), condensate added, half retained (cubasen.F90:466-493)
        q_old = q_new
        T_new0 = (s_new - geo_half[:, j]) * rcpd_inv
        T_adj, q_adj = cuadjtq_condense(T_new0, q_new, p_half[:, j])
        zdq = jnp.maximum(q_old - q_adj, 0.0)
        l_new = ret_frac * (l_u[:, j + 1] + zdq)
        # freezing correction ZLGLAC = 0: liquid-only saturation (FOEALFCU unavailable)
        T_new = T_adj
        s_new = c_pd_arr * T_new + geo_half[:, j]

        # buoyancy on half levels (cubasen.F90:499-508); AD safety: the
        # environmental virtual temperature is floored positive before the
        # division (zero/negative operands on inactive branches must stay
        # finite under autodiff).
        tvu = (1.0 + retv * q_adj - l_new) * T_new
        tven = (1.0 + retv * q_h[:, j]) * T_h[:, j]
        buoh_new = (tvu - tven) * cg / jnp.maximum(tven, tiny)
        buof = 0.5 * (buoh_new + buoh)
        # kinetic-energy recurrence, ZAW = ZBW = 1 (cubasen.F90:511-513)
        w2_new = (w2 * (1.0 - 2.0 * zmix) + 2.0 * buof * dz) / (1.0 + 2.0 * zmix)
        cape_new = cape + jnp.maximum(0.0, buof * dz)      # cubasen.F90:521

        # first layer with liquid water: exact cloud base (cubasen.F90:524-551).
        # Saturation-deficit interpolation on the liquid specific-humidity curve
        # (declared departure: FOEEWM coefficients differ, FOEALFA-R5LES/R5IES
        # blend unavailable, cubasen.F90:527, 532-535).  AD safety: the
        # temperature is clamped to the physical range before both saturation
        # calls, the half-level pressure in ZDTDP is floored positive, and the
        # interpolation denominator is floored away from zero on inactive
        # branches.
        cond_first = (l_new > 0.0) & (ilab[:, j + 1] == 1)
        T_cb = jnp.clip(T_u[:, j + 1], t_min, t_max)
        zqsu = saturation_specific_humidity(T_cb, p_half[:, j + 1])
        zdqsdT = dqsat_dT(T_cb, p_half[:, j + 1])
        zdq_cb = jnp.minimum(0.0, q_u[:, j + 1] - zqsu)
        zdtdp = r_d * T_u[:, j + 1] / (c_pd_arr * jnp.maximum(p_half[:, j + 1], tiny))
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
        # scratch label lifecycle (cubasen.F90:265-268, 615-629): the previous
        # carried ilab(:, j) is left where the level is not accepted (no
        # zeroing, no fresh creation), so an accepted terminal-level label
        # from a previous departure survives
        labj = ilab[:, j]
        labj = jnp.where(case_top | case_bot, 2, labj)

        # stop at w2 < 0; ICTOP/LLDCUM use the UPDATED l_below
        # (cubasen.F90:615-621, strict '<'); the internal ICTOP default is
        # the caller's (cubasen.F90:323-325) and is not touched here
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
    q_u: jnp.ndarray         # (ncol, nlev) kg/kg specific humidity (PQU)
    l_u: jnp.ndarray         # (ncol, nlev) kg/kg, same moist-mass basis (PLU)
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
                           ustar, land_frac, dq_dt_adv, cfg, elig_active=None,
                           scratch=None):
    """Initialise the departure-level parcel and scan carry (cubasen.F90:314-435).

    Nested-scan form (codex r14 item 2): ``k_dep`` is a TRACED int32 scalar
    (or (ncol,) array) and ``is_surface`` a TRACED bool; both the surface and
    the elevated branch are evaluated (both are cheap) and selected with
    jnp.where, so this function is the first step of the departure lax.scan
    in ifs_departure_search.  All ``arr[:, k_dep]``-style static slices are
    dynamic gathers ``arr[idx, k_dep]`` with indices clipped to the valid
    range; the gathered results only matter on the branch (and columns) where
    they are selected, so clipped-index gathers on the discarded branch are
    harmless (k_dep = 0 never occurs: the departure scan runs N-1 ... 1).

    q_v here is SPECIFIC HUMIDITY (PQEN, cubasen.F90:72), the same basis
    as at the ifs_departure_search entry -- there is no conversion seam.
    dq_dt_adv is a SPECIFIC-humidity tendency [kg/kg/s] (PTENQA) on the
    same basis.

    The departure temperature is returned EXPLICITLY as ``"T_dep"`` (review
    finding 1): it is the source's ZTU(JKK) -- the single ZTEXC excess over
    the environment temperature on the plain elevated branch
    (cubasen.F90:397), the literal DOUBLE ZTEXC excess in the mixed-layer
    branch (:414) and the surface formula (:366).  It is preserved separately
    from ``"s"`` (the singly perturbed ZSUH) because the source stores both;
    the departure-level buoyancy ``"buoh"`` is computed from T_dep, and the
    ascent stores T_dep (not a reconstruction from s) into the scratch T_u at
    the departure level.

    Internal defaults follow the source: ICBOT = JKK = k_dep,
    ICTOP = KLEV-1 <-> our nlev-2 (cubasen.F90:323-325; IFS KLEV-1 is the
    second-lowest half level, our index nlev-2).  These are translated to
    the public -1 sentinel only in the returned TestAscent for unsuccessful
    columns (cubasen.F90:664-668).  With the internal ICTOP default nlev-2,
    the surface-profile copy JK >= JKT (cubasen.F90:673-678) copies the
    BOTTOM TWO levels when no top was diagnosed, matching the source.

    ``scratch`` carries the PERSISTENT scratch arrays (T_u, q_u, l_u, w2h,
    ilab) initialised once before the outer search (cubasen.F90:263-271:
    ILAB=0, ZTU=PTENH, ZQU=PQENH, ZLU=0, ZWU2H=0) and threaded through every
    departure by ifs_departure_search; the source never resets them between
    candidates.  They are passed through unchanged here -- only the ascent
    writes the departure level into them (dynamic scatter ``.at[idx,
    k_dep].set``).

    elig_active carries the pre-ascent eligibility (only ZKHVFL < 0 columns
    are active for the surface departure, cubasen.F90:367; only columns with
    full-level departure pressure p_full[:, k_dep] > departure_top_pa and not
    yet deep-resolved are active for elevated departures, :655/:735), so that
    inactive columns accumulate no CAPE and MAXVAL(ZCAPE) matches the source
    (:741).  The pressure thresholds departure_top_pa / qadv_land_min_pa are
    documented approximations of the source's fixed level indices NJKT1 /
    NJKT6 (C:310, 392 are full-level indices) and use the departure's
    FULL-level pressure consistently.

    The elevated loop runs with k_dep >= 1 (the departure scan runs
    N-1 ... 1), so every k_dep - 1 gather below is inside the source's index
    domain (JKK = KLEV-1..2, :302); the k_dep == N-2 excess inheritance
    (cubasen.F90:380-391, 396-397, 408-409) and the ordered mixed-layer
    offsets (+1, 0, -1) with the pre-add < 50 hPa test (cubasen.F90:400,
    424-430) and the literal double temperature excess (:380-414) are
    preserved exactly.  The nested-scan version of this routine is checked
    bit-for-bit against the previous unrolled implementation."""
    nlev = T.shape[1]
    T_h, q_h, s_h = env_half
    ncol = T.shape[0]
    dt = T.dtype
    idx = jnp.arange(ncol)
    k_dep = jnp.asarray(k_dep, jnp.int32)
    is_surface = jnp.asarray(is_surface, bool)
    # clipped dynamic level indices (k_dep in 1..nlev-1, so k_dep-1 >= 0;
    # k_dep+1 <= nlev is inside the half arrays of length nlev+1)
    k0 = jnp.clip(k_dep, 0, nlev)
    km1 = jnp.clip(k_dep - 1, 0, nlev)

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
    # max(zws, TINY) and the flux divisions on floored zrho*zws_f, so inactive /
    # zero-wind branches remain finite for autodiff.
    zws_f = cfg.surface_ws_factor * jnp.maximum(zws, TINY) ** _WS_EXP       # :352
    zws_fd = jnp.maximum(zrho * zws_f, TINY)
    ztex_s = jnp.clip(-cfg.surface_flux_factor * pahfs
                      / (zws_fd * constants.c_pd),
                      cfg.parcel_dT_excess_min_K, cfg.parcel_dT_excess_max_K)  # :360,362
    zqex_s = jnp.clip(-cfg.surface_flux_factor * pqhfl / zws_fd,
                      cfg.parcel_dq_excess_min, cfg.parcel_dq_excess_max)      # :361,363
    ztex_s = jnp.where(zkhvfl < 0.0, ztex_s, 0.0)
    zqex_s = jnp.where(zkhvfl < 0.0, zqex_s, 0.0)

    # ---- surface branch (cubasen.F90:346-370); inactive where ZKHVFL >= 0
    # (:367,370).  All indices are the static surface level nlev-1 (JKK = KLEV).
    zws_u = zws_f
    w2_sfc = zws_u ** 2 + cfg.surface_w2_add                # :368
    q_u_s = q_h[:, nlev - 1] + zqex_s                       # :364
    s_u_s = s_h[:, nlev - 1] + constants.c_pd * ztex_s      # :365
    T_dep_s = (s_h[:, nlev - 1] - geo_half[:, nlev - 1]) * _RCPD_INV + ztex_s  # ZTU, :366
    l_u_s = jnp.zeros(ncol, dt)                             # :367
    active_s = zkhvfl < 0.0                                 # LLGO_ON (:367)
    # the surface departure level is nlev-1 (the first scan step passes it
    # as k_dep already; no override needed, only the traced is_surface
    # selects this branch)

    # ---- elevated branch (cubasen.F90:372-435), dynamic gathers at k_dep
    ztexc = jnp.asarray(cfg.parcel_dT_excess_min_K, dt)     # :373-374
    zqexc = jnp.asarray(cfg.parcel_dq_excess_min, dt)
    inherit = k_dep == nlev - 2                             # JKK == KLEV-1 (:380)
    ztexc = jnp.where(inherit, jnp.maximum(ztexc, ztex_s), ztexc)  # :396-397
    zqexc = jnp.where(inherit, jnp.maximum(zqexc, zqex_s), zqexc)
    ztexc = jnp.where(inherit, jnp.minimum(ztexc, cfg.parcel_dT_excess_max_K), ztexc)  # :408
    zqexc = jnp.where(inherit, jnp.minimum(zqexc, cfg.parcel_dq_excess_max), zqexc)
    # land-only advective moistening (cubasen.F90:415-417), land_frac-weighted
    # DOCUMENTED APPROXIMATION of the source's binary LDLAND mask (exact
    # fidelity requires the caller's binary land mask).
    # JKK > NJKT6 selects levels at pressures ABOVE ~700 hPa (below it);
    # qadv_land_min_pa is a documented approximation of the fixed level
    # index NJKT6 and uses the departure's FULL-level pressure; the RH
    # gate uses the HALF-level humidity of the departure over the
    # FULL-level saturation: ZQENH(JKK)/PQSEN(JKK) (:392).
    qs_dep = saturation_specific_humidity(T[idx, k0], p_full[idx, k0])
    adv = jnp.minimum(cfg.land_qadv_cap,
                      jnp.maximum(0.0, dq_dt_adv[idx, k0] * cfg.land_qadv_timescale_s))
    gate = ((k_dep < nlev - 2) & (p_full[idx, k0] > cfg.qadv_land_min_pa)
            & (q_h[idx, k0] / qs_dep < cfg.land_qadv_rh_max))           # :415
    zqexc = zqexc + land_frac * jnp.where(gate, adv, 0.0)                # :416
    q_u_e = q_h[idx, k0] + zqexc
    s_u_e = s_h[idx, k0] + constants.c_pd * ztexc
    # ZTU = (ZSENH - PGEOH)*ZRCPD + ZTEXC built from the ENVIRONMENT dry
    # static energy (cubasen.F90:397): single excess; preserved separately
    # from s_u_e (which carries the same single excess here, but see the
    # mixed-layer branch below where the two differ -- the literal double
    # ZTEXC in ZTU, :414)
    T_dep_e = (s_h[idx, k0] - geo_half[idx, k0]) * _RCPD_INV + ztexc      # :397
    # mixed layer for parcels within 60 hPa of the surface (cubasen.F90:400,
    # 424-430).  cfg.mixed_layer_gate selects the condition:
    #   "half_above"  : the literal source test
    #                    PAPH(KLEV+1) - PAPH(JKK-1) < 60 hPa (:400),
    #                    i.e. our p_half[:, nlev] - p_half[idx, k_dep-1];
    #                    with uniform ~33 hPa layers the first elevated
    #                    candidate tests 3*dp ~ 100 hPa and is never mixed.
    #   "cell_centre" : the resolution-aware DEPARTURE from the source,
    #                    PAPH(KLEV+1) - PAPH(JKK) < 60 hPa using the
    #                    departure's FULL-level (cell-centre) pressure
    #                    p_full[idx, k_dep], which enables the lowest
    #                    elevated candidate on L137-like grids.  Note
    #                    (finding 9) that a "962 hPa departure" named by
    #                    its full level still LAUNCHES at the half level
    #                    p_half[k_dep].
    # The three-sample 50 hPa accumulation (cubasen.F90:424-430) is the
    # source's in BOTH modes.  k_dep >= 1 always here (the departure scan
    # runs N-1 ... 1), so k_dep-1 is in the source's index domain.
    if cfg.mixed_layer_gate == "half_above":
        mixed = p_half[:, nlev] - p_half[idx, km1] < cfg.mixed_layer_depth_pa  # :400
    else:  # "cell_centre"
        mixed = p_half[:, nlev] - p_full[idx, k0] < cfg.mixed_layer_depth_pa
    # accumulate over exactly the three IFS half levels JK = JKK+1, JKK,
    # JKK-1 (our half indices k_dep+1, k_dep, k_dep-1) IN THAT ORDER, each
    # weighted by ZWORK2 = PAPH(JK) - PAPH(JK-1), while the accumulated span
    # ZWORK1 < 50 hPa (the pre-add test, cubasen.F90:425); divide by ZWORK1
    # (the span actually accumulated).
    # AD safety: the division by the accumulated span is floored and the
    # mixed values are selected only where the span is positive.
    qw = jnp.zeros(ncol, dt); sw = jnp.zeros(ncol, dt)
    span = jnp.zeros(ncol, dt)
    for off in (1, 0, -1):                                 # :424-430
        l = jnp.clip(k_dep + off, 0, nlev)                 # k_dep+1, k_dep, k_dep-1
        lm1 = jnp.clip(k_dep + off - 1, 0, nlev)
        dp = p_half[idx, l] - p_half[idx, lm1]             # ZWORK2 (:426)
        inc = span < cfg.mixed_layer_span_pa               # ZWORK1 < 50 hPa (:425)
        w = jnp.where(inc, dp, 0.0)
        qw = qw + q_h[idx, l] * w
        sw = sw + s_h[idx, l] * w
        span = span + w
    span_pos = span > 0.0
    q_m = qw / jnp.maximum(span, TINY) + zqexc
    s_m = sw / jnp.maximum(span, TINY) + constants.c_pd * ztexc
    T_m = (s_m - geo_half[idx, k0]) * _RCPD_INV + ztexc    # literal double ZTEXC, :412-414
    take_m = mixed & span_pos
    q_u_e = jnp.where(take_m, q_m, q_u_e)
    s_u_e = jnp.where(take_m, s_m, s_u_e)
    T_dep_e = jnp.where(take_m, T_m, T_dep_e)
    l_u_e = jnp.zeros(ncol, dt)
    w2_e = jnp.full(ncol, cfg.elevated_w2, dt)             # :435
    active_e = jnp.ones(ncol, bool) if elig_active is None else elig_active

    # ---- branch selection on the TRACED is_surface (both branches are
    # cheap; jnp.where over both, as allowed by the codex spec)
    q_u = jnp.where(is_surface, q_u_s, q_u_e)
    s_u = jnp.where(is_surface, s_u_s, s_u_e)
    T_dep = jnp.where(is_surface, T_dep_s, T_dep_e)
    l_u = jnp.where(is_surface, l_u_s, l_u_e)
    w2 = jnp.where(is_surface, w2_sfc, w2_e)
    active = jnp.where(is_surface, active_s, active_e)

    # buoyancy at the departure half level (cubasen.F90:427-431), computed
    # from the PRESERVED departure temperature T_dep (review finding 1);
    # dynamic gathers at the traced k_dep (for the surface departure
    # k_dep = nlev-1, exactly the source's static surface level)
    tven = (1.0 + _RETV * q_h[idx, k0]) * (s_h[idx, k0] - geo_half[idx, k0]) * _RCPD_INV
    tvu = (1.0 + _RETV * q_u) * T_dep
    buoh = (tvu - tven) * constants.g / tven

    if scratch is None:
        # standalone use: fresh scratch (cubasen.F90:263-271)
        scratch = {"T_u": jnp.zeros((ncol, nlev), dt),
                   "q_u": jnp.zeros((ncol, nlev), dt),
                   "l_u": jnp.zeros((ncol, nlev), dt),
                   "w2h": jnp.zeros((ncol, nlev), dt),
                   "ilab": jnp.zeros((ncol, nlev), jnp.int32)}

    carry = {
        "q": q_u.astype(dt), "s": s_u.astype(dt), "w2": w2.astype(dt),
        "buoh": buoh.astype(dt), "active": active,
        # the source's ZTU(JKK), preserved separately from s (review finding 1)
        "T_dep": T_dep.astype(dt),
        # internal source defaults: ICBOT = JKK, ICTOP = KLEV-1 <-> nlev-2
        # (cubasen.F90:323-325); translated to -1 on unsuccessful return;
        # with this default the surface copy JK >= JKT takes the bottom TWO
        # levels (cubasen.F90:673-678)
        "icbot": jnp.full(ncol, k_dep, jnp.int32),
        "ictop": jnp.full(ncol, nlev - 2, jnp.int32),
        "lldcum": jnp.zeros(ncol, bool),
        # persistent scratch, passed through untouched (review finding 3);
        # only the ascent writes the departure level into them
        "T_u": jnp.asarray(scratch["T_u"], dt),
        "q_u": jnp.asarray(scratch["q_u"], dt),
        "l_u": jnp.asarray(scratch["l_u"], dt),
        "w2h": jnp.asarray(scratch["w2h"], dt),
        "ilab": jnp.asarray(scratch["ilab"], jnp.int32),
    }
    return carry


def ifs_departure_search(T, q_v, p_full, p_half, geo_full, geo_half,
                         shf_w_m2, lhf_w_m2, ustar, land_frac, dq_dt_adv, cfg):
    """Full departure search (cubasen.F90:314-735).

    Nested-scan structure (codex r14 item 2): the candidate-departure loop is
    a single ``lax.scan`` over the departures N-1 ... 1 (JKK = KLEV..2 of the
    source's KLEV..NJKT1 loop, cubasen.F90:314) with a TRACED k_dep
    (k_dep = N-1-step; the first step is the surface departure with the
    traced is_surface flag), carrying the persistent scratch (T_u, q_u, l_u,
    w2h, ilab), the selected outputs (ldcum, ktype, k_dpl, k_cbot, k_ctop,
    w_base, the output profiles Tu/qu/lu, klab, w2, w2_surface, ldsc,
    k_botsc), the first-deep/resolved flags, the surface diagnostics and the
    maximum CAPE; no stacked candidate history is returned (ys = None,
    unroll = 1).  The per-departure ascent is a fixed-length masked scan
    (see _test_ascent_from_departure).  REGRESSION ANCHOR: the nested-scan
    version is checked bit-for-bit against the previous unrolled
    implementation (Python loop over static k_dep), including the ``n = 1``
    behaviour.  Cost note: the arithmetic is O(N^2) per column (N-1
    departures x N-1 masked ascent levels); reverse-mode AD memory over the
    nested scans is to be measured.

    API humidity convention (oracle: the vendored OpenIFS source): ``q_v``
    is SPECIFIC humidity [kg/kg], exactly the source's PQEN ("PROVISIONAL
    ENVIRONMENT SPEC. HUMIDITY", cubasen.F90:72; PQENH likewise :62, set
    from PQEN at cuinin.F90:187, with PQU initialised from PQENH at
    cuinin.F90:211).  NO unit conversion happens at either boundary: the
    half-level environment, saturation and ascent all work in specific
    humidity on this single moist-mass basis, and the returned ``q_u`` and
    ``l_u`` profiles are PQU / PLU on the SAME moist-mass basis as PQEN
    (cubasen.F90:677-678 copies ZQU/ZLU straight into PQU/PLU;
    cuascn.F90:526/534 combine PQENH and PQU with the same ZDMFEN/ZDMFDE
    weights, :570-572 call CUADJTQ on PQU directly, :618 moves
    ZQOLD - PQU into PLU).  ``dq_dt_adv`` is a specific-humidity tendency
    [kg/kg/s] (PTENQA) on the same basis; there is no mixing-ratio seam.

    Surface flux sign convention: shf_w_m2 / lhf_w_m2 follow the IFS PAHFS /
    PQHFL convention, NEGATIVE = upward (into the atmosphere); the latent
    mass flux is PQHFL [kg m-2 s-1] = lhf_w_m2 / L_v with NO extra minus
    (cubasen.F90:335).

    Scratch lifecycle (review finding 3; cubasen.F90:263-271, 296-303): the
    scratch arrays T_u (= T_h), q_u (= q_h), l_u = 0, w2h = 0, ilab = 0
    (int32) are created ONCE before the candidate scan and carried through
    every departure: they are passed to _init_departure_parcel / the ascent
    in the init dict, and the ascent's returned arrays become the scratch
    for the NEXT departure (scan carry).  The source never resets them
    between candidates (the LLRESET fill C:697-713 writes the OUTPUT arrays
    klab/PTU/PQU/PLU, not the scratch).

    Land perturbation: land_frac weighting is an explicitly chosen
    DOCUMENTED APPROXIMATION of the source's binary LDLAND mask
    (cubasen.F90:415-417), which applies the full increment wherever LDLAND
    is true; exact fidelity requires the caller's binary land mask.

    Pressure thresholds: departure_top_pa (NJKT1 ~ 350 hPa) and
    qadv_land_min_pa (NJKT6 ~ 700 hPa) are DOCUMENTED APPROXIMATIONS of the
    source's fixed full-level indices (cubasen.F90:310, 392); both are
    evaluated on the departure's FULL-level pressure p_full[:, k_dep]
    consistently (note I:165 / C:310's JKK > NJKT6 EXCLUDES the boundary
    level itself, unlike a plain above-700 hPa inclusion).

    mixed_layer_gate: "half_above" reproduces the literal source condition
    PAPH(KLEV+1)-PAPH(JKK-1) < 60 hPa (cubasen.F90:400; never satisfied on
    ~33 hPa layers, so no elevated departure is ever mixed-layer initialised
    there); "cell_centre" is a DOCUMENTED DEPARTURE that reinterprets
    eligibility by the departure cell centre,
    PAPH(KLEV+1)-PAPH(JKK) < 60 hPa, enabling the lowest elevated candidate
    on L137-like grids.  In both modes the parcel still LAUNCHES at the half
    level p_half[k_dep] (finding 9: a "962 hPa departure" named by its full
    level is not the parcel's actual launch pressure).  The three-sample
    50 hPa accumulation (cubasen.F90:424-430) is unchanged in both modes.

    Outputs beyond the selected ascent: ``w2_surface`` is PWU2H, the SURFACE
    test's velocity profile, always retained (cubasen.F90:358, 559-560);
    ``ldsc`` / ``k_botsc`` are the LDSC / KBOTSC boundary-layer cloud
    outputs of the surface test (cubasen.F90:598-607, 639-645), with the
    surface profiles copied for ALL columns at levels >= ictop
    (:671-683) -- with the internal ICTOP default nlev-2 (KLEV-1, :325) the
    copy takes the BOTTOM TWO levels when no top was diagnosed.  ktype is
    derived from the ACCEPTED cloud depth (depth >= depth_split_pa -> deep,
    cumastrn.F90:517-521), not from the departure identity.  Internal
    defaults ICBOT = JKK / ICTOP = KLEV-1 <-> nlev-2 (cubasen.F90:323-325)
    are translated to the public -1 sentinel for unsuccessful columns
    (:664-668)."""
    # public entry: accept array-likes (numpy inputs would otherwise meet
    # traced indices inside the scans and fail with a tracer conversion)
    T, q_v, p_full, p_half, geo_full, geo_half = (jnp.asarray(a) for a in (T, q_v, p_full, p_half, geo_full, geo_half))
    shf_w_m2, lhf_w_m2, ustar, land_frac, dq_dt_adv = (jnp.asarray(a) for a in (shf_w_m2, lhf_w_m2, ustar, land_frac, dq_dt_adv))
    if cfg.mixed_layer_gate not in ("half_above", "cell_centre"):
        raise ValueError(
            f"IFSTestAscentConfig.mixed_layer_gate must be 'half_above' or "
            f"'cell_centre', got {cfg.mixed_layer_gate!r}")

    ncol, nlev = T.shape
    # q_v IS specific humidity, exactly PQEN (cubasen.F90:72) -- no conversion
    q = q_v
    env_half = half_level_env(T, q, p_full, p_half, geo_full, geo_half, cfg)
    T_h, q_h, s_h = env_half
    idx = jnp.arange(ncol)

    # surface fluxes gate ONLY the surface departure (LLGO_ON=.FALSE. at :367);
    # PQHFL = lhf/L_v with NO extra minus (negative = upward)
    zrho = p_half[:, nlev] / (constants.R_d * T[:, nlev - 1] * (1.0 + _RETV * q[:, nlev - 1]))
    zkhvfl = (shf_w_m2 * _RCPD_INV + _RETV * T[:, nlev - 1] * (lhf_w_m2 / constants.L_v)) / zrho
    col_go = zkhvfl < 0.0                                   # :367

    lev = jnp.arange(nlev)[None, :]
    ones_ncol = jnp.ones(ncol, bool)
    # persistent scratch (review finding 3; cubasen.F90:263-271): created ONCE
    # before the candidate scan (ILAB=0, ZTU=PTENH, ZQU=PQENH, ZLU=0, ZWU2H=0
    # -- here initialised from the half-level environment, the first call's
    # PTU/PQU/PLU/KLAB equivalent) and carried through every departure; the
    # ascent's returned arrays become the next departure's scratch, never
    # reset (the LLRESET fill C:697-713 writes the OUTPUT arrays, not these)
    # OUTPUT profiles (what the source's PTU/PQU/PLU/KLAB hold on exit)
    # resolved tracks only DEEP-resolved columns (LLFIRST at :695-717, 733);
    # a shallow surface result does NOT stop the search
    carry0 = {
        "ldcum": jnp.zeros(ncol, T.dtype),
        "ktype": jnp.zeros(ncol, jnp.int32),
        "kdpl": jnp.full(ncol, -1, jnp.int32),              # public sentinel
        "kcbot": jnp.full(ncol, -1, jnp.int32),
        "kctop": jnp.full(ncol, -1, jnp.int32),
        "wbase": jnp.zeros(ncol, T.dtype),
        "T_u": T_h[:, :nlev].copy(),
        "q_u": q_h[:, :nlev].copy(),
        "l_u": jnp.zeros((ncol, nlev), T.dtype),
        "ilab": jnp.zeros((ncol, nlev), jnp.int32),
        "w2h": jnp.zeros((ncol, nlev), T.dtype),
        "Tu": T_h[:, :nlev].copy(),
        "qu": q_h[:, :nlev].copy(),
        "lu": jnp.zeros((ncol, nlev), T.dtype),
        "klab": jnp.zeros((ncol, nlev), jnp.int32),
        "cape_out": jnp.zeros(ncol, T.dtype),
        "w2_out": jnp.zeros((ncol, nlev), T.dtype),
        "w2_sfc_out": jnp.zeros((ncol, nlev), T.dtype),
        "ldsc": jnp.zeros(ncol, bool),
        "kbotsc": jnp.full(ncol, -1, jnp.int32),
        "resolved": jnp.zeros(ncol, bool),
    }

    def step(carry, t):
        # candidate loop: surface level, then elevated k_dep = nlev-2 .. 1
        # (JKK = KLEV..2 of the source's KLEV..NJKT1 loop, cubasen.F90:314);
        # k_dep = 0 is never a departure, so every k_dep - 1 index below is
        # valid.  DYNAMIC (traced) k_dep and is_surface.
        k_dep = jnp.asarray(nlev - 1 - t, jnp.int32)
        is_surface = t == 0
        # eligibility BEFORE the ascent: only ZKHVFL < 0 columns are active
        # for the surface departure (cubasen.F90:367); elevated eligibility:
        # NJKT1 pressure floor (:314/:655) AND not yet deep-resolved
        # (LLGO_ON = .NOT.LLDEEP, :655/:735); passed as the initial `active`
        # carry so inactive columns accumulate no CAPE
        cand = p_full[:, k_dep] > cfg.departure_top_pa        # NJKT1 (:314)
        cand = jnp.where(is_surface, ones_ncol, cand)
        elig = jnp.where(is_surface, col_go, cand & carry["resolved"].__xor__(True) & cand)
        elig = jnp.where(is_surface, col_go, cand & ~carry["resolved"])
        init = _init_departure_parcel(k_dep, is_surface, T, q, p_full, p_half,
                                      geo_full, geo_half, env_half, shf_w_m2,
                                      lhf_w_m2, ustar, land_frac, dq_dt_adv,
                                      cfg, elig_active=elig,
                                      scratch={"T_u": carry["T_u"],
                                               "q_u": carry["q_u"],
                                               "l_u": carry["l_u"],
                                               "w2h": carry["w2h"],
                                               "ilab": carry["ilab"]})
        ilab, ztu, zqu, zlu, w2h, icbot, ictop, lldcum, zcape = \
            _test_ascent_from_departure(k_dep, is_surface, T, q, p_full, p_half,
                                        geo_full, geo_half, env_half, init, cfg)
        # the ascent's returned arrays are the scratch for the NEXT departure
        # (the source never resets them; review finding 3)
        jkb = jnp.clip(icbot, 0, nlev - 1); jkt = jnp.clip(ictop, 0, nlev - 1)
        depth = p_half[idx, jkb] - p_half[idx, jkt]
        wb = jnp.sqrt(jnp.maximum(w2h[idx, jkb], 0.0))       # :670 / :716
        # PCAPE = MAXVAL(ZCAPE(JL,:)) over ALL departures (cubasen.F90:741)
        cape_out = jnp.maximum(carry["cape_out"], zcape)

        # ---- surface step: shallow acceptance, Sc diagnostics, PWU2H
        lldeep_s = depth > cfg.depth_split_pa                # '>' (:666)
        sel_s = lldcum & ~lldeep_s & col_go                  # :666-667, gated by ZKHVFL<0 (:367)
        # LDSC / KBOTSC from the surface test for ALL columns
        # (cubasen.F90:598-607, 639-645): LLDSC/LL_LDBASE are set when the
        # surface ascent found a cloud base, i.e. ICBOT was updated from
        # its internal default JKK = nlev-1 (base levels are <= nlev-2)
        lldsc_flag = icbot < (nlev - 1)
        ldsc_s = jnp.where(lldsc_flag, jnp.ones(ncol, bool), carry["ldsc"])
        kbotsc_s = jnp.where(lldsc_flag, icbot, jnp.full(ncol, -1, jnp.int32))
        # PWU2H: the SURFACE test's velocity profile, always retained
        # (cubasen.F90:358, 559-560)
        w2_sfc_out = jnp.where(is_surface, w2h, carry["w2_sfc_out"])
        # copy surface ascent values for JK >= JKT for ALL columns
        # (cubasen.F90:671-683); ictop carries the internal default
        # nlev-2 (= KLEV-1, :325) when no top was found, so the BOTTOM
        # TWO levels copy (C:673-678)
        m_s = lev >= ictop[:, None]

        # ---- elevated step: first-deep reset (LLRESET :693-713, LLFIRST at :733)
        lldeep_e = depth >= cfg.depth_split_pa               # '>=' (:691)
        # only the FIRST deep elevated departure resets and REPLACES any
        # shallow surface result; on LLDEEP & LLFIRST the source copies
        # LDCUM = LLDCUM (:722-726), which `sel` (containing lldcum)
        # reproduces without forcing it
        sel_e = lldcum & lldeep_e & cand & ~carry["resolved"]  # :695-717
        resolved = carry["resolved"] | (sel_e & ~is_surface)   # :733
        # LLRESET fill: ascent inside [JKT, KDPL], half-level environment
        # (klab 1) outside, klab 0 above JKT (cubasen.F90:697-713) -- the
        # environment fill uses the HALF-level values T_h/q_h (:711-713)
        # and writes the OUTPUT arrays only, NOT the persistent scratch
        inside = (lev >= jkt[:, None]) & (lev <= k_dep)
        m_e = sel_e[:, None] & inside
        env_m = (~is_surface) & sel_e[:, None] & ~inside
        zero_above = (~is_surface) & sel_e[:, None] & (lev < jkt[:, None])
        # deep acceptance clears the Sc outputs (cubasen.F90:727-728)
        ldsc_e = jnp.where(sel_e, jnp.zeros(ncol, bool), carry["ldsc"])
        kbotsc_e = jnp.where(sel_e, jnp.full(ncol, -1, jnp.int32), carry["kbotsc"])

        # ---- branch selection on the traced is_surface
        sel = jnp.where(is_surface, sel_s, sel_e)
        m = jnp.where(is_surface, m_s, m_e)
        ldsc = jnp.where(is_surface, ldsc_s, ldsc_e)
        kbotsc = jnp.where(is_surface, kbotsc_s, kbotsc_e)

        # ktype from the ACCEPTED cloud depth, not the departure identity:
        # a selected surface parcel with depth exactly == depth_split_pa is
        # deep by cumastrn's '>=' (cumastrn.F90:517-521)
        kt = jnp.where(depth >= cfg.depth_split_pa, 1, 2)
        klab = carry["klab"]
        Tu, qu, lu = carry["Tu"], carry["qu"], carry["lu"]
        klab = jnp.where(env_m, 1, klab)
        Tu = jnp.where(env_m, T_h[:, :nlev], Tu)
        qu = jnp.where(env_m, q_h[:, :nlev], qu)
        lu = jnp.where(env_m, 0.0, lu)
        klab = jnp.where(zero_above, 0, klab)
        ldcum = jnp.where(sel, 1.0, carry["ldcum"])          # = LLDCUM via sel
        ktype = jnp.where(sel, kt, carry["ktype"])
        kdpl = jnp.where(sel, k_dep, carry["kdpl"])
        kcbot = jnp.where(sel, icbot, carry["kcbot"])
        kctop = jnp.where(sel, ictop, carry["kctop"])
        wbase = jnp.where(sel, wb, carry["wbase"])
        w2_out = jnp.where(sel[:, None], w2h, carry["w2_out"])
        klab = jnp.where(m, ilab, klab)
        Tu = jnp.where(m, ztu, Tu); qu = jnp.where(m, zqu, qu)
        lu = jnp.where(m, zlu, lu)

        return {"ldcum": ldcum, "ktype": ktype, "kdpl": kdpl,
                "kcbot": kcbot, "kctop": kctop, "wbase": wbase,
                "T_u": ztu, "q_u": zqu, "l_u": zlu, "w2h": w2h, "ilab": ilab,
                "Tu": Tu, "qu": qu, "lu": lu, "klab": klab,
                "cape_out": cape_out, "w2_out": w2_out,
                "w2_sfc_out": w2_sfc_out, "ldsc": ldsc, "kbotsc": kbotsc,
                "resolved": resolved}, ()

    final, _ = lax.scan(step, carry0, jnp.arange(nlev - 1, dtype=jnp.int32),
                        unroll=1)

    ldcum = final["ldcum"]; ktype = final["ktype"]
    kdpl = final["kdpl"]; kcbot = final["kcbot"]; kctop = final["kctop"]
    wbase = final["wbase"]; Tu = final["Tu"]; qu = final["qu"]
    lu = final["lu"]; klab = final["klab"]; cape_out = final["cape_out"]
    w2_out = final["w2_out"]; w2_sfc_out = final["w2_sfc_out"]
    ldsc = final["ldsc"]; kbotsc = final["kbotsc"]

    # PQU / PLU leave on the SAME moist-mass basis as PQEN (cubasen.F90:677-678)
    return TestAscent(ldcum, ktype, kdpl, kcbot, kctop, wbase, Tu, qu, lu,
                      klab, cape_out, w2_out, w2_sfc_out, ldsc, kbotsc)


def _hydrostatic_geopotential(T, q, p_half):
    """Hydrostatic geopotential (surface-relative, Phi_half[:, -1] = 0) from
    full-level temperature T and SPECIFIC humidity q on the half-level grid
    p_half (ncol, nlev+1), surface-last: Phi_half[:, k+1] = Phi_half[:, k] +
    R_d*T_v[:, k]*ln(p_half[:, k+1]/p_half[:, k]) integrated downward from
    the surface (Phi_half[:, nlev] = 0); Phi_full[:, k] is the arithmetic
    mean of the adjacent half values.  Used only by
    ifs_departure_search_refined to build the geopotential of the refined
    grid; the input T_v uses the shared virtual_temperature helper."""
    dphi = constants.R_d * virtual_temperature(T, q) \
        * jnp.log(p_half[:, 1:] / p_half[:, :-1])         # positive, dp>0
    phi_int = jnp.flip(jnp.cumsum(jnp.flip(dphi, axis=1), axis=1), axis=1)
    phi_half = jnp.concatenate(
        [phi_int, jnp.zeros_like(phi_int[:, :1])], axis=1)  # [:, nlev] = 0
    phi_full = 0.5 * (phi_half[:, :-1] + phi_half[:, 1:])
    return phi_half, phi_full


def _layer_centre(p_half):
    """Layer-centre pressures of a half-level grid (ncol, nlev+1) ->
    (ncol, nlev): 0.5*(p_half[:, :-1] + p_half[:, 1:]).  This is THE
    location convention of ifs_departure_search_refined: every level value
    is its LAYER MEAN located at this half-level midpoint, DERIVED FROM
    p_half -- never the caller's p_full, which on a log-midpoint or
    IFS-hybrid grid differs from it by a fraction of a layer.  Both the
    r == 1 arm and every r > 1 path of the refined wrapper build the
    pressure they hand to the source-literal ifs_departure_search with
    this one expression; native ifs_departure_search keeps the source's
    PAP convention (the model's full-level pressure, as handed in) and is
    unchanged."""
    return 0.5 * (p_half[:, :-1] + p_half[:, 1:])


def _refine_half_levels(p_half, r):
    """Refined half levels: r equal-pressure sub-layers per parent layer.

    Fractions j/r for j = 0..r-1 keep EVERY parent interface exactly once
    (p_r[:, 0] == p_half[:, 0] and p_r[:, k*r] == p_half[:, k] for all k,
    surface-last layout, half levels (ncol, nlev+1)); the surface half
    level p_half[:, nlev] is appended unchanged, so the result is
    (ncol, nlev*r + 1), every sub-layer has strictly positive thickness,
    and r = 1 reproduces p_half exactly.
    """
    nlev = p_half.shape[1] - 1
    h_par = jnp.repeat(jnp.arange(nlev), r)              # static parent index
    frac = jnp.tile(jnp.arange(r), nlev).astype(p_half.dtype) / r
    lo_h = p_half[:, h_par]
    dp_h = p_half[:, h_par + 1] - lo_h
    return jnp.concatenate(
        [lo_h + frac * dp_h, p_half[:, nlev:nlev + 1]], axis=1)


def _fv_minmod_refine(x, c_par, h_par_dp, c_child, k_par):
    """Delta-p-conservative minmod-limited piecewise-linear refinement.

    Parent values x (ncol, nlev) are LAYER MEANS on the parent layers with
    pressure centres c_par and pressure thicknesses h_par_dp; c_child
    (ncol, nlev*r) are the sub-layer centres and k_par (nlev*r,) the static
    parent index of each child.  Returns the sub-layer means

        x_kj = x_k + m_k * (c_kj - c_k),
        m_k = minmod(d_minus / (c_k - c_{k-1}),
                     d_plus  / (c_{k+1} - c_k),
                     2*d_minus / h_k, 2*d_plus / h_k),
        d_minus = x_k - x_{k-1},  d_plus = x_{k+1} - x_k,

    where minmod returns the smallest-magnitude argument when all four
    arguments share a sign and 0 otherwise; the 2*d/h arguments are the
    endpoint bound that keeps the reconstruction inside the neighbour
    range.  m_k is 0 exactly on the top and bottom parent layers.  The
    children of a parent are equal-pressure sub-layers whose centres
    average to the parent centre, so sum_j x_kj * dp/r == x_k * dp to
    round-off; children are bounded by the neighbouring parent values
    (non-negative when the parents are), the operator is exact for
    profiles affine in pressure and the identity at r = 1.  Static index
    arithmetic only -- JAX-traceable, no branching on traced values, no
    data-dependent shapes.  q is NOT clipped against saturation: a scalar
    limiter cannot guarantee sub-saturation and clipping would break the
    per-layer conservation.
    """
    nlev = x.shape[1]
    ones = jnp.ones_like(c_par[:, :1])
    diff = x[:, 1:] - x[:, :-1]
    d_minus = jnp.concatenate([x[:, :1], diff], axis=1)   # x_k - x_{k-1}
    d_plus = jnp.concatenate([diff, x[:, -1:]], axis=1)   # x_{k+1} - x_k
    dc = c_par[:, 1:] - c_par[:, :-1]
    dc_minus = jnp.concatenate([ones, dc], axis=1)        # c_k - c_{k-1}
    dc_plus = jnp.concatenate([dc, ones], axis=1)         # c_{k+1} - c_k
    s1 = d_minus / dc_minus
    s2 = d_plus / dc_plus
    s3 = 2.0 * d_minus / h_par_dp
    s4 = 2.0 * d_plus / h_par_dp
    sgn = jnp.sign(s1)
    same = ((sgn == jnp.sign(s2)) & (sgn == jnp.sign(s3))
            & (sgn == jnp.sign(s4)) & (sgn != 0.0))
    mag = jnp.minimum(jnp.minimum(jnp.abs(s1), jnp.abs(s2)),
                      jnp.minimum(jnp.abs(s3), jnp.abs(s4)))
    m = jnp.where(same, sgn * mag, 0.0)
    interior = (jnp.arange(nlev) > 0) & (jnp.arange(nlev) < nlev - 1)
    m = jnp.where(interior, m, 0.0)                       # flat top/bottom
    return x[:, k_par] + m[:, k_par] * (c_child - c_par[:, k_par])


def ifs_departure_search_refined(T, q_v, p_full, p_half, geo_full, geo_half,
                                 shf_w_m2, lhf_w_m2, ustar, land_frac,
                                 dq_dt_adv, cfg):
    """REFINED-COLUMN trigger (codex design r14, item 3): run the UNCHANGED
    source-literal ``ifs_departure_search`` on a vertically refined copy of
    each column and map the results back to the parent levels.

    CONVENTION (one meaning, both arms): this function treats every level
    value it is handed (T, q_v, dq_dt_adv) as its LAYER MEAN, located at
    the half-level midpoint _layer_centre(p_half) derived from p_half
    rather than trusting the caller's p_full, which is NOT used as a
    location anywhere here.  The GEOPOTENTIAL is derived here too, by
    hydrostatic integration of p_half, so both the pressure and the
    geopotential a level value is paired with come from one geometry;
    the geo_full / geo_half arguments are consequently unused, as is
    p_full, and all three are kept for signature parity with the native
    search (callers pass the same argument tuple to both).  Both the
    r == 1 arm and every r > 1 path therefore hand the source-literal
    search a layer-centre pressure and a module-built geopotential; the
    native
    ``ifs_departure_search`` keeps the source's PAP convention (the
    model's full-level pressure, as handed in) and is unchanged.

    Two of that search's pressure tests are PORT-LOCAL rather than
    source-literal, so they are covered by this convention too and get
    layer centres through this wrapper (codex, correcting an earlier
    audit): the NJKT1/NJKT2 departure and test-top bounds, which the
    source precomputes ONCE as level INDICES from a standard pressure
    profile (sucumf.F90:284, ``IF(STPRE(JLEV) > 350.E2)NJKT1=JLEV``)
    rather than testing PAP per column, and the optional ``cell_centre``
    mixed-layer gate, where the source uses a half-level difference
    (cubasen.F90:400, ``PAPH(KLEV+1)-PAPH(JKK-1) < 60.E2``).  The
    saturation calls ARE source-literal: satur.F90:120 divides by
    PAPRSF, the full-level pressure.

    PURPOSE: the coarse-grid departure sampling defect.  On ~33 hPa
    layers (L60-like grids) the first elevated departure launches with
    cloud-layer air: the cuinin half-level rules hand the parcel the
    humidity of the full level ABOVE the departure and the MAX(s_above,
    s_below) of the adjacent (33 hPa-thick) neighbours, so the test
    parcel starts already mixed through the whole subcloud layer and the
    literal cubasen trigger misclassifies the boundary layer.  Refining
    the column by r sub-layers lets the same UNCHANGED code resolve the
    lowest levels: at parent interfaces the humidity now comes from the
    adjacent refined full level above, and MAX(s_above, s_below) uses
    refined neighbours.

    GATE TEST (to be written): native L60 vs L30 refined x2 on a frozen
    sounding -- identical ktype, departure pressure within one parent
    layer, cloud top within 50 hPa.  If the gate fails, the follow-up is
    a Delta-p-CONSERVATIVE limited reconstruction, because plain
    interpolation between full-level points is NOT conservative (the
    piecewise-linear interpolant reproduces the parent POINT values but
    not their layer means, so column mass/energy integrals drift by
    O(1/r) of the vertical curvature).

    COST: O((rN)^2) per column (the nested departure x ascent scans run
    on the refined grid), plus O(rN) reconstruction.

    RECONSTRUCTION (r = cfg.column_refine, static; r = 1 is the identity
    and calls ifs_departure_search directly): refined half levels split
    each parent layer evenly in PRESSURE; refined full levels are the
    midpoints of the refined half intervals.  Dry static energy
    s = c_p T + Phi, SPECIFIC humidity q and dq_dt_adv are reconstructed
    from the parent LAYER MEANS with the Delta-p-conservative
    piecewise-linear minmod-limited finite-volume operator
    (_fv_minmod_refine) on the parent-layer pressure geometry taken from
    p_half (layer centres and thicknesses, NOT p_full): per-parent mass is
    conserved to round-off, every child value is bounded by the
    neighbouring parent values, and profiles affine in pressure are
    reproduced exactly in the interior.  Temperature is DERIVED from s,
    T = (s - Phi_refined)/c_p, and nothing else is recomputed.  The
    refined geopotential is hydrostatic from the refined T and q
    (_hydrostatic_geopotential, cumulative R_d T_v ln(p) from the
    surface); the circularity T(s, Phi(T)) is broken with a single
    provisional pass that reconstructs the parent T with the same
    operator purely to seed Phi (the final s-derived T then closes the
    consistency to within the reconstruction error).

    RESTRICTION (map back to the parent levels): refined half indices
    snap via round(h_r / r) (refined half levels at multiples of r
    coincide with parent interfaces); the full-level indices k_dpl /
    k_cbot / k_ctop / k_botsc (surface-last FULL-level indices in this
    module) snap via k_r // r (a refined full index lies in parent layer
    k_r // r), preserving the -1 sentinel.  The profile and integer
    outputs (T_u/q_u/l_u/klab/w2/w2_surface) are SAMPLED at the refined
    level whose half level coincides with the parent half level, i.e.
    refined index k*r for parent k.  The refined ktype, w_base,
    cape_test, ldcum and ldsc are preserved; NO reclassification is done
    after snapping.

    The per-column surface fluxes, ustar and land_frac are unchanged
    (they are level-independent).  All refinement arithmetic is
    vectorised across columns (no Python loop over columns); the static
    refinement index arrays (parent-layer indices, sub-layer fractions,
    snap tables) are built once from cfg.column_refine, and the
    interpolation WEIGHTS are traced functions of the pressures
    computed per column.  Output dtype = input dtype throughout."""
    r = int(cfg.column_refine)
    if r < 1:
        raise ValueError(
            f"IFSTestAscentConfig.column_refine must be >= 1, got {r}")

    # The GEOPOTENTIAL is a level value too, so it gets the same treatment as
    # the pressure: derived here from p_half by the module's own hydrostatic
    # integration rather than taken from the caller, in BOTH arms.  Leaving
    # the caller's geo_full in place would reintroduce the very seam this
    # convention removes -- the r > 1 path already inverts the reconstructed
    # dry static energy against a module-built refined geopotential, so a
    # caller whose geopotential sits on a different geometry would have its
    # temperature offset by the difference (GLM review).
    geo_half, geo_full = _hydrostatic_geopotential(T, q_v, p_half)

    if r == 1:
        # SAME convention as every r > 1 path: the level values' location
        # is the layer centre derived from p_half, not the caller's
        # p_full.  The old early return forwarded the caller's p_full,
        # which made the refinement's anchoring a function of r on any
        # grid where p_full != mid(p_half).
        return ifs_departure_search(T, q_v, _layer_centre(p_half), p_half,
                                    geo_full, geo_half, shf_w_m2, lhf_w_m2,
                                    ustar, land_frac, dq_dt_adv, cfg)

    ncol, nlev = T.shape
    c_pd = constants.c_pd
    # q_v IS specific humidity, exactly PQEN (cubasen.F90:72) -- no conversion
    q = q_v

    # ---- refined grids (static index arithmetic, traced pressures)
    # half levels: r sub-layers per parent layer, even in pressure, at
    # fractions j/r for j = 0..r-1 so that EVERY parent interface appears
    # exactly once (the refined half levels at multiples of r coincide
    # with the parent interfaces, including the model top p_half[:, 0]);
    # the surface half level p_half[:, nlev] is appended unchanged, every
    # sub-layer has positive thickness, and r = 1 reproduces p_half
    p_half_r = _refine_half_levels(p_half, r)
    # full levels: sub-layer centres, the same _layer_centre convention
    p_full_r = _layer_centre(p_half_r)

    # ---- reconstruction (traced, computed once per column): the
    # Delta-p-conservative finite-volume piecewise-linear minmod operator
    # (_fv_minmod_refine), parent values as LAYER MEANS on the parent
    # half-level geometry (layer centres/thicknesses from p_half, NOT
    # p_full); children are the sub-layer means at the refined sub-layer
    # centres.  This replaces the old clipped linear ramp between parent
    # FULL levels, which was half-cell shifted and NOT conservative.
    c_par = _layer_centre(p_half)                        # parent layer centres
    h_par_dp = p_half[:, 1:] - p_half[:, :-1]            # parent thicknesses
    k_par = jnp.repeat(jnp.arange(nlev), r)              # (nlev*r,) static

    def _fv(x):
        return _fv_minmod_refine(x, c_par, h_par_dp, p_full_r, k_par)

    s_r = _fv(c_pd * T + geo_full)                       # dry static energy
    q_r = _fv(q)                                         # specific humidity
    dq_r = _fv(dq_dt_adv)                                # reconstructed like q
    # provisional T (same operator) only to SEED the hydrostatic Phi; the
    # final T below is derived from s (T = (s - Phi_refined)/c_p)
    T_prov = _fv(T)
    phi_half_r, phi_full_r = _hydrostatic_geopotential(T_prov, q_r, p_half_r)
    T_r = (s_r - phi_full_r) / c_pd                      # derive temperature

    # ---- run the UNCHANGED source-literal search on the refined column
    q_v_r = q_r                                 # stays specific humidity (PQEN)
    res = ifs_departure_search(T_r, q_v_r, p_full_r, p_half_r,
                               phi_full_r, phi_half_r, shf_w_m2,
                               lhf_w_m2, ustar, land_frac, dq_r, cfg)

    # ---- restriction: snap indices to valid parent levels, preserving -1
    def snap_full(k):
        k = jnp.asarray(k, jnp.int32)
        return jnp.where(k >= 0, k // r, jnp.asarray(-1, jnp.int32))

    # half-level-profile sampling: refined index k*r has its half level
    # coincident with the parent half level k
    kk = jnp.arange(nlev) * r                            # static

    return TestAscent(
        ldcum=res.ldcum,
        ktype=res.ktype,                                 # preserved, not reclassified
        k_dpl=snap_full(res.k_dpl),
        k_cbot=snap_full(res.k_cbot),
        k_ctop=snap_full(res.k_ctop),
        w_base=res.w_base,                               # preserved
        T_u=res.T_u[:, kk],
        q_u=res.q_u[:, kk],
        l_u=res.l_u[:, kk],
        klab=res.klab[:, kk],
        cape_test=res.cape_test,                         # preserved
        w2=res.w2[:, kk],
        w2_surface=res.w2_surface[:, kk],
        ldsc=res.ldsc,                                   # preserved
        k_botsc=snap_full(res.k_botsc),
    )
