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
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_dT
from legoesm.atmosphere.physics._shared import virtual_temperature


# --- IFS cubasen / sucumf defaults (OpenIFS main 8f6f722; sucumf.F90 lines cited) ---
_WS_EXP = 0.3333            # ZWS**.3333, cubasen.F90:352
_RLMIN = 1.0e-12            # RLMIN of yoecumf applied at the Sc cloud base, cubasen.F90:565
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
    qadv_land_min_pa: float = 70000.0     # NJKT6 ~ 700 hPa [Pa]


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
    """cuadjtq.F90:324-345, KCALL=3 branch: two successive UNCLIPPED corrections
    ZCOND1 = (q - qsat)/(1 + (L_v/c_pd) dqsat/dT); T += (L_v/c_pd) ZCOND1; q -= ZCOND1
    (second pass uses the updated T)."""
    for _ in range(2):
        qs = saturation_mixing_ratio(T, p)
        dqs_dT = saturation_mixing_ratio_dT(T, p)
        zcond1 = (q - qs) / (1.0 + (constants.L_v / constants.c_pd) * dqs_dT)
        T = T + (constants.L_v / constants.c_pd) * zcond1
        q = q - zcond1
    return T, q


def _half_level_env(T, q_v, p_full, p_half, geo_full, geo_half, cfg):
    """Half-level environment (ZTENH, ZQENH, ZSENH) ported exactly from cuinin.F90:150-192,
    with ZSENH = RCPD*PTENH + PGEOH (cubasen.F90:306).
    Index map: IFS half JK (1..KLEV) <-> our half h = JK-1; our h = nlev is the surface
    (IFS KLEV+1, PGEOH/PAPH only). Surface half index nlev is filled with the lowest
    full level's T and q and s = c_pd*T + geo_half[:, nlev]; cubasen never reads it."""
    c_pd = constants.c_pd
    ncol, nlev = T.shape
    # dry static energy on full levels (cuinin.F90:156-157)
    s_full = c_pd * T + geo_full  # (:156-157)
    T_h = jnp.zeros((ncol, nlev + 1), T.dtype)
    q_h = jnp.zeros((ncol, nlev + 1), T.dtype)
    # interior half levels h = 1..nlev-1 (IFS JK = 2..KLEV)
    # T from MAX of adjacent full-level s (cuinin.F90:156-157), q and qs from level above (:158-159)
    s_max = jnp.maximum(s_full[:, :-1], s_full[:, 1:])
    T_int = (s_max - geo_half[:, 1:nlev]) / c_pd
    q_int = q_v[:, :-1]
    qs_above = saturation_mixing_ratio(T[:, :-1], p_full[:, :-1])  # PQSEN of the level above (:159)

    # cuadjtq KCALL=3 correction for h = 1..nlev-2 with p_half > ~60 hPa
    # (cuinin.F90:163 CYCLE skips JK >= KLEV-1 and JK < NJKT2; :164-175)
    h_lo, h_hi = 1, nlev - 3  # IFS JK = 2..KLEV-2 <-> h = 1..nlev-3 (static slice)
    p_int = p_half[:, h_lo:h_hi + 1]
    mask = p_int > cfg.test_top_pa  # traced pressure mask (NJKT2 ~ 60 hPa, sucumf.F90:285)
    T_corr, qs_corr = _cuadjtq_pair(T_int[:, h_lo:h_hi + 1], qs_above[:, h_lo:h_hi + 1],
                                    p_int)  # (:165-175)
    T_int = T_int.at[:, h_lo:h_hi + 1].set(jnp.where(mask, T_corr, T_int[:, h_lo:h_hi + 1]))
    # PQENH = MIN(PQEN(JK-1), PQSEN(JK-1)) + (PQSENH - PQSEN(JK-1)), clipped >= 0 (:178-180)
    q_corr = jnp.maximum(
        jnp.minimum(q_v[:, :-1], qs_above)[:, h_lo:h_hi + 1]
        + (qs_corr - qs_above[:, h_lo:h_hi + 1]),
        0.0)
    q_int = q_int.at[:, h_lo:h_hi + 1].set(jnp.where(mask, q_corr, q_int[:, h_lo:h_hi + 1]))

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
    """CUADJTQ (KCALL=1, condensation only), cuadjtq.F90:165-186: first
    correction with ZCOND = MAX(0,(q-qsat)/(1+(L/cp) dqsat/dT)), second
    correction ZCOND1 unclipped but zeroed where ZCOND == 0.  Mixing-ratio
    form: the IFS ZCOR specific-humidity factors are absorbed by our
    mixing-ratio saturation function (departure, cuadjtq.F90:170-180)."""
    lvcp = constants.L_v / constants.c_pd
    qs = saturation_mixing_ratio(T, p)
    dqs = saturation_mixing_ratio_dT(T, p)
    zcond = jnp.maximum(0.0, (q - qs) / (1.0 + lvcp * dqs))   # cuadjtq.F90:171
    T1 = T + lvcp * zcond
    q1 = q - zcond
    qs1 = saturation_mixing_ratio(T1, p)
    dqs1 = saturation_mixing_ratio_dT(T1, p)
    zcond1 = jnp.where(zcond > 0.0,                            # cuadjtq.F90:182
                       (q1 - qs1) / (1.0 + lvcp * dqs1), 0.0)
    T2 = T1 + lvcp * zcond1
    q2 = q1 - zcond1
    return T2, q2


def _test_ascent_from_departure(k_dep, is_surface, T, q_v, p_full, p_half,
                                geo_full, geo_half, env_half, parcel_init, cfg):
    """One test ascent from departure level k_dep upward (IFS DO JK=JKK-1,JKT2,-1,
    cubasen.F90:437-600). lax.scan over the k_dep levels above the departure;
    levels with p_full <= cfg.test_top_pa are masked out (NJKT2 ~ 60 hPa)."""
    nlev = T.shape[1]
    T_h, q_h, s_h = env_half
    # candidate ascent levels: our indices k_dep-1 .. 0 (surface-last: upward)
    js = jnp.arange(k_dep - 1, -1, -1)

    def body(carry, j):
        q, s, w2, buoh, active, icbot, ictop, lldcum, cape = (
            carry["q"], carry["s"], carry["w2"], carry["buoh"], carry["active"],
            carry["icbot"], carry["ictop"], carry["lldcum"], carry["cape"])
        T_u, q_u, l_u, w2h, ilab = (carry["T_u"], carry["q_u"], carry["l_u"],
                                    carry["w2h"], carry["ilab"])
        work = active & (p_full[:, j] > cfg.test_top_pa)  # NJKT2 bound

        dz = (geo_half[:, j] - geo_half[:, j + 1]) * _G_INV
        qf = 0.5 * (q_h[:, j + 1] + q_h[:, j])
        sf = 0.5 * (s_h[:, j + 1] + s_h[:, j])
        if is_surface:
            # 1/z mixing for the shallow/surface departure (cubasen.F90:437, 446-451);
            # KINDEX==KLEV-1 branches are dead on Earth (NJKT1 < KLEV-1)
            zeps = cfg.entr_test_c1 / ((geo_full[:, j] - geo_half[:, nlev]) * _G_INV) \
                + cfg.entr_test_c2
            zmix = jnp.minimum(1.0, 0.5 * dz * zeps)           # cubasen.F90:450-452
            tmp = 1.0 / (1.0 + zmix)
            q_new = (q * (1.0 - zmix) + 2.0 * zmix * qf) * tmp  # cubasen.F90:456-458
            s_new = (s * (1.0 - zmix) + 2.0 * zmix * sf) * tmp  # cubasen.F90:459-461
        else:
            # deep-test mixing 0.4*ENTRORG*dz*min(1,(qsat/qsat_sfc)**3) (cubasen.F90:472-478)
            qsat_j = saturation_mixing_ratio(T[:, j], p_full[:, j])
            qsat_sfc = saturation_mixing_ratio(T[:, nlev - 1], p_full[:, nlev - 1])
            zmix = jnp.minimum(1.0, cfg.deep_test_mix_factor * cfg.entr_deep_base * dz
                               * jnp.minimum(1.0, (qsat_j / qsat_sfc) ** _QSAT_RATIO_EXP))
            q_new = q * (1.0 - zmix) + qf * zmix               # cubasen.F90:483
            s_new = s * (1.0 - zmix) + sf * zmix               # cubasen.F90:484

        # condensation (CUADJTQ), condensate added, half retained (cubasen.F90:466-493)
        q_old = q_new
        T_new0 = (s_new - geo_half[:, j]) * _RCPD_INV
        T_adj, q_adj = _cuadjtq_condense(T_new0, q_new, p_half[:, j])
        zdq = jnp.maximum(q_old - q_adj, 0.0)
        l_new = cfg.test_condensate_retained * (l_u[:, j + 1] + zdq)
        # freezing correction ZLGLAC = 0: liquid-only saturation (FOEALFCU unavailable)
        T_new = T_adj
        s_new = constants.c_pd * T_new + geo_half[:, j]

        # buoyancy on half levels (cubasen.F90:499-508)
        tvu = (1.0 + _RETV * q_adj - l_new) * T_new
        tven = (1.0 + _RETV * q_h[:, j]) * T_h[:, j]
        buoh_new = (tvu - tven) * constants.g / tven
        buof = 0.5 * (buoh_new + buoh)
        # kinetic-energy recurrence, ZAW = ZBW = 1 (cubasen.F90:511-513)
        w2_new = (w2 * (1.0 - 2.0 * zmix) + 2.0 * buof * dz) / (1.0 + 2.0 * zmix)
        cape_new = cape + jnp.maximum(0.0, buof * dz)          # cubasen.F90:521

        # first layer with liquid water: exact cloud base (cubasen.F90:524-551).
        # Saturation-deficit interpolation on the liquid curve (departure:
        # FOEEWM/FOEALFA-R5LES/R5IES blend unavailable, cubasen.F90:527, 532-535).
        cond_first = (l_new > 0.0) & (ilab[:, j + 1] == 1)
        zqsu = saturation_mixing_ratio(T_u[:, j + 1], p_half[:, j + 1])
        zdqsdT = saturation_mixing_ratio_dT(T_u[:, j + 1], p_half[:, j + 1])
        zdq_cb = jnp.minimum(0.0, q_u[:, j + 1] - zqsu)
        zdtdp = constants.R_d * T_u[:, j + 1] / (constants.c_pd * p_half[:, j + 1])
        zcb = p_half[:, j + 1] + zdq_cb / (zdqsdT * zdtdp)
        pdtop = zcb - p_half[:, j]
        pdbot = p_half[:, j + 1] - zcb
        case_top = cond_first & (pdtop > pdbot) & (w2 > 0.0)   # cubasen.F90:548
        case_bot = cond_first & (pdtop <= pdbot) & (w2_new > 0.0)  # cubasen.F90:555
        jkb = jnp.minimum(nlev - 2, j + 1)                     # MIN(KLEV-1,JK+1)
        icbot_new = jnp.where(case_top, jkb, icbot)
        icbot_new = jnp.where(case_bot, j, icbot_new)
        l_below = jnp.where(case_top, _RLMIN, l_u[:, j + 1])   # cubasen.F90:553
        labj = jnp.where(case_top | case_bot, 2, 0)

        # stop at w2 < 0; ICTOP set when the level below carried condensate
        # (cubasen.F90:575-586, strict '<')
        stop = w2_new < 0.0
        ictop_new = jnp.where(stop & (l_u[:, j + 1] > 0.0), j, ictop)
        lldcum_new = jnp.where(stop, l_u[:, j + 1] > 0.0, lldcum)
        active_new = active & ~stop
        labj = jnp.where(work & ~stop, jnp.where(l_new > 0.0, 2, 1), labj)  # 588-592

        # store this level (masked by 'work')
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

    final, _ = lax.scan(body, parcel_init, js)
    return (final["ilab"], final["T_u"], final["q_u"], final["l_u"], final["w2h"],
            final["icbot"], final["ictop"], final["lldcum"], final["cape"])


class TestAscent(NamedTuple):
    ldcum: jnp.ndarray       # (ncol,) bool-valued float
    ktype: jnp.ndarray       # (ncol,) int32: 1 deep, 2 shallow, 0 none
    k_dpl: jnp.ndarray       # (ncol,) int32, -1 when none
    k_cbot: jnp.ndarray
    k_ctop: jnp.ndarray
    w_base: jnp.ndarray      # PWUBASE [m/s]
    T_u: jnp.ndarray         # (ncol, nlev) K
    q_u: jnp.ndarray         # (ncol, nlev) kg/kg
    l_u: jnp.ndarray         # (ncol, nlev) kg/kg
    klab: jnp.ndarray        # (ncol, nlev) int32
    cape_test: jnp.ndarray   # (ncol,) J/kg
    w2: jnp.ndarray          # (ncol, nlev) m2/s2


def _init_departure_parcel(k_dep, is_surface, T, q_v, p_full, p_half,
                           geo_full, geo_half, env_half, shf_w_m2, lhf_w_m2,
                           ustar, land_frac, dq_dt_adv, cfg):
    """Initialise the departure-level parcel and scan carry (cubasen.F90:314-435)."""
    nlev = T.shape[1]
    T_h, q_h, s_h = env_half
    ncol = T.shape[0]
    dt = T.dtype

    def _surface_fluxes():
        # ZRHO from the surface half level (cubasen.F90:333)
        zrho = p_half[:, nlev] / (constants.R_d * T[:, nlev - 1]
                                  * (1.0 + _RETV * q_v[:, nlev - 1]))
        pahfs = shf_w_m2                                    # PAHFS (neg = up)
        pqhfl = -lhf_w_m2 / constants.L_v                   # PQHFL = -lhf/L_v
        zkhvfl = (pahfs * _RCPD_INV + _RETV * T[:, nlev - 1] * pqhfl) / zrho  # :335
        zust = jnp.maximum(ustar, cfg.ustar_min)            # :336
        # ZWS = ZUST**3 - 1.5*RKAP*ZKHVFL*(PGEOH(KLEV)-PGEOH(KLEV+1))/PTEN(KLEV)
        # (cubasen.F90:337-338) -- no 1/c_pd factor
        zws = zust ** 3 - cfg.surface_flux_factor * constants.kappa_von_karman \
            * zkhvfl * (geo_half[:, nlev - 1] - geo_half[:, nlev]) \
            / T[:, nlev - 1]                                # :337-338
        return zrho, pahfs, pqhfl, zkhvfl, zws

    zrho, pahfs, pqhfl, zkhvfl, zws = _surface_fluxes()
    # surface excesses (0 unless ZKHVFL < 0; kept for the KLEV-1 inheritance, :396-401)
    zws_f = cfg.surface_ws_factor * jnp.maximum(zws, 0.0) ** _WS_EXP  # :352
    ztex_s = jnp.clip(-cfg.surface_flux_factor * pahfs
                      / (zrho * zws_f * constants.c_pd),
                      cfg.parcel_dT_excess_min_K, cfg.parcel_dT_excess_max_K)  # :360,362
    zqex_s = jnp.clip(-cfg.surface_flux_factor * pqhfl / (zrho * zws_f),
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
        # land-only advective moistening (cubasen.F90:415-417), land_frac-weighted
        qs_dep = saturation_mixing_ratio(T[:, k_dep], p_full[:, k_dep])
        adv = jnp.minimum(cfg.land_qadv_cap,
                          jnp.maximum(0.0, dq_dt_adv[:, k_dep] * cfg.land_qadv_timescale_s))
        gate = ((k_dep < nlev - 2) & (p_full[:, k_dep] < cfg.qadv_land_min_pa)
                & (q_v[:, k_dep] / qs_dep < cfg.land_qadv_rh_max))       # :415
        zqexc = zqexc + land_frac * jnp.where(gate, adv, 0.0)            # :416
        q_u = q_h[:, k_dep] + zqexc
        s_u = s_h[:, k_dep] + constants.c_pd * ztexc
        # ZTU = (ZSENH - PGEOH)*ZRCPD + ZTEXC built from the ENVIRONMENT dry
        # static energy (cubasen.F90:397): single excess
        T_u = (s_h[:, k_dep] - geo_half[:, k_dep]) * _RCPD_INV + ztexc   # :397
        # mixed layer for parcels within 60 hPa of the surface (cubasen.F90:419,424-430):
        # accumulate over exactly the three IFS half levels JK = JKK+1, JKK,
        # JKK-1 (our half indices k_dep+1, k_dep, k_dep-1), each weighted by
        # ZWORK2 = PAPH(JK) - PAPH(JK-1), while the accumulated span ZWORK1
        # < 50 hPa; divide by ZWORK1 (the span actually accumulated)
        mixed = p_half[:, nlev] - p_half[:, k_dep - 1] < cfg.mixed_layer_depth_pa  # :419
        qw = jnp.zeros(ncol, dt); sw = jnp.zeros(ncol, dt)
        span = jnp.zeros(ncol, dt)
        for l in (k_dep + 1, k_dep, k_dep - 1):             # :424-430
            if (l - 1) >= 0 and l <= nlev:
                dp = p_half[:, l] - p_half[:, l - 1]        # ZWORK2 (:426)
                inc = span < cfg.mixed_layer_span_pa         # ZWORK1 < 50 hPa (:425)
                w = jnp.where(inc, dp, 0.0)
                qw = qw + q_h[:, l] * w
                sw = sw + s_h[:, l] * w
                span = span + w
        q_m = qw / span + zqexc
        s_m = sw / span + constants.c_pd * ztexc
        T_m = (s_m - geo_half[:, k_dep]) * _RCPD_INV + ztexc  # literal double ZTEXC, :412-414
        q_u = jnp.where(mixed, q_m, q_u)
        s_u = jnp.where(mixed, s_m, s_u)
        T_u = jnp.where(mixed, T_m, T_u)
        l_u = jnp.zeros(ncol, dt)
        w2 = jnp.full(ncol, cfg.elevated_w2, dt)            # :435
        active = jnp.ones(ncol, bool)
    # buoyancy at the departure half level (cubasen.F90:427-431)
    tven = (1.0 + _RETV * q_h[:, k_dep]) * (s_h[:, k_dep] - geo_half[:, k_dep]) * _RCPD_INV
    tvu = (1.0 + _RETV * q_u) * T_u
    buoh = (tvu - tven) * constants.g / tven

    carry = {
        "q": q_u.astype(dt), "s": s_u.astype(dt), "w2": w2.astype(dt),
        "buoh": buoh.astype(dt), "active": active,
        "icbot": jnp.full(ncol, -1, jnp.int32), "ictop": jnp.full(ncol, -1, jnp.int32),
        "lldcum": jnp.zeros(ncol, bool), "cape": jnp.zeros(ncol, dt),
        "T_u": T.at[:, k_dep].set(jnp.where(active, T_u, T[:, k_dep]).astype(dt)),
        "q_u": q_v.at[:, k_dep].set(jnp.where(active, q_u, q_v[:, k_dep]).astype(dt)),
        "l_u": jnp.zeros((ncol, nlev), dt).at[:, k_dep].set(l_u),
        "w2h": jnp.zeros((ncol, nlev), dt).at[:, k_dep].set(w2.astype(dt)),
        "ilab": jnp.zeros((ncol, nlev), jnp.int32).at[:, k_dep].set(
            jnp.where(active, 1, 0).astype(jnp.int32)),       # ILAB=1, :345/:375
    }
    return carry


def ifs_departure_search(T, q_v, p_full, p_half, geo_full, geo_half,
                         shf_w_m2, lhf_w_m2, ustar, land_frac, dq_dt_adv, cfg):
    """Full departure search (cubasen.F90:314-735)."""
    ncol, nlev = T.shape
    env_half = _half_level_env(T, q_v, p_full, p_half, geo_full, geo_half, cfg)
    idx = jnp.arange(ncol)

    # surface fluxes gate ONLY the surface departure (LLGO_ON=.FALSE. at :367)
    zrho = p_half[:, nlev] / (constants.R_d * T[:, nlev - 1] * (1.0 + _RETV * q_v[:, nlev - 1]))
    zkhvfl = (shf_w_m2 * _RCPD_INV + _RETV * T[:, nlev - 1] * (-lhf_w_m2 / constants.L_v)) / zrho
    col_go = zkhvfl < 0.0                                   # :367

    ldcum = jnp.zeros(ncol, T.dtype)
    ktype = jnp.zeros(ncol, jnp.int32)
    kdpl = jnp.full(ncol, -1, jnp.int32)
    kcbot = jnp.full(ncol, -1, jnp.int32)
    kctop = jnp.full(ncol, -1, jnp.int32)
    wbase = jnp.zeros(ncol, T.dtype)
    Tu = T.copy(); qu = q_v.copy(); lu = jnp.zeros((ncol, nlev), T.dtype)
    klab = jnp.zeros((ncol, nlev), jnp.int32)
    cape_out = jnp.zeros(ncol, T.dtype)
    w2_out = jnp.zeros((ncol, nlev), T.dtype)
    # resolved tracks only DEEP-resolved columns (LLFIRST at :695-717,733);
    # a shallow surface result does NOT stop the search
    resolved = jnp.zeros(ncol, bool)

    lev = jnp.arange(nlev)[None, :]
    for k_dep in range(nlev - 1, -1, -1):                   # JKK = KLEV..NJKT1 (:314)
        is_surface = k_dep == nlev - 1
        if not is_surface:
            # before the elevated departures LLDEEP=.FALSE. (:652) and
            # LLGO_ON = .NOT.LLDEEP (:655) re-enable EVERY column; the only
            # elevated mask is the NJKT1 pressure floor (:314)
            cand = p_full[:, k_dep] > cfg.departure_top_pa   # NJKT1 (:314)
        init = _init_departure_parcel(k_dep, is_surface, T, q_v, p_full, p_half,
                                      geo_full, geo_half, env_half, shf_w_m2,
                                      lhf_w_m2, ustar, land_frac, dq_dt_adv, cfg)
        ilab, ztu, zqu, zlu, w2h, icbot, ictop, lldcum, zcape = \
            _test_ascent_from_departure(k_dep, is_surface, T, q_v, p_full, p_half,
                                        geo_full, geo_half, env_half, init, cfg)
        jkb = jnp.clip(icbot, 0, nlev - 1); jkt = jnp.clip(ictop, 0, nlev - 1)
        depth = p_half[idx, jkb] - p_half[idx, jkt]
        wb = jnp.sqrt(jnp.maximum(w2h[idx, jkb], 0.0))       # :670 / :716
        # PCAPE = MAXVAL(ZCAPE(JL,:)) over ALL departures (cubasen.F90:741)
        cape_out = jnp.maximum(cape_out, zcape)
        if is_surface:
            lldeep = depth > cfg.depth_split_pa              # '>' (:666)
            sel = lldcum & ~lldeep & col_go                  # :666-667, gated by ZKHVFL<0 (:367)
        else:
            lldeep = depth >= cfg.depth_split_pa             # '>=' (:691)
            # only the FIRST deep elevated departure resets (LLRESET :693-713,
            # LLFIRST=.FALSE. at :733) and REPLACES any shallow surface result
            sel = lldcum & lldeep & cand & ~resolved         # :695-717
            resolved = resolved | sel                        # :733
        kt = jnp.where(is_surface, 2, 1)
        ldcum = jnp.where(sel, 1.0, ldcum)
        ktype = jnp.where(sel, kt, ktype)                    # shallow 2 -> deep 1
        kdpl = jnp.where(sel, jnp.full(ncol, k_dep, jnp.int32), kdpl)
        kcbot = jnp.where(sel, icbot, kcbot)
        kctop = jnp.where(sel, ictop, kctop)
        wbase = jnp.where(sel, wb, wbase)
        w2_out = jnp.where(sel[:, None], w2h, w2_out)
        if is_surface:
            # copy ascent values for JK >= JKT (cubasen.F90:673-683); the source
            # copies for all columns, but the consumer only reads LDCUM columns,
            # so masking by the selected shallow columns is equivalent
            m = sel[:, None] & (lev >= jkt[:, None])
        else:
            # LLRESET fill: ascent inside [JKT, KDPL], environment (klab 1)
            # outside, klab 0 above JKT (cubasen.F90:697-713)
            inside = (lev >= jkt[:, None]) & (lev <= k_dep)
            m = sel[:, None] & inside
            env_m = sel[:, None] & ~inside
            klab = jnp.where(env_m, 1, klab)
            Tu = jnp.where(env_m, T, Tu); qu = jnp.where(env_m, q_v, qu)
            lu = jnp.where(env_m, 0.0, lu)
            klab = jnp.where(sel[:, None] & (lev < jkt[:, None]), 0, klab)
        klab = jnp.where(m, ilab, klab)
        Tu = jnp.where(m, ztu, Tu); qu = jnp.where(m, zqu, qu)
        lu = jnp.where(m, zlu, lu)

    return TestAscent(ldcum, ktype, kdpl, kcbot, kctop, wbase, Tu, qu, lu,
                      klab, cape_out, w2_out)
