"""Generate the CAM6 MG2 warm-rain oracle fixture by RUNNING the oracle.

Oracle: ESCOMP/CAM ``cam_cesm2_1_rel_60`` (commit
a03b84b7c4e34f965b115686f22a043b85739e56), ``src/physics/cam/micro_mg2_0.F90``
(full ``micro_mg_tend``), ``micro_mg_utils.F90`` and ``wv_sat_methods.F90``
(CAM's Goff-Gratch saturation) compiled UNMODIFIED.  Settings are CAM6 + CLUBB: precip_frac_method
'in_cloud', allow_sed_supersat=.false., microp_uniform=.false., nccons with
the column droplet number, legoESM's constants (``mg2_driver.F90``).

Two parts, written to one ``.npz``:
  * ``col_*``: six warm columns (CAM L32, published-sounding shapes; built
    by :func:`build_columns`, reused from the 2026-10-18 tier-1 audit) run
    through the whole ``micro_mg_tend`` at relvar 1, 2, 10, dt = 600 s;
  * ``pt_*``: a point grid through the warm-rain kernels
    (``mg2_procs_driver.F90``).

``--prognostic-nc`` instead writes the prognostic-droplet-number fixture
(``mg2_prognostic_nc.npz``): the same columns run with ``nccons_in=.false.``
(per-level grid-mean nc incl. a cloudy cell with nc = 0, per-level relvar
spanning [0.001, 10], npccn = 0) plus a point set at low density, where MG2's
size reset leaves ncic*rho below 1 m^-3.

``--prognostic-nc-act`` writes ``mg2_prognostic_nc_act.npz`` for the droplet-number
budget (stage S10.2): the prognostic columns with per-level npccn (positive,
negative, zero; also in clear air), plus three copies of ``trop_deep``: ``capped``
(cloud fraction qc/8e-3, so in-cloud water exceeds MG2's 5e-3 cap), ``witness``
(very high nc, so MG2's size reset LOWERS ncic, and a q_r scan, npccn = 0: the qc
mass limiter binds where the number limiter does not) and ``limbind`` (nc x 1e-3,
npccn = 0: the number limiter binds).  Run through the probe build ``mg2_tap``
(build.sh), whose 16th output column is nctend right after the activation
re-apply (micro_mg2_0.F90:2022).

``--prognostic-nc-sed`` writes ``mg2_prognostic_nc_sed.npz`` for the cloud-droplet
sedimentation and the number clear (stage S10.3): the activation columns plus
``thin`` (32 layers of 50 Pa, so MG2's droplet sub-step count exceeds 1, and a
cloud fraction falling downward through the cloud, so part of the incoming flux
evaporates) and ``clearnc`` (cloud water below qsmall with droplets left, and
clear-air activation), through ``mg2_tap`` (nctend after sedimentation, before
micro_mg2_0.F90:2501, and the droplet sub-step count).

Usage (CPU node; needs gfortran and the CAM checkout)::

    python scripts/validate/oracle/morrison_mg2_warm_rain/make_fixture.py \\
        --cam /burg/glab/users/pg2328/oracles/CAM --build /tmp/mg2_bld \\
        --out tests/atmosphere/microphysics/oracle_fixtures/mg2_warm_rain.npz
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT = 600.0          # CAM6 macmic sub-step: 1800 s / cld_macmic_num_steps = 3
RELVARS = (1.0, 2.0, 10.0)
COL_FIELDS = ("T", "q_v", "q_c", "cf", "q_r", "N_r", "N_c", "p_full",
              "p_half", "rho")
ORACLE_COLS = ("tlat", "qvlat", "qctend", "qrtend", "nctend", "nrtend",
               "nevapr", "pratot", "prctot", "qrsedten", "qcsedten", "umr",
               "qcrat", "rflx_bot", "lflx_bot")
PT_IN = ("qcic", "ncic", "rho", "relvar", "qr", "nr", "T", "p", "qfrac",
         "lcldm", "precip_frac", "qiic")
PT_OUT = ("prc", "nprc", "nprc1", "ncic_adj", "pra", "npra", "nragg", "lamr",
          "n0r", "nric", "qric", "pre", "qvl")


def build_columns():
    """The six tier-1 audit columns (level 0 = model top)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm import constants as C
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0
    from legoesm.thermo import saturation_specific_humidity
    hyai = np.array(CAM6_L32_HYAI)
    hybi = np.array(CAM6_L32_HYBI)
    nlev = len(hyai) - 1

    def qsat(T, p):
        return np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))

    cols = []

    def base(ps, Ts, gam, Ttrop):
        ph = hyai * CAM6_L32_P0 + hybi * ps
        pf = 0.5 * (ph[1:] + ph[:-1])
        T = np.maximum(Ts * (pf / ps) ** (C.R_d * gam / C.g), Ttrop)
        return ph, pf, T

    def lam_to_nr(lam, rho, qr):  # exponential PSD: N_vol = lam^3 rho q/(pi rho_w)
        return lam ** 3 * rho * qr / (math.pi * C.rho_water)

    def finish(name, ph, pf, T, rh, qc, cf, qr, lam, Nc):
        rho = pf / (C.R_d * T)
        rh = np.where(T < 253.0, np.minimum(rh, 0.3), rh)   # ice paths idle
        qv = rh * qsat(T, pf)
        warm = T > 275.0
        qc = np.where(warm, qc, 0.0)
        qr = np.where(warm, qr, 0.0)
        cf = np.where(qc > 0, cf, 0.0)
        Nr = np.where(qr > 0, lam_to_nr(lam, rho, qr), 0.0)
        cols.append(dict(name=name, p_half=ph, p_full=pf, T=T, q_v=qv, q_c=qc,
                         cf=cf, q_r=qr, N_r=Nr, N_c=np.full(nlev, Nc), rho=rho))

    # tropical deep convective (warm-rain part below the freezing level)
    ph, pf, T = base(101000.0, 300.0, 6.2e-3, 195.0)
    P = pf / 100.0
    rh = np.where(P > 950, 0.80, np.where(P > 900, 0.85, 0.5))
    cl = (P <= 900) & (P >= 600)
    rh = np.where(cl, 1.0 - 0.05 * (1 - 0.35), rh)
    rh = np.where(P < 400, 0.3, rh)
    qc = np.where(cl, 0.35 * 1.0e-3, 0.0)
    qr = np.where(P >= 600, np.interp(P, [600, 900, 1100], [2e-4, 6e-4, 5e-4]), 0.0)
    finish("trop_deep", ph, pf, T, rh, qc, 0.35, qr, 2500.0, 50.0e6)

    # subtropical stratocumulus (DYCOMS-II RF02-shaped)
    ps = 101780.0
    ph = hyai * CAM6_L32_P0 + hybi * ps
    pf = 0.5 * (ph[1:] + ph[:-1])
    P = pf / 100.0
    kap = C.R_d / C.c_pd
    th_l = np.where(P >= 925, 288.3, 297.0 + 0.04 * (925 - P))
    qt = np.where(P >= 925, 9.45e-3, 1.5e-3)
    T = th_l * (pf / 1.0e5) ** kap
    for _ in range(60):
        qs = qsat(T, pf)
        ql = np.maximum(qt - qs, 0.0)
        Tn = th_l * (pf / 1.0e5) ** kap + C.L_v / C.c_pd * ql  # latent-ok: tier-1 audit column recipe (fixture inputs)
        T = 0.5 * T + 0.5 * Tn
    T = np.maximum(T, 200.0)
    qs = qsat(T, pf)
    ql = np.maximum(qt - qs, 0.0)
    cfc = 0.95
    rh = np.where(ql > 0, 1.0 - 0.05 * (1 - cfc), np.minimum(qt / qs, 0.9))
    qr = np.where(P >= 925, np.where(ql > 0, 4e-5, 1.5e-5), 0.0)
    finish("subtrop_sc", ph, pf, T, rh, ql, cfc, qr, 15000.0, 55.0e6)

    # midlatitude continental warm stratiform rain
    ph, pf, T = base(100000.0, 295.0, 6.5e-3, 210.0)
    P = pf / 100.0
    cl = (P <= 820) & (P >= 680)
    rh = np.where(cl, 1.0 - 0.05 * (1 - 0.7), np.where(P > 820, 0.75, 0.5))
    qc = np.where(cl, 0.7 * 4.0e-4, 0.0)
    qr = np.where(P >= 680, np.interp(P, [680, 820, 1100], [1.5e-4, 3e-4, 2e-4]), 0.0)
    finish("midlat_cont", ph, pf, T, rh, qc, 0.7, qr, 3000.0, 200.0e6)

    # trade cumulus (RICO-shaped)
    ph, pf, T = base(101540.0, 299.2, 6.0e-3, 200.0)
    P = pf / 100.0
    cl = (P <= 950) & (P >= 800)
    rh = np.where(cl, 0.92, np.where(P > 950, 0.80, 0.6))
    qc = np.where(cl, 0.06 * 1.5e-3, 0.0)
    qr = np.where(P >= 800, np.where(cl, 3e-5, 2e-5), 0.0)
    finish("trade_cu", ph, pf, T, rh, qc, 0.06, qr, 4000.0, 70.0e6)

    # edge / threshold column
    ph, pf, T = base(100000.0, 290.0, 6.5e-3, 210.0)
    rho = pf / (C.R_d * T)
    nl = nlev
    rh = np.full(nl, 0.3)
    qc = np.zeros(nl)
    cf = np.zeros(nl)
    qr = np.zeros(nl)
    lam = np.full(nl, 5000.0)
    k = nl - 1
    qc[k] = 5e-9; cf[k] = 0.1; rh[k] = 0.97
    qc[k - 1] = 5e-10; cf[k - 1] = 0.1; rh[k - 1] = 0.97
    qc[k - 2] = 2e-6; cf[k - 2] = 0.0; rh[k - 2] = 0.97
    qr[k - 2] = 1e-4
    qr[k - 3] = 1e-16; lam[k - 3] = 5000.0
    qr[k - 4] = 5e-4; lam[k - 4] = 800.0
    qc[k - 5] = 3e-3; cf[k - 5] = 0.15; qr[k - 5] = 2.5e-3; lam[k - 5] = 2500.0
    rh[k - 5] = 0.99
    qc[k - 6] = 4e-4; cf[k - 6] = 1.0; rh[k - 6] = 1.02; qr[k - 6] = 1e-5
    Nr = np.where(qr > 0, lam ** 3 * rho * qr / (math.pi * C.rho_water), 0.0)
    qv = rh * qsat(T, pf)
    cols.append(dict(name="edge", p_half=ph, p_full=pf, T=T, q_v=qv, q_c=qc,
                     cf=np.where(qc > 0, cf, 0.0), q_r=qr, N_r=Nr,
                     N_c=np.full(nl, 100.0e6), rho=rho))

    # polar dry
    ph, pf, T = base(100500.0, 245.0, 5.0e-3, 200.0)
    finish("polar_dry", ph, pf, T, np.full(nlev, 0.6), np.zeros(nlev), 0.0,
           np.zeros(nlev), 3000.0, 30.0e6)
    return cols


def build_points():
    """Point grid through the warm-rain kernels (columns of PT_IN)."""
    from legoesm import constants as C
    T, p = 285.0, 85000.0
    rho = p / (C.R_d * T)
    fracs = ((1.0, 1.0, 0.0), (0.3, 0.3, 0.0), (0.3, 0.8, 0.0),
             (1.0e-4, 0.5, 0.0), (0.3, 0.8, 2.0e-6))   # lcldm, precip_frac, qiic
    pts = []
    for qcic in (0.0, 1e-9, 2e-8, 5e-7, 1e-4, 5e-3):
        for Nc in (30e6, 300e6):
            for qr_ip in (0.0, 1e-17, 1e-6, 1e-4, 8e-3, 2e-2):
                for lam in (900.0, 3000.0, 1.0e5):
                    for relvar in (1.0, 10.0):
                        for lcldm, pf, qiic in fracs:
                            for qfrac in (0.7, 1.05):
                                nr_ip = lam ** 3 * qr_ip / (math.pi * C.rho_water)
                                pts.append((qcic, Nc / rho, rho, relvar, qr_ip * pf,
                                            nr_ip * pf, T, p, qfrac, lcldm, pf, qiic))
    return np.array(pts)


PROG_RELVARS = (0.001, 0.3, 1.0, 2.5, 10.0)


def build_prognostic_columns(cols):
    """Per-level grid-mean nc [1/kg] and relvar for the --prognostic-nc fixture.

    nc = N_c/rho * lcldm where there is cloud water (the specified columns'
    in-cloud number made grid-mean), scaled by 0.5..2 along the column; the
    edge column's fully cloudy level (q_c = 4e-4, cf = 1) gets nc = 0.
    relvar cycles through PROG_RELVARS by level."""
    out = []
    for c in cols:
        nlev = len(c["T"])
        lcl = np.maximum(c["cf"], 1e-4)
        scale = np.linspace(0.5, 2.0, nlev)
        nc = np.where(c["q_c"] > 0, c["N_c"] / c["rho"] * lcl * scale, 0.0)
        if c["name"] == "edge":
            nc[nlev - 7] = 0.0
        relvar = np.array([PROG_RELVARS[k % len(PROG_RELVARS)] for k in range(nlev)])
        out.append(dict(c, nc_kg=nc, relvar=relvar))
    return out


def build_prognostic_points():
    """Kernel points at rho 1.0 and 0.005 kg/m^3, ncic 0 .. 3e7 /kg (PT_IN columns)."""
    T, p = 285.0, 85000.0
    pts = []
    for rho in (1.0, 0.005):
        for qcic in (1e-8, 2e-8, 1e-6, 1e-4):
            for ncic in (0.0, 1.0e-3, 1.0, 3.0e7):
                for relvar in (0.001, 1.0, 10.0):
                    pts.append((qcic, ncic, rho, relvar, 0.0, 0.0, T, p, 0.7, 1.0, 1.0, 0.0))
    return np.array(pts)


ACT_NPCCN_CLOUDY = ("zero", 1.0e3, 1.0e5, "half_sink", "over_sink")
ACT_NPCCN_CLEAR = (0.0, 1.0e3, -1.0e2)
WITNESS_QR = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)


def build_activation_columns(cols):
    """``--prognostic-nc-act`` columns: per-level npccn [1/kg/s] on the prognostic
    columns, plus the ``witness`` and ``limbind`` copies of trop_deep (npccn = 0)."""
    out = []
    for c in cols:
        npccn = np.zeros_like(c["T"])
        ic = il = 0
        for k in range(len(npccn)):
            if c["q_c"][k] > 0:
                m = ACT_NPCCN_CLOUDY[ic % len(ACT_NPCCN_CLOUDY)]
                ic += 1
                if m == "zero":
                    npccn[k] = 0.0
                elif m == "half_sink":
                    npccn[k] = -0.5 * c["nc_kg"][k] / DT
                elif m == "over_sink":
                    npccn[k] = -2.0 * c["nc_kg"][k] / DT
                else:
                    npccn[k] = m
            else:
                npccn[k] = ACT_NPCCN_CLEAR[il % len(ACT_NPCCN_CLEAR)]
                il += 1
        out.append(dict(c, npccn=npccn))
    base = next(c for c in cols if c["name"] == "trop_deep")
    cloudy = np.flatnonzero(base["q_c"] > 0)
    lcl = np.maximum(base["cf"], 1e-4)
    wit = dict(base, name="witness", npccn=np.zeros_like(base["T"]),
               nc_kg=np.where(base["q_c"] > 0, 1.0e12 * lcl, 0.0),
               q_r=base["q_r"].copy(), N_r=base["N_r"].copy())
    for j, k in enumerate(cloudy):
        qr = WITNESS_QR[j % len(WITNESS_QR)]
        wit["q_r"][k] = qr
        wit["N_r"][k] = base["rho"][k] * qr / (math.pi / 6.0 * 1000.0 * 5.0e-4 ** 3)
    lim = dict(base, name="limbind", npccn=np.zeros_like(base["T"]),
               nc_kg=base["nc_kg"] * 1.0e-3)
    # in-cloud water above MG2's 5e-3 cap (micro_mg2_0.F90:1226): cf = qc/8e-3
    cf_cap = np.where(base["q_c"] > 0, np.clip(base["q_c"] / 8.0e-3, 1e-4, 1.0), base["cf"])
    # no rain, so accretion cannot saturate the nc limiter and the cap shows in nprc1
    cap = dict(base, name="capped", cf=cf_cap, npccn=np.zeros_like(base["T"]),
               q_r=np.zeros_like(base["T"]), N_r=np.zeros_like(base["T"]),
               nc_kg=np.where(base["q_c"] > 0, base["N_c"] / base["rho"] * cf_cap, 0.0))
    return out + [wit, lim, cap]


def build_sedimentation_columns(cols):
    """``--prognostic-nc-sed`` columns: the activation columns plus ``thin`` and ``clearnc``."""
    from legoesm import constants as C
    base = next(c for c in cols if c["name"] == "trop_deep")
    nlev = len(base["T"])
    ph = 90000.0 + 50.0 * np.arange(nlev + 1)
    pf = 0.5 * (ph[1:] + ph[:-1])
    T = np.full(nlev, 285.0)
    rho = pf / (C.R_d * T)
    cloud = (np.arange(nlev) >= 8) & (np.arange(nlev) < 24)
    cf = np.where(cloud, np.linspace(0.9, 0.2, nlev), 0.0)
    qc = np.where(cloud, 5.0e-4, 0.0)
    nc_vol = 3.0e7
    thin = dict(base, name="thin", T=T, p_half=ph, p_full=pf, rho=rho, cf=cf, q_c=qc,
                q_v=np.full(nlev, 8.0e-3), q_r=np.zeros(nlev), N_r=np.zeros(nlev),
                N_c=np.full(nlev, nc_vol), nc_kg=np.where(cloud, nc_vol / rho * cf, 0.0),
                relvar=np.ones(nlev), npccn=np.zeros(nlev))
    tiny = base["q_c"] > 0
    clear = dict(base, name="clearnc", q_c=np.where(tiny, 5.0e-19, 0.0),
                 nc_kg=np.where(tiny, 1.0e6, 0.0), npccn=np.where(tiny, 0.0, 1.0e3))
    return cols + [thin, clear]


def build_melt_freeze_columns(cols):
    """``--prognostic-nc-melt`` columns (S10.4): ``meltice`` (cloud ice across 0 C,
    incl. heat-limited partial melts and 0 < qi < qsmall) and ``homog`` (cloud
    water across -40 C, incl. heat-limited partial freezing and 0 < qc < qsmall).
    No rain or snow, so MG2's instant snow melt and rain freezing (M:2537-2612)
    leave the pools the ice melt reads unchanged."""
    from legoesm import constants as C
    base = next(c for c in cols if c["name"] == "trop_deep")
    nlev = len(base["T"])
    ph = 60000.0 + 500.0 * np.arange(nlev + 1)
    pf = 0.5 * (ph[1:] + ph[:-1])
    z = np.zeros(nlev)

    def col(name, T, qc, qi, nc_kg, ni_kg, cf, qv):
        rho = pf / (C.R_d * T)
        cf = np.maximum(cf, 1.0e-4)   # CAM hands MG2 fractions >= mincld (clear cells NaN otherwise)
        return dict(base, name=name, T=T, p_half=ph, p_full=pf, rho=rho, cf=cf,
                    q_c=qc, q_v=np.full(nlev, qv), q_r=z, N_r=z,
                    N_c=nc_kg * rho, nc_kg=nc_kg, q_i=qi, ni_kg=ni_kg,
                    relvar=np.ones(nlev), npccn=z)

    T_m = np.linspace(273.0, 275.5, nlev)
    qi_m = np.tile([3.0e-3, 1.0e-3, 1.0e-4, 1.0e-6, 5.0e-19, 0.0, 2.0e-3, 5.0e-4], nlev // 8 + 1)[:nlev]
    qc_m = np.tile([0.0, 2.0e-4, 0.0, 1.0e-5], nlev // 4 + 1)[:nlev]
    cf_m = np.where(qc_m + qi_m > 0, 0.6, 0.0)
    melt = col("meltice", T_m, qc_m, qi_m, np.where(qc_m > 0, 5.0e7, 0.0),
               np.where(qi_m > 0, 1.0e5, 0.0), cf_m, 4.0e-3)
    T_h = np.linspace(229.0, 237.0, nlev)
    qc_h = np.tile([1.0e-3, 3.0e-4, 1.0e-5, 3.0e-19, 0.0, 2.0e-3, 5.0e-4, 1.0e-4], nlev // 8 + 1)[:nlev]
    cf_h = np.where(qc_h > 0, 0.5, 0.0)
    homog = col("homog", T_h, qc_h, z, np.where(qc_h > 0, 8.0e7, 0.0), z, cf_h, 1.0e-4)
    return [melt, homog]


def build_snow_melt_columns(cols):
    """``--prognostic-nc-snow`` columns (S10.4b): ``snowmelt``, snow across +2 C, incl.
    heat-limited partial melts and 0 < qs < qsmall; no cloud ice, T > 0 C, so
    MG2's ice processes and rain freezing are idle; ``snowfall``, snow above
    the +2 C level falling into it (the post-sedimentation melt point)."""
    from legoesm import constants as C
    from legoesm.thermo import saturation_specific_humidity
    import jax.numpy as jnp
    base = next(c for c in cols if c["name"] == "trop_deep")
    nlev = len(base["T"])
    ph = 60000.0 + 500.0 * np.arange(nlev + 1)
    pf = 0.5 * (ph[1:] + ph[:-1])
    z = np.zeros(nlev)
    T = np.linspace(274.5, 279.5, nlev)
    qs = np.tile([3.0e-3, 1.0e-3, 2.0e-4, 1.0e-6, 5.0e-19, 0.0, 6.0e-3, 5.0e-4], nlev // 8 + 1)[:nlev]
    ns = np.where(qs > 0, 2.0e4, 0.0)
    rho = pf / (C.R_d * T)
    qv = 0.95 * np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(pf)))
    snow = dict(base, name="snowmelt", T=T, p_half=ph, p_full=pf, rho=rho,
                cf=np.full(nlev, 1.0e-4), q_c=z, q_v=qv, q_r=z, N_r=z, N_c=z, nc_kg=z,
                q_i=z, ni_kg=z, q_s=qs, ns_kg=ns, relvar=np.ones(nlev), npccn=z)
    # cold snow above +2 C air: sedimentation carries it down to the post-step melt
    T2 = np.linspace(268.0, 281.0, nlev)
    qs2 = np.where(T2 < 274.0, 2.0e-3, 0.0)
    rho2 = pf / (C.R_d * T2)
    qv2 = 0.95 * np.asarray(saturation_specific_humidity(jnp.asarray(T2), jnp.asarray(pf)))
    fall = dict(snow, name="snowfall", T=T2, rho=rho2, q_v=qv2, q_s=qs2,
                ns_kg=np.where(qs2 > 0, 1.0e4, 0.0))
    return [snow, fall]


def write_prognostic_column_inputs(cols, path):
    nlev = len(cols[0]["T"])
    with open(path, "w") as f:
        f.write(f"{len(cols)} {nlev} {DT!r}\n")
        for c in cols:
            f.write(f"{0.0:.17e}\n")       # ncnst: unused with nccons_in=.false.
            pdel = np.diff(c["p_half"])
            for k in range(nlev):
                row = [c["T"][k], c["q_v"][k], c["q_c"][k], c.get("q_i", np.zeros(nlev))[k],
                       c["nc_kg"][k], c.get("ni_kg", np.zeros(nlev))[k],
                       c["q_r"][k], c.get("q_s", np.zeros(nlev))[k], c["N_r"][k] / c["rho"][k],
                       c.get("ns_kg", np.zeros(nlev))[k],
                       c["relvar"][k], 1.0, c["p_full"][k], pdel[k],
                       c["cf"][k], c["cf"][k], c["cf"][k], 1.0, c.get("npccn", np.zeros(nlev))[k]]
                if "naai" in c:
                    row.append(c["naai"][k])
                f.write(" ".join(f"{x:.17e}" for x in row) + "\n")


def write_column_inputs(cols, path, relvar):
    """``relvar``: a scalar, or a per-level ``(nlev,)`` profile."""
    nlev = len(cols[0]["T"])
    relvar = np.broadcast_to(np.asarray(relvar, dtype=np.float64), (nlev,))
    with open(path, "w") as f:
        f.write(f"{len(cols)} {nlev} {DT!r}\n")
        for c in cols:
            rho = c["rho"]
            ncnst = float(c["N_c"][0])
            f.write(f"{ncnst:.17e}\n")
            pdel = np.diff(c["p_half"])
            lcl = np.maximum(c["cf"], 1e-4)
            for k in range(nlev):
                row = [c["T"][k], c["q_v"][k], c["q_c"][k], 0.0,
                       ncnst / rho[k] * lcl[k] if c["q_c"][k] > 0 else 0.0, 0.0,
                       c["q_r"][k], 0.0, c["N_r"][k] / rho[k], 0.0,
                       relvar[k], 1.0, c["p_full"][k], pdel[k],
                       c["cf"][k], c["cf"][k], c["cf"][k], 1.0]
                f.write(" ".join(f"{x:.17e}" for x in row) + "\n")


def load_column_outputs(path, ncol, nlev):
    lines = Path(path).read_text().split("\n")
    prect = np.array([float(line.split()[1]) for line in lines[:ncol]])
    body = np.array([[float(x) for x in line.split()] for line in lines[ncol:]
                     if line.strip()]).reshape(ncol, nlev, -1)
    out = {k: body[:, :, i + 2] for i, k in enumerate(ORACLE_COLS)}
    extra = body.shape[-1] - len(ORACLE_COLS) - 2   # mg2_tap probe columns
    if extra >= 1:
        out["nctend_stage"] = body[:, :, len(ORACLE_COLS) + 2]    # after :2022
    if extra >= 3:
        out["nctend_sed"] = body[:, :, len(ORACLE_COLS) + 3]      # before :2501
        out["nstepc"] = body[:, :, len(ORACLE_COLS) + 4]          # droplet sed sub-steps
    if extra >= 20:   # S10.4: before the ice melt (M:2616) / after homog freezing
        b = len(ORACLE_COLS) + 5
        for i, k in enumerate(("pre_dumc", "pre_dumnc", "pre_dumi", "pre_dumni", "pre_Tp",
                               "pre_qctend", "pre_nctend", "pre_qitend", "pre_nitend",
                               "pre_tlat", "post_qctend", "post_nctend", "post_qitend",
                               "post_nitend", "post_tlat", "qitend", "nitend")):
            out[k] = body[:, :, b + i]
    if extra >= 38:   # S10.4b snow melt taps: driver columns 36..53
        names = ("snow1_qs", "snow1_ns", "snow1_qr", "snow1_nr", "snow1_tlat",
                 "snow2_dums", "snow2_dumns", "snow2_Tp", "snow2_qstend", "snow2_nstend",
                 "snow2_qrtend", "snow2_nrtend", "snow2_tlat", "snow3_qstend", "snow3_nstend",
                 "snow3_qrtend", "snow3_nrtend", "snow3_tlat")
        for i, k in enumerate(names):
            out[k] = body[:, :, 1 + 36 + i]      # body = (ic, k, driver column j at index j + 1)
    if extra >= 76:   # S10.6 ice / vapour limiter taps: driver columns 54..91
        names = (("ilim1_" + k for k in ILIM1) , ("ilim2_" + k for k in ("pre", "mnuccr", "mnuccri", "pracs", "nnuccri")),
                 ("ilim3_" + k for k in ILIM3))
        j = 54
        for group in names:
            for k in group:
                out[k] = body[:, :, 1 + j]
                j += 1
    if extra >= 81:   # 2026-10-22 tap_frz: driver columns 92..96
        for i, k in enumerate(("frz_mnuccr", "frz_nnuccr", "frz_lamr", "frz_qric", "frz_nric")):
            out[k] = body[:, :, 1 + 92 + i]
    out["prect"] = prect
    return out


ILIM1 = ("vap_dep", "ice_sublim", "mnuccd", "berg", "msacwi", "nsacwi", "prci", "nprci", "bergs",
         "psacws", "prds", "qcrat", "q", "t", "qi", "ni", "qs", "qc", "qvi", "precip_frac", "icldm",
         "lcldm", "nnuccd", "prai", "nprai")
ILIM3 = ("vap_dep", "ice_sublim", "mnuccd", "prci", "nprci", "prds", "pre", "nnuccd")


def _git(path, *args):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(path), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _meta(a):
    return dict(
        oracle="ESCOMP/CAM", tag="cam_cesm2_1_rel_60", sha=_git(a.cam, "rev-parse", "HEAD"),
        files=["src/physics/cam/micro_mg2_0.F90", "src/physics/cam/micro_mg_utils.F90",
               "src/physics/cam/wv_sat_methods.F90"],
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        legoesm_sha=_git(HERE, "rev-parse", "HEAD"),
        generator=("python scripts/validate/oracle/morrison_mg2_warm_rain/"
                   f"make_fixture.py --cam <CAM> --build <dir> --out {a.out.name}"
                   + (" --prognostic-nc" if a.prognostic_nc else "")
                   + (" --prognostic-nc-act" if a.prognostic_nc_act else "")
                   + (" --prognostic-nc-sed" if a.prognostic_nc_sed else "")
                   + (" --prognostic-nc-melt" if getattr(a, "prognostic_nc_melt", False) else "")
                   + (" --prognostic-nc-snow" if getattr(a, "prognostic_nc_snow", False) else "")
                   + (" --prognostic-nc-rime" if getattr(a, "prognostic_nc_rime", False) else "")
                   + (" --prognostic-nc-ice" if getattr(a, "prognostic_nc_ice", False) else "")
                   + (" --prognostic-nc-icelim" if getattr(a, "prognostic_nc_icelim", False) else "")
                   + (" --prognostic-nc-frz" if getattr(a, "prognostic_nc_frz", False) else "")))


def main_prognostic(a):
    cols = build_prognostic_columns(build_columns())
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols]) for k in COL_FIELDS + ("nc_kg", "relvar")}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fin, fout = a.build / "prog_col_in.txt", a.build / "prog_col_out.txt"
    write_prognostic_column_inputs(cols, fin)
    subprocess.run([str(a.build / "mg2_driver"), str(fin), str(fout), "F", "P"], check=True)
    o = load_column_outputs(fout, ncol, nlev)
    for k in ORACLE_COLS:
        fx[f"ora_{k}"] = o[k]
    fx["ora_rain_sfc_flux"] = o["rflx_bot"][:, -1]
    pts = build_prognostic_points()
    pin, pout = a.build / "prog_pt_in.txt", a.build / "prog_pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_procs"), str(pin), str(pout)], check=True)
    po = np.loadtxt(pout)
    for i, k in enumerate(PT_IN):
        fx[f"pt_in_{k}"] = pts[:, i]
    for i, k in enumerate(PT_OUT):
        fx[f"pt_out_{k}"] = po[:, i]
    meta = _meta(a)
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


def main_prognostic_act(a):
    cols = build_activation_columns(build_prognostic_columns(build_columns()))
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols])
          for k in COL_FIELDS + ("nc_kg", "relvar", "npccn")}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fin, fout = a.build / "act_col_in.txt", a.build / "act_col_out.txt"
    write_prognostic_column_inputs(cols, fin)
    subprocess.run([str(a.build / "mg2_tap"), str(fin), str(fout), "F", "P"], check=True)
    o = load_column_outputs(fout, ncol, nlev)
    for k in ORACLE_COLS + ("nctend_stage", "nctend_sed", "nstepc"):
        fx[f"ora_{k}"] = o[k]
    meta = _meta(a)
    meta["probe"] = ("build.sh tap/: micro_mg2_0.F90 + nctend after :2022, nctend before :2501, "
                     "droplet sedimentation nstep (8 added lines)")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


def main_prognostic_sed(a):
    cols = build_sedimentation_columns(
        build_activation_columns(build_prognostic_columns(build_columns())))
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols])
          for k in COL_FIELDS + ("nc_kg", "relvar", "npccn")}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fin, fout = a.build / "sed_col_in.txt", a.build / "sed_col_out.txt"
    write_prognostic_column_inputs(cols, fin)
    subprocess.run([str(a.build / "mg2_tap"), str(fin), str(fout), "F", "P"], check=True)
    o = load_column_outputs(fout, ncol, nlev)
    for k in ORACLE_COLS + ("nctend_stage", "nctend_sed", "nstepc"):
        fx[f"ora_{k}"] = o[k]
    meta = _meta(a)
    meta["probe"] = ("build.sh tap/: micro_mg2_0.F90 + nctend after :2022, nctend before :2501, "
                     "droplet sedimentation nstep (8 added lines)")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


def main_prognostic_melt(a):
    cols = build_melt_freeze_columns(build_prognostic_columns(build_columns()))
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols])
          for k in COL_FIELDS + ("nc_kg", "relvar", "npccn", "q_i", "ni_kg")}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fin, fout = a.build / "melt_col_in.txt", a.build / "melt_col_out.txt"
    write_prognostic_column_inputs(cols, fin)
    subprocess.run([str(a.build / "mg2_tap"), str(fin), str(fout), "F", "P"], check=True)
    o = load_column_outputs(fout, ncol, nlev)
    for k in o:
        fx[f"ora_{k}"] = o[k]
    meta = _meta(a)
    meta["probe"] = ("build.sh tap/: micro_mg2_0.F90 + S10.2/S10.3 taps + tap_pre before M:2616 "
                     "and tap_post before M:2687 (27 added lines)")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


def main_prognostic_snow(a):
    cols = build_snow_melt_columns(build_prognostic_columns(build_columns()))
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols])
          for k in COL_FIELDS + ("nc_kg", "relvar", "npccn", "q_i", "ni_kg", "q_s", "ns_kg")}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fin, fout = a.build / "snow_col_in.txt", a.build / "snow_col_out.txt"
    write_prognostic_column_inputs(cols, fin)
    subprocess.run([str(a.build / "mg2_tap"), str(fin), str(fout), "F", "P"], check=True)
    o = load_column_outputs(fout, ncol, nlev)
    for k in o:
        fx[f"ora_{k}"] = o[k]
    meta = _meta(a)
    meta["probe"] = ("build.sh tap/: micro_mg2_0.F90 + S10.2-S10.4 taps + tap_snow1 after the "
                     "snow-melt pre-apply (M:1178), tap_snow2/3 around the post-sedimentation "
                     "snow melt (M:2535, M:2569) (51 added lines)")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


RIME_IN = ("qcic", "ncic_kg", "rho", "qs", "ns_kg", "precip_frac", "T")
RIME_OUT = ("psacws_raw", "npsacws", "psacws_left", "msacwi", "nsacwi", "lams", "n0s",
            "uns", "nsic_adj", "ncic_adj")
RIME_RHOSN = 250.0   # MG2 rhosn; the test runs our PSD at this density


def build_rime_points():
    """Points spanning T from -13 to +1 C (offset off every MG2 corner), in-cloud
    droplets up to the 5e-3 cap, snow from trace to the 0.01 in-precip cap, and
    snow numbers placing the PSD slope below, inside and above its bounds."""
    rng = np.random.default_rng(105)
    T = 260.15 + 0.0731 + 0.25 * np.arange(57)
    pts = []
    for t in T:
        for _ in range(6):
            qc = 10.0 ** rng.uniform(-7, np.log10(5.0e-3))
            nc = 10.0 ** rng.uniform(7, 9)
            rho = rng.uniform(0.4, 1.2)
            pf = rng.choice([1.0, 0.35])
            qs = 10.0 ** rng.uniform(-8, -1.5)
            lam = 10.0 ** rng.uniform(np.log10(300.0), np.log10(2.0e5))
            ns = lam ** 3 * min(qs / pf, 0.01) / (np.pi * RIME_RHOSN) * pf
            pts.append((qc, nc, rho, qs, ns, pf, t))
    return np.array(pts)


def main_prognostic_rime(a):
    pts = build_rime_points()
    pin, pout = a.build / "rime_pt_in.txt", a.build / "rime_pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_rime"), str(pin), str(pout)], check=True)
    po = np.loadtxt(pout)
    fx = {f"in_{k}": pts[:, i] for i, k in enumerate(RIME_IN)}
    fx.update({f"out_{k}": po[:, i] for i, k in enumerate(RIME_OUT)})
    fx["rhosn"] = np.array(RIME_RHOSN)
    meta = _meta(a)
    meta["probe"] = "mg2_rime_driver.F90: unmodified accrete_cloud_water_snow + secondary_ice_production"
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


ICE_IN = ("T", "q", "qi", "ni_kg", "qcic", "qs", "ns_kg", "precip_frac", "icldm", "lcldm", "p")
ICE_OUT = ("vap_dep", "ice_sublim", "berg", "bergs", "prds", "prci", "nprci", "qvl", "qvi",
           "lami", "n0i", "lams", "n0s")
ICE_DCS = 500.0e-6    # mg2_ice_driver dcs (CAM6 mg2 namelist default)


def build_ice_points():
    """Points from -38 to +3 C (offset off 0 C) with vapour from ice-subsaturated
    to water-supersaturated, cloud ice from trace (incl. < qsmall) to 2 g/kg with
    ice and snow slopes below, inside and above their bounds, with and without
    cloud liquid, at full and partial cloud / precipitation fractions."""
    from legoesm.thermo import saturation_specific_humidity, saturation_specific_humidity_ice
    import jax.numpy as jnp
    rng = np.random.default_rng(106)
    pts = []
    for t in 235.15 + 0.0731 + 0.5 * np.arange(83):
        for _ in range(4):
            p = rng.uniform(4.0e4, 9.0e4)
            qvi = float(saturation_specific_humidity_ice(jnp.array(t), jnp.array(p)))
            qvl = float(saturation_specific_humidity(jnp.array(t), jnp.array(p)))
            q = rng.uniform(0.7 * qvi, 1.05 * max(qvl, qvi))
            cf = rng.choice([1.0, 0.3])
            qi = rng.choice([0.0, 5.0e-19]) if rng.uniform() < 0.1 else 10.0 ** rng.uniform(-7, np.log10(2.0e-3))
            lam = 10.0 ** rng.uniform(np.log10(500.0), np.log10(2.0e6))
            ni = lam ** 3 * qi / cf / (np.pi * 500.0) * cf
            qcic = 0.0 if rng.uniform() < 0.3 else 10.0 ** rng.uniform(-6, np.log10(5.0e-3))
            pf = rng.choice([1.0, 0.4]) if cf < 1 else 1.0
            pf = max(pf, cf)
            qs = 10.0 ** rng.uniform(-8, -2)
            lams = 10.0 ** rng.uniform(np.log10(300.0), np.log10(2.0e5))
            ns = lams ** 3 * min(qs / pf, 0.01) / (np.pi * 250.0) * pf
            pts.append((t, q, qi, ni, qcic, qs, ns, pf, cf, cf, p))
    return np.array(pts)


def main_prognostic_ice(a):
    pts = build_ice_points()
    pin, pout = a.build / "ice_pt_in.txt", a.build / "ice_pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_ice"), str(pin), str(pout)], check=True)
    po = np.loadtxt(pout)
    fx = {f"in_{k}": pts[:, i] for i, k in enumerate(ICE_IN)}
    fx.update({f"out_{k}": po[:, i] for i, k in enumerate(ICE_OUT)})
    fx["dcs"] = np.array(ICE_DCS)
    fx["rhosn"] = np.array(250.0)
    meta = _meta(a)
    meta["probe"] = ("mg2_ice_driver.F90: unmodified ice_deposition_sublimation, "
                     "bergeron_process_snow, evaporate_sublimate_precip, ice_autoconversion")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


def build_ice_limiter_columns(cols):
    """``--prognostic-nc-icelim`` columns (S10.6): cold mixed-phase levels drawn at
    random so that each MG2 ice / vapour limiter binds somewhere: tiny cloud water
    under strong Bergeron (q_c limiter), ice nucleation (naai) close to ice
    saturation (deposition limiter), trace ice sublimating (ice limiter), crystals
    near the ice->snow size (number limiter), trace snow in dry clear air (snow
    limiter) and rain + snow + ice evaporating into near-saturated air (the
    water-supersaturation limiter)."""
    from legoesm import constants as C
    from legoesm.thermo import saturation_specific_humidity, saturation_specific_humidity_ice
    import jax.numpy as jnp
    base = next(c for c in cols if c["name"] == "trop_deep")
    nlev = len(base["T"])
    out = []
    for j in range(8):
        rng = np.random.default_rng(1060 + j)
        ph = 30000.0 + 1500.0 * np.arange(nlev + 1)
        pf = 0.5 * (ph[1:] + ph[:-1])
        T = rng.uniform(238.0, 272.5, nlev)
        rho = pf / (C.R_d * T)
        qvl = np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(pf)))
        qvi = np.asarray(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(pf)))
        f = rng.uniform(0.6, 1.0, nlev)
        qv = np.where(rng.uniform(size=nlev) < 0.5, qvi * f, qvi + (qvl - qvi) * rng.uniform(0, 1.02, nlev))
        cf = rng.choice([1.0, 0.5, 0.2], nlev)
        qc = np.where(rng.uniform(size=nlev) < 0.6, 10.0 ** rng.uniform(-9, -3.5, nlev), 0.0) * cf
        qi = np.where(rng.uniform(size=nlev) < 0.85, 10.0 ** rng.uniform(-9, -3.3, nlev), 0.0)
        lam = 10.0 ** rng.uniform(np.log10(800.0), 6.0, nlev)
        ni = lam ** 3 * qi / (np.pi * 500.0)
        qs = np.where(rng.uniform(size=nlev) < 0.7, 10.0 ** rng.uniform(-9, -3, nlev), 0.0)
        lams = 10.0 ** rng.uniform(np.log10(600.0), 4.5, nlev)
        ns = lams ** 3 * qs / (np.pi * 250.0)
        qr = np.where(rng.uniform(size=nlev) < 0.4, 10.0 ** rng.uniform(-8, -3.5, nlev), 0.0)
        nr = qr * rho * 1.0e6
        nc = np.where(qc > 0, rng.uniform(2e7, 2e8, nlev), 0.0)
        naai = np.where(rng.uniform(size=nlev) < 0.5, rng.uniform(1e3, 3e5, nlev), 0.0)
        z = np.zeros(nlev)
        out.append(dict(base, name=f"icelim{j}", T=T, p_half=ph, p_full=pf, rho=rho, cf=cf,
                        q_c=qc, q_v=qv, q_r=qr, N_r=nr, N_c=nc * rho, nc_kg=nc, q_i=qi, ni_kg=ni,
                        q_s=qs, ns_kg=ns, relvar=np.ones(nlev), npccn=z, naai=naai))
    for j in range(4):
        # targeted: (j < 2) rain + snow evaporating into nearly water-saturated
        # cloudy air (supersaturation limiter); (j >= 2) ice nucleation and
        # deposition close to ice saturation (deposition limiter)
        rng = np.random.default_rng(1070 + j)
        ph = 30000.0 + 1500.0 * np.arange(nlev + 1)
        pf = 0.5 * (ph[1:] + ph[:-1])
        z = np.zeros(nlev)
        if j < 2:
            T = np.where(np.arange(nlev) < nlev // 2, rng.uniform(262.0, 285.0, nlev),
                         rng.uniform(273.4, 288.0, nlev))
        else:
            T = rng.uniform(228.0, 262.0, nlev)
        rho = pf / (C.R_d * T)
        qvl = np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(pf)))
        qvi = np.asarray(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(pf)))
        if j < 2:
            # cloudy upper half; clear, nearly water-saturated lower half that
            # inherits the precipitation fraction (rain/snow evaporate there)
            top = np.arange(nlev) < nlev // 2
            qv = np.where(top, qvl, qvl * rng.uniform(0.997, 0.9999, nlev))
            cf = np.where(top, rng.uniform(0.8, 0.98, nlev), 1.0e-4)
            qc = np.where(top, 10.0 ** rng.uniform(-6, -4, nlev) * cf, 0.0)
            qr = 10.0 ** rng.uniform(-4.5, -3, nlev)
            qs = 10.0 ** rng.uniform(-5, -3.5, nlev)
            qi = np.where(top, 10.0 ** rng.uniform(-6, -4, nlev), 0.0)
            naai = z
        else:
            qv = qvi * rng.uniform(1.06, 1.15, nlev)
            cf = rng.choice([1.0, 0.6], nlev)
            qc = z
            qr = z
            qs = 10.0 ** rng.uniform(-7, -5, nlev)
            qi = 10.0 ** rng.uniform(-4.5, -3, nlev)
            naai = rng.uniform(1e5, 1e6, nlev)
        lam = 10.0 ** rng.uniform(3.0, 4.0, nlev)
        ni = lam ** 3 * qi / (np.pi * 500.0)
        ns = (10.0 ** rng.uniform(3.2, 4.2, nlev)) ** 3 * qs / (np.pi * 250.0)
        nc = np.where(qc > 0, 1.0e8, 0.0)
        out.append(dict(base, name=f"icelim_t{j}", T=T, p_half=ph, p_full=pf, rho=rho, cf=cf,
                        q_c=qc, q_v=qv, q_r=qr, N_r=qr * rho * 1.0e6, N_c=nc * rho, nc_kg=nc,
                        q_i=qi, ni_kg=ni, q_s=qs, ns_kg=ns, relvar=np.ones(nlev), npccn=z,
                        naai=naai))
    return out


def build_rain_freezing_columns(cols):
    """``--prognostic-nc-frz`` columns (2026-10-22, 15a/3a): supercooled rain at
    -21..-1 C with mean drop sizes 1/lamr of 300-700 um, half of the levels at
    430-520 um around MG2's Dcs = 500 um (MG2 routes frozen rain to cloud ice
    where 1/lamr < Dcs; rain above 500 um is pinned to lamr = 1/500 um), in
    water-saturated full cloud (no rain evaporation)."""
    from legoesm import constants as C
    from legoesm.thermo import saturation_specific_humidity
    import jax.numpy as jnp
    base = next(c for c in cols if c["name"] == "trop_deep")
    nlev = len(base["T"])
    out = []
    for j in range(6):
        rng = np.random.default_rng(1080 + j)
        ph = 40000.0 + 1500.0 * np.arange(nlev + 1)
        pf = 0.5 * (ph[1:] + ph[:-1])
        T = rng.uniform(252.0, 272.0, nlev)
        rho = pf / (C.R_d * T)
        qv = np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(pf)))
        cf = np.ones(nlev)
        qc = 10.0 ** rng.uniform(-7, -4, nlev)
        qr = 10.0 ** rng.uniform(-6, -3, nlev)
        d = np.where(rng.uniform(size=nlev) < 0.5, rng.uniform(430e-6, 520e-6, nlev),
                     rng.uniform(300e-6, 700e-6, nlev))
        nr = (1.0 / d) ** 3 * rho * qr / (np.pi * C.rho_water)      # per volume
        qi = 10.0 ** rng.uniform(-8, -5, nlev)
        ni = (10.0 ** rng.uniform(4, 5, nlev)) ** 3 * qi / (np.pi * 500.0)
        qs = 10.0 ** rng.uniform(-8, -5, nlev)
        ns = (10.0 ** rng.uniform(3.2, 4.2, nlev)) ** 3 * qs / (np.pi * 250.0)
        nc = np.full(nlev, 1.0e8)
        z = np.zeros(nlev)
        out.append(dict(base, name=f"frz{j}", T=T, p_half=ph, p_full=pf, rho=rho, cf=cf,
                        q_c=qc, q_v=qv, q_r=qr, N_r=nr, N_c=nc * rho, nc_kg=nc, q_i=qi,
                        ni_kg=ni, q_s=qs, ns_kg=ns, relvar=np.ones(nlev), npccn=z, naai=z))
    return out


def main_prognostic_icelim(a, builder=None, tag="icelim"):
    cols = (builder or build_ice_limiter_columns)(build_prognostic_columns(build_columns()))
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols])
          for k in COL_FIELDS + ("nc_kg", "relvar", "npccn", "q_i", "ni_kg", "q_s", "ns_kg", "naai")}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fin, fout = a.build / f"{tag}_col_in.txt", a.build / f"{tag}_col_out.txt"
    write_prognostic_column_inputs(cols, fin)
    subprocess.run([str(a.build / "mg2_tap"), str(fin), str(fout), "F", "P", "I"], check=True)
    o = load_column_outputs(fout, ncol, nlev)
    for k in o:
        fx[f"ora_{k}"] = o[k]
    meta = _meta(a)
    meta["probe"] = ("build.sh tap/: micro_mg2_0.F90 + taps incl. S10.6 tap_ilim1 before M:1588, "
                     "tap_ilim2 before M:1687, tap_ilim3 before M:1855, tap_frz before M:1645 (66 added lines); naai input")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prognostic-nc", action="store_true")
    ap.add_argument("--prognostic-nc-act", action="store_true")
    ap.add_argument("--prognostic-nc-sed", action="store_true")
    ap.add_argument("--prognostic-nc-melt", action="store_true")
    ap.add_argument("--prognostic-nc-snow", action="store_true")
    ap.add_argument("--prognostic-nc-rime", action="store_true")
    ap.add_argument("--prognostic-nc-ice", action="store_true")
    ap.add_argument("--prognostic-nc-icelim", action="store_true")
    ap.add_argument("--prognostic-nc-frz", action="store_true")
    a = ap.parse_args(argv)
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.cam), str(a.build)],
                   check=True)
    if a.prognostic_nc:
        return main_prognostic(a)
    if a.prognostic_nc_act:
        return main_prognostic_act(a)
    if a.prognostic_nc_sed:
        return main_prognostic_sed(a)
    if a.prognostic_nc_melt:
        return main_prognostic_melt(a)
    if a.prognostic_nc_snow:
        return main_prognostic_snow(a)
    if a.prognostic_nc_rime:
        return main_prognostic_rime(a)
    if a.prognostic_nc_ice:
        return main_prognostic_ice(a)
    if a.prognostic_nc_icelim:
        return main_prognostic_icelim(a)
    if a.prognostic_nc_frz:
        return main_prognostic_icelim(a, build_rain_freezing_columns, "frz")
    cols = build_columns()
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols]) for k in COL_FIELDS}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fx["relvars"] = np.array(RELVARS)
    for rv in RELVARS:
        fin = a.build / f"col_in_rv{rv:g}.txt"
        fout = a.build / f"col_out_rv{rv:g}.txt"
        write_column_inputs(cols, fin, rv)
        subprocess.run([str(a.build / "mg2_driver"), str(fin), str(fout), "F"],
                       check=True)
        o = load_column_outputs(fout, ncol, nlev)
        for k in ("prctot", "pratot", "nevapr", "qrtend", "nrtend", "qrsedten",
                  "qcrat"):
            fx[f"ora_rv{rv:g}_{k}"] = o[k]
        fx[f"ora_rv{rv:g}_rain_sfc_flux"] = o["rflx_bot"][:, -1]
    pts = build_points()
    pin = a.build / "pt_in.txt"
    pout = a.build / "pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_procs"), str(pin), str(pout)], check=True)
    po = np.loadtxt(pout)
    for i, k in enumerate(PT_IN):
        fx[f"pt_in_{k}"] = pts[:, i]
    for i, k in enumerate(PT_OUT):
        fx[f"pt_out_{k}"] = po[:, i]
    meta = _meta(a)
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


if __name__ == "__main__":
    main()
