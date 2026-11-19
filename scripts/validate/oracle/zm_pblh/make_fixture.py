"""Generate tests/atmosphere/oracle_fixtures/zm_pblh_cam.npz by RUNNING CAM6 ZM.

Oracle: ESCOMP/CAM tag cam_cesm2_1_rel_60, SHA a03b84b7c4e34f965b115686f22a043b85739e56,
src/physics/cam/zm_conv.F90 (zm_convi, zm_convr, zm_conv_evap, momtran,
convtran) compiled by build.sh after patch_oracle.py (P1 drops the dead
zm_mphy call, P2 makes the ientropy Brent tolerances variables, P3/P4 add
diagnostic copies only).  driver.F90 reproduces zm_conv_intr.F90::zm_conv_tend
(zm_convr(delt=0.5*ztodt) -> physics_update T,Q with qneg3 1e-12 ->
zm_conv_evap(ztodt) -> momtran -> convtran on the MG2 convtran1
constituents legoESM carries).  stubs.F90 supplies CAM infrastructure;
patch_wv_sat.py (W1) lets CAM's own wv_sat_svp_trans run on legoESM's curves.

Variant (the tier-1 audit's V1, oracle_audit/zm_deep_convection, with both
saturation calls CAM's own): legoESM constants (physconst mode 1), CAM's
unmodified qsat_water (Goff-Gratch) for the plume and CAM's table qsat
(estblf on the 1-K table of the Goff-Gratch svp_trans) for zm_conv_evap
(wv_saturation sat_mode 2), and the ientropy Brent solve tightened to tol 1e-11 K / EPS
1e-15 so it converges to the root the port's Newton solve finds.  Heights are the port's own (compute_column_geometry, the CAM-SE
geopotential_t form).  Everything else is the stock CAM code path.

The columns are the tier-1 set (3 MPAS AMIP day-1 L32 columns from
tests/atmosphere/hydrostatic/unit/fixtures/zm_cam6_real_zero_humidity_columns.npz
plus 12 constructed soundings) with pressures taken from the production
cam_l32 coordinate at each column's surface pressure, so the MPAS bridge and
the oracle see identical inputs.  Each column carries a PBL height (pblh);
column ``nocturnal_elevated`` (pblh 150 m, moist layer at 850-800 hPa) is the
one where the launch bound decides whether the column convects.  Cloud
liquid/ice are the tier-1 profiles (2e-4 and 5e-5 times the cloud fraction);
the droplet/ice/rain numbers and rain/snow masses are constructed profiles of
distinct vertical shape, so a tracer routed to the wrong name cannot pass.

    sbatch --wait scripts/validate/oracle/zm_pblh/build.sh    (from the repo root)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
OUT_NPZ = os.path.join(REPO, "tests", "atmosphere", "oracle_fixtures", "zm_pblh_cam.npz")
REAL_COLS = os.path.join(REPO, "tests", "atmosphere", "hydrostatic", "unit", "fixtures",
                         "zm_cam6_real_zero_humidity_columns.npz")
DT = 1800.0  # CAM6 physics step (ztodt)
# V1: (const_mode, sat_mode, ientropy tol [K], ientropy EPS)
VARIANT = (1, 2, 1e-11, 1e-15)
# convtran1 tracers: (fixture key, legoESM tracer name, oracle output name)
TRACERS = (("qc", "q_c", "dqc_convtran"), ("qi", "q_i", "dqi_convtran"),
           ("nc", "N_c", "dnc_convtran"), ("ni", "N_i", "dni_convtran"),
           ("qr", "q_r", "dqr_convtran"), ("qs", "q_s", "dqs_convtran"),
           ("nr", "N_r", "dnr_convtran"))
ORACLE_TAG = "cam_cesm2_1_rel_60"
ORACLE_SHA = "a03b84b7c4e34f965b115686f22a043b85739e56"


def build_columns():
    """Tier-1 columns (oracle_audit/zm_deep_convection/harness.py) on cam_l32."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.core.precision import PrecisionPolicy, set_policy
    # fp64 policy: the cam_l32 coordinate coefficients follow the policy dtype
    set_policy(PrecisionPolicy.fp64())
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.thermo import saturation_specific_humidity

    coord = make_cam6_l32_levels()
    nlev = coord.n_levels

    def pres(ps):
        # strongly-typed float64 p_s, as the bridge receives the state's p_s
        p_s = jnp.asarray([ps], dtype=jnp.float64)
        return (np.asarray(coord.pressure_at_half(p_s))[0],
                np.asarray(coord.pressure_at_full(p_s))[0])

    fx = np.load(REAL_COLS)
    cols = []
    for j in range(3):
        ph, pf = pres(float(fx["p_half32"][j][-1]))
        cols.append(dict(name=f"real_cell{int(fx['cols32'][j])}", T=fx["T32"][j], q=fx["q32"][j],
                         p_half=ph, p_full=pf, land=[0.0, 0.3, 1.0][j],
                         pblh=[700.0, 500.0, 1200.0][j]))

    def make(name, ps, Ts, lapse_kmk, Ttrop, rh_fn, land, pblh, inversion=None, zero_q_above=None):
        ph, pf = pres(ps)
        z = -7500.0 * np.log(pf / ps)
        T = np.maximum(Ts - lapse_kmk * 1e-3 * z, Ttrop)
        ztrop = (Ts - Ttrop) / (lapse_kmk * 1e-3)
        T = np.where(z > ztrop, Ttrop + 1.5e-3 * (z - ztrop), T)
        if inversion is not None:
            p_bot, p_top, dT = inversion
            T = T + np.where(pf <= p_top, dT, np.where(pf < p_bot, dT * (p_bot - pf) / (p_bot - p_top), 0.0))
        q = rh_fn(pf / ps) * np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(pf)))
        q = np.where(pf < 1.0e4, np.minimum(q, 3.0e-6), q)
        if zero_q_above is not None:
            q = np.where(pf < zero_q_above, 0.0, q)
        return dict(name=name, T=T, q=q, p_half=ph, p_full=pf, land=land, pblh=pblh)

    cols += [
        make("trop_ocean_deep", 1.0110e5, 300.5, 6.2, 192.0,
             lambda s: np.clip(0.30 + 0.55 * s ** 2, 0.0, 0.9), 0.0, 600.0),
        make("midlat_land_summer", 0.9850e5, 304.0, 7.2, 215.0,
             lambda s: np.clip(0.25 + 0.45 * s ** 3, 0.0, 0.9), 1.0, 1800.0),
        make("cin_capped", 1.0130e5, 299.0, 6.0, 200.0,
             lambda s: np.where(s > 0.88, 0.92, 0.25), 0.5, 900.0, inversion=(8.9e4, 8.4e4, 3.5)),
        make("polar_dry", 0.9900e5, 254.0, 4.0, 212.0, lambda s: 0.8 * np.ones_like(s), 1.0, 300.0),
        make("desert_high_lcl", 0.9500e5, 316.0, 9.0, 213.0, lambda s: 0.08 + 0.05 * s, 1.0, 2500.0),
        make("trop_zero_q_aloft", 1.0080e5, 300.0, 6.3, 193.0,
             lambda s: np.clip(0.35 + 0.5 * s ** 2, 0.0, 0.9), 0.2, 450.0, zero_q_above=3.0e4),
        make("trop_ocean_strong", 1.0120e5, 302.5, 6.0, 190.0,
             lambda s: np.where(s > 0.85, 0.92, 0.45 + 0.35 * s), 0.0, 500.0),
        make("midlat_supercell", 0.9700e5, 307.0, 8.0, 213.0,
             lambda s: np.where(s > 0.8, 0.8, 0.3), 1.0, 2000.0),
        make("monsoon_land", 0.9900e5, 305.0, 6.8, 195.0,
             lambda s: np.clip(0.55 + 0.35 * s, 0.0, 0.92), 0.8, 1000.0),
        make("nocturnal_elevated", 0.9800e5, 289.0, 6.5, 205.0,
             lambda s: np.where((s < 0.93) & (s > 0.78), 0.95, np.where(s >= 0.93, 0.6, 0.4)),
             1.0, 150.0, inversion=(0.98e5, 0.90e5, 9.0)),
        make("cold_deep_ocean", 1.0000e5, 276.0, 7.5, 210.0,
             lambda s: np.clip(0.5 + 0.45 * s, 0.0, 0.95), 0.0, 400.0),
        make("all_snow_land", 0.9900e5, 272.5, 7.8, 212.0,
             lambda s: np.clip(0.55 + 0.42 * s, 0.0, 0.97), 1.0, 350.0),
    ]

    ncol = len(cols)
    out = {k: np.zeros((ncol, nlev)) for k in ("T", "q", "p_full", "u", "v", "cld")}
    out["p_half"] = np.zeros((ncol, nlev + 1))
    out["land"] = np.zeros(ncol)
    out["pblh"] = np.zeros(ncol)
    for i, c in enumerate(cols):
        for k in ("T", "q", "p_full", "p_half"):
            out[k][i] = c[k]
        s = c["p_full"] / c["p_half"][-1]
        out["u"][i] = (-4.0 + 3.0 * i) + 28.0 * (1.0 - s) ** 1.5 * (1 if i % 2 == 0 else -1)
        out["v"][i] = 2.0 * np.sin(np.pi * s) * (i - 3) + 6.0 * (1.0 - s)
        pf = c["p_full"]
        out["cld"][i] = np.clip(0.3 * np.exp(-((pf - 3.0e4) / 1.5e4) ** 2)
                                + 0.5 * np.exp(-((pf - 8.5e4) / 5.0e3) ** 2), 0.0, 1.0)
        out["land"][i] = c["land"]
        out["pblh"][i] = c["pblh"]
    pf, s = out["p_full"], out["p_full"] / out["p_half"][:, -1:]
    out["qc"] = 2.0e-4 * out["cld"]                                   # kg/kg
    out["qi"] = 5.0e-5 * out["cld"]                                   # kg/kg
    out["nc"] = 5.0e7 * out["cld"] * s                                # 1/kg
    out["ni"] = 2.0e5 * out["cld"] * (1.0 - s)                        # 1/kg
    out["qr"] = 2.0e-5 * np.exp(-((pf - 7.0e4) / 1.5e4) ** 2)         # kg/kg
    out["qs"] = 1.0e-5 * np.exp(-((pf - 4.0e4) / 1.0e4) ** 2)         # kg/kg
    out["nr"] = 1.0e4 * np.exp(-((pf - 7.5e4) / 1.0e4) ** 2)          # 1/kg
    out["p_s"] = out["p_half"][:, -1].copy()
    # exactly what the hydrostatic/MPAS bridge hands zm_convr as pref_edge
    out["pref_edge"] = np.asarray(coord.pressure_at_half(jnp.asarray(constants.p_ref, dtype=jnp.float64)))
    out["names"] = np.array([c["name"] for c in cols])
    return out


def heights(T, q, p_full, p_half):
    """The port's own z / zf (zhang_mcfarlane_convection builds them the same way)."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.convection._zm_cam6 import Q_MIN_VAPOR
    from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry
    qf = jnp.maximum(jnp.asarray(q), Q_MIN_VAPOR)
    dz, _, z = compute_column_geometry(jnp.asarray(T), jnp.asarray(p_full), jnp.asarray(p_half), q_v=qf)
    zf = jnp.concatenate([jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((T.shape[0], 1))], axis=1)
    return np.asarray(z), np.asarray(zf)


def write_oracle_input(c, z, zf, path):
    from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
    cfg = ZhangMcFarlaneConfig()
    cm, sm, tol, eps = VARIANT
    ncol, nlev = c["T"].shape
    fmt = lambda a: " ".join(f"{x:.17e}" for x in np.ravel(a))  # noqa: E731
    L = [f"{ncol} {nlev}", f"{DT:.17e}", f"{cm} {sm} {tol:.17e} {eps:.17e}",
         f"{cfg.c0_lnd:.17e} {cfg.c0_ocn:.17e} {cfg.ke:.17e} 1.0e-5 {cfg.momcu:.17e} "
         f"{cfg.momcd:.17e} {int(cfg.num_cin)}",
         fmt(c["pref_edge"])]
    for i in range(ncol):
        L += [f"{c['land'][i]:.17e} {c['pblh'][i]:.17e} 0.0",
              fmt(c["T"][i]), fmt(np.maximum(c["q"][i], 1e-12)), fmt(c["p_full"][i]),
              fmt(c["p_half"][i]), fmt(np.diff(c["p_half"][i])), fmt(z[i]), fmt(zf[i]),
              fmt(c["u"][i]), fmt(c["v"][i]), fmt(c["cld"][i])]
        L += [fmt(c[key][i]) for key, _, _ in TRACERS]
    open(path, "w").write("\n".join(L) + "\n")


def read_oracle(path):
    d = {}
    toks = open(path).read().split("\n")
    i = 0
    while i < len(toks):
        ln = toks[i].split()
        if not ln:
            i += 1
            continue
        if ln[0] == "DIMS":
            d["ncol"], d["nlev"], d["lengath"] = map(int, ln[1:4])
            i += 1
        elif ln[0] == "LIMCNV":
            d["limcnv"] = int(ln[1])
            i += 1
        elif ln[0] in ("F1", "I1"):
            n = int(ln[2])
            d[ln[1]] = np.array([float(x) for x in toks[i + 1:i + 1 + n]])
            i += 1 + n
        elif ln[0] == "F2":
            n, m = int(ln[2]), int(ln[3])
            d[ln[1]] = np.array([float(x) for x in toks[i + 1:i + 1 + n * m]]).reshape(n, m)
            i += 1 + n * m
        else:
            raise ValueError(f"unparsed oracle output line: {ln}")
    return d


# Oracle outputs kept in the fixture (the production-contract fields).
KEEP = ("cape", "prec", "snow", "mb", "qtnd", "heat", "dlf", "ql", "mcon", "evap_tend_s",
        "evap_tend_q", "ntprprd", "dudt", "dvdt", "seten", "lcl", "lel", "mx_buoy") + tuple(
            out for _, _, out in TRACERS)


def main(exe, workdir, cam_dir):
    c = build_columns()
    z, zf = heights(c["T"], c["q"], c["p_full"], c["p_half"])
    inp = os.path.join(workdir, "oracle_in.txt")
    outp = os.path.join(workdir, "oracle_out.txt")
    write_oracle_input(c, z, zf, inp)
    r = subprocess.run([exe, inp, outp], capture_output=True, text=True, check=True)
    print(r.stdout.strip())
    o = read_oracle(outp)
    ncol = c["T"].shape[0]
    if o["ncol"] != ncol:
        raise RuntimeError(f"oracle returned {o['ncol']} columns, expected {ncol}")
    ideep = np.zeros(ncol, bool)
    ideep[[int(i) - 1 for i in o["ideep"] if i > 0]] = True
    sha = subprocess.run(["git", "-C", cam_dir, "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    if sha != ORACLE_SHA:
        raise RuntimeError(f"CAM checkout is {sha}, expected {ORACLE_SHA} ({ORACLE_TAG})")
    md5 = hashlib.md5(open(os.path.join(cam_dir, "src/physics/cam/zm_conv.F90"), "rb").read()).hexdigest()
    md5w = hashlib.md5(open(os.path.join(cam_dir, "src/physics/cam/wv_sat_methods.F90"),
                            "rb").read()).hexdigest()
    lego_sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    meta = (f"oracle=ESCOMP/CAM tag={ORACLE_TAG} sha={sha} zm_conv.F90 md5={md5} "
            f"wv_sat_methods.F90 md5={md5w}; "
            f"variant const_mode,sat_mode,ient_tol,ient_eps={VARIANT}; dt={DT}; "
            f"generator=scripts/validate/oracle/zm_pblh/build.sh (make_fixture.py) at legoESM {lego_sha}")
    save = {k: c[k] for k in ("T", "q", "p_s", "p_full", "p_half", "u", "v", "cld", "land", "pblh",
                              "pref_edge", "names") + tuple(key for key, _, _ in TRACERS)}
    save.update({f"oracle_{k}": o[k] for k in KEEP})
    save["oracle_ideep"] = ideep
    save["oracle_limcnv"] = np.asarray(o["limcnv"])
    save["dt"] = np.asarray(DT)
    save["meta"] = np.asarray(meta)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, **save)
    print("wrote", OUT_NPZ, os.path.getsize(OUT_NPZ), "bytes; deep columns:",
          [str(n) for n, d in zip(c["names"], ideep) if d])
    print(meta)


if __name__ == "__main__":
    main(*sys.argv[1:4])
