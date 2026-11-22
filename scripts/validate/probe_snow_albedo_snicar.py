"""Probe: spring snow albedo of the production land vs the CLM5.0 SNICAR oracle,
clean and with CLM5.0's own spring BC + dust loading, 45-70N land, MAM.

Three instruments, one table:
  1. CESM2 CMIP6 historical (CLM5.0 = the CAM6 land) monthly history, MAM 1979-1983,
     land 45-70N, full-snow cell-months: flux-weighted snow-only albedo
     (SNOFSR* / FSDS*), the SW weights, mean FSDS, and the top-layer / lower-layer
     BC and dust concentrations in snow.
  2. The verbatim CLM5.0 SNICAR oracle (scripts/validate/clm50_snicar_oracle.F90:
     CESM2.1.5 SnowSnicarMod compiled unmodified) driving a two-layer pack through a
     synthetic winter -> spring -> melt temperature history: CLM5.0 grain growth, then
     SNICAR_RT VIS/NIR direct+diffuse albedo for clean and dirty snow.
  3. The production land snow albedo (LandAlbedoConfig with the calibration scalars,
     150-day melt-equivalent e-folding, 5000 K temperature clock -- the resolved
     amip_production values) aged by update_snow_age through the SAME history.

PRE-REGISTERED (2026-11-22, written before the first run).  Scored on the
spring-stage mean (days 61-105 of the history), cosz 0.45, broadband with the
CLM5.0 history's own VIS/NIR x direct/diffuse SW weights for both models:
  CONFIRMS soot/dust matters:  SNICAR(clean) - SNICAR(dirty) >= 0.03
                               AND ours - SNICAR(dirty)      >= 0.03
  REFUTES:                     |ours - SNICAR(dirty)|        <= 0.02
Anything else: inconclusive.

Run on a compute node (needs netcdf-fortran and the compiler that built it; FC, default ifx).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants

REPO = Path(__file__).resolve().parents[2]
CESM_SNICAR = Path("/glade/u/home/branson/cesm2.1.5/components/clm/src/biogeophys/SnowSnicarMod.F90")
INPUT = Path("/glade/campaign/cesm/cesmdata/inputdata")
OPTICS = INPUT / "lnd/clm2/snicardata/snicar_optics_5bnd_c090915.nc"     # clm5_0 default
AGING = INPUT / "lnd/clm2/snicardata/snicar_drdt_bst_fit_60_c070416.nc"
AERDEP = INPUT / "atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1849-2015_0.9x1.25_CMIP6_c180924.nc"
HIST = Path("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
            "b.e21.BHIST.f09_g17.CMIP6-historical.001/lnd/proc/tseries/month_1")
HPRE = "b.e21.BHIST.f09_g17.CMIP6-historical.001.clm2.h0"
MONTH_END = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])   # noleap

YEARS = (1979, 1983)
LAT = (45.0, 70.0)
MIN_FSNO = 0.9
MIN_LANDFRAC = 0.5
MAX_SWE = 1000.0          # kg/m2 per cell-month: drops ice-sheet / glacier-capped columns
RHO_SNOW = 250.0          # kg/m3, sets layer dz only (CLM aging reads density from it)
DT = 1800.0
COSZ = (0.3, 0.45, 0.6)
COSZ_PRIMARY = 0.45
# (days, T_top [K], dT/dz [K/m], liquid mass fraction, forcing air T [K]).
# T_top is the top-layer centre temperature AND the temperature our age clock sees;
# deeper layers are set so CLM's own diagnosed top-layer gradient equals dT/dz.
SCHEDULE = ((60, 255.0, 20.0, 0.0, 255.0),                     # winter, cold, dry
            (30, 265.0, 10.0, 0.0, 265.0),                     # early spring, dry
            (15, constants.T_freeze, 0.0, 0.05, 275.0))        # melting
SPRING_DAYS = (61, 105)
FORC_T0 = 255.0           # air T at the snowfall that built the pack (fresh radius)


def _mam(tdays, year0):
    yr = year0 + np.floor(tdays / 365.0).astype(int)
    mo = 1 + np.searchsorted(MONTH_END, tdays % 365.0, side="right")
    return (yr >= YEARS[0]) & (yr <= YEARS[1]) & np.isin(mo, (3, 4, 5))


def hist_stats() -> dict:
    import xarray as xr
    names = ["FSNO", "SNOFSRVD", "SNOFSRVI", "SNOFSRND", "SNOFSRNI",
             "FSDSVD", "FSDSVI", "FSDSND", "FSDSNI", "FSDS",
             "SNOBCMSL", "SNOBCMCL", "SNODSTMSL", "SNODSTMCL", "H2OSNO_TOP", "H2OSNO"]
    d = {}
    units = {"SNOBCMSL": "kg/m2", "SNOBCMCL": "kg/m2", "SNODSTMSL": "kg/m2",
             "SNODSTMCL": "kg/m2", "H2OSNO_TOP": "kg/m2", "H2OSNO": "mm"}
    for v in names:
        ds = xr.open_dataset(HIST / f"{HPRE}.{v}.185001-201412.nc", decode_times=False)
        if ds["time"].attrs.get("calendar") != "noleap":
            raise SystemExit(f"{v}: calendar is not noleap")
        if v in units and ds[v].attrs.get("units") != units[v]:
            raise SystemExit(f"{v}: units {ds[v].attrs.get('units')!r} != {units[v]!r}")
        tb = ds["time_bounds"].values
        sel = _mam(tb[:, 0], 1850)                                 # interval START date
        if not sel.any():
            raise SystemExit(f"no MAM {YEARS} months in {v}")
        d[v] = ds[v].isel(time=np.where(sel)[0]).values.astype(np.float64)
        dur = (tb[sel, 1] - tb[sel, 0]).astype(np.float64)        # days in each month
        if v == "FSNO":
            lat = ds["lat"].values
            area = ds["area"].values.astype(np.float64)
            lfrac = ds["landfrac"].values.astype(np.float64)
    band = (lat >= LAT[0]) & (lat <= LAT[1])
    w2 = np.where(band[:, None] & (lfrac >= MIN_LANDFRAC), area * lfrac, 0.0)
    keys = (("vd", "SNOFSRVD", "FSDSVD"), ("vi", "SNOFSRVI", "FSDSVI"),
            ("nd", "SNOFSRND", "FSDSND"), ("ni", "SNOFSRNI", "FSDSNI"))
    m = (d["FSNO"] >= MIN_FSNO) & (w2[None] > 0) & (d["H2OSNO"] <= MAX_SWE)
    for v in names:
        m &= np.isfinite(d[v])
    w = np.where(m, w2[None] * dur[:, None, None], 0.0)          # area x month length
    if w.sum() == 0:
        raise SystemExit("empty mask")

    def wsum(x):
        return float((np.where(m, x, 0.0) * w).sum())

    out = {"n_cell_months": int(m.sum()),
           "alb_note": "SNOFSR*/FSDS* proxy: SNOFSR* is averaged only over snow-layer "
                       "times, FSDS* over all times; diagnostic, not scored"}
    tot_down = sum(wsum(d[f]) for _, _, f in keys)
    out["alb"] = {k: wsum(d[r]) / wsum(d[f]) for k, r, f in keys}
    out["alb"]["broadband"] = sum(wsum(d[r]) for _, r, _ in keys) / tot_down
    out["sw_weight"] = {k: wsum(d[f]) / tot_down for k, _, f in keys}
    out["fsds_mean_W_m2"] = wsum(d["FSDS"]) / float(w.sum())
    top, col = wsum(d["H2OSNO_TOP"]), wsum(d["H2OSNO"])
    out["swe_mean_kg_m2"] = col / float(w.sum())
    out["swe_top_mean_kg_m2"] = top / float(w.sum())
    for sp, a, b in (("bc", "SNOBCMSL", "SNOBCMCL"), ("dst", "SNODSTMSL", "SNODSTMCL")):
        out[f"{sp}_top_kg_kg"] = wsum(d[a]) / top
        out[f"{sp}_lower_kg_kg"] = (wsum(d[b]) - wsum(d[a])) / (col - top)
    return out


def dep_split() -> dict:
    """Deposition-weighted species fractions over the probe's land domain.  An
    ASSUMED snowpack composition (CLM scavenges species differently); the
    all-hydrophilic / all-hydrophobic BC bracket is reported alongside."""
    import xarray as xr
    ds = xr.open_dataset(AERDEP, decode_times=False)
    date = ds["date"].values.astype(int)                           # yyyymmdd
    yr, mo = date // 10000, (date // 100) % 100
    sel = np.where((yr >= YEARS[0]) & (yr <= YEARS[1]) & np.isin(mo, (3, 4, 5)))[0]
    if sel.size != 3 * (YEARS[1] - YEARS[0] + 1):
        raise SystemExit(f"aerosol deposition file: {sel.size} MAM months selected")
    h = xr.open_dataset(HIST / f"{HPRE}.FSNO.185001-201412.nc", decode_times=False)
    if not np.allclose(h["lat"].values, ds["lat"].values) or not np.allclose(h["lon"].values, ds["lon"].values):
        raise SystemExit("deposition grid differs from the history grid")
    lat = ds["lat"].values
    band = (lat >= LAT[0]) & (lat <= LAT[1])
    lf = h["landfrac"].values.astype(np.float64)
    wl = np.where(band[:, None] & (lf >= MIN_LANDFRAC), h["area"].values * lf, 0.0)

    def tot(v):
        x = ds[v].isel(time=sel).values.mean(axis=0)
        if not np.all(np.isfinite(x)) or np.any(x < 0):
            raise SystemExit(f"{v}: non-finite or negative deposition")
        return float((x * wl).sum())

    phi = tot("BCDEPWET") + tot("BCPHIDRY")
    pho = tot("BCPHODRY")
    dst = [tot(f"DSTX0{i}DD") + tot(f"DSTX0{i}WD") for i in range(1, 5)]
    return {"bc_phi_frac": phi / (phi + pho), "dst_bin_frac": [x / sum(dst) for x in dst]}


def build_oracle(work: Path) -> Path:
    (work / "shr_assert.h").write_text(
        "#define SHR_ASSERT(assert, msg)\n#define SHR_ASSERT_ALL(assert, msg)\n"
        "#define SHR_ASSERT_ALL_EXT(assert, msg)\n")
    ff = subprocess.check_output(["nf-config", "--fflags"], text=True).split()
    fl = subprocess.check_output(["nf-config", "--flibs"], text=True).split()
    src = REPO / "scripts/validate/clm50_snicar_oracle.F90"
    head, body = src.read_text().split("program clm50_snicar_oracle", 1)
    (work / "stubs.F90").write_text(head)
    (work / "main.F90").write_text("program clm50_snicar_oracle" + body)
    # netcdf-fortran's .mod files must come from the same compiler (Derecho: Intel ifx)
    fc = os.environ.get("FC", "ifx")
    flags = (["-O0", "-fpp", "-module", str(work)] if fc in ("ifx", "ifort")
             else ["-O0", "-cpp", "-ffree-line-length-none", "-J", str(work)])
    common = [fc] + flags + [f"-I{work}"] + ff
    for f in (work / "stubs.F90", CESM_SNICAR, work / "main.F90"):
        subprocess.check_call(common + ["-c", str(f), "-o", str(work / (f.stem + ".o"))], cwd=work)
    exe = work / "clm50_snicar_oracle"
    subprocess.check_call([fc, "-o", str(exe), str(work / "stubs.o"),
                           str(work / "SnowSnicarMod.o"), str(work / "main.o")] + fl, cwd=work)
    return exe


def layer_temps(T_top, g, swe_top, swe_low):
    """Layer-centre temperatures [top, lower, soil] such that CLM5.0 SnowAge_grain's
    diagnosed top-layer gradient |t1 - (t2 dz1 + t1 dz2)/(dz1+dz2)| / dz1 equals g
    (it reduces to |t1 - t2| / (dz1 + dz2)).  The soil node sits at the lower-layer
    temperature: the gradient is imposed inside the snow only (vote 2026-11-22, a/a/a).
    Dry segments must keep the snow layers below freezing."""
    dz1, dz2 = swe_top / RHO_SNOW, swe_low / RHO_SNOW
    t2 = T_top + g * (dz1 + dz2)
    return T_top, t2, t2


def run_oracle(exe: Path, swe_top: float, swe_low: float, cases: list) -> np.ndarray:
    lines = [str(OPTICS), str(AGING), f"{DT} 2 {FORC_T0} {int(86400 / DT)}",
             f"{swe_top} {swe_low}", f"{swe_top / RHO_SNOW} {swe_low / RHO_SNOW}",
             "0.2 0.3", str(len(cases))]
    for top, low in cases:
        lines.append(" ".join(f"{x:.6e}" for x in list(top) + list(low)))
    lines += [str(len(COSZ)), " ".join(map(str, COSZ)), str(len(SCHEDULE))]
    for nd, T, g, lq, ft in SCHEDULE:
        lines.append(f"{int(nd * 86400 / DT)} {lq} {ft} "
                     + " ".join(f"{t:.6f}" for t in layer_temps(T, g, swe_top, swe_low)))
    p = subprocess.run([str(exe)], input="\n".join(lines) + "\n", text=True, capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"oracle failed ({p.returncode}): {p.stderr[-2000:]}")
    rows = np.array([[float(x) for x in ln.split()] for ln in p.stdout.splitlines() if ln.strip()])
    ndays = sum(s[0] for s in SCHEDULE)
    ncol = 2 + len(cases) * (2 * len(COSZ) + 2)
    if rows.ndim != 2 or rows.shape != (ndays + 1, ncol):
        raise SystemExit(f"oracle output shape {rows.shape}, expected {(ndays + 1, ncol)}")
    if not np.array_equal(rows[:, 0], np.arange(ndays + 1) * int(86400 / DT)):
        raise SystemExit("oracle output steps are not one per day")
    return rows


def check_results(res: dict, r_top: np.ndarray, ours: np.ndarray) -> None:
    """Fatal instrument checks: finite, albedo in [0, 1], impurities only darken,
    grain radius non-decreasing with no snowfall."""
    for n, r in res.items():
        for b, x in r.items():
            if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
                raise SystemExit(f"self-check: {n}/{b} non-finite or outside [0, 1]")
    if not np.all(np.isfinite(ours)) or np.any(ours < 0) or np.any(ours > 1):
        raise SystemExit("self-check: production albedo non-finite or outside [0, 1]")
    for n in res:
        if n == "clean":
            continue
        for b in ("dir_vis", "dir_nir", "dif_vis", "dif_nir"):
            if np.any(res[n][b] > res["clean"][b] + 1e-9):
                raise SystemExit(f"self-check: {n} brighter than clean in {b}")
    if not np.all(np.isfinite(r_top)) or np.any(np.diff(r_top) < -1e-9):
        raise SystemExit("self-check: grain radius non-finite or decreased with no new snow")


def ours_daily():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.land.clm_surface_map import biophysics_lmip_albedo_scalars
    from legoesm.land.snow_budget import update_snow_age
    from legoesm.surface_albedo import LandAlbedoConfig, snow_albedo

    # resolved amip_production: calibration scalars, then land_snow_tau_days=150 and
    # snow_age_activation_K=5000 applied last (model_driver order)
    cfg = LandAlbedoConfig()._replace(**biophysics_lmip_albedo_scalars())
    cfg = cfg._replace(tau_snow_decay=150.0 * 86400.0, snow_age_activation_K=5000.0)
    swe = jnp.full(1, 100.0)
    age = jnp.zeros(1)
    alb = [float(snow_albedo(age, cfg)[0])]
    step = jax.jit(lambda a, T: update_snow_age(swe, a, jnp.zeros(1), DT, T_snow=T,
                                                age_activation_K=cfg.snow_age_activation_K))
    for nd, T, *_ in SCHEDULE:
        Tv = jnp.full(1, T)
        for _ in range(nd):
            for _ in range(int(86400 / DT)):
                age = step(age, Tv)
            alb.append(float(snow_albedo(age, cfg)[0]))
    return np.asarray(alb), cfg


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from legoesm.land.canopy.radiative_transfer import broadband_albedo

    h = hist_stats()
    sp = dep_split()
    swe_top = h["swe_top_mean_kg_m2"]
    swe_low = h["swe_mean_kg_m2"] - swe_top
    if not (swe_top > 0 and swe_low > 0):
        raise SystemExit(f"non-positive layer mass: top {swe_top}, lower {swe_low}")
    for nd, T, g, lq, ft in SCHEDULE:
        t = layer_temps(T, g, swe_top, swe_low)
        if lq == 0.0 and max(t[:2]) >= constants.T_freeze:
            raise SystemExit(f"dry segment reaches freezing: {t}")
    phi = sp["bc_phi_frac"]

    def conc(bc, dst, f_phi=phi):
        c = [bc * f_phi, bc * (1 - f_phi), 0.0, 0.0] + [dst * f for f in sp["dst_bin_frac"]]
        if not all(np.isfinite(c)) or min(c) < 0:
            raise SystemExit(f"bad concentration {c}")
        return c

    zero = [0.0] * 8
    T_, L_ = (h["bc_top_kg_kg"], h["dst_top_kg_kg"]), (h["bc_lower_kg_kg"], h["dst_lower_kg_kg"])
    cases = {
        "clean": (zero, zero),
        "dirty": (conc(*T_), conc(*L_)),
        "dirty_bc_all_phi": (conc(*T_, 1.0), conc(*L_, 1.0)),
        "dirty_bc_all_pho": (conc(*T_, 0.0), conc(*L_, 0.0)),
        "bc_only": (conc(T_[0], 0.0), conc(L_[0], 0.0)),
        "dust_only": (conc(0.0, T_[1]), conc(0.0, L_[1])),
    }
    with tempfile.TemporaryDirectory() as td:
        rows = run_oracle(build_oracle(Path(td)), swe_top, swe_low, list(cases.values()))
    fc = os.environ.get("FC", "ifx")
    fc_ver = subprocess.run([fc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]

    w = h["sw_weight"]
    nz = len(COSZ)
    per = 2 * nz + 2
    res, bb_cz = {}, {}
    for k, name in enumerate(cases):
        blk = rows[:, 2 + k * per: 2 + (k + 1) * per]
        fvis, fnir = blk[:, 2 * nz], blk[:, 2 * nz + 1]
        bbz = {cz: (w["vd"] * blk[:, 2 * j] + w["nd"] * blk[:, 2 * j + 1]
                    + w["vi"] * fvis + w["ni"] * fnir) for j, cz in enumerate(COSZ)}
        j = COSZ.index(COSZ_PRIMARY)
        res[name] = {"dir_vis": blk[:, 2 * j], "dir_nir": blk[:, 2 * j + 1],
                     "dif_vis": fvis, "dif_nir": fnir, "bb": bbz[COSZ_PRIMARY]}
        bb_cz[name] = bbz
    ours, cfg = ours_daily()
    ours_internal = np.asarray(broadband_albedo(ours, ours))
    check_results(res, rows[:, 1], ours)

    d0, d1 = SPRING_DAYS
    S = lambda x: float(np.mean(x[d0:d1 + 1]))
    fsds = h["fsds_mean_W_m2"]
    clean, dirty, o = S(res["clean"]["bb"]), S(res["dirty"]["bb"]), S(ours)
    darkening, gap = clean - dirty, o - dirty
    verdict = ("CONFIRMS" if (darkening >= 0.03 and gap >= 0.03)
               else "REFUTES" if abs(gap) <= 0.02 else "INCONCLUSIVE")
    bands = lambda n: {b: S(res[n][b]) for b in ("dir_vis", "dir_nir", "dif_vis", "dif_nir", "bb")}
    summary = {
        "provenance": {
            "git_sha": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
            "sha256": {str(f): _sha(f) for f in (Path(__file__), REPO / "scripts/validate/clm50_snicar_oracle.F90",
                                               CESM_SNICAR, OPTICS, AGING)},
            "compiler": fc_ver, "oracle_source": str(CESM_SNICAR), "optics": str(OPTICS),
            "aging": str(AGING), "history": f"{HIST}/{HPRE}.*", "aerdep": str(AERDEP),
            "years": YEARS, "lat": LAT, "min_fsno": MIN_FSNO, "min_landfrac": MIN_LANDFRAC,
            "max_swe": MAX_SWE, "rho_snow": RHO_SNOW, "dt": DT, "schedule": SCHEDULE,
            "layer_temps_K": [layer_temps(T, g, swe_top, swe_low) for _, T, g, *_ in SCHEDULE],
            "spring_days": SPRING_DAYS, "cosz_primary": COSZ_PRIMARY, "forc_t0": FORC_T0,
            "ours_cfg": {k: float(getattr(cfg, k)) for k in
                         ("alpha_snow_max", "alpha_snow_min", "tau_snow_decay",
                          "snow_age_activation_K")}},
        "cesm2_history": h, "dep_split": sp,
        "spring_mean": {"ours_band": o, "ours_model_broadband": S(ours_internal),
                        "snicar": {n: bands(n) for n in cases},
                        "snicar_bb_by_cosz": {n: {str(cz): S(v) for cz, v in bb_cz[n].items()}
                                              for n in cases},
                        "r_top_um": [float(rows[d0, 1]), float(rows[d1, 1])]},
        "fresh_day0": {"ours": float(ours[0]), "snicar_clean_bb": float(res["clean"]["bb"][0]),
                       "snicar_dirty_bb": float(res["dirty"]["bb"][0]),
                       "r_top_um": float(rows[0, 1])},
        "darkening_clean_minus_dirty": darkening, "darkening_W_m2": darkening * fsds,
        "gap_ours_minus_dirty": gap, "gap_W_m2": gap * fsds,
        "W_m2_note": "albedo difference x CESM2 mean FSDS over the mask: an equivalent "
                     "snow-surface SW difference, not a diagnosed regional forcing",
        "verdict": verdict,
        "verdict_scope": "synthetic snowfall-free snow-only column; not the production "
                         "cell albedo (no cover blending, canopy, or fresh-snow events)",
        "daily": {"r_top_um": rows[:, 1].tolist(), "ours": ours.tolist(),
                  **{f"snicar_{n}_bb": res[n]["bb"].tolist() for n in cases}},
    }
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "daily"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
