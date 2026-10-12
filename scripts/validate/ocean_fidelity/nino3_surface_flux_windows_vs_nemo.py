#!/usr/bin/env python
"""Surface heat and momentum input to the equatorial Pacific, ours vs NEMO, per 5-day window.

NEMO: the 5-day means it wrote (SBC file: qsr_oce, qns_oce, taum, utau_ao).

Ours: the run did not store its fluxes, so they are REBUILT through the SAME
production call the tripole run makes every step (``compute_omip2_surface_forcing``,
bilinear CORE-II sampling, absolute winds as both cards run), once per 6-hourly
forcing record whose centre lies in the window, averaged.  The SST it sees is the
run's own window-mean top-level temperature (``T_mean``).  No diurnal cycle is
applied to the solar term: NEMO's ln_dm2dc is mean-preserving (sbcdcy.F90:87-143,
rscal), and OURS is CHECKED here (control 2: the run's own per-step dm2dc windows
over one day must average to 1).  The production tripole builds the bulk with
ice_albedo=None and applies the open-water albedo later, in
blend_ice_ocean_forcing(sw_partition="raw_core2", alpha_ocean=alpha_ocean_broadband)
(run_omip_core2 --prognostic-sea-ice path); the rebuild applies the same
(1 - alpha_ocean_broadband) and refuses boxes that carry any sea ice.  This rebuild is NOT the flux the model
applied step by step (instantaneous SST, ramp, ice); it is labelled as such.

CONTROL (decides whether a qns difference means anything): the same production
bulk fed NEMO's OWN window-mean SST (grid_T ``tos``).  If that reproduces NEMO's
qns_oce, the formulas and forcing sampling agree and any remaining ours-NEMO qns
gap is the SST feedback; if it does not, the gap is a forcing/bulk mismatch.

Prints, per window and box (area-weighted): qsr, qns, taum, ocean zonal stress.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parent))

BOXES = {"nino3_5S5N_210E270E": (-5.0, 5.0, 210.0, 270.0),
         "eq_2S2N_160E270E": (-2.0, 2.0, 160.0, 270.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", required=True, type=Path)
    ap.add_argument("--nemo-sbc", required=True, type=Path)
    ap.add_argument("--nemo-gridt", required=True, type=Path)
    ap.add_argument("--windows", default="0,1,2,3,4,5")
    ap.add_argument("--mesh", default=str(_REPO / "data/grids/eORCA1.2_mesh_mask.nc"))
    ap.add_argument("--forcing-path", default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()
    sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"[provenance] git {sha} | {' '.join(sys.argv)}")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm import constants
    from legoesm.ocean.forcing.core2 import load_core2_nyf
    from legoesm.ocean.coupler.omip2_applicator import (
        compute_omip2_surface_forcing, dm2dc_sw_factor)
    from so_freshwater_budget import align_output_to_mesh

    m = nc.Dataset(a.mesh)
    area = np.asarray(m["e1t"][0] * m["e2t"][0], dtype=np.float64)
    gphit = np.asarray(m["gphit"][0], dtype=np.float64)
    m.close()

    forcing = load_core2_nyf(cache_dir=Path(a.forcing_path) if a.forcing_path else None,
                             allow_synthetic=False)
    t_rec = np.asarray(forcing.time_s, dtype=np.float64)

    sbc = nc.Dataset(a.nemo_sbc); gt = nc.Dataset(a.nemo_gridt)
    lat_n = np.asarray(sbc["nav_lat"][:], dtype=np.float64)
    sj, si = align_output_to_mesh(gphit, lat_n)
    tb = np.asarray(sbc["time_centered_bounds"][:], dtype=np.float64)

    def nrec(ds, name, r):
        x = np.ma.filled(np.ma.masked_invalid(ds[name][r]), np.nan).astype(np.float64)
        out = np.full(area.shape, np.nan); out[sj, si] = x
        return out

    def boxmean(f, lat, lon, b, valid):
        la, lb, oa, ob = b
        msk = (lat >= la) & (lat <= lb) & (lon >= oa) & (lon <= ob) & valid & np.isfinite(f)
        return float(np.sum(f[msk] * area[msk]) / np.sum(area[msk]))

    # control 2: our dm2dc factor, run-identical windows (step k integrates
    # [(k-1)dt, k dt], day-of-year from the mid-step time), one full day at dt=150.
    dt = 150.0
    g0 = SimpleNamespace(lat_T=np.radians(gphit), lon_T=np.radians(np.asarray(nc.Dataset(a.mesh)["glamt"][0], dtype=np.float64)))
    facs = []
    for step in range(1, int(86400 / dt) + 1):
        t_mid = (step - 0.5) * dt
        t_lo = ((t_mid % 86400.0) - 0.5 * dt) / 86400.0
        facs.append(np.asarray(dm2dc_sw_factor(g0, (int((t_mid / 86400.0) % 365.0) + 1, 365.0, t_lo, t_lo + dt / 86400.0))))
    fm = np.mean(facs, axis=0)
    eq = np.abs(gphit) <= 5.0
    print(f"[control 2] our dm2dc factor, day-1 mean over 576 steps, |lat|<=5: min {fm[eq].min():.6f} max {fm[eq].max():.6f} (must be 1)")
    alpha_o = float(constants.alpha_ocean_broadband)

    hdr = (f"{'win(d)':>8s} {'box':20s} | {'qsr N':>7s} {'ours':>7s} | {'qns N':>7s} {'bulk@N':>7s} "
           f"{'ours':>7s} | {'taum N':>7s} {'bulk@N':>7s} {'ours':>7s} | {'utau N':>7s} {'ours':>7s}")
    print(hdr)
    rec = {}
    for k in (int(x) for x in a.windows.split(",")):
        d0, d1 = 5.0 * k, 5.0 * k + 5.0
        win_n = ((tb[k, 0] - tb[0, 0]) / 86400.0, (tb[k, 1] - tb[0, 0]) / 86400.0)
        if abs(win_n[0] - d0) > 1e-6 or abs(win_n[1] - d1) > 1e-6:
            raise SystemExit(f"NEMO record {k} spans days {win_n}, not {d0}-{d1}")
        f = a.snapshot_dir / ("snapshot_final.npz" if d1 == 30.0 else f"snapshot_day{int(d1):04d}.npz")
        if not f.exists():
            print(f"{d0:4.0f}-{d1:<3.0f} (our snapshot {f.name} missing, skipped)")
            continue
        z = np.load(f)
        if abs(float(z["time_days"]) - d1) > 1e-6 or abs(float(z["flux_mean_window_s"]) - 5 * 86400.0) > 1.0:
            raise SystemExit(f"{f.name}: time {float(z['time_days'])}, window {float(z['flux_mean_window_s'])} s")
        lat = np.asarray(z["lat_T"], dtype=np.float64); lon = np.asarray(z["lon_T"], dtype=np.float64) % 360.0
        wet = np.asarray(z["land_mask"], dtype=np.float64) > 0.5
        ice = np.asarray(z["ice_concentration"], dtype=np.float64)
        grid = SimpleNamespace(lat_T=np.radians(lat), lon_T=np.radians(lon))
        sst_o = np.asarray(z["T_mean"][..., 0], dtype=np.float64)
        tos_n = nrec(gt, "tos", k)
        sst_c = np.where(np.isfinite(tos_n), tos_n, sst_o)
        recs = np.where((t_rec >= d0 * 86400.0) & (t_rec < d1 * 86400.0))[0]
        acc = {s: {v: 0.0 for v in ("qsr", "qns", "taum", "utau")} for s in ("ours", "bulkN")}
        for r in recs:
            for s, sst in (("ours", sst_o), ("bulkN", sst_c)):
                st = SimpleNamespace(T=SimpleNamespace(data=sst[..., None]))
                sf = compute_omip2_surface_forcing(st, forcing=forcing, idx_t=int(r), grid=grid,
                                                   grid_type="tripole")
                sw = np.asarray(sf.sw_down); qn = np.asarray(sf.q_net)
                tx = np.asarray(sf.tau_x); ty = np.asarray(sf.tau_y)
                acc[s]["qsr"] += sw * (1.0 - alpha_o); acc[s]["qns"] += qn - sw
                acc[s]["taum"] += np.hypot(tx, ty); acc[s]["utau"] += tx
        for s in acc:
            for v in acc[s]:
                acc[s][v] = acc[s][v] / len(recs)
        N = {v: nrec(sbc, n, k) for v, n in (("qsr", "qsr_oce"), ("qns", "qns_oce"),
                                               ("taum", "taum"), ("utau", "utau_ao"))}
        valid = wet & np.isfinite(N["qns"])
        for bn, b in BOXES.items():
            g = lambda fld: boxmean(fld, lat, lon, b, valid)
            if g(ice) > 0.0:
                raise SystemExit(f"{bn}: sea ice present ({g(ice):.3g}); the albedo rebuild is open-water only")
            rec.setdefault(bn, {})[str(k)] = {
                "nemo": {v: g(N[v]) for v in N},
                "ours": {v: g(acc["ours"][v]) for v in acc["ours"]},
                "bulk_at_nemo_sst": {v: g(acc["bulkN"][v]) for v in acc["bulkN"]}}
            print(f"{d0:4.0f}-{d1:<3.0f} {bn:20s} | {g(N['qsr']):7.2f} {g(acc['ours']['qsr']):7.2f} | "
                  f"{g(N['qns']):7.2f} {g(acc['bulkN']['qns']):7.2f} {g(acc['ours']['qns']):7.2f} | "
                  f"{g(N['taum']):7.4f} {g(acc['bulkN']['taum']):7.4f} {g(acc['ours']['taum']):7.4f} | "
                  f"{g(N['utau']):7.4f} {g(acc['ours']['utau']):7.4f}")
        print(f"         SST box means nino3: NEMO tos {boxmean(tos_n, lat, lon, BOXES['nino3_5S5N_210E270E'], valid):.3f} "
              f"ours T_mean(k=0) {boxmean(sst_o, lat, lon, BOXES['nino3_5S5N_210E270E'], valid):.3f} C; "
              f"{len(recs)} forcing records")
    print("utau: NEMO utau_ao = stress ON the ocean; ours = the bulk's tau_x as returned "
          "(ATMOSPHERIC convention per compute_omip2_surface_forcing docstring) -- compare signs before magnitudes.")
    if a.json_out:
        import json
        a.json_out.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
