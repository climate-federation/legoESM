"""Last-window heat residual split by term, row-1 run vs baseline (offline).

V: net volume change (snapshot eta) and the part of dH carried by thickness
   change, dH = rho cp sum(Tbar*dh + hbar*dT) (exact for two endpoints).
   Known answer: synthetic eta += 1 cm -> dV = 0.01 m * wet area exactly.
F: freshwater that enters at a non-zero temperature (P-E enter at 0 degC
   and carry none): runoff at max(SST,0) from the runner's own loader,
   averaged over the window's steps with its own month sampler; ice melt
   and the global normalisation removal at SST, from the run's daily
   [fwbudget] lines.  Known answer: loader runoff total at day 30 must equal
   the log's R.
APPROXIMATIONS (reviewers, 2026-11-15): F terms are products of window means
(flux x mean SST, no covariance), runoff at mean SST not its spread layers or
the model's smooth floor, ice heat at area-mean SST of end-of-window ice
cells, and the daily log lines are instantaneous and include the two
duplicated cyclic columns.  The absolute F values are estimates; the
row-minus-baseline differences rest on both runs sharing identical forcing.

Usage (all paths required; the snapshot directory and the directory holding
the run's ``run.log`` may differ when a leg was re-scored elsewhere):

    omip_row1_heat_terms.py --row-dir R --row-log-dir RL \
        --base-dir B --base-log-dir BL --domain-cfg domain_cfg.nc
"""

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

_FWB = re.compile(r"\[fwbudget\] step=(\d+) day=([\d.]+) \w+: P=([-+\d.]+) E=([-+\d.]+) "
                  r"R=([-+\d.]+) ice=([-+\d.]+) net\(P-E\+R\+ice\)=([-+\d.]+)")


def parse_fwbudget(lines):
    """The run log's daily ``[fwbudget]`` lines as regex matches, in order."""
    return [m for m in map(_FWB.search, lines) if m]


def thickness_heat_split(Ta, Tb, ha, hb, area, k, dt_win):
    """Heat-content change rate [W] split into thickness and temperature parts.

    ``H_b - H_a = k sum(Tbar (hb - ha) + hbar (Tb - Ta)) area`` exactly for two
    endpoints (bars are endpoint means); returns ``(H_dh, H_dT)``.
    """
    Tbar, hbar = 0.5 * (Ta + Tb), 0.5 * (ha + hb)
    w = np.asarray(area)[..., None]
    H_dh = k * (Tbar * (hb - ha) * w).sum() / dt_win
    H_dT = k * (hbar * (Tb - Ta) * w).sum() / dt_win
    return H_dh, H_dT


def capture(ht, run, dcfg):
    cap = {}
    real = ht.close_windows

    def spy(snaps, zc, rho_0, c_p, *a, **kw):
        cap.update(snaps=snaps, zc=zc, rho_0=rho_0, c_p=c_p)
        cap["rows"] = real(snaps, zc, rho_0, c_p, *a, **kw)
        return cap["rows"]
    ht.close_windows = spy
    try:
        ht.main(["--run-dir", run, "--domain-cfg", dcfg, "--max-residual-wm2", "1e9"])
    finally:
        ht.close_windows = real
    return cap


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, what in (("--row-dir", "row-1 run snapshot dir (run_manifest.json)"),
                       ("--row-log-dir", "dir holding the row-1 run.log"),
                       ("--base-dir", "baseline run snapshot dir"),
                       ("--base-log-dir", "dir holding the baseline run.log"),
                       ("--domain-cfg", "NEMO domain_cfg.nc")):
        p.add_argument(name, required=True, help=what)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import omip_tripole_heat_closure as ht
    import scripts.run.run_omip_core2 as rc
    from legoesm.ocean.vertical import compute_layer_thickness

    RUNS = {"row": (args.row_dir, args.row_log_dir),
            "base": (args.base_dir, args.base_log_dir)}
    DCFG = args.domain_cfg
    fin = str(Path(args.row_dir) / "snapshot_final.npz")
    lat = ht._snap_lat(fin)
    lon = ht._snap_lon(fin)
    raw = ht.load_snapshot(fin)
    _, rargs, rho0, csw = ht._manifest_args(args.row_dir)
    runoff12 = rc.load_runoff_monthly(
        SimpleNamespace(area=raw["cell_area"]), "tripole", lat, lon, rargs.mesh,
        land_mask=raw["land_mask"], spread_passes=int(rargs.runoff_spread_passes),
        exclude_isf=rargs.isf, regrid=rargs.runoff_regrid)
    runoff12 = np.asarray(runoff12, dtype=np.float64)

    out = {}
    for name, (gdir, logdir) in RUNS.items():
        cap = capture(ht, gdir, DCFG)
        a, b = cap["snaps"][-2], cap["snaps"][-1]
        dt_win = float(b["flux_mean_window_s"])
        n = int(b["flux_mean_n_steps"])
        dt = dt_win / n
        wet = b["land_mask"] > 0.5
        area = b["cell_area"] * wet
        A = area.sum()
        resid = cap["rows"][-1]["residual_Wm2"]

        # V: volume and thickness-change heat
        ha = np.asarray(compute_layer_thickness(a["eta"], a["H_bathy"], cap["zc"](a)))
        hb = np.asarray(compute_layer_thickness(b["eta"], b["H_bathy"], cap["zc"](b)))
        dV = ((hb - ha).sum(-1) * area).sum() / dt_win
        a_syn = {**a, "eta": a["eta"] + 0.01}
        hs = np.asarray(compute_layer_thickness(a_syn["eta"], a["H_bathy"], cap["zc"](a)))
        ka = ((hs - ha).sum(-1) * area).sum() / (0.01 * A)
        k = cap["rho_0"] * cap["c_p"]
        H_dh, H_dT = thickness_heat_split(a["T"], b["T"], ha, hb, area, k, dt_win)
        H_dh, H_dT = H_dh / A, H_dT / A
        dH_tool = cap["rows"][-1]["dHdt_W"] / A

        # F: runoff at max(SST,0) over the window's steps
        s0, s1 = int(a["step"]), int(b["step"])
        Rw = sum(rc.month_sample(runoff12, s, dt, rargs.forcing_time_interp)
                 for s in range(s0 + 1, s1 + 1)) / (s1 - s0)
        sst = np.asarray(np.load(Path(gdir) / "snapshot_final.npz")["T_mean"][..., 0], dtype=np.float64)
        R_end = rc.month_sample(runoff12, s1, dt, rargs.forcing_time_interp)
        R_sv_loader = float((R_end * area).sum()) / 1e9
        F_run = float((csw * Rw * np.maximum(sst, 0.0) * area).sum()) / A

        with open(Path(logdir) / "run.log") as fh:
            lines = parse_fwbudget(fh)
        win = [m for m in lines if s0 < int(m.group(1)) <= s1]
        R_log_end = next(float(m.group(5)) for m in lines if int(m.group(1)) == s1)
        ice_sv = np.mean([float(m.group(6)) for m in win])
        net_sv = np.mean([float(m.group(7)) for m in win])
        icy = wet & (b["ice_concentration"] > 0.15)
        sst_ice = float((sst * area * icy).sum() / (area * icy).sum())
        sst_glob = float((sst * area).sum() / A)
        F_ice = csw * ice_sv * 1e9 * sst_ice / A
        F_norm = -csw * net_sv * 1e9 * sst_glob / A
        out[name] = dict(resid=resid, dV_sv=dV / 1e6, ka=ka, H_dh=H_dh, H_dT=H_dT,
                         sum_split=H_dh + H_dT, dH_tool=dH_tool, R_loader=R_sv_loader,
                         R_log=R_log_end, F_run=F_run, ice_sv=ice_sv, sst_ice=sst_ice,
                         F_ice=F_ice, net_sv=net_sv, sst_glob=sst_glob, F_norm=F_norm,
                         F_tot=F_run + F_ice + F_norm,
                         resid_after_F=resid - (F_run + F_ice + F_norm), nwin=len(win))

    for name, o in out.items():
        print(f"[{name}] controls: eta+1cm -> dV/(0.01*A) = {o['ka']:.12f} (1 exact); "
              f"split sum {o['sum_split']:+.6f} vs tool dH/dt {o['dH_tool']:+.6f} W/m2; "
              f"runoff loader {o['R_loader']:.4f} Sv vs log R {o['R_log']:.4f} Sv; fwbudget lines {o['nwin']}")
        print(f"[{name}] residual {o['resid']:+.4f} W/m2 | net volume change {o['dV_sv']:+.4f} Sv | "
              f"dH from thickness change {o['H_dh']:+.4f}, from T change {o['H_dT']:+.4f} W/m2")
        print(f"[{name}] freshwater heat: runoff {o['F_run']:+.4f}, ice {o['F_ice']:+.4f} "
              f"({o['ice_sv']:+.3f} Sv at {o['sst_ice']:+.2f} C), normalisation {o['F_norm']:+.4f} "
              f"({o['net_sv']:+.3f} Sv removed at {o['sst_glob']:.2f} C) -> total {o['F_tot']:+.4f}; "
              f"residual left {o['resid_after_F']:+.4f} W/m2")
    r, s = out["row"], out["base"]
    print("[row-base] " + "  ".join(f"{k} {r[k] - s[k]:+.4f}" for k in
                                    ("resid", "dV_sv", "H_dh", "H_dT", "F_run", "F_ice", "F_norm",
                                     "resid_after_F")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
