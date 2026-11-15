"""Row-1 heat-gate diagnosis: is the residual the probe or the model?

Runs the heat-closure tool's own main() with close_windows captured, so every
check below uses the tool's exact snapshots and cell rebuild (z_coord_for).
A) probe cells vs domain_cfg e3t_0 (independent oracle), every wet cell.
B) known-answer: synthetic window (T += depth-varying dT, eta fixed) whose
   heat change is computed from domain_cfg e3t_0; residual must be ~0.
   Control: same window through mesh_mask-style cells must NOT close.
C) row restart_leg1 vs snapshot_day0015 heat content (same state twice).
D) days 25-30 per-column (dH/dt - q_net - geo) by latitude band, both runs,
   and row-minus-baseline split by thick-bottom / geometry-changed columns.

Usage (all paths required; the row run must carry restart_leg1.npz and the
day-15/day-20 snapshots that check C reads):

    omip_row1_heat_diag.py --row-dir R --base-dir B --domain-cfg domain_cfg.nc
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402


def h_of(s, pc):
    from legoesm.ocean.vertical import compute_layer_thickness
    return np.asarray(compute_layer_thickness(s["eta"], s["H_bathy"], pc), dtype=np.float64)


def col_heat(s, pc, rho_0, c_p):
    """Per-column heat content [J] of a snapshot dict on cells ``pc``."""
    wet = s["land_mask"] > 0.5
    return rho_0 * c_p * (s["T"] * h_of(s, pc)).sum(-1) * s["cell_area"] * wet


def band_sums(col, lat, bands):
    """``col`` summed over each half-open latitude band ``[lo, hi)``."""
    return [float(col[(lat >= lo) & (lat < hi)].sum()) for lo, hi in bands]


def capture(ht, run, dcfg):
    cap = {}
    real = ht.close_windows

    def spy(snaps, z_coord_for, rho_0, c_p, flux_geo, isf_fwf_for, L, **kw):
        cap.update(snaps=snaps, zc=z_coord_for, rho_0=rho_0, c_p=c_p,
                   geo=flux_geo, isf=isf_fwf_for, L=L, kw=kw)
        rows = real(snaps, z_coord_for, rho_0, c_p, flux_geo, isf_fwf_for, L, **kw)
        cap["rows"] = rows
        return rows
    ht.close_windows = spy
    try:
        ht.main(["--run-dir", run, "--domain-cfg", dcfg, "--max-residual-wm2", "1e9"])
    finally:
        ht.close_windows = real
    return cap


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--row-dir", required=True, help="row-1 run dir")
    p.add_argument("--base-dir", required=True, help="baseline run dir")
    p.add_argument("--domain-cfg", required=True, help="NEMO domain_cfg.nc")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import xarray as xr
    import omip_tripole_heat_closure as ht
    from legoesm.ocean.init_tripole import nemo_domain_offset
    ROW, BASE, DCFG = args.row_dir, args.base_dir, args.domain_cfg
    row, base = capture(ht, ROW, DCFG), capture(ht, BASE, DCFG)
    rs = row["snaps"]
    s30 = rs[-1]
    lat = ht._snap_lat(str(Path(ROW) / "snapshot_final.npz"))
    lon = ht._snap_lon(str(Path(ROW) / "snapshot_final.npz"))

    # ---- A: probe cells vs domain_cfg e3t_0
    with xr.open_dataset(DCFG, mask_and_scale=False) as ds:
        e3t = np.asarray(ds["e3t_0"].values, dtype=np.float64)
        kb = np.asarray(ds["bottom_level"].values).squeeze().astype(np.int64)
        dlat = np.asarray(ds["gphit"].values, dtype=np.float64).squeeze()
        dlon = np.asarray(ds["glamt"].values, dtype=np.float64).squeeze()
    while e3t.ndim > 3:
        e3t = e3t[0]
    j0, i0 = nemo_domain_offset(lat, lon, dlat, dlon, DCFG)
    nz, nj, ni = e3t.shape
    pc = row["zc"](s30)
    hp = np.asarray(pc.h_partial, dtype=np.float64)
    truth = np.zeros_like(hp)
    k = np.arange(nz)
    truth[j0:j0 + nj, i0:i0 + ni, :] = np.where(k[None, None, :] < kb[..., None],
                                                 np.moveaxis(e3t, 0, -1), 0.0)
    wet = s30["land_mask"] > 0.5
    d = np.abs(hp - truth)[wet]
    print(f"[A] probe cells vs domain_cfg e3t_0: wet cols {wet.sum()}, max |dh| {d.max():.3e} m, "
          f"cells >1e-6 m: {(d > 1e-6).sum()}; thick bottom cells (h>dz_ref): "
          f"{(hp[wet] > np.asarray(pc.dz_ref)[None, :] + 1e-6).sum()}")

    # ---- B: known answer on the real row geometry (window day 25 -> 30)
    a, b = rs[-2], rs[-1]
    dT = (0.01 + 0.002 * k)[None, None, :] * np.ones_like(a["T"])
    b_syn = {**b, "T": a["T"] + dT, "eta": a["eta"], "S": a.get("S", a["T"] * 0)}
    Htrue = truth.sum(-1)
    h_true = truth * np.where(Htrue > 0, (a["eta"] + Htrue) / np.where(Htrue > 0, Htrue, 1), 0)[..., None]
    dt_win = float(b["flux_mean_window_s"])
    Q = row["rho_0"] * row["c_p"] * (dT * h_true * (a["cell_area"] * wet)[..., None]).sum() / dt_win
    A_wet = (a["cell_area"] * wet).sum()
    b_syn["q_net_mean"] = np.full_like(b["q_net_mean"], Q / A_wet)
    r_syn = ht.close_windows([a, b_syn], row["zc"], row["rho_0"], row["c_p"], 0.0, None, row["L"])[0]
    print(f"[B] known-answer window, tool cells: residual {r_syn['residual_Wm2']:+.3e} W/m2 "
          f"(signal {Q / A_wet:.3e} W/m2)")


    def mesh_cells(s):
        zs, pcm = ht.build_z_coord(row_e3t, row_gdept, s["H_bathy"])
        return pcm


    with xr.open_dataset(DCFG, decode_times=False) as ds:
        row_e3t = np.asarray(ds["e3t_1d"]).squeeze().astype(np.float64)
        row_gdept = (np.asarray(ds["gdept_1d"]).squeeze().astype(np.float64) if "gdept_1d" in ds else None)
    r_ctl = ht.close_windows([a, b_syn], mesh_cells, row["rho_0"], row["c_p"], 0.0, None, row["L"])[0]
    print(f"[B-control] same window, interface-rule cells: residual {r_ctl['residual_Wm2']:+.3e} W/m2 "
          f"(must be clearly non-zero)")

    # ---- C: restart_leg1 vs snapshot_day0015 (row)
    s15 = next(s for s in rs if int(round(float(s["time_days"]))) == 15)
    rst = ht.restart_as_snapshot(str(Path(ROW) / "restart_leg1.npz"),
                                 ht.load_snapshot(str(Path(ROW) / "snapshot_day0020.npz")))
    rst = ht.drop_cyclic_overlap(rst)
    Hs = ht.heat_content(s15, row["zc"](s15), row["rho_0"], row["c_p"])[0]
    Hr = ht.heat_content(rst, row["zc"](rst), row["rho_0"], row["c_p"])[0]
    print(f"[C] row day-15: snapshot H {Hs:.12e} J, restart H {Hr:.12e} J, "
          f"diff {(Hr - Hs) / dt_win / A_wet:+.3e} W/m2-equivalent over a window; "
          f"max|dT| {np.abs(rst['T'] - s15['T'])[s15['land_mask'] > 0.5].max():.3e}")

    # ---- D: day 25-30 band map, both runs
    bands = [(-90, -45), (-45, -23), (-23, 23), (23, 45), (45, 90)]


    def colres(cap):
        a, b = cap["snaps"][-2], cap["snaps"][-1]
        dt = float(b["flux_mean_window_s"])
        wet_b = b["land_mask"] > 0.5
        dH = (col_heat(b, cap["zc"](b), cap["rho_0"], cap["c_p"])
              - col_heat(a, cap["zc"](a), cap["rho_0"], cap["c_p"])) / dt
        geo = cap["geo"] * cap["kw"].get("geo_scale", 1.0) * b["cell_area"] * wet_b * (b["H_bathy"] > 0)
        return (dH - b["q_net_mean"] * b["cell_area"] * wet_b - geo), b, cap["rows"][-1]


    cr, br, rr = colres(row)
    cb, bb, rb = colres(base)
    print(f"[D] tool d25-30 residual: row {rr['residual_Wm2']:+.4f}, base {rb['residual_Wm2']:+.4f} W/m2 "
          f"(isf term excluded from per-column map; global isf {rr['Q_isf_latent_W']:.3e} W)")
    Aw = (br["cell_area"] * (br["land_mask"] > 0.5)).sum()
    for (lo, hi), sr, sb in zip(bands, band_sums(cr, lat, bands), band_sums(cb, lat, bands)):
        print(f"[D] band {lo:+4d}..{hi:+4d}: row {sr / Aw:+.4f}  base {sb / Aw:+.4f}  "
              f"row-base {(sr - sb) / Aw:+.4f} W/m2 (global-area normalised)")
    pcr = row["zc"](br)
    thick = (np.asarray(pcr.h_partial) > np.asarray(pcr.dz_ref)[None, None, :] + 1e-6).any(-1)
    pcb = base["zc"](bb)
    changed = np.any(np.abs(np.asarray(pcr.h_partial) - np.asarray(pcb.h_partial)) > 1e-6, axis=-1)
    for name, m in [("thick-bottom cols", thick), ("geometry-changed cols", changed),
                    ("unchanged cols", ~changed)]:
        m = m & (br["land_mask"] > 0.5)
        print(f"[D] {name:22s} n={m.sum():6d}: row-base {(cr[m].sum() - cb[m].sum()) / Aw:+.4f} W/m2")
    print("[D] NOTE per-column residual includes lateral heat transport divergence; "
          "only sums over closed sets are budgets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
