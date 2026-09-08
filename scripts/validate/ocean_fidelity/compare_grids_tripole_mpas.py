"""Direct tripole-vs-MPAS comparison on a common lat-lon grid.

The two grids cannot be compared cell-to-cell (tripole is 331x360 curvilinear,
MPAS is an unstructured Voronoi mesh), so both are IDW-regridded onto the same
1-degree lat-lon target with the SAME helper the NEMO scorecard uses
(compare_omip_nemo.regrid_curv_to_latlon), and only cells resolved on BOTH are
scored.

WHAT THIS ANSWERS: "do our two grids agree with each other" -- the standing
cross-grid question. It is NOT a NEMO-fidelity statement: to make the pair
matched at all, both arms had to drop --dm2dc, --isf, --bbl-adv, --sw-rgb-chl,
--gateway-transports, --iwm and the four --tke-* knobs (MPAS hard-errors on
each), and both use the annual-WOA IC because the driver refuses
--nemo-monthly-init on MPAS. So BOTH arms are degraded from the NEMO-faithful
tripole configuration, equally. Never compare these numbers to the
NEMO-faithful runs.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/burg-archive/glab/users/pg2328/legoESM/scripts/validate")
from compare_omip_nemo import regrid_curv_to_latlon, _load_legoesm

R = "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tripole", required=True)
    p.add_argument("--mpas", required=True)
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--out", default="/burg-archive/glab/users/pg2328/legoESM/omip_nemo/xgrid_compare.png")
    a = p.parse_args()

    tgt_lat = np.arange(-89.5, 90.0, a.res_deg)
    tgt_lon = np.arange(0.5, 360.0, a.res_deg)
    fields = {}
    for tag, path in (("tripole", a.tripole), ("MPAS", a.mpas)):
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: missing snapshot {path}")
        L = _load_legoesm(path)
        sst, oc1 = regrid_curv_to_latlon(L["sst"], L["lat"], L["lon"], L["mask"],
                                         tgt_lat, tgt_lon)
        sss, oc2 = regrid_curv_to_latlon(L["sss"], L["lat"], L["lon"], L["mask"],
                                         tgt_lat, tgt_lon)
        fields[tag] = dict(sst=sst, sss=sss, oc=(oc1 > 0.5) & (oc2 > 0.5))
        print(f"[load] {tag}: {L['sst'].shape} -> {sst.shape}, "
              f"{int(fields[tag]['oc'].sum())} resolved cells")

    common = fields["tripole"]["oc"] & fields["MPAS"]["oc"]
    w = np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(common, float)
    w = np.where(common, w, 0.0)
    print(f"\n[common] {int(common.sum())} cells scored on BOTH grids")
    out = {}
    for v in ("sst", "sss"):
        d = fields["tripole"][v] - fields["MPAS"][v]
        bias = float(np.sum(w * np.where(common, d, 0.0)) / np.sum(w))
        rmse = float(np.sqrt(np.sum(w * np.where(common, d, 0.0) ** 2) / np.sum(w)))
        A = fields["tripole"][v][common]; B = fields["MPAS"][v][common]
        corr = float(np.corrcoef(A, B)[0, 1])
        out[v] = (bias, rmse, corr, d)
        u = "C" if v == "sst" else "psu"
        print(f"  {v.upper():4s} tripole-minus-MPAS   bias {bias:+.4f} {u}   "
              f"RMSE {rmse:.4f} {u}   corr {corr:.5f}")
    # zonal means, area-weighted, identical bands both grids
    edges = np.arange(-90, 91, 2.0)
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for r_, v in enumerate(("sst", "sss")):
        u = "degC" if v == "sst" else "psu"
        for c_, (t, F) in enumerate((("tripole", fields["tripole"][v]),
                                     ("MPAS", fields["MPAS"][v]))):
            m = ax[r_, c_].pcolormesh(tgt_lon, tgt_lat, np.where(common, F, np.nan),
                                      cmap="viridis", shading="auto")
            ax[r_, c_].set_title(f"{t} {v.upper()} [{u}]", fontsize=10)
            plt.colorbar(m, ax=ax[r_, c_], shrink=0.8)
        bias, rmse, corr, d = out[v]
        lim = float(np.nanpercentile(np.abs(np.where(common, d, np.nan)), 98))
        m = ax[r_, 2].pcolormesh(tgt_lon, tgt_lat, np.where(common, d, np.nan),
                                 cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax[r_, 2].set_title(f"tripole - MPAS  bias {bias:+.3f}  RMSE {rmse:.3f}  "
                            f"corr {corr:.4f}", fontsize=10)
        plt.colorbar(m, ax=ax[r_, 2], shrink=0.8)
    fig.suptitle("Cross-grid agreement, day 30, matched configuration "
                 "(annual-WOA IC, card-default TKE, no iwm/isf/dm2dc/bbl/rgb) — "
                 "NOT a NEMO-fidelity comparison", fontsize=12)
    fig.tight_layout(); fig.savefig(a.out, dpi=115)
    print(f"\n[map] wrote {a.out}")
    # zonal profile figure
    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
    for i, v in enumerate(("sst", "sss")):
        zc, zt, zm = [], [], []
        for k in range(len(edges) - 1):
            sel = common & (tgt_lat[:, None] >= edges[k]) & (tgt_lat[:, None] < edges[k + 1])
            if sel.sum() < 5: continue
            ww = w[sel]
            zc.append(0.5 * (edges[k] + edges[k + 1]))
            zt.append(float(np.sum(ww * fields["tripole"][v][sel]) / np.sum(ww)))
            zm.append(float(np.sum(ww * fields["MPAS"][v][sel]) / np.sum(ww)))
        ax2[i].plot(zc, zt, "o-", ms=3, label="tripole")
        ax2[i].plot(zc, zm, "s--", ms=3, label="MPAS")
        ax2[i].set_xlabel("latitude [degN]")
        ax2[i].set_ylabel(f"{v.upper()} [{'degC' if v=='sst' else 'psu'}]")
        ax2[i].set_title(f"zonal mean {v.upper()} (identical bands + weights)", fontsize=10)
        ax2[i].legend(); ax2[i].grid(alpha=0.3)
    fig2.tight_layout()
    z_out = a.out.replace(".png", "_zonal.png")
    fig2.savefig(z_out, dpi=125); print(f"[map] wrote {z_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
