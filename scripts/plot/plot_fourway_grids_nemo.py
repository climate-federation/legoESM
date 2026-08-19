"""One figure per field with ALL grids: tripole | MPAS | FESOM2 | NEMO.

The pairwise three-way scorecards already carry the numbers; this plotter
puts all four models on ONE common footing (same regrid, same
all-four-resolved mask) so the visual comparison the user asked for is a
single figure per field: a map row (four models), a difference row (each
minus NEMO), and a combined zonal-mean + zonal-bias panel.

Reuses the comparator's loaders/regridder/mask machinery verbatim
(compare_omip_nemo + compare_three_way_nemo); no new numerics beyond a
four-way conjunction mask.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "validate"))
sys.path.insert(0, str(_HERE.parents[1] / "validate" / "ocean_fidelity"))
from compare_omip_nemo import _load_legoesm, _load_nemo, regrid_curv_to_latlon  # noqa: E402
from compare_three_way_nemo import _lego_mld, build_ocean_mask  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tripole", required=True)
    p.add_argument("--mpas", required=True)
    p.add_argument("--fesom", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-month", type=int, default=1)
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tgt_lat = -90.0 + a.res_deg / 2.0 + a.res_deg * np.arange(int(180.0 / a.res_deg))
    tgt_lon = a.res_deg / 2.0 + a.res_deg * np.arange(int(360.0 / a.res_deg))

    srcs = {"tripole": _load_legoesm(a.tripole),
            "MPAS": _load_legoesm(a.mpas),
            "FESOM2": _load_legoesm(a.fesom),
            "NEMO": _load_nemo(a.nemo_gridt, -1, month=a.nemo_month)}

    def rg(S, field, mask=None):
        return regrid_curv_to_latlon(np.asarray(field), S["lat"], S["lon"],
                                     S["mask"] if mask is None else mask,
                                     tgt_lat, tgt_lon)

    fields = {"SST": ("degC", {k: S["sst"] for k, S in srcs.items()}),
              "SSS": ("psu", {k: S["sss"] for k, S in srcs.items()})}
    mlds = {}
    for k, S in srcs.items():
        if k == "NEMO":
            mlds[k] = S.get("mld")
        else:
            mlds[k] = _lego_mld(S)
    if all(v is not None for v in mlds.values()):
        fields["MLD"] = ("m", mlds)

    regridded, covs = {}, {}
    for name, (unit, per_src) in fields.items():
        regridded[name] = {}
        for k, f in per_src.items():
            S = srcs[k]
            if name == "MLD":
                g, oc = regrid_curv_to_latlon(
                    np.nan_to_num(np.asarray(f), nan=0.0), S["lat"], S["lon"],
                    np.isfinite(np.asarray(f)).astype(np.float64),
                    tgt_lat, tgt_lon)
            else:
                g, oc = rg(S, f)
            regridded[name][k] = g
            covs.setdefault(k, []).append(oc > 0.5)

    coverage = np.logical_and.reduce([np.logical_and.reduce(v) for v in covs.values()])
    ocean = build_ocean_mask(coverage, tuple(srcs.values()), tgt_lat, tgt_lon,
                             "nearest")
    print(f"[mask] {int(ocean.sum())} cells resolved on all four sources")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["tripole", "MPAS", "FESOM2", "NEMO"]
    for name, (unit, _) in fields.items():
        G = {k: np.where(ocean, regridded[name][k], np.nan) for k in order}
        finite = np.logical_and.reduce([np.isfinite(G[k]) for k in order])
        for k in order:
            G[k] = np.where(finite, G[k], np.nan)
        stack = np.concatenate([G[k][finite] for k in order])
        vmin, vmax = np.percentile(stack, [1, 99])
        dmax = float(np.percentile(np.abs(np.concatenate(
            [(G[k] - G["NEMO"])[finite] for k in order[:3]])), 99))

        fig, ax = plt.subplots(2, 4, figsize=(24, 8))
        for j, k in enumerate(order):
            im = ax[0][j].pcolormesh(tgt_lon, tgt_lat, G[k], vmin=vmin,
                                     vmax=vmax, cmap="RdYlBu_r", shading="auto")
            ax[0][j].set_title(f"{k} {name}", fontsize=11)
            plt.colorbar(im, ax=ax[0][j], shrink=0.8)
        for j, k in enumerate(order[:3]):
            im = ax[1][j].pcolormesh(tgt_lon, tgt_lat, G[k] - G["NEMO"],
                                     vmin=-dmax, vmax=dmax, cmap="RdBu_r",
                                     shading="auto")
            ax[1][j].set_title(f"{k} - NEMO", fontsize=11)
            plt.colorbar(im, ax=ax[1][j], shrink=0.8)
        # last panel: combined zonal means + biases
        a4 = ax[1][3]
        cnt = finite.sum(axis=1)
        good = cnt >= 5
        zm = {}
        for k in order:
            z = np.full(tgt_lat.shape, np.nan)
            z[good] = np.where(finite, G[k], 0.0).sum(axis=1)[good] / cnt[good]
            zm[k] = z
        a4.plot(zm["tripole"], tgt_lat, lw=1.4, label="tripole")
        a4.plot(zm["MPAS"], tgt_lat, lw=1.4, ls="--", label="MPAS")
        a4.plot(zm["FESOM2"], tgt_lat, lw=1.4, ls="-.", label="FESOM2")
        a4.plot(zm["NEMO"], tgt_lat, lw=2.0, color="k", label="NEMO")
        a4.set_title(f"zonal-mean {name} [{unit}]", fontsize=11)
        a4.legend(fontsize=8)
        a4.grid(alpha=0.3)
        fig.suptitle(
            f"{name} [{unit}] — tripole / MPAS / FESOM2 / NEMO month {a.nemo_month} "
            "(cells resolved on all four; FESOM2 runs its published protocol — "
            "three-model comparison, not a controlled pair)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"{name}_fourway.png", dpi=110)
        plt.close(fig)
        print(f"[map] {out / f'{name}_fourway.png'}")

        fig, axz = plt.subplots(1, 1, figsize=(6, 7))
        for k, ls in (("tripole", "-"), ("MPAS", "--"), ("FESOM2", "-.")):
            axz.plot(zm[k] - zm["NEMO"], tgt_lat, ls=ls, lw=1.4,
                     label=f"{k} - NEMO")
        axz.axvline(0, color="k", lw=0.8)
        axz.set_title(f"zonal-mean bias vs NEMO — {name} [{unit}]")
        axz.set_ylabel("latitude")
        axz.legend(fontsize=9)
        axz.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{name}_fourway_zonalbias.png", dpi=110)
        plt.close(fig)
        print(f"[map] {out / f'{name}_fourway_zonalbias.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
