#!/usr/bin/env python
"""Per-RCE ranking + profile panels from scm_rce_per_sst_profiles.py outputs.

Reads every per-SST JSON in a directory and produces, per SST (295/300/305),
a two-panel T and q_v figure with all schemes over that SST's CRM, plus a
printed ranking of schemes by thermo score at each SST. Tropospheric (<=15 km).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOT_TOP_KM = 15.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--glob", default="per_sst_*.json")
    ap.add_argument("--ssts", default="295,300,305")
    ap.add_argument("--out-prefix", type=Path, required=True)
    a = ap.parse_args(argv)

    ssts = [int(x) for x in a.ssts.split(",")]
    data = {}
    for p in sorted(glob.glob(str(a.in_dir / a.glob))):
        d = json.loads(Path(p).read_text())
        data[d["scheme"]] = d["per_sst"]
    if not data:
        raise SystemExit(f"no {a.glob} in {a.in_dir}")

    # --- ranking per SST ---
    print("Per-RCE ranking by thermo score (lower = closer to CRM):")
    for sst in ssts:
        rows = [(s, data[s][str(sst)]["thermo"]) for s in data
                if str(sst) in data[s]]
        rows.sort(key=lambda r: r[1])
        print(f"\n  {sst} K:")
        for i, (s, v) in enumerate(rows, 1):
            print(f"    {i:2d}. {s:16s} "
                  + ("BLOWUP" if v == float("inf") else f"{v:.2f}"))

    # --- profile panels per SST ---
    cmap = plt.get_cmap("turbo")
    for sst in ssts:
        schemes = [s for s in data if data[s].get(str(sst), {}).get("T_profile")]
        schemes.sort(key=lambda s: data[s][str(sst)]["thermo"])
        if not schemes:
            continue
        colors = {s: cmap(i / max(1, len(schemes) - 1))
                  for i, s in enumerate(schemes)}
        fig, (axT, axQ) = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
        ref = data[schemes[0]][str(sst)]
        z = np.asarray(ref["z_m"]) / 1000.0
        m = z <= PLOT_TOP_KM
        for s in schemes:
            ps = data[s][str(sst)]
            c = colors[s]
            axT.plot(np.asarray(ps["T_profile"])[m], z[m], color=c, lw=1.5,
                     label=f"{s} ({ps['thermo']:.1f})")
            axQ.plot(np.asarray(ps["qv_profile"])[m] * 1000.0, z[m],
                     color=c, lw=1.5)
        axT.plot(np.asarray(ref["T_ref"])[m], z[m], "k", lw=2.6,
                 label="SAM CRM", zorder=10)
        axQ.plot(np.asarray(ref["qv_ref"])[m] * 1000.0, z[m], "k", lw=2.6,
                 zorder=10)
        axT.set_xlabel("Temperature [K]")
        axQ.set_xlabel("Water vapour [g/kg]")
        axT.set_ylabel("Height [km]")
        axT.set_ylim(0, PLOT_TOP_KM)
        for ax in (axT, axQ):
            ax.grid(alpha=0.25)
        axT.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"SCM convection (joint-tuned) vs SAM CRM at RCE{sst} "
                     "(nonrotating)\nordered best-to-worst by thermo score",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = Path(f"{a.out_prefix}_{sst}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
