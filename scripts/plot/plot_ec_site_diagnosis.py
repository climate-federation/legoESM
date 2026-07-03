#!/usr/bin/env python
"""Summary 'diagnosis' figure for the EC-site LE/H/EF gap (US-MMS, JJA daytime).

Two panels that together make the case that the residual latent-heat error is the
canopy<->soil EVAPORATION PARTITION, not net radiation / albedo / WUE:

  (A) LE partition: transpiration vs soil evaporation for the DifferBESS oracle
      (matches obs) vs legoESM — DifferBESS soil-evap ~5% of ET, legoESM ~70-85%.
  (B) Surface radiation vs the EC four-way measurements — net Rn, reflected SW
      and outgoing LW all match observations (Rn r=0.999), ruling out a radiation
      or albedo error.

Values are the summer-daytime means from this study's diagnostic probes
(DifferBESS production output + the legoESM LE-partition / radiation-benchmark
probes); see docs/ec_site_offline_run.md.  Throwaway/summary plotter.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "diagnostics/ec_site_diagnosis"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "mathtext.default": "regular",
})
C_T, C_S = "#117733", "#CC6677"   # transpiration / soil-evap (color-blind safe)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- Panel A: LE partition (US-MMS, JJA daytime, W/m2) ----
# transpiration, soil evaporation
part = {"DifferBESS\n(oracle)": (184.0, 9.9), "legoESM": (54.0, 143.0)}
labels = list(part)
tr = np.array([part[k][0] for k in labels])
so = np.array([part[k][1] for k in labels])
x = np.arange(len(labels))
axA.bar(x, tr, 0.6, label="transpiration", color=C_T, edgecolor="0.3", lw=0.3)
axA.bar(x, so, 0.6, bottom=tr, label="soil evaporation", color=C_S,
        edgecolor="0.3", lw=0.3)
axA.axhline(193, color="k", ls="--", lw=1.0)
axA.text(-0.45, 168, "observed\ntotal LE", fontsize=8, ha="left", va="top")
for i, k in enumerate(labels):
    frac = 100 * so[i] / (tr[i] + so[i])
    axA.text(i, tr[i] + so[i] + 8, f"soil = {frac:.0f}% of ET",
             ha="center", fontsize=9.5, fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(labels)
axA.set_ylabel("latent heat flux (W m$^{-2}$)")
axA.set_title("(A) LE partition — JJA daytime, US-MMS")
axA.legend(loc="upper left"); axA.set_ylim(0, 230)

# ---- Panel B: surface radiation vs EC four-way obs (JJA daytime, W/m2) ----
comps = ["net Rn", "reflected SW", "outgoing LW"]
mod = np.array([326.0, 65.0, 457.0]); obs = np.array([336.0, 70.0, 442.0])
xb = np.arange(len(comps)); w = 0.38
axB.bar(xb - w/2, obs, w, label="EC observed", color="0.25",
        edgecolor="0.3", lw=0.3)
axB.bar(xb + w/2, mod, w, label="legoESM", color="#0072B2",
        edgecolor="0.3", lw=0.3)
axB.set_xticks(xb); axB.set_xticklabels(comps)
axB.set_ylabel("flux (W m$^{-2}$)")
axB.set_title("(B) Surface radiation vs EC four-way ($r_{Rn}$=0.999)")
axB.legend(loc="upper left")
axB.text(0.0, 350, "albedo 0.14 vs 0.16", fontsize=8, ha="center")

fig.suptitle("The LE/H/EF gap is the canopy$\\rightarrow$soil evaporation "
             "partition — NOT radiation, albedo, or WUE", fontsize=11.5, y=1.03)
fig.tight_layout()
p = os.path.join(OUT, "ec_site_le_partition_diagnosis.png")
fig.savefig(p); plt.close(fig)
print(f"  plot -> {p}")
