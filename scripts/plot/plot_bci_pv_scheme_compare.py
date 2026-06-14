"""Compare BCI KE growth on MPAS under different PV flux schemes vs lat-lon.

Confirms the hypothesis that the TRiSK enstrophy-conserving PV flux is the
dominant damping source for the BCI mode on MPAS at strong forcing.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results/ocean/eady_uniform")

RUNS = [
    ("lat-lon tvd (reference)",
     RESULTS / "latlon_channel/100x50/tvd_nosponge_200d",
     dict(color="C3", lw=2.5)),
    ("MPAS pv=enstrophy (600d → blowup d252)",
     RESULTS / "mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_600d",
     dict(color="C0", lw=2.0)),
    ("MPAS pv=energy (no ζ-damp — blew d57)",
     RESULTS / "mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_pvenergy_apvm300_80d",
     dict(color="C1", lw=1.2, ls="--")),
    ("MPAS pv=energy + K_ζ=2e11 (200d → d99)",
     RESULTS / "mpas_channel/20km/tvd_mpas_20km_U08_pvenergy_Kzeta2e11_200d",
     dict(color="C2", lw=1.5)),
    ("MPAS pv=energy + K_ζ=3e11 (200d → d143)",
     RESULTS / "mpas_channel/20km/tvd_mpas_20km_U08_pvenergy_Kzeta3e11_200d",
     dict(color="C4", lw=2.0)),
    ("MPAS pv=energy + K_ζ=1e12 (over-damped)",
     RESULTS / "mpas_channel/20km/tvd_mpas_20km_U08_pvenergy_Kzeta1e12_80d",
     dict(color="C7", lw=1.0, ls=":")),
]


def load(p):
    csv_path = p / "mean_timeseries.csv"
    if not csv_path.exists():
        return None
    cols = {}
    with csv_path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            for k, v in row.items():
                if k is None or k.startswith("#"):
                    continue
                cols.setdefault(k, []).append(float(v))
    out = {k: np.asarray(v) for k, v in cols.items()}
    # Mask NaN rows (post-blowup)
    good = np.isfinite(out["mean_ke"])
    out = {k: v[good] for k, v in out.items()}
    return out


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for label, d, style in RUNS:
    ts = load(d)
    if ts is None:
        continue
    t = ts["time_days"]
    axes[0].plot(t, ts["max_speed"], label=label, **style)
    axes[1].semilogy(t, np.clip(ts["mean_ke"], 1e-6, None), label=label, **style)
    axes[2].plot(t, ts["max_abs_eta"], label=label, **style)

axes[0].set_xlabel("time (d)"); axes[0].set_ylabel("max |u| (m/s)")
axes[0].set_title("max_speed"); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 300)

axes[1].set_xlabel("time (d)"); axes[1].set_ylabel("mean KE (m²/s²)")
axes[1].set_title("mean_ke (log)"); axes[1].grid(alpha=0.3, which="both")
axes[1].set_xlim(0, 300)

axes[2].set_xlabel("time (d)"); axes[2].set_ylabel("max |η| (m)")
axes[2].set_title("max_abs_eta — checkerboard proxy")
axes[2].grid(alpha=0.3); axes[2].set_xlim(0, 300)

axes[0].legend(fontsize=8, loc="upper left", frameon=False)

fig.suptitle(
    "MPAS BCI growth recovered: pv=energy + biharmonic ζ-damping (K_ζ·∇⁴ζ)\n"
    "(strong Eady forcing U=0.8, 20 km)",
    fontsize=11)
fig.tight_layout()

out = RESULTS / "bci_pv_scheme_compare.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
