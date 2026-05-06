#!/usr/bin/env python
"""Plot time series comparing two dissipation sweep runs."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_timeseries(run_dir):
    """Load time series from restart files."""
    restart_dir = Path(run_dir) / "latlon" / "180x360"
    files = sorted(restart_dir.glob("restart_day*.npz"))
    if not files:
        print(f"No restart files in {restart_dir}")
        return None

    days, sst_mean, sss_mean = [], [], []
    T_100m, T_500m, S_100m, S_500m = [], [], [], []
    eta_mean, eta_std, max_u, max_v = [], [], [], []
    ke_mean = []

    for f in files:
        d = np.load(f)
        mask = d["land_mask"]
        T = d["T"]       # (180, 360, 20)
        S = d["S"]
        eta = d["eta"]
        u = d["u"]       # (180, 361, 20)
        v = d["v"]       # (181, 360, 20)
        t = float(d["time_days"])

        ocean = mask > 0.5

        days.append(t)

        # SST/SSS = surface layer
        sst_mean.append(np.mean(T[:, :, 0][ocean]))
        sss_mean.append(np.mean(S[:, :, 0][ocean]))

        # Depth indices: 20 levels over 5500m
        # level 2 ≈ ~137m, level 5 ≈ ~550m (rough)
        T_100m.append(np.mean(T[:, :, 2][ocean]))
        T_500m.append(np.mean(T[:, :, 5][ocean]))
        S_100m.append(np.mean(S[:, :, 2][ocean]))
        S_500m.append(np.mean(S[:, :, 5][ocean]))

        # SSH
        eta_mean.append(np.mean(eta[ocean]))
        eta_std.append(np.std(eta[ocean]))

        # Max velocities
        max_u.append(np.max(np.abs(u)))
        max_v.append(np.max(np.abs(v)))

        # Mean KE (surface, on T-grid approx)
        u_t = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_t = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        ke = 0.5 * (u_t**2 + v_t**2)
        ke_mean.append(np.mean(ke[ocean]))

    return {
        "days": np.array(days),
        "sst": np.array(sst_mean),
        "sss": np.array(sss_mean),
        "T_100m": np.array(T_100m),
        "T_500m": np.array(T_500m),
        "S_100m": np.array(S_100m),
        "S_500m": np.array(S_500m),
        "eta_mean": np.array(eta_mean),
        "eta_std": np.array(eta_std),
        "max_u": np.array(max_u),
        "max_v": np.array(max_v),
        "ke_mean": np.array(ke_mean),
    }


def main():
    runs = {
        r"Baseline (A_h=2×10$^5$)": "results/jra55_10yr_production",
        r"Run C: A$_h$=5e4, eq_boost=5": "results/sweep_C_eqboost5",
        r"Run D: A$_h$=3e4, eq_boost=7": "results/sweep_D_eqboost7",
    }

    data = {}
    for label, path in runs.items():
        ts = load_timeseries(path)
        if ts is not None:
            data[label] = ts

    if not data:
        print("No data found!")
        return

    colors = ["C0", "C1", "C2", "C3"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

    for i, (label, ts) in enumerate(data.items()):
        years = ts["days"] / 365.25
        c = colors[i % len(colors)]

        axes[0, 0].plot(years, ts["sst"], c, label=label)
        axes[0, 1].plot(years, ts["sss"], c, label=label)
        axes[1, 0].plot(years, ts["T_100m"], c, label=label)
        axes[1, 1].plot(years, ts["T_500m"], c, label=label)
        axes[2, 0].plot(years, ts["eta_mean"], c, label=label)
        axes[2, 1].plot(years, ts["eta_std"], c, label=label)
        axes[3, 0].plot(years, ts["ke_mean"], c, label=label)
        axes[3, 1].plot(years, np.maximum(ts["max_u"], ts["max_v"]), c, label=label)

    axes[0, 0].set_ylabel("SST [°C]")
    axes[0, 0].set_title("Mean SST")
    axes[0, 1].set_ylabel("SSS [PSU]")
    axes[0, 1].set_title("Mean SSS")
    axes[1, 0].set_ylabel("T [°C]")
    axes[1, 0].set_title("T at ~100m")
    axes[1, 1].set_ylabel("T [°C]")
    axes[1, 1].set_title("T at ~500m")
    axes[2, 0].set_ylabel("η [m]")
    axes[2, 0].set_title("Mean SSH")
    axes[2, 1].set_ylabel("η std [m]")
    axes[2, 1].set_title("SSH variability")
    axes[3, 0].set_ylabel("KE [m²/s²]")
    axes[3, 0].set_title("Mean surface KE")
    axes[3, 1].set_ylabel("|u| [m/s]")
    axes[3, 1].set_title("Max velocity")

    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    axes[0, 0].legend(fontsize=8)

    fig.suptitle("Dissipation Sweep: Time Series Comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/sweep_timeseries.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
