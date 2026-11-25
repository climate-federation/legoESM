"""Figures, summaries and the JSON manifest for one built dataset."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metadata, quality_control

OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")


def profiles(ds, path):
    summer = ds.sel(time=ds.time.dt.month.isin([6, 7, 8]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for hour, colour, label in ((12, OKABE_ITO[1], "12 UTC"), (21, OKABE_ITO[0], "21 UTC"), (9, OKABE_ITO[2], "09 UTC")):
        pick = summer.sel(time=summer.time.dt.hour == hour).median("time")
        speed = (pick.u ** 2 + pick.v ** 2) ** 0.5
        for ax, field in zip(axes, (pick.T, pick.q * 1e3, speed)):
            good = field.notnull()
            ax.plot(field[good], field.height[good], "o-", color=colour, label=label)
    for ax, name in zip(axes, ("Temperature (degC)", "Specific humidity (g/kg)", "Wind speed (m/s)")):
        ax.set_xlabel(name)
    axes[0].set_ylabel("Height (m)")
    axes[0].legend()
    fig.suptitle("SGP Central Facility, June to August medians")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def fluxes(ds, path):
    series = (("ECOR", "H_surface", "LE_surface"), ("EBBR", "H_ebbr", "LE_ebbr"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    summer = ds.sel(time=ds.time.dt.month.isin([6, 7, 8]))
    for colour, (label, h, le) in zip(OKABE_ITO, series + (("Tower 4 m", None, None),)):
        for ax, name, level in ((axes[0], h, "H"), (axes[1], le, "LE")):
            field = summer[name] if name else summer[level].sel(height=4)
            diurnal = field.groupby("time.hour").median()
            ax.plot(diurnal.hour, diurnal, color=colour, label=label)
    axes[0].set_title("Sensible heat flux")
    axes[1].set_title("Latent heat flux")
    for ax in axes:
        ax.set_xlabel("Hour (UTC)")
    axes[0].set_ylabel("W/m2")
    axes[0].legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def first_valid(variable):
    present = variable.notnull().any([d for d in variable.dims if d != "time"])
    times = variable.time.values[present.values]
    return str(times[0]) if times.size else ""


def run(root, ds):
    out = root / "report"
    out.mkdir(exist_ok=True)
    profiles(ds, out / "profiles.png")
    fluxes(ds, out / "fluxes.png")
    checks = quality_control.checks(ds)
    (out / "validation.json").write_text(json.dumps(checks, indent=1))
    rows = [{"variable": v, "first_valid": first_valid(ds[v]),
             "fraction_valid": float(ds[v].notnull().mean()), "units": ds[v].attrs.get("units")} for v in ds.data_vars]
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(
        {"files": len(metadata.read(root)), "statuses": pd.Series([r["status"] for r in metadata.read(root)]).value_counts().to_dict()}, indent=1))
    return checks
