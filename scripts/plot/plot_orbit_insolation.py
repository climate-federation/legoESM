"""Daily-mean TOA insolation (latitude x day) for idealized and paleo orbits.

Every orbit goes through the run's resolver (``legoesm.driver.orbit``) and the
model's own ``daily_mean_insolation``, so the maps show what a run would see.
PMIP4 calendar: vernal equinox fixed at 21 March, fixed-day months.

usage (compute node): python plot_orbit_insolation.py <out_dir>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
from legoesm.driver.orbit import experiment_orbit

jax.config.update("jax_enable_x64", True)

OBLIQUITIES = (0.0, 23.44, 60.0, 90.0)
JJA_DAYS = (152.0, 243.0)  # fixed-day 1 Jun - 31 Aug, noleap


def orbit_of(**kw):
    cfg = dict(orbital_insolation=True, orbit_year=None, orbit_eccentricity=None,
               orbit_obliquity_deg=None, orbit_long_perihelion_deg=None)
    cfg.update(kw)
    return experiment_orbit(SimpleNamespace(**cfg))


def insolation_map(orbit, lat_deg, days):
    lat = jnp.deg2rad(jnp.asarray(lat_deg))
    return np.asarray(jax.vmap(
        lambda d: daily_mean_insolation(lat, d, orbit=orbit))(jnp.asarray(days)))


def main(out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(-90.0, 90.0, 181)
    days = np.arange(1.0, 366.0)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    jja = (days >= JJA_DAYS[0]) & (days <= JJA_DAYS[1])
    i65 = int(np.argmin(abs(lat - 65.0)))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, obl in zip(axes, OBLIQUITIES):
        q = insolation_map(orbit_of(orbit_obliquity_deg=obl), lat, days)
        pc = ax.pcolormesh(days, lat, q.T, vmin=0, vmax=1400, cmap="magma",
                           shading="auto")
        ax.set_title(f"obliquity {obl:g} deg (present e, perihelion)")
        ax.set_xlabel("day of year")
        print(f"obliquity {obl:5.2f}: max {q.max():7.1f} W/m2 at lat "
              f"{lat[np.unravel_index(q.argmax(), q.shape)[1]]:+.0f}, "
              f"global-annual mean {np.mean(q * np.cos(np.deg2rad(lat))) / np.mean(np.cos(np.deg2rad(lat))):.2f}")
    axes[0].set_ylabel("latitude")
    fig.colorbar(pc, ax=axes, label="daily-mean TOA insolation [W/m2]")
    fig.suptitle(f"Insolation vs obliquity (legoESM {sha})")
    fig.savefig(out / "insolation_obliquity.png", dpi=110)

    q6 = insolation_map(orbit_of(orbit_year=-4050), lat, days)
    q0 = insolation_map(orbit_of(orbit_year=1850), lat, days)
    dq = q6 - q0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lim = float(np.abs(dq).max())
    pc = ax.pcolormesh(days, lat, dq.T, vmin=-lim, vmax=lim, cmap="RdBu_r",
                       shading="auto")
    ax.contour(days, lat, dq.T, levels=np.arange(-60, 61, 10), colors="k",
               linewidths=0.4)
    fig.colorbar(pc, label="6 ka minus 1850 [W/m2]")
    ax.set_xlabel("day of year (fixed-day calendar, equinox 21 Mar)")
    ax.set_ylabel("latitude")
    ax.set_title(f"Mid-Holocene minus pre-industrial insolation (legoESM {sha})")
    fig.savefig(out / "insolation_6ka_minus_1850.png", dpi=110)
    print(f"6ka orbit {orbit_of(orbit_year=-4050)}")
    print(f"1850 orbit {orbit_of(orbit_year=1850)}")
    print(f"6ka-1850 JJA mean at 65N: {dq[jja, i65].mean():+.1f} W/m2; "
          f"max {dq.max():+.1f}, min {dq.min():+.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
