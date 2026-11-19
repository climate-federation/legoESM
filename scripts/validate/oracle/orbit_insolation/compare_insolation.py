"""Model daily-mean TOA insolation vs climlab and published orbital tables.

Three modes, run under two interpreters (climlab lives in a throwaway env):
  model   <out.npz>                     legoESM env: orbits + insolation maps
  climlab <model.npz> <out.npz>         climlab env: same orbits, climlab maps
                                        and climlab's Berger/Laskar elements
  compare <model.npz> <climlab.npz> <png_dir>   legoESM env: tables + PNGs

Calendar conventions (named, not tuned away):
  legoESM: day 1.0 = 1 Jan 00Z, vernal equinox at calday 80.5, 365-day year.
  climlab: day_type=1, equinox at day 80, days_per_year 365.2422 by default.
"matched" comparison feeds climlab day-0.5 and a 365-day year; "raw" uses
climlab defaults on the same day numbers.
"""

from __future__ import annotations

import sys

import numpy as np

LAT = np.linspace(-90.0, 90.0, 181)
DAYS = np.arange(1.0, 366.0)
# PMIP4 Table 1 (Otto-Bliesner et al. 2017, GMD 10:3979): ecc, obliquity,
# perihelion-180; vernal equinox fixed to noon 21 March.
PMIP4 = {"1850": (0.016764, 23.459, 100.33), "6ka": (0.018682, 24.105, 0.87),
         "127ka": (0.039378, 24.040, 275.41)}
YEARS = {"1850": 1850, "6ka": 1950 - 6000, "127ka": 1950 - 127000}
OBLIQUITY_CASES = (0.0, 60.0, 90.0)


def model(out):
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
    from legoesm.driver.orbit import experiment_orbit

    jax.config.update("jax_enable_x64", True)

    def orbit(**kw):
        cfg = dict(orbital_insolation=True, orbit_year=None, orbit_eccentricity=None,
                   orbit_obliquity_deg=None, orbit_long_perihelion_deg=None)
        cfg.update(kw)
        return experiment_orbit(SimpleNamespace(**cfg))

    cases = {"present": orbit()}
    cases.update({k: orbit(orbit_year=y) for k, y in YEARS.items()})
    cases.update({f"obl{o:g}": orbit(orbit_obliquity_deg=o) for o in OBLIQUITY_CASES})
    lat = jnp.deg2rad(jnp.asarray(LAT))
    res = {"names": np.array(list(cases)), "S0": constants.S_0}
    for k, o in cases.items():
        res[f"orb_{k}"] = np.array([o.eccentricity, o.obliquity_deg, o.long_perihelion_deg])
        res[f"q_{k}"] = np.asarray(jax.vmap(
            lambda d: daily_mean_insolation(lat, d, orbit=o))(jnp.asarray(DAYS)))
    np.savez(out, **res)


def climlab_side(model_npz, out):
    import inspect

    import climlab
    from climlab.solar.insolation import daily_insolation
    from climlab.solar.orbital import OrbitalTable

    m = np.load(model_npz)
    res = {"climlab_version": climlab.__version__}
    if "days_per_year" not in inspect.signature(daily_insolation).parameters:
        raise RuntimeError("this climlab cannot take a 365-day year; no matched calendar")
    res["has_days_per_year"] = True
    for k in m["names"]:
        ecc, obl, peri = m[f"orb_{k}"]
        orb = {"ecc": ecc, "obliquity": obl, "long_peri": peri}
        # climlab takes 1-D lat, day and returns (lat, day).
        raw = daily_insolation(LAT, DAYS, orb=orb, S0=float(m["S0"]))
        matched = daily_insolation(LAT, DAYS - 0.5, orb=orb,
                                   S0=float(m["S0"]), days_per_year=365.0)
        for name, q in (("raw", raw), ("matched", matched)):
            q = np.asarray(q)
            assert q.shape == (LAT.size, DAYS.size), q.shape
            res[f"{name}_{k}"] = q.T
    berger = OrbitalTable
    for k, y in YEARS.items():
        kyear = (y - 1950) / 1000.0
        o = berger.interp(kyear=kyear)
        res[f"berger_{k}"] = np.array([float(o["ecc"]), float(o["obliquity"]),
                                       float(o["long_peri"])])
    try:
        from climlab.solar.orbital.long import OrbitalTable as LaskarTable
        for k, y in YEARS.items():
            o = LaskarTable.interp(kyear=(y - 1950) / 1000.0)
            res[f"laskar_{k}"] = np.array([float(o["ecc"]), float(o["obliquity"]),
                                           float(o["long_peri"])])
    except Exception as exc:  # report, never substitute
        res["laskar_error"] = repr(exc)
    np.savez(out, **res)


def compare(model_npz, climlab_npz, png_dir):
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, c = np.load(model_npz), np.load(climlab_npz, allow_pickle=True)
    out = Path(png_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"climlab {c['climlab_version']}  days_per_year arg: {bool(c['has_days_per_year'])}")
    print("orbital elements: model | PMIP4 table | climlab Berger(1991 table) | climlab Laskar04")
    for k in YEARS:
        e, o, p = m[f"orb_{k}"]
        pe, po, pp = PMIP4[k]
        line = (f"  {k:6s} model e={e:.6f} obl={o:.3f} peri-180={(p - 180) % 360:.2f} | "
                f"PMIP4 {pe:.6f} {po:.3f} {pp:.2f} | "
                f"d(e)={e - pe:+.1e} d(obl)={o - po:+.4f} d(peri)={((p - 180) % 360) - pp:+.3f}")
        for src in ("berger", "laskar"):
            if f"{src}_{k}" in c:
                be, bo, bp = c[f"{src}_{k}"]
                line += f" | {src} {be:.6f} {bo:.3f} {(bp - 180) % 360:.2f}"
        print(line)
    if "laskar_error" in c:
        print(f"  Laskar table unavailable: {c['laskar_error']}")

    jja = (DAYS >= 152) & (DAYS <= 243)
    ja = (DAYS >= 182) & (DAYS <= 243)
    jun = (DAYS >= 152) & (DAYS <= 181)
    i65 = int(np.argmin(abs(LAT - 65)))
    band = (LAT >= 40) & (LAT <= 50)
    for k in ("6ka", "127ka"):
        d = m[f"q_{k}"] - m["q_1850"]
        dc = c[f"matched_{k}"] - c["matched_1850"]
        print(f"  {k}-1850 model: 65N June {d[jun, i65].mean():+.1f}, 65N JJA {d[jja, i65].mean():+.1f}, "
              f"40-50N Jul-Aug {d[ja][:, band].mean():+.1f} W/m2 | climlab(matched) "
              f"65N June {dc[jun, i65].mean():+.1f}, 40-50N Jul-Aug {dc[ja][:, band].mean():+.1f}")

    print("model - climlab, max |diff| W/m2 (tolerance 0.1):")
    names = list(m["names"])
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.5),
                             sharex=True, sharey=True)
    for j, k in enumerate(names):
        for i, kind in enumerate(("matched", "raw")):
            d = m[f"q_{k}"] - c[f"{kind}_{k}"]
            mx = float(np.abs(d).max())
            if i == 0:
                print(f"  {k:8s} matched {mx:.3e}  raw {float(np.abs(m[f'q_{k}'] - c[f'raw_{k}']).max()):.3e}"
                      f"{'  <-- FINDING' if mx > 0.1 else ''}")
            ax = axes[i, j]
            lim = max(mx, 1e-12)
            pc = ax.pcolormesh(DAYS, LAT, d.T, vmin=-lim, vmax=lim, cmap="RdBu_r",
                               shading="auto")
            ax.set_title(f"{k} {kind}\nmax {mx:.2g}", fontsize=8)
            fig.colorbar(pc, ax=ax, fraction=0.05)
    axes[0, 0].set_ylabel("lat (matched calendar)")
    axes[1, 0].set_ylabel("lat (raw climlab calendar)")
    fig.suptitle("legoESM minus climlab daily-mean TOA insolation [W/m2]")
    fig.savefig(out / "insolation_minus_climlab.png", dpi=100)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"model": model, "climlab": climlab_side, "compare": compare}[mode](*args)
