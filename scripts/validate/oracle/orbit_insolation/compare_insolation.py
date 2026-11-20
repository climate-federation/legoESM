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
climlab defaults on the same day numbers (a half-day plus year-length offset in
the day index, so up to ~5 W/m2 where insolation changes fastest, e.g. spring
polar edges; a convention mismatch, not physics).

climlab's solar longitude is Berger's cubic series (invalid at high e), so the
e=0.9 case feeds climlab the solar longitude from an independent numpy
bisection Kepler solve (day_type=2); its series result is reported separately.
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
HIGH_ECC = 0.9


def kepler_true_longitude_deg(day, ecc, long_peri_deg):
    """Independent reference: Kepler's equation by bisection (numpy fp64),
    PMIP4 anchoring (true longitude 0 at calday 80.5, 365-day year)."""
    w = np.deg2rad(long_peri_deg)
    e_eq = 2.0 * np.arctan2(np.sqrt(1 - ecc) * np.sin(-w / 2), np.sqrt(1 + ecc) * np.cos(-w / 2))
    m = np.mod(e_eq - ecc * np.sin(e_eq) + (day - 80.5) * 2 * np.pi / 365.0, 2 * np.pi)
    lo, hi = np.zeros_like(m), np.full_like(m, 2 * np.pi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        above = mid - ecc * np.sin(mid) - m > 0
        lo, hi = np.where(above, lo, mid), np.where(above, mid, hi)
    ea = 0.5 * (lo + hi)
    nu = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(ea / 2), np.sqrt(1 - ecc) * np.cos(ea / 2))
    return np.rad2deg(nu + w) % 360.0


def model(out):
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.atmosphere.physics.radiation.solar import (
        daily_mean_insolation, earth_sun_distance_factor, solar_declination,
    )
    from legoesm.driver.orbit import experiment_orbit

    jax.config.update("jax_enable_x64", True)

    def orbit(**kw):
        cfg = dict(orbital_insolation=True, orbit_year=None, orbit_eccentricity=None,
                   orbit_obliquity_deg=None, orbit_long_perihelion_deg=None,
                   orbit_solver="berger_series")
        cfg.update(kw)
        return experiment_orbit(SimpleNamespace(**cfg))

    cases = {"present": orbit()}
    cases.update({k: orbit(orbit_year=y) for k, y in YEARS.items()})
    cases.update({f"obl{o:g}": orbit(orbit_obliquity_deg=o) for o in OBLIQUITY_CASES})
    cases["e09"] = orbit(orbit_solver="kepler", orbit_eccentricity=HIGH_ECC)
    lat = jnp.deg2rad(jnp.asarray(LAT))
    res = {"names": np.array(list(cases)), "S0": constants.S_0}
    for k, o in cases.items():
        res[f"orb_{k}"] = np.array([o.eccentricity, o.obliquity_deg, o.long_perihelion_deg])
        res[f"q_{k}"] = np.asarray(jax.vmap(
            lambda d: daily_mean_insolation(lat, d, orbit=o))(jnp.asarray(DAYS)))
        res[f"decl_{k}"] = np.asarray(solar_declination(jnp.asarray(DAYS), orbit=o))
        res[f"fac_{k}"] = np.asarray(earth_sun_distance_factor(jnp.asarray(DAYS), o))
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
        if ecc > 0.1:  # climlab's series is invalid here: exact longitude in
            res[f"series_{k}"] = np.asarray(matched).T
            matched = daily_insolation(
                LAT, kepler_true_longitude_deg(DAYS, ecc, peri), orb=orb,
                S0=float(m["S0"]), day_type=2)
        for name, q in (("raw", raw), ("matched", matched)):
            q = np.asarray(q)
            assert q.shape == (LAT.size, DAYS.size), q.shape
            res[f"{name}_{k}"] = q.T
    # Detection control: a planted 0.05-day calendar error on the present day.
    ecc, obl, peri = m["orb_present"]
    res["control_present"] = np.asarray(daily_insolation(
        LAT, DAYS - 0.5 + 0.05, orb={"ecc": ecc, "obliquity": obl, "long_peri": peri},
        S0=float(m["S0"]), days_per_year=365.0)).T
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

    def cap_shortfall(k):
        # Model's AD-safe clip of cos(sunset angle) to +-(1-1e-7), minus the
        # unclipped integral, from the model's own declination and (a/r)^2.
        phi = np.deg2rad(LAT)[None, :]
        dl = m[f"decl_{k}"][:, None]
        f = m[f"fac_{k}"][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = -np.sin(phi) * np.sin(dl) / (np.cos(phi) * np.cos(dl))
        ratio = np.where(np.isfinite(ratio), ratio, np.sign(-np.sin(phi) * np.sin(dl)) * 1e30)

        def q(h):
            return np.maximum(float(m["S0"]) / np.pi * f * (
                h * np.sin(phi) * np.sin(dl) + np.cos(phi) * np.cos(dl) * np.sin(h)), 0.0)

        return (q(np.arccos(np.clip(ratio, -(1 - 1e-7), 1 - 1e-7)))
                - q(np.arccos(np.clip(ratio, -1.0, 1.0))))

    print("residual after removing the predicted sunset-cap shortfall (W/m2):")
    for k in m["names"]:
        r = m[f"q_{k}"] - (c[f"matched_{k}"] + cap_shortfall(k))
        print(f"  {k:8s} max |residual| {float(np.abs(r).max()):.2e}")
    r = m["q_present"] - (c["control_present"] + cap_shortfall("present"))
    print(f"  control (planted 0.05-day calendar shift, present): max |residual| "
          f"{float(np.abs(r).max()):.2e}  <- must exceed 0.1 to prove detection")

    print("model - climlab (matched calendar), max |diff| W/m2 (tolerance 0.1):")
    for k in m["names"]:
        q, qc = m[f"q_{k}"], c[f"matched_{k}"]
        d = q - qc
        mx = float(np.abs(d).max())
        extra = (f"  [climlab series at this e: max |diff| "
                 f"{float(np.abs(q - c[f'series_{k}']).max()):.1f}]"
                 if f"series_{k}" in c else "")
        print(f"  {k:8s} {mx:.3e}  raw-calendar {float(np.abs(q - c[f'raw_{k}']).max()):.2e}"
              f"{'  <-- FINDING' if mx > 0.1 else ''}{extra}")
        sol = int(np.argmin(abs(DAYS - 172)))
        print(f"           realism: max {q.max():.1f} at lat {LAT[np.unravel_index(q.argmax(), q.shape)[1]]:+.0f} "
              f"day {DAYS[np.unravel_index(q.argmax(), q.shape)[0]]:.0f}; day172 90N {q[sol, -1]:.1f} "
              f"vs equator {q[sol, 90]:.1f}; zero cells {np.mean(q == 0.0):.3f}")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True,
                                 constrained_layout=True)
        vmax = float(max(q.max(), qc.max()))
        for ax, field, title in ((axes[0], q, "legoESM"), (axes[1], qc, "climlab")):
            pc = ax.pcolormesh(DAYS, LAT, field.T, vmin=0.0, vmax=vmax, cmap="magma",
                               shading="auto")
            ax.contour(DAYS, LAT, field.T, levels=[1e-6], colors="c", linewidths=0.6)
            ax.set_title(title)
            ax.set_xlabel("day of year")
        fig.colorbar(pc, ax=axes[:2], label="daily-mean TOA insolation [W/m2]")
        lim = max(mx, 1e-12)
        pd = axes[2].pcolormesh(DAYS, LAT, d.T, vmin=-lim, vmax=lim, cmap="RdBu_r",
                                shading="auto")
        axes[2].set_title(f"legoESM minus climlab (max |d| {mx:.2g})")
        axes[2].set_xlabel("day of year")
        fig.colorbar(pd, ax=axes[2], label="W/m2")
        axes[0].set_ylabel("latitude")
        e, o, p = m[f"orb_{k}"]
        fig.suptitle(f"{k}: e={e:.4f} obliquity={o:.2f} perihelion={p:.1f} deg "
                     "(matched calendar; cyan = polar-night edge)")
        fig.savefig(out / f"climlab_{k}.png", dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"model": model, "climlab": climlab_side, "compare": compare}[mode](*args)
