"""Jet-decay / small-scale-energy probe for the AMIP decks (spectral vs lat-lon).

Runs ``run_amip.main(argv)`` unchanged and prints, at run entry ("day0") and
exit ("end"), for the 120-200 hPa band:
  * point max |u|, max zonal-mean u (and its lat / pressure), area-weighted KE;
  * the cos(lat)-weighted zonal-wavenumber KE spectrum of u and v on band
    levels: the share of eddy KE (m >= 1) in the tail band ``TAIL_M`` and the
    log-log slope over ``SLOPE_M``.

``TAIL_M`` is the top of what the T31 spectral state RETAINS: the spectral PE
dealiases at total wavenumber floor(0.667 * 31) = 20, so zonal m > 20 is zero
on that lane by construction (a tail band above it reads ~1e-32 on every arm
and cannot rank them). The same band is used on lat-lon so the two grids are
compared on the same scales. A spectral model whose retained cut (its own
grid n_max and dealiasing fraction) is below ``TAIL_M[1]`` is refused. The
spectrum is taken on the fixed model levels whose reference-pressure
(p_s = constants.p_ref) value lies in the band, so every call samples the
same population; jet and KE use each column's own pressure.

Grid must have exact latitude rows (Gaussian / lat-lon). Prints one provenance
line (git sha, dirty count, argv) first.

  python scripts/validate/spectral_jet_decay_probe.py --config <deck> ... (run_amip args)
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

BAND_HPA = (120.0, 200.0)
TAIL_M = (16, 20)
SLOPE_M = (8, 20)


def band_diagnostics(lat, lon, u, v, p_hpa, spec_levels, band=BAND_HPA,
                     tail_m=TAIL_M, slope_m=SLOPE_M) -> dict:
    """Band jet / KE / zonal-spectrum numbers from column fields.

    lat, lon: (ncol,) radians; u, v, p_hpa: (ncol, nlev). Jet and KE use the
    columns' own pressure; the spectrum uses the fixed level set
    ``spec_levels`` (same population at every call, every column).
    """
    from legoesm.ocean.diagnostics import zonal_power_spectrum

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    u = np.asarray(u)
    v = np.asarray(v)
    p = np.asarray(p_hpa)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("non-finite wind")
    if p.shape != u.shape or v.shape != u.shape or u.ndim != 2 or u.shape[0] != lat.size:
        raise ValueError(f"shape mismatch u {u.shape} v {v.shape} p {p.shape} lat {lat.shape}")
    spec_levels = np.asarray(spec_levels, dtype=int)
    if spec_levels.size == 0:
        raise ValueError("no spectrum levels")
    inb = (p >= band[0]) & (p <= band[1])
    rows, inv = np.unique(np.round(lat, 8), return_inverse=True)
    cnt = np.bincount(inv)
    if cnt.min() != cnt.max():
        raise ValueError(f"latitude rows uneven: {cnt.min()}..{cnt.max()}")
    nlon = int(cnt[0])
    if tail_m[1] > nlon // 2:
        raise ValueError(f"tail band {tail_m} beyond Nyquist of {nlon} longitudes")
    nk = u.shape[1]
    ubar = np.stack([np.bincount(inv, u[:, k]) / cnt for k in range(nk)], 1)
    pbar = np.stack([np.bincount(inv, p[:, k]) / cnt for k in range(nk)], 1)
    bb = (pbar >= band[0]) & (pbar <= band[1])
    j = np.unravel_index(np.argmax(np.where(bb, ubar, -np.inf)), ubar.shape)
    w = np.cos(lat)[:, None] * inb
    ke = float(np.sum(w * 0.5 * (u ** 2 + v ** 2)) / np.sum(w))

    spec = np.zeros(nlon // 2 + 1)
    wsum = 0.0
    for r in range(rows.size):
        idx = np.where(inv == r)[0]
        idx = idx[np.argsort(lon[idx])]
        c = math.cos(rows[r])
        for f in (u, v):
            _, P = zonal_power_spectrum(f[idx][:, spec_levels].T, 1.0, detrend=False)
            spec += c * 0.5 * np.asarray(P)
        wsum += c
    spec /= wsum
    eddy = spec[1:].sum()
    m = np.arange(slope_m[0], slope_m[1] + 1)
    return {
        "point_max_u": float(np.max(np.where(inb, np.abs(u), -np.inf))),
        "zonal_max_u": float(ubar[j]),
        "zonal_max_lat_deg": float(np.degrees(rows[j[0]])),
        "zonal_max_p_hpa": float(pbar[j]),
        "ke_band": ke,
        "nlon": nlon,
        "spectrum": spec,
        "tail_frac": float(spec[tail_m[0]:tail_m[1] + 1].sum() / eddy),
        "slope": float(np.polyfit(np.log(m), np.log(spec[m]), 1)[0]),
    }


def retained_zonal_cut(n_max: int, dealiasing_fraction: float) -> int:
    """Highest total (so zonal) wavenumber a spectral PE state keeps."""
    if dealiasing_fraction <= 0:
        return int(n_max)
    return int(math.floor(dealiasing_fraction * n_max))


def _report(driver, tag):
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    dyn = driver.model
    sgrid = getattr(dyn, "grid", None)
    n_max = getattr(sgrid, "n_max", None)
    if n_max is not None:
        cut = retained_zonal_cut(n_max, float(dyn.config.dealiasing_fraction))
        if cut < TAIL_M[1]:
            raise SystemExit(f"PROBE tail band {TAIL_M} above retained cut {cut} (n_max {n_max})")
    st = driver.state
    p = np.asarray(jax.vmap(driver.sigma.pressure_at_full)(jnp.asarray(st.p_s.data))) / 100
    p_ref = np.asarray(driver.sigma.pressure_at_full(jnp.asarray(constants.p_ref))) / 100
    lev = np.where((p_ref >= BAND_HPA[0]) & (p_ref <= BAND_HPA[1]))[0]
    grid = driver.grid
    d = band_diagnostics(grid.latCell, grid.lonCell, st.u.data, st.v.data, p, lev)
    print(f"PROBE {tag} spectrum nlon {d['nlon']} levels {lev.tolist()} "
          f"tail_frac_m{TAIL_M[0]}_{TAIL_M[1]}_of_eddy {d['tail_frac']:.4e} "
          f"slope_m{SLOPE_M[0]}_{SLOPE_M[1]} {d['slope']:.3f}")
    print(f"PROBE {tag} point_max_u_band {d['point_max_u']:.3f} zonal_max_u_band "
          f"{d['zonal_max_u']:.3f} at lat {d['zonal_max_lat_deg']:.2f} p "
          f"{d['zonal_max_p_hpa']:.1f} ke_band {d['ke_band']:.3f}")


def main(argv):
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "-uno"],
                           capture_output=True, text=True).stdout.count("\n")
    print(f"PROBE provenance sha {sha} dirty {dirty} argv {argv}")
    from legoesm.driver import model_driver as md
    orig = md.ModelDriver.run

    def _wrap(self, *a, **k):
        _report(self, "day0")
        r = orig(self, *a, **k)
        _report(self, "end")
        return r

    md.ModelDriver.run = _wrap
    sys.path.insert(0, str(root / "scripts" / "run"))
    import run_amip
    run_amip.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
