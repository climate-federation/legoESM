"""Probe: WMO tropopause (forcing.surface_utils) on real FV3-duo checkpoints.

Measures, per daily checkpoint: fraction of columns falling back to the
climatological tropopause (and their |lat|, jump vs 8 nearest neighbours),
WMO p_tp percentiles, and day-to-day change of the tropopause LAYER index
(>=1/>=2 levels, by latitude band, A->B->A flip-backs).  Checkpoints carry
``T`` (ncol, nlev) TOA-first, ``p_s`` (ncol,), ``meta_vgrid`` = (hyai, hybi)
with hyai normalised by 1e5 Pa; full level = mean of the bounding half levels.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python check_wmo_tropopause_fv3duo.py \
        --mesh duo_c24_mesh_latlon.npz checkpoint_day_0001.npz ...
"""

import argparse
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.forcing.surface_utils import wmo_tropopause_pressure


def load(f):
    d = np.load(f)
    a, b = d["meta_vgrid"]
    a = a * 1.0e5 if a.max() < 1.0 else a  # normalised hyai -> Pa
    ph = a[None, :] + b[None, :] * d["p_s"][:, None]
    return d["T"], 0.5 * (ph[:, 1:] + ph[:, :-1]), ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="npz with lat, lon [rad]")
    ap.add_argument("ckpts", nargs="+", help="consecutive checkpoints")
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=Path(__file__).parent).stdout.strip()
    print(f"code {sha}; x64={jax.config.jax_enable_x64}; inputs {a.ckpts}")
    ll = np.load(a.mesh)
    lat, lon = ll["lat"], ll["lon"]
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    nbr = np.argsort(-(xyz @ xyz.T), axis=1)[:, 1:9]
    alat = np.abs(np.degrees(lat))
    p_clim = (300.0 - 200.0 * np.cos(lat) ** 2) * 100.0
    levs = []
    for f in a.ckpts:
        temp, pf, ph = load(f)
        pw = np.asarray(wmo_tropopause_pressure(jnp.asarray(pf), jnp.asarray(temp)))
        fb = np.isnan(pw)
        pe = np.where(fb, p_clim, pw)
        levs.append(np.array([np.searchsorted(ph[i], pe[i]) - 1 for i in range(len(pe))]))
        jump = np.abs(pe - np.median(pe[nbr], axis=1)) / 100
        print(f"{Path(f).name}: fallback {fb.mean():.2%} ({fb.sum()}/{fb.size})"
              + (f" |lat| {np.round(alat[fb], 0)[:20]} nbr-jump hPa median"
                 f" {np.median(jump[fb]):.1f} max {jump[fb].max():.1f}" if fb.any() else "")
              + f"; WMO p_tp hPa p5/50/95 {np.round(np.percentile(pw[~fb], [5, 50, 95]) / 100, 1)}"
              f"; median WMO-clim {np.median(pw[~fb] - p_clim[~fb]) / 100:.1f} hPa"
              f"; p_tp>400 hPa {(pw[~fb] > 4e4).sum()} cols")
    lev = np.array(levs)
    d = np.abs(np.diff(lev, axis=0))
    for name, m in (("all", alat >= 0), ("|lat|<30", alat < 30),
                    ("30-60", (alat >= 30) & (alat < 60)), (">=60", alat >= 60)):
        print(f"layer change between consecutive ckpts [{name}, {m.sum()} cols]:"
              f" >=1 {np.mean(d[:, m] >= 1):.1%}, >=2 {np.mean(d[:, m] >= 2):.1%}")
    if len(lev) > 2:
        chg = lev[1:-1] != lev[:-2]
        back = chg & (lev[2:] == lev[:-2])
        print(f"A->B->A flip-backs: {back.sum() / max(chg.sum(), 1):.1%} of changes")


if __name__ == "__main__":
    main()
