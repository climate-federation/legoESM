"""Validate plane-LES profiles against the jax-alfa oracle's physical targets.

The jax-alfa LES oracle (Bou-Zeid LASD SGS) ships no reference dumps, so the
shared ground truth is **Monin-Obukhov similarity theory (MOST)** — the
surface-layer scaling every faithful ABL LES (oracle and legoESM alike) must
reproduce in the neutral limit:

  * log-law mean wind          U(z) = (u_*/κ) ln(z/z0)
  * non-dimensional shear      φ_m(z) = (κz/u_*) dU/dz → 1  (neutral surface layer)
  * resolved velocity variances σ_w/u_* ≈ 1.25, σ_u/u_* ≈ 2.4 (Stull 1988)
  * constant-flux layer        |τ|(z) ≈ u_*² near the surface, → 0 at the BL top
  * friction velocity          u_* from the lowest-level resolved stress

Reads the ``final_profiles.npz`` written by ``run_les_plane.py`` and reports each
diagnostic against its MOST target with a tolerance, plus an overall verdict.

Usage
-----
.. code-block:: bash

   .venv/bin/python scripts/validate/validate_les_vs_oracle.py \\
       results/les_neutral/final_profiles.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from legoesm import constants

_KAPPA = constants.kappa_von_karman

# MOST neutral surface-layer targets (Stull 1988, Table 9.x; Garratt 1992).
_SIGMA_W_OVER_USTAR = 1.25
_SIGMA_U_OVER_USTAR = 2.40


def log_law_fit(z, U, z0, z_lo=None, z_hi=None):
    """Least-squares u_* from U = (u_*/κ) ln(z/z0) over a surface-layer band."""
    z = np.asarray(z); U = np.asarray(U)
    order = np.argsort(z)
    z, U = z[order], U[order]
    z_lo = z_lo or z[z > 0].min()
    z_hi = z_hi or 0.2 * z.max()            # surface layer ~ lowest 10-20% of BL
    band = (z >= z_lo) & (z <= z_hi) & (z > z0)
    x = np.log(z[band] / z0)                # U = (u_*/κ) x → slope = u_*/κ
    slope = np.polyfit(x, U[band], 1)[0]
    return slope * _KAPPA, band.sum()


def phi_m(z, U, u_star):
    """Non-dimensional shear φ_m = (κ z / u_*) dU/dz (→ 1 in the neutral SL)."""
    z = np.asarray(z); U = np.asarray(U)
    order = np.argsort(z)
    z, U = z[order], U[order]
    dUdz = np.gradient(U, z)
    return z, _KAPPA * z / max(u_star, 1e-9) * dUdz


def _check(name, value, target, tol, unit=""):
    ok = np.isfinite(value) and abs(value - target) <= tol
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:<28} {value:8.3f}{unit}  "
          f"(target {target:.3f} ± {tol:.3f})")
    return ok


def validate(npz_path: Path) -> int:
    d = np.load(npz_path, allow_pickle=True)
    case = str(d["case"]) if "case" in d else "?"
    z = d["z"]; u = d["u"]; v = d["v"]
    z0 = float(d["z0"]) if "z0" in d else 0.1
    U = np.sqrt(u ** 2 + v ** 2)
    u_star_flux = float(d["u_star"]) if "u_star" in d else np.nan
    ww = d["ww"] if "ww" in d else d.get("wvar")
    uu = d["uu"] if "uu" in d else None

    print(f"== LES vs MOST oracle targets: case={case}  file={npz_path.name} ==")
    print(f"  resolved-stress u_*        = {u_star_flux:.3f} m/s   z0 = {z0} m")

    results = []
    # 1. log-law u_* vs the resolved-stress u_* (the two must agree if the
    #    surface layer is in MOST equilibrium).
    u_star_log, npts = log_law_fit(z, U, z0)
    print(f"  log-law-fit u_* ({npts} pts) = {u_star_log:.3f} m/s")
    if np.isfinite(u_star_flux) and u_star_flux > 1e-3:
        results.append(_check("u_*(log) / u_*(flux)",
                              u_star_log / u_star_flux, 1.0, 0.35))

    # 2. φ_m ≈ 1 averaged over the surface layer.
    zc, phim = phi_m(z, U, u_star_flux if u_star_flux > 1e-3 else u_star_log)
    sl = (zc > z0) & (zc <= 0.2 * z.max())
    phim_sl = float(np.nanmean(phim[sl])) if sl.any() else np.nan
    results.append(_check("phi_m (surface-layer mean)", phim_sl, 1.0, 0.5))

    # 3. resolved variance similarity (needs developed turbulence).
    u_star = u_star_flux if u_star_flux > 1e-3 else u_star_log
    if u_star > 1e-3:
        # Sample the SURFACE LAYER (z ~ 0.1·BL depth), NOT the wall-adjacent cell:
        # w' → 0 at the wall by the rigid BC, so σ_w at the first cell is
        # artificially low; the MOST similarity value applies in the surface
        # layer above the immediate wall. (Plain `argmin(z)` understates σ_w/u_*.)
        order = np.argsort(z)
        zs = z[order]
        k = int(order[np.argmin(np.abs(zs - 0.1 * zs.max()))])
        sig_w = float(np.sqrt(max(ww[k], 0.0))) / u_star
        results.append(_check("sigma_w / u_* (z~0.1h)", sig_w,
                              _SIGMA_W_OVER_USTAR, 0.6))
        if uu is not None:
            sig_u = float(np.sqrt(max(uu[k], 0.0))) / u_star
            results.append(_check("sigma_u / u_* (z~0.1h)", sig_u,
                                  _SIGMA_U_OVER_USTAR, 1.2))

    n_pass = sum(results)
    verdict = "PASS" if n_pass == len(results) and results else "PARTIAL/FAIL"
    print(f"  --> {n_pass}/{len(results)} diagnostics within tolerance  [{verdict}]")
    if u_star_flux < 0.05:
        print("  NOTE: u_* < 0.05 m/s — turbulence not yet developed; run longer "
              "(GPU) before trusting the similarity comparison.")
    return 0 if verdict == "PASS" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="final_profiles.npz from run_les_plane")
    args = p.parse_args()
    if not args.npz.exists():
        print(f"missing: {args.npz}", file=sys.stderr)
        return 1
    return validate(args.npz)


if __name__ == "__main__":
    sys.exit(main())
