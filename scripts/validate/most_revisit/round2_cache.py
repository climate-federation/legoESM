#!/usr/bin/env python
"""Cache the benchmark-v5 gradient samples (zeta, phi_obs) per site, kind and day parity, so round-2 candidates
are fitted and scored on exactly the v5 samples without re-reading the tower tables.

Same call and settings as candidate_formulations.main (offset-corrected arm, min u* 0.10, max z_g 50 m);
parity 1 = odd UTC days (fit), parity 0 = even days (score). Also stores u*, H, 1/L per sample for the
self-correlation test.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import flux_gradient_sites as fs
import standard_formulations as sf


def second_level_inv_l(ds, min_ustar):
    """1/L from the next sonic above the lowest above-canopy one (independent u*, H, LE); NaN where absent or failing QC."""
    z = np.nanmedian(ds.z_f.values, axis=0)
    low = fs.lowest_flux_level(ds)
    above = np.where(z > z[low], z, np.nan)
    if not np.isfinite(above).any():
        return np.full(ds.sizes["time"], np.nan)
    j = int(np.nanargmin(above))
    f = {k: ds[k].values[:, j].astype(float) for k in ("ustar", "h", "le", "t_flux")}
    length = fs.obukhov_length(f["ustar"], fs.buoyancy_flux(f["h"], f["le"], f["t_flux"], ds.pressure.values.astype(float)), f["t_flux"])
    return np.where(fs.clean_flux(f, min_ustar), 1.0 / length, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ustar", type=float, required=True)
    ap.add_argument("--max-zg", type=float, required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    arrays = {}
    for path in sorted(glob.glob(f"{a.tables}/*.nc")):
        name = os.path.basename(path)[:-3]
        if name in fs.EXCLUDED:
            continue
        ds = fs.open_site(path)
        inv_l2 = second_level_inv_l(ds, a.min_ustar)
        for parity in (0, 1):
            pairs_, flux = sf.site_pairs(ds, a.min_ustar, True, a.max_zg, day_parity=parity)
            for kind, plist in pairs_.items():
                if not plist:
                    continue
                for field in ("zeta", "phi", "z1", "z2", "diff", "scale", "inv_l"):
                    arrays[f"{name}|{kind}|{parity}|{field}"] = np.concatenate([p[field][p["keep"]] for p in plist])
                for field in ("ustar", "h", "rho", "theta"):
                    arrays[f"{name}|{kind}|{parity}|{field}"] = np.concatenate([flux[field][p["keep"]] for p in plist])
                arrays[f"{name}|{kind}|{parity}|inv_l2"] = np.concatenate([inv_l2[p["keep"]] for p in plist])
                arrays[f"{name}|{kind}|{parity}|time"] = np.concatenate([ds.time.values.astype(float)[p["keep"]] for p in plist])
                arrays[f"{name}|{kind}|{parity}|pair"] = np.concatenate([np.full(int(p["keep"].sum()), k) for k, p in enumerate(plist)])
        print(name, flush=True)
    np.savez_compressed(a.out, **arrays)
    print("WROTE", a.out, len(arrays))


if __name__ == "__main__":
    main()
