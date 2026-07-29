"""Stitch NEMO per-rank restart tiles into global fields (run via SLURM).

The RUN_TRD restart tiles carry the PROGNOSTIC state the trend re-run
started from — including ``en`` (the zdftke TKE field), which no XIOS output
stream provides. Rebuilding it enables the EXACT Mode-A closure test: feed
NEMO's own (tn, sn, un, vn, en) at step 8760 to the legoESM TKE kernel and
compare one prognostic en-step's K_H against NEMO's ``avt`` of step 8761
(record 0 of the hourly trend file) — same state, same carry, closure vs
closure with nothing else in the loop.

Tiles use the standard DOMAIN_position_first/last attributes (1-based,
inclusive; halo 0 in these files), stitched into (nlev, ny_global,
nx_global) with NaN outside written regions.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np


def rebuild(pattern: str, fields: list[str]) -> dict:
    import netCDF4 as nc

    tiles = sorted(glob.glob(pattern))
    if not tiles:
        raise SystemExit(f"no tiles match {pattern}")
    out: dict[str, np.ndarray] = {}
    for path in tiles:
        ds = nc.Dataset(path)
        nxg, nyg = (int(v) for v in ds.DOMAIN_size_global)
        x0, y0 = (int(v) for v in ds.DOMAIN_position_first)
        x1, y1 = (int(v) for v in ds.DOMAIN_position_last)
        hx0, hy0 = (int(v) for v in ds.DOMAIN_halo_size_start)
        hx1, hy1 = (int(v) for v in ds.DOMAIN_halo_size_end)
        for name in fields:
            if name not in ds.variables:
                continue
            v = ds.variables[name]
            a = v[:]
            a = a.filled(np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
            a = np.squeeze(a)                      # drop time_counter
            if name not in out:
                shape = ((a.shape[0], nyg, nxg) if a.ndim == 3
                         else (nyg, nxg))
                out[name] = np.full(shape, np.nan, dtype=np.float64)
            # strip halos (0 here, but keep the general form), then place
            sl_y = slice(hy0, a.shape[-2] - hy1 or None)
            sl_x = slice(hx0, a.shape[-1] - hx1 or None)
            core = a[..., sl_y, sl_x]
            out[name][..., y0 - 1 + hy0:y1 - hy1, x0 - 1 + hx0:x1 - hx1] = core
        ds.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="glob for the oce restart tiles, e.g. .../ORCA1_00008760_restart_oce_00*.nc")
    ap.add_argument("--fields", nargs="+",
                    default=["tn", "sn", "un", "vn", "sshn", "en"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = rebuild(args.pattern, args.fields)
    for k, v in out.items():
        wet = np.isfinite(v).sum()
        print(f"{k}: shape={v.shape} finite={wet} "
              f"absmax={np.nanmax(np.abs(v)):.4e}")
    np.savez_compressed(args.out, **out)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
