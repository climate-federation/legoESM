"""Does the emitted NEMO trend set CLOSE the state change?

Before any operator can be tested against ``ttrd_totad`` / ``ttrd_ldf``, the
reconstruction those tests rely on has to be justified: the pre-operator
state is recovered by unwinding trends from the saved post-step state, and
that is only valid if the emitted trends account for (nearly) all of the
step's tendency.  ``ttrd_tot`` is empty in every RUN_TRD file, so closure has
to be measured directly:

    residual = (X(r) - X(r-1))/dt  -  sum(available trends at r)

Reported as an RMS ratio against the RMS of the state change, globally and
per depth band.  A small residual means the missing processes (surface
fluxes, bbl, damping, the free-surface/qco geometry terms) are negligible for
this window and the unwinding is sound; a large one names how much of the
budget is unaccounted, and any Stage-C/D number has to carry that caveat.

Usage:
    python scripts/validate/ocean_fidelity/nemo_trend_closure.py \
        --tfile .../ORCA1_1h_..._trd1h_T.nc --rec 1
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tfile", required=True)
    p.add_argument("--rec", type=int, default=1)
    p.add_argument("--dt", type=float, default=3600.0)
    a = p.parse_args()

    import netCDF4 as nc
    ds = nc.Dataset(a.tfile)

    def g(name, r):
        if name not in ds.variables:
            return None
        x = ds.variables[name][r]
        x = x.filled(np.nan) if np.ma.isMaskedArray(x) else np.asarray(x)
        return x.astype(np.float64)

    out = {}
    for tag, state_var, trends in (
            ("T", "votemper", ("ttrd_totad", "ttrd_ldf", "ttrd_zdf")),
            ("S", "vosaline", ("strd_totad", "strd_ldf", "strd_zdf"))):
        x0, x1 = g(state_var, a.rec - 1), g(state_var, a.rec)
        if x0 is None or x1 is None:
            print(f"[{tag}] state variable missing")
            continue
        dxdt = (x1 - x0) / a.dt
        avail, names = np.zeros_like(dxdt), []
        for tv in trends:
            v = g(tv, a.rec)
            if v is None or not np.isfinite(v).any():
                print(f"[{tag}] {tv}: absent/empty -- NOT in the sum")
                continue
            avail = avail + np.nan_to_num(v, nan=0.0)
            names.append(tv)
        wet = np.isfinite(dxdt) & np.isfinite(x0)
        res = np.where(wet, dxdt - avail, np.nan)
        rms = lambda v: float(np.sqrt(np.nanmean(np.where(wet, v, np.nan) ** 2)))
        r_state, r_res, r_tr = rms(dxdt), rms(res), rms(avail)
        print(f"[{tag}] summed {names}")
        print(f"[{tag}] rms d{tag}/dt {r_state:.4e} | rms trends {r_tr:.4e} | "
              f"rms residual {r_res:.4e}  -> residual/state = {r_res / r_state:.3f}")
        # Depth-banded: the surface band carries the unmodelled surface flux.
        nz = dxdt.shape[0]
        for k0, k1, lab in ((0, 1, "level 0 (surface)"), (1, 10, "levels 1-9"),
                            (10, nz, "levels 10+")):
            sl = slice(k0, k1)
            w = wet[sl]
            if not w.any():
                continue
            rs = float(np.sqrt(np.nanmean(np.where(w, dxdt[sl], np.nan) ** 2)))
            rr = float(np.sqrt(np.nanmean(np.where(w, res[sl], np.nan) ** 2)))
            print(f"    {lab:18s} rms state {rs:.4e}  residual {rr:.4e}  "
                  f"ratio {rr / rs if rs else float('nan'):.3f}")
        out[tag] = (r_state, r_res)
    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
