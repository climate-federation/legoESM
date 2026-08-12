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
            ("T", "votemper", ("ttrd_totad", "ttrd_ldf", "ttrd_iso",
                               "ttrd_zdf", "ttrd_bbl", "ttrd_qsr", "ttrd_qns")),
            ("S", "vosaline", ("strd_totad", "strd_ldf", "strd_iso",
                               "strd_zdf", "strd_bbl"))):
        x0, x1 = g(state_var, a.rec - 1), g(state_var, a.rec)
        if x0 is None or x1 is None:
            print(f"[{tag}] state variable missing")
            continue
        # key_qco VARIABLE-VOLUME form (trdtra.F90; codex tendency-instrument
        # review RED-2): NEMO's reported trend is
        #     (e3t_aa*X_aa - e3t_bb*X_bb) / (e3t_mm*dt),
        # NOT the raw (X_aa - X_bb)/dt -- the free-surface thickness change
        # rides in the trend.  Comparing the raw state change against the
        # trends left a residual LARGER than the state change itself
        # (measured 2026-08-12: ratio 2.8 for T).  Use the thickness-weighted
        # form when e3t is available, and say which form was used.
        e0, e1 = g("e3t", a.rec - 1), g("e3t", a.rec)
        if e0 is not None and e1 is not None:
            emid = 0.5 * (e0 + e1)
            dxdt = np.where(emid > 0, (e1 * x1 - e0 * x0) / (np.where(emid > 0, emid, 1.0) * a.dt), np.nan)
            form = "thickness-weighted (qco)"
        else:
            dxdt = (x1 - x0) / a.dt
            form = "RAW state difference (no e3t in file)"
        print(f"[{tag}] state-change form: {form}")
        avail, names = np.zeros_like(dxdt), []
        for tv in trends:
            v = g(tv, a.rec)
            if v is None or not np.isfinite(v).any():
                print(f"[{tag}] {tv}: absent/empty -- NOT in the sum")
                continue
            avail = avail + np.nan_to_num(v, nan=0.0)
            names.append(tv)
            # PER-TERM magnitude by band: a single term that is huge in the
            # interior is a units/definition problem, not a closure gap
            # (measured 2026-08-12: adding qsr+qns fixed the SURFACE residual
            # 5.8 -> 0.46 but blew the interior up to 37, so one added term
            # carries an interior signal it should not).
            nz_ = v.shape[0]
            b = []
            for k0, k1, lab in ((0, 1, "sfc"), (1, 10, "1-9"), (10, nz_, "10+")):
                seg = v[k0:k1]
                b.append(f"{lab} {float(np.sqrt(np.nanmean(seg ** 2))):.2e}")
            print(f"    [{tag}] {tv:12s} rms by band: " + "  ".join(b))
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
