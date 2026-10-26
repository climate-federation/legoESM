"""Compare one restart family across two oracle run directories, per tile.

Prints max |a - b| and the relative difference (``full_step_oracle_parity.rel``,
over the larger peak) for every requested variable on every tile, then
exits 1 if any exceeds ``--max-rel`` (default 0.0 = bitwise).  ``--moved``
names variables that MUST differ somewhere (a passenger that did not move
under a baroclinic wave is a defect, not a match).

Usage::

    python compare_oracle_restarts.py --a <run A> --b <run B> \
        --file fv_core.res --vars u,v,T,delp
    python compare_oracle_restarts.py --a <zerostep> --b <1step> \
        --file fv_tracer.res --vars sphum,liq_wat --moved sphum,liq_wat --max-rel inf
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "full_step_oracle_parity", os.path.join(_HERE, "full_step_oracle_parity.py"))
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)


def load(run_dir: str, family: str, names: list[str]) -> list[dict]:
    import netCDF4
    out = []
    for t in range(1, 7):
        p = os.path.join(run_dir, "RESTART", f"{family}.tile{t}.nc")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        with netCDF4.Dataset(p) as ds:
            out.append({nm: np.asarray(ds.variables[nm][0], dtype=np.float64)
                        for nm in names})
    return out


def compare(a: list[dict], b: list[dict], names: list[str]) -> dict:
    """{name: (max abs, max rel, moved)} over the six tiles."""
    res = {}
    for nm in names:
        mx = rel = 0.0
        for t in range(6):
            x, y = a[t][nm], b[t][nm]
            if x.shape != y.shape:
                raise SystemExit(f"{nm} tile {t + 1}: shapes {x.shape} vs {y.shape}")
            mx = max(mx, float(np.abs(x - y).max()))
            rel = max(rel, parity.rel(x, y))
        res[nm] = (mx, rel, mx > 0.0)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--file", required=True, help="fv_core.res | fv_tracer.res")
    ap.add_argument("--vars", required=True)
    ap.add_argument("--max-rel", type=float, default=0.0,
                    help="largest allowed relative difference; 0 = bitwise")
    ap.add_argument("--moved", default="",
                    help="comma list of variables that must differ somewhere")
    args = ap.parse_args(argv)
    names = [v for v in args.vars.split(",") if v]
    moved = {v for v in args.moved.split(",") if v}
    res = compare(load(args.a, args.file, names), load(args.b, args.file, names),
                  names)
    bad = []
    for nm, (mx, rel, did_move) in res.items():
        verdict = "BITWISE" if mx == 0.0 else f"max abs {mx:.3e} rel {rel:.3e}"
        print(f"  {args.file} {nm:8s} {verdict}")
        if rel > args.max_rel:
            bad.append(f"{nm} rel {rel:.3e} > {args.max_rel:g}")
        if nm in moved and not did_move:
            bad.append(f"{nm} did not move")
    if bad:
        print("COMPARE FAILED: " + "; ".join(bad))
        return 1
    print("COMPARE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
