#!/usr/bin/env python
"""Compare the fp32 and f64 final-state .npz from _diag_fp32_vs_f64_closeness.py.

Asserts the two runs agree to ~fp32 precision (the expected single-precision
floor over a few steps), confirming fp32 computes the same physics as f64 — not
a divergent/broken path.  Pure NumPy; safe to run on the login node.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--fp32", required=True)
    p.add_argument("--f64", required=True)
    # ~fp32 precision over a few steps. atol covers the near-zero fields
    # (eta/v O(1e-3-1e-2)); rtol covers the O(1-30) fields (T, u). Same order
    # as the SPMD re-association floor (atol=2e-4, rtol=1e-3) — a divergent
    # fp32 path would blow far past this.
    p.add_argument("--atol", type=float, default=5.0e-3)
    p.add_argument("--rtol", type=float, default=5.0e-3)
    args = p.parse_args()

    a = np.load(args.fp32)
    b = np.load(args.f64)
    ok = True
    print(f"{'field':>6} {'max_abs_diff':>14} {'max_rel_diff':>14} "
          f"{'max|f64|':>12}  verdict")
    for nm in ("u", "v", "eta", "T", "S"):
        x = a[nm].astype(np.float64)
        y = b[nm].astype(np.float64)
        if x.shape != y.shape:
            print(f"{nm:>6}  SHAPE MISMATCH {x.shape} vs {y.shape}")
            ok = False
            continue
        absd = np.max(np.abs(x - y))
        denom = np.maximum(np.abs(y), 1e-30)
        reld = np.max(np.abs(x - y) / denom)
        scale = np.max(np.abs(y))
        # Pass if within atol OR rtol elementwise (np.allclose semantics).
        passed = bool(np.allclose(x, y, atol=args.atol, rtol=args.rtol))
        ok = ok and passed
        print(f"{nm:>6} {absd:14.4e} {reld:14.4e} {scale:12.4e}  "
              f"{'PASS' if passed else 'FAIL'}")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            print(f"{nm:>6}  NON-FINITE present")
            ok = False

    print(f"\n[compare] atol={args.atol:g} rtol={args.rtol:g} -> "
          f"{'ALL PASS — fp32 matches f64 to single precision' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
