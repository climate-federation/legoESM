"""Per-leaf difference of two tiled_m6_model_gate --save-npz references.

Used to attribute a bit-pattern change between two flat references to
lowering (diffuse, ~1e-16..1e-14 relative, no growth) versus a defect
(structured: rings, seams, whole faces).  Prints, per leaf: cells that
differ, max |a-b|, max |a-b|/max|a|, and the argmax location.
Exit 1 if any leaf exceeds --rel-tol.
"""
import argparse
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rel-tol", type=float, default=1e-12)
    args = ap.parse_args(argv)
    A, B = np.load(args.a), np.load(args.b)
    assert set(A.files) == set(B.files), (set(A.files) ^ set(B.files))
    worst = 0.0
    for k in sorted(A.files):
        a, b = A[k], B[k]
        if a.dtype.kind not in "fc":
            print(f"{k:24s} non-float, equal={np.array_equal(a, b)}")
            continue
        d = np.abs(a - b)
        scale = np.nanmax(np.abs(a)) if a.size else 0.0
        rel = float(np.nanmax(d) / scale) if scale > 0 else float(np.nanmax(d))
        nd = int(np.count_nonzero(d > 0))
        loc = np.unravel_index(int(np.nanargmax(d)), d.shape) if nd else None
        worst = max(worst, rel)
        print(f"{k:24s} differ {nd:9d}/{d.size:<9d} max|d| {np.nanmax(d):.3e} "
              f"rel {rel:.3e} at {loc}")
    print(f"WORST rel {worst:.3e} (tol {args.rel_tol:.1e}): "
          f"{'PASS' if worst <= args.rel_tol else 'FAIL'}")
    return 0 if worst <= args.rel_tol else 1


if __name__ == "__main__":
    sys.exit(main())
