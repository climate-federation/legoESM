#!/usr/bin/env python
"""Block-level state twin: ours vs the instrumented Zenodo model.

Loads dyncore_b<blk>_<name>_t<T>.dat (oracle, end-of-dyn_core state per
dt_atmos block, k=1) and the runner's --dump-state-out npz, and reports
per block the max abs state difference split into a CORNER BAND (cells
within --corner-band of a face corner, default 6) and the INTERIOR —
the campaign's defect signature is corner-band growth with a faithful
interior.  Also prints each side's max|pt-1| (pt==1 constancy is a
weighting-free transport invariant for case-8, oracle-fidelity Rule 6).

Signs are FIT at block 0 (IC) per field family and held fixed: under
the oracle's do_schmidt 180-degree rotation the case-8 winds map to
their negative and the scalars to themselves; a bad fit at block 0 is
FATAL, not patched over.

Comparison window: COMPUTE domain only (halos have different staleness
conventions by design).  Missing blocks/fields are FATAL when present
on one side only.  Can also compare two runner npz files
(--ours vs --control, both npz) for the chaos-floor calibration run
(oracle-fidelity Rule 3): pass --control instead of --fort-dir.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# staggering of each dumped field: (stag_i, stag_j)
STAG = {"u": (0, 1), "v": (1, 0), "delp": (0, 0), "pt": (0, 0)}


def read_fort(path: Path):
    with open(path) as f:
        ilo, ihi, jlo, jhi = (int(x) for x in f.readline().split())
        a = np.full((ihi - ilo + 1, jhi - jlo + 1), np.nan)
        for line in f:
            i, j, v = line.split()
            a[int(i) - ilo, int(j) - jlo] = float(v)
    return a, ilo, jlo


def load_fort_dir(d: Path):
    out = {}
    for p in sorted(d.glob("dyncore_b*.dat")):
        m = re.match(r"dyncore_b(\d+)_(\w+)_t(\d)\.dat", p.name)
        if not m:
            continue
        out[(int(m.group(1)), m.group(2), int(m.group(3)))] = read_fort(p)
    return out


def compute_crop(arr, lo_i, lo_j, n, stag_i, stag_j):
    """Slice the Fortran-windowed array to the compute domain
    [1..n+stag] per axis."""
    i0, j0 = 1 - lo_i, 1 - lo_j
    return arr[i0:i0 + n + stag_i, j0:j0 + n + stag_j]


def corner_band_mask(n, stag_i, stag_j, band):
    """True where the compute-domain cell is within `band` cells
    (Chebyshev) of one of the four face corners."""
    ii = np.arange(1, n + stag_i + 1)[:, None] * np.ones(
        (1, n + stag_j), dtype=int)
    jj = np.arange(1, n + stag_j + 1)[None, :] * np.ones(
        (n + stag_i, 1), dtype=int)
    d = np.full(ii.shape, 10 ** 9)
    for ci in (1, n + stag_i):
        for cj in (1, n + stag_j):
            d = np.minimum(d, np.maximum(np.abs(ii - ci),
                                         np.abs(jj - cj)))
    return d <= band


def fit_sign(a, b):
    """+1/-1 minimizing mean|a - s*b|; FATAL-quality check is caller's."""
    dp = float(np.abs(a - b).mean())
    dm = float(np.abs(a + b).mean())
    return (1.0, dp) if dp <= dm else (-1.0, dm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fort-dir", default=None)
    ap.add_argument("--control", default=None,
                    help="second runner npz instead of a fortran dir "
                         "(chaos-floor calibration)")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--corner-band", type=int, default=6)
    ap.add_argument("--expect-blocks", default="0-14,18,36,54,72,90,108,126,144",
                    help="REQUIRED inventory: every listed block must "
                         "carry u/v/delp/pt x t1..t6 on BOTH sides or "
                         "the verdict is refused (codex r2 P0: a run "
                         "that died early must never read as clean)")
    ap.add_argument("--expect-dt-atmos", type=float, default=1200.0)
    ap.add_argument("--expect-n-split", type=int, default=7)
    args = ap.parse_args(argv)
    if (args.fort_dir is None) == (args.control is None):
        print("FATAL: exactly one of --fort-dir / --control required")
        return 1

    expect_blocks = []
    for tok in args.expect_blocks.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            expect_blocks.extend(range(int(a), int(b) + 1))
        else:
            expect_blocks.append(int(tok))

    ours = np.load(args.ours, allow_pickle=False)
    n = int(ours["n"])
    ng = int(ours["ng"])
    lo = 1 - ng
    # provenance gate (codex r2 P1-2): a stale npz or wrong cadence
    # makes every block after b0 a false "divergence"
    for key, want in (("dt_atmos", args.expect_dt_atmos),
                      ("n_split", args.expect_n_split)):
        if key in ours.files:
            got = float(ours[key])
            if abs(got - want) > 1e-9:
                print(f"FATAL: ours {key}={got} != expected {want}")
                return 1
        else:
            print(f"FATAL: ours npz lacks {key} provenance")
            return 1

    def our_block(b, k, t):
        key = f"b{b}_{k}_t{t}"
        return ours[key] if key in ours.files else None

    if args.control:
        ctrl = np.load(args.control, allow_pickle=False)

        def ref_block(b, k, t):
            key = f"b{b}_{k}_t{t}"
            if key not in ctrl.files:
                return None
            a = ctrl[key]
            return a, lo, lo
        blocks = sorted({int(m.group(1)) for f in ours.files
                         if (m := re.match(r"b(\d+)_", f))})
        label = "OURS vs CONTROL (chaos floor)"
    else:
        fort = load_fort_dir(Path(args.fort_dir))
        if not fort:
            print("FATAL: no dyncore_b*.dat in", args.fort_dir)
            return 1

        def ref_block(b, k, t):
            return fort.get((b, k, t))
        blocks = sorted({b for (b, _, _) in fort})
        label = "OURS vs ORACLE"

    fatal = []
    # ---- inventory gate: EVERY expected record on BOTH sides ----
    missing = []
    for b in expect_blocks:
        for k in STAG:
            for t in range(1, 7):
                if ref_block(b, k, t) is None:
                    missing.append(f"ref b{b} {k} t{t}")
                if our_block(b, k, t) is None:
                    missing.append(f"ours b{b} {k} t{t}")
    if missing:
        print(f"FATAL: incomplete twin inventory — {len(missing)} of "
              f"{len(expect_blocks) * 24} expected records missing "
              "(run died early or wrong schedule); first few:")
        for m in missing[:8]:
            print("  -", m)
        return 1
    blocks = [b for b in blocks if b in set(expect_blocks)]
    print(f"inventory OK: {len(expect_blocks)} blocks x 24 records "
          "on both sides")
    # ---- sign fit at block 0 ----
    signs = {}
    for fam, fields in (("wind", ("u", "v")), ("scalar", ("delp", "pt"))):
        votes = []
        for k in fields:
            for t in range(1, 7):
                r = ref_block(0, k, t)
                o = our_block(0, k, t)
                if r is None or o is None:
                    print(f"FATAL: block-0 {k} t{t} missing "
                          f"(ref={r is not None} ours={o is not None})")
                    return 1
                fa, ilo, jlo = r
                si, sj = STAG[k]
                fc = compute_crop(fa, ilo, jlo, n, si, sj)
                oc = compute_crop(o, lo, lo, n, si, sj)
                if fc.shape != oc.shape:
                    print(f"FATAL: block-0 {k} t{t} shape {oc.shape} "
                          f"vs {fc.shape}")
                    return 1
                s, res = fit_sign(oc, fc)
                votes.append((s, res, float(np.abs(fc).max())))
        ss = {v[0] for v in votes if v[2] > 1e-12}  # ignore all-zero fields
        if len(ss) > 1:
            print(f"FATAL: inconsistent block-0 sign votes for {fam}: "
                  f"{votes}")
            return 1
        signs[fam] = ss.pop() if ss else 1.0
        worst_res = max(v[1] for v in votes)
        print(f"SIGN {fam}: {signs[fam]:+.0f} (worst block-0 residual "
              f"{worst_res:.3e})")
        if worst_res > 1e-9:
            fatal.append(f"block-0 {fam} residual {worst_res:.3e} — IC "
                         "not equivalent, later blocks uninterpretable")
    sgn = {"u": signs["wind"], "v": signs["wind"],
           "delp": signs["scalar"], "pt": signs["scalar"]}

    print(f"\n=== {label} ===")
    cb = f"corner<={args.corner_band}"
    print(f"{'blk':>4s} {'fld':>5s} {'max_abs':>12s} {cb:>12s} "
          f"{'interior':>12s} {'arg(t,i,j)':>16s} {'|pt-1| o/r':>22s}")
    curve = {}
    for b in blocks:
        pt_o = pt_r = 0.0
        for k in ("u", "v", "delp", "pt"):
            si, sj = STAG[k]
            band = corner_band_mask(n, si, sj, args.corner_band)
            mx = mxc = mxi = 0.0
            arg = None
            for t in range(1, 7):
                r = ref_block(b, k, t)
                o = our_block(b, k, t)
                if r is None and o is None:
                    continue
                if r is None or o is None:
                    fatal.append(f"block {b} {k} t{t} one-sided")
                    continue
                fa, ilo, jlo = r
                fc = compute_crop(fa, ilo, jlo, n, si, sj)
                oc = compute_crop(o, lo, lo, n, si, sj)
                if fc.shape != oc.shape:
                    fatal.append(f"block {b} {k} t{t} shape")
                    continue
                if not (np.all(np.isfinite(fc)) and
                        np.all(np.isfinite(oc))):
                    fatal.append(f"block {b} {k} t{t} non-finite")
                d = np.abs(oc - sgn[k] * fc)
                if float(d.max()) > mx:
                    mx = float(d.max())
                    kk = np.unravel_index(np.argmax(d), d.shape)
                    arg = (t, int(kk[0]) + 1, int(kk[1]) + 1)
                mxc = max(mxc, float(d[band].max()))
                if (~band).any():
                    mxi = max(mxi, float(d[~band].max()))
                if k == "pt":
                    pt_o = max(pt_o, float(np.abs(oc - 1.0).max()))
                    pt_r = max(pt_r, float(np.abs(fc - 1.0).max()))
            if arg is not None:
                extra = (f" {pt_o:.3e}/{pt_r:.3e}" if k == "pt" else "")
                print(f"{b:4d} {k:>5s} {mx:12.4e} {mxc:12.4e} "
                      f"{mxi:12.4e} {str(arg):>16s}{extra}")
                curve.setdefault(k, []).append((b, mx, mxc, mxi))
                if b == 0 and mx > 1e-9:
                    fatal.append(f"block-0 {k} max_abs {mx:.3e} — IC "
                                 "not equivalent (tile map / cadence / "
                                 "constants)")
            if k == "pt" and max(pt_o, pt_r) > 1e-9:
                fatal.append(f"block {b}: pt constancy violated "
                             f"(ours {pt_o:.3e} / ref {pt_r:.3e}) — "
                             "transport invariant broken on a side")

    print("\n=== corner-band vs interior growth (u) ===")
    for b, mx, mxc, mxi in curve.get("u", []):
        ratio = mxc / mxi if mxi > 0 else float("inf")
        print(f"  blk {b:4d}: corner {mxc:.4e}  interior {mxi:.4e}  "
              f"ratio {ratio:8.2f}")
    print("\nREADING: FP-compounding grows smoothly from ~1e-13 and "
          "stays corner/interior-neutral (compare the --control "
          "chaos-floor run); a STRUCTURAL defect shows corner-band "
          "growth ORDERS above interior at the same block.  Judge "
          "ours-vs-oracle only against the ours-vs-control floor "
          "(oracle-fidelity Rule 3).")
    if fatal:
        print("\nFATAL findings:")
        for f in fatal:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
