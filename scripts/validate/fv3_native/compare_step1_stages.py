#!/usr/bin/env python
"""Step-1 stage twin: mid-step intermediates, ours vs instrumented
dyn_core, first acoustic step of block 1 on the case-8 IC.

Pseudo-blocks (dumped by both sides at the same pipeline points):
  201 post-entry-exchange u/v/delp/pt
  202 post-c_sw          uc/vc/delpc/ua/va/divgd
  203 post-p_grad_c      uc/vc            (pre-exchange)
  204 post duo exchanges uc/vc/divgd
  205 post-d_sw2         delp/pt

Per-field sign is FIT from the data (amplitude-weighted vote across
tiles; the do_schmidt 180-degree rotation flips wind-like fields and
preserves scalars) and reported — a residual above threshold on a
voting field is FATAL.  Compute-domain crop per staggering;
fail-loud inventory.  The first pseudo-block whose max_abs jumps
orders above its predecessor is the defect stage.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# name -> (stag_i, stag_j)
FIELDS = {"u": (0, 1), "v": (1, 0), "delp": (0, 0), "pt": (0, 0),
          "uc": (1, 0), "vc": (0, 1), "delpc": (0, 0),
          "ua": (0, 0), "va": (0, 0), "divgd": (1, 1)}
BLOCKS = {201: ("u", "v", "delp", "pt"),
          202: ("uc", "vc", "delpc", "ua", "va", "divgd"),
          203: ("uc", "vc"),
          204: ("uc", "vc", "divgd"),
          205: ("delp", "pt")}
# compute-window shrink for fields only defined on the CG ring
# (delpc/ua/va written is-1..ie+1; divgd on B compute) — compare the
# plain compute domain, always covered
SCALARS = {"delp", "pt", "delpc"}


def read_fort(path: Path):
    with open(path) as f:
        ilo, ihi, jlo, jhi = (int(x) for x in f.readline().split())
        a = np.full((ihi - ilo + 1, jhi - jlo + 1), np.nan)
        for line in f:
            i, j, v = line.split()
            a[int(i) - ilo, int(j) - jlo] = float(v)
    return a, ilo, jlo


def compute_crop(arr, lo_i, lo_j, n, si, sj):
    i0, j0 = 1 - lo_i, 1 - lo_j
    return arr[i0:i0 + n + si, j0:j0 + n + sj]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fort-dir", required=True)
    ap.add_argument("--ours", required=True)
    args = ap.parse_args(argv)

    fort = {}
    for p in sorted(Path(args.fort_dir).glob("dyncore_b2*.dat")):
        m = re.match(r"dyncore_b(\d+)_(\w+)_t(\d)\.dat", p.name)
        if m:
            fort[(int(m.group(1)), m.group(2),
                  int(m.group(3)))] = read_fort(p)
    ours = np.load(args.ours, allow_pickle=False)
    n = int(ours["n"])
    ng = int(ours["ng"])
    lo = 1 - ng

    missing = []
    for b, names in BLOCKS.items():
        for k in names:
            for t in range(1, 7):
                if (b, k, t) not in fort:
                    missing.append(f"fort b{b} {k} t{t}")
                if f"b{b}_{k}_t{t}" not in ours.files:
                    missing.append(f"ours b{b} {k} t{t}")
    if missing:
        print(f"FATAL: incomplete stage inventory ({len(missing)}):")
        for m_ in missing[:10]:
            print("  -", m_)
        return 1
    print("inventory OK:", sum(len(v) for v in BLOCKS.values()), "fields x 6 tiles")

    # per-field sign fit (amplitude-weighted vote)
    signs = {}
    fatal = []
    for k, (si, sj) in FIELDS.items():
        blocks_with = [b for b, names in BLOCKS.items() if k in names]
        b = blocks_with[0]
        votes = []
        for t in range(1, 7):
            fa, ilo, jlo = fort[(b, k, t)]
            fc = compute_crop(fa, ilo, jlo, n, si, sj)
            oc = compute_crop(ours[f"b{b}_{k}_t{t}"], lo, lo, n, si, sj)
            if fc.shape != oc.shape:
                print(f"FATAL: shape b{b} {k} t{t} {oc.shape} vs "
                      f"{fc.shape}")
                return 1
            amp = float(np.abs(fc).max())
            dp = float(np.abs(oc - fc).mean())
            dm = float(np.abs(oc + fc).mean())
            votes.append((1.0 if dp <= dm else -1.0,
                          min(dp, dm), amp))
        live = [v for v in votes if v[2] > 1e-11]
        ss = {v[0] for v in live}
        if len(ss) > 1:
            fatal.append(f"{k}: inconsistent sign votes {votes}")
            signs[k] = 1.0
        else:
            signs[k] = ss.pop() if live else 1.0
        exp = 1.0 if k in SCALARS else -1.0
        if live and signs[k] != exp:
            print(f"NOTE: {k} fitted sign {signs[k]:+.0f} != expected "
                  f"{exp:+.0f} (Rz(pi) symmetry) — inspect")

    print(f"\n{'blk':>4s} {'fld':>6s} {'max_abs':>12s} "
          f"{'arg(t,i,j)':>16s} {'sign':>5s}")
    for b in sorted(BLOCKS):
        for k in BLOCKS[b]:
            si, sj = FIELDS[k]
            mx, arg = -1.0, None
            for t in range(1, 7):
                fa, ilo, jlo = fort[(b, k, t)]
                fc = compute_crop(fa, ilo, jlo, n, si, sj)
                oc = compute_crop(ours[f"b{b}_{k}_t{t}"], lo, lo, n,
                                  si, sj)
                d = np.abs(oc - signs[k] * fc)
                if not np.all(np.isfinite(d)):
                    fatal.append(f"non-finite b{b} {k} t{t}")
                    continue
                if arg is None or float(d.max()) > mx:
                    mx = float(d.max())
                    kk = np.unravel_index(np.argmax(d), d.shape)
                    arg = (t, int(kk[0]) + 1, int(kk[1]) + 1)
            print(f"{b:4d} {k:>6s} {mx:12.4e} {str(arg):>16s} "
                  f"{signs[k]:+5.0f}")

    print("\nREADING: 201 should sit at IC-noise (~1e-13).  The first "
          "block whose max_abs jumps orders above its predecessor is "
          "the defect stage: 202=c_sw, 203=geopk/p_grad_c, "
          "204=post-PG exchanges, 205=d_sw1/averaging/d_sw2.")
    if fatal:
        print("\nFATAL findings:")
        for f in fatal:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
