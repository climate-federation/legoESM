#!/usr/bin/env python
"""Per-stage tangent-gain comparison: ours vs oracle, same-frame diffs.

Ours: twin_b72_gain.py npz (<arm>_<stage>_<name>_t<T>).  Oracle: three
run dirs (base/sym/anti) each holding dyncore_b201..205 stage dumps
from the instrumented dyn_core at LEGOESM_STAGE_BLK, plus the next
block-state dump as the final stage.  For each stage/field:
||perturbed - baseline||_max within each side — NO cross-model mapping
(each side diffs against its own baseline; the mapping gauge cancels).

The stage where OURS' response magnitude exceeds the ORACLE's by a
growing factor is the amplifier stage.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

STAGES = [("201", ("u", "v", "delp", "pt")),
          ("202", ("uc", "vc", "delpc", "ua", "va", "divgd")),
          ("203", ("uc", "vc")),
          ("204", ("uc", "vc", "divgd")),
          ("205", ("delp", "pt")),
          ("300", ("u", "v", "delp", "pt"))]


def read_fort(path: Path):
    with open(path) as f:
        f.readline()
        vals = {}
        for line in f:
            i, j, v = line.split()
            vals[(int(i), int(j))] = float(v)
    return vals


def load_oracle_dir(d: Path, next_block: int):
    """{(stage, name, tile): dict{(i,j): v}}"""
    out = {}
    for p in sorted(d.glob("dyncore_b*.dat")):
        m = re.match(r"dyncore_b(\d+)_(\w+)_t(\d)\.dat", p.name)
        if not m:
            continue
        blk = int(m.group(1))
        if 201 <= blk <= 205:
            key = (str(blk), m.group(2), int(m.group(3)))
        elif blk == next_block:
            key = ("300", m.group(2), int(m.group(3)))
        else:
            continue
        out[key] = read_fort(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--oracle-base", required=True)
    ap.add_argument("--oracle-sym", required=True)
    ap.add_argument("--oracle-anti", required=True)
    ap.add_argument("--next-block", type=int, default=73)
    args = ap.parse_args(argv)

    ours = np.load(args.ours, allow_pickle=False)
    fatal = []
    orc = {arm: load_oracle_dir(Path(getattr(args, f"oracle_{arm}")),
                                args.next_block)
           for arm in ("base", "sym", "anti")}

    print(f"{'stage':>6s} {'field':>6s} {'arm':>5s} {'ours_max':>12s} "
          f"{'oracle_max':>12s} {'ratio':>10s}")
    for st, names in STAGES:
        for nm in names:
            for arm in ("sym", "anti"):
                o_mx = f_mx = None
                for t in range(1, 7):
                    kb = f"base_{st}_{nm}_t{t}"
                    kp = f"{arm}_{st}_{nm}_t{t}"
                    if kb in ours.files and kp in ours.files:
                        d = float(np.abs(ours[kp] - ours[kb]).max())
                        o_mx = d if o_mx is None else max(o_mx, d)
                    else:
                        fatal.append(f"ours missing {st} {nm} t{t}")
                    fb = orc["base"].get((st, nm, t))
                    fp = orc[arm].get((st, nm, t))
                    if fb is not None and fp is not None:
                        common = fb.keys() & fp.keys()
                        if not common:
                            fatal.append(f"oracle empty {st} {nm} t{t}")
                            continue
                        d = max(abs(fp[k] - fb[k]) for k in common)
                        f_mx = d if f_mx is None else max(f_mx, d)
                    else:
                        fatal.append(f"oracle missing {st} {nm} t{t}")
                if o_mx is None or f_mx is None:
                    continue
                ratio = o_mx / f_mx if f_mx > 0 else float("inf")
                print(f"{st:>6s} {nm:>6s} {arm:>5s} {o_mx:12.4e} "
                      f"{f_mx:12.4e} {ratio:10.3f}")

    print("\nREADING: both sides start from the same eps=1e-4 delp "
          "bump at tile-3 vertex cells.  ratio ~1 = stage responds "
          "identically; a stage where the ratio departs (and keeps "
          "departing downstream) is where OUR composition amplifies "
          "relative to the oracle.  201 delp must read exactly eps at "
          "the bump cells on both sides (injection check).")
    if fatal:
        seen = sorted(set(fatal))
        print(f"\nFATAL/incomplete ({len(seen)} unique):")
        for f in seen[:12]:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
