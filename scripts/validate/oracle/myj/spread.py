"""The MYJ oracle's own round-off spread: rerun it with every float input
moved 1 ulp (random direction per element, 20 seeds) and report, per output
field and kept step, max over seeds of max|perturbed - base| / max|base|.

usage: python spread.py --audit <ysu_pbl/inputs.npz> --build <build> [--seeds 20]
"""
from __future__ import annotations

import argparse
import tempfile

import numpy as np

from make_fixture import KEEP_STEPS, audit_columns, run


def perturb(cols, rng):
    out = []
    for c in cols:
        d = {}
        for k, v in c.items():
            if isinstance(v, np.ndarray) or (isinstance(v, float) and k not in ("dt",)):
                a = np.asarray(v, float)
                up = rng.random(a.shape) < 0.5
                p = np.where(up, np.nextafter(a, np.inf), np.nextafter(a, -np.inf))
                # zeros stay zero (structural: absent species, zero fluxes)
                p = np.where(a == 0.0, 0.0, p)
                d[k] = p if isinstance(v, np.ndarray) else float(p)
            else:
                d[k] = v
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    cols = audit_columns(np.load(a.audit))
    drv = f"{a.build}/myj_driver"
    base = run(drv, cols, tempfile.mkdtemp())
    spread = {}
    for seed in range(a.seeds):
        rec = run(drv, perturb(cols, np.random.default_rng(seed)), tempfile.mkdtemp())
        for s in KEEP_STEPS:
            for k in base[(0, s)]:
                if not k.startswith("out_"):
                    continue
                b = np.stack([np.atleast_1d(base[(c, s)][k]) for c in range(len(cols))])
                p = np.stack([np.atleast_1d(rec[(c, s)][k]) for c in range(len(cols))])
                e = float(np.max(np.abs(p - b)) / max(np.max(np.abs(b)), 1e-300))
                spread[(s, k)] = max(spread.get((s, k), 0.0), e)
    for (s, k), e in sorted(spread.items()):
        print(f"spread s{s} {k} {e:.3e}")


if __name__ == "__main__":
    main()
