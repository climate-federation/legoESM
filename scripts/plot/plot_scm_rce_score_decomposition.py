#!/usr/bin/env python
"""Ranked score DECOMPOSITION: temperature term vs humidity term per scheme.

The single combined thermo_score hides the trade each scheme makes.  This shows,
per scheme (5-seed mean +/- seed sd), the masked tropospheric temperature term
and the mass-weighted log-humidity term side by side, ordered by the combined
score.  A scheme with a small T bar and a large q bar (e.g. Bechtold) has the
right thermal structure but a moist bias -- a fact the scalar ranking erases.

Both terms are in the score's own normalised units (T tol 1 K, logq tol 0.10),
so their bar heights are directly comparable: that comparability is the whole
point of the equal-weight quadrature the score uses.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _collect(root: Path, seeds, suffix):
    d = {}
    for s in seeds:
        for p in glob.glob(str(root / f"arm_thermo_physical_{suffix}_seed{s}"
                                   / "scheme_*.json")):
            j = json.loads(Path(p).read_text())
            t = j["tuned"]
            r = d.setdefault(j["scheme"], {"T": [], "q": [], "c": []})
            r["T"].append(t["thermo_T_term"])
            r["q"].append(t["thermo_q_term"])
            r["c"].append(t["thermo_score"])
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--arm-suffix", default="f0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    d = _collect(a.arm_dir, a.seeds, a.arm_suffix)
    names = sorted(d, key=lambda n: np.mean(d[n]["c"]))
    Tm = [np.mean(d[n]["T"]) for n in names]
    Ts = [np.std(d[n]["T"]) for n in names]
    qm = [np.mean(d[n]["q"]) for n in names]
    qs = [np.std(d[n]["q"]) for n in names]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(y - 0.2, Tm, 0.38, xerr=Ts, color="#c0392b",
            label="temperature term (masked troposphere, tol 1 K)",
            error_kw=dict(lw=0.8))
    ax.barh(y + 0.2, qm, 0.38, xerr=qs, color="#2471a3",
            label="log-humidity term (mass-weighted, tol 0.10)",
            error_kw=dict(lw=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}. {n}" for i, n in enumerate(names)])
    ax.invert_yaxis()
    ax.set_xlabel("normalised score term (lower = closer to SAM CRM)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("SCM convection vs SAM CRM, RCEMIP RCE300 (nonrotating, "
                 "5-seed mean)\nranked by combined score; bars decompose it "
                 "into temperature and humidity", fontsize=10)
    for i, n in enumerate(names):
        if n == "bechtold":
            ax.axhspan(i - 0.45, i + 0.45, color="gold", alpha=0.18, zorder=0)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out} ({len(names)} schemes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
