#!/usr/bin/env python
"""Round-2 figure: phi_m and phi_h vs zeta for Dyer, J3 (v5 global fit) and the round-2 winners, over the
median-over-sites observed phi per zeta bin (even days, the scored half). Okabe-Ito colours."""
from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import round2 as r2  # noqa: E402

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
EDGES = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])


def site_bins(data, kind):
    """Per stability side: bin centres (signed zeta) and median over sites of each site's median phi in the bin."""
    out = []
    for sign in (-1, 1):
        rows = []
        for v in data.values():
            z, ph = v[kind][0]
            row = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                sel = (sign * z >= lo) & (sign * z < hi) & (ph > 0)
                row.append(np.median(ph[sel]) if sel.sum() >= 30 else np.nan)
            rows.append(row)
        rows = np.array(rows)
        n = np.isfinite(rows).sum(0)
        out.append((sign * np.sqrt(EDGES[:-1] * EDGES[1:]), np.nanmedian(rows, 0), np.nanpercentile(rows, 25, 0),
                    np.nanpercentile(rows, 75, 0), n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--winners", required=True, help="json {kind: [[label, spec, params], ...]}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    data = r2.load(a.samples)
    j3 = json.load(open(a.bench))["global_fit"]["J3"]
    ref = {"m": [("Dyer 1974 (legoESM default)", r2.DYER["m"], []),
                 ("J3 round 1 (3 params)", {"stable": "1 + a_inf*z + (a0 - a_inf)*z/(1 + z/zc)", "unstable": "(1 + 16*x)**-0.25",
                                            "params": {"a0": 0, "a_inf": 0, "zc": 0}}, [j3["a0"], j3["a_inf"], j3["zc"]])],
           "h": [("Dyer 1974 (legoESM default)", r2.DYER["h"], []),
                 ("J3 round 1", {"stable": "pr0 + a_inf*z + (a0 - a_inf)*z/(1 + z/zc)", "unstable": "pr0*(1 + 16*x)**-0.5",
                                 "params": {"a0": 0, "a_inf": 0, "zc": 0, "pr0": 0}}, [j3["a0"], j3["a_inf"], j3["zc"], j3["pr0"]])]}
    winners = json.load(open(a.winners))
    zeta = np.concatenate([-np.logspace(1, -2, 200), np.logspace(-2, 1, 200)])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, kind in zip(axes, "mh"):
        for (zc, med, q25, q75, n) in site_bins(data, kind):
            ok = np.isfinite(med)
            ax.errorbar(zc[ok], med[ok], yerr=[med[ok] - q25[ok], q75[ok] - med[ok]], fmt="o", color="0.45", ms=5, capsize=2,
                        label="towers: median over sites (bar: 25-75% of sites)" if zc[0] < 0 else None)
        curves = ref[kind] + [tuple(w) for w in winners[kind]]
        for i, (label, spec, p) in enumerate(curves):
            f, _, _ = r2.build(spec)
            ax.plot(zeta, np.asarray(f(jnp.asarray(zeta), jnp.asarray(p or [0.0]))), color=OKABE_ITO[[0, 2, 6, 5, 1, 7][i]],
                    ls=["-", "--", "-", "-.", ":", "-"][i], lw=2, label=label)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("log")
        ax.set_xlabel("zeta = z/L")
        ax.set_ylabel("phi_m" if kind == "m" else "phi_h")
        ax.set_title("momentum" if kind == "m" else "heat (|zeta| < 0.1 not scored)")
        ax.grid(color="0.9")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Round 2: winners vs Dyer and J3 (fits on odd UTC days; data bins = even days, 22 towers)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
