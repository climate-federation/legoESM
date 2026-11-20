#!/usr/bin/env python
"""Flux-profile relationships of the new equations (J3, round-2 winners) against Dyer, the 12 standard schemes and
the towers' held-out (even-day) phi: momentum and heat, unstable and stable, plus the observed/predicted ratio."""
from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import round2 as r2  # noqa: E402
import standard_formulations as sf  # noqa: E402
from plot_round2 import EDGES, OKABE_ITO, reference_curves  # noqa: E402

MIN_PER_BIN = 30  # as plot_round2.site_bins
CENTRES = np.sqrt(EDGES[:-1] * EDGES[1:])


def per_site(data, kind, sign):
    """sites x bins matrix of each site's median phi on even days (NaN where fewer than MIN_PER_BIN samples)."""
    rows = []
    for v in data.values():
        z, ph = v[kind][0]
        rows.append([np.median(ph[s]) if (s := (sign * z >= lo) & (sign * z < hi) & (ph > 0)).sum() >= MIN_PER_BIN
                     else np.nan for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    return np.array(rows)


def curve(spec, p, zeta):
    f, _, _ = r2.build(spec)
    return np.asarray(f(jnp.asarray(zeta), jnp.asarray(p or [0.0])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--winners", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    data = r2.load(a.samples)
    ref = reference_curves(a.bench)
    winners = json.load(open(a.winners))
    xs = np.logspace(-2, 1, 300)
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for r, kind in enumerate("mh"):
        new = ref[kind] + [tuple(w) for w in winners[kind]]
        for c, sign in enumerate((-1, 1)):
            rows = per_site(data, kind, sign)
            med = np.nanmedian(rows, 0)
            ax, axr = axes[r, c], axes[r, c + 2]
            for row in rows:
                ax.plot(CENTRES, row, color="0.8", lw=0.8)
            ok = np.isfinite(med)
            ax.errorbar(CENTRES[ok], med[ok], yerr=[med[ok] - np.nanpercentile(rows, 25, 0)[ok],
                                                     np.nanpercentile(rows, 75, 0)[ok] - med[ok]],
                        fmt="o", color="0.3", ms=6, capsize=3, zorder=5,
                        label=f"towers, median of {np.isfinite(rows).any(1).sum()} sites (bar 25-75%); thin: each site")
            for name in sf.SCHEMES:
                if name != "dyer1974":
                    ax.plot(xs, sf.phi(name, kind, sign * xs), color="#56B4E9", lw=0.8, alpha=0.6,
                            label="11 other standard schemes" if name == "businger1971" else None)
            for i, (label, spec, p) in enumerate(new):
                col, ls = OKABE_ITO[[0, 2, 6, 5, 1, 7][i]], ["-", "--", "-", "-.", ":", "-"][i]
                ax.plot(xs, curve(spec, p, sign * xs), color=col, ls=ls, lw=2.2, label=label)
                ratio = rows / curve(spec, p, sign * CENTRES)[None, :]
                rm = np.nanmedian(ratio, 0)
                axr.plot(CENTRES[ok], rm[ok], color=col, ls=ls, marker="o", lw=2, label=label)
            for x in (ax, axr):
                x.set_xscale("log")
                x.set_yscale("log")
                x.set_xlabel("-z/L (unstable)" if sign < 0 else "z/L (stable)")
                x.grid(color="0.9", which="both")
            ax.set_ylabel(f"phi_{kind}")
            ax.set_title(f"{'momentum' if kind == 'm' else 'heat'}, {'unstable' if sign < 0 else 'stable'}")
            axr.axhline(1, color="0.5", lw=1)
            axr.set_ylim(0.2, 2.5)
            axr.set_ylabel(f"observed / predicted phi_{kind} (median over sites)")
            axr.set_title(f"ratio, {'momentum' if kind == 'm' else 'heat'}, {'unstable' if sign < 0 else 'stable'}")
            if c == 0:
                ax.legend(fontsize=7, loc="upper right")
            else:
                axr.legend(fontsize=7, loc="upper left")
    fig.suptitle("New flux-profile equations vs Dyer, 11 standard schemes and towers (fits on odd UTC days; data = even "
                 "days). Heat |z/L| < 0.1 not scored.")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
