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
from flux_gradient import KAPPA  # noqa: E402
from plot_round2 import EDGES, OKABE_ITO, reference_curves  # noqa: E402

MIN_PER_BIN = 30  # as plot_round2.site_bins
ZETA_MAX_PLOT = 9.9  # standard schemes clip z/L at 10; stay inside


def per_site(data, kind, sign, fn=None):
    """sites x bins matrices on even days: median phi (or median phi/fn(zeta) when fn is given, each sample at its own
    zeta) and median |zeta|; NaN where a site has fewer than MIN_PER_BIN samples in the bin."""
    val, zm = [], []
    for v in data.values():
        z, ph = v[kind][0]
        if fn is None:
            y = ph
        else:
            pred = fn(z)
            if not np.all(np.isfinite(pred) & (pred > 0)):
                raise ValueError("predicted phi must be finite and positive at every sample")
            y = ph / pred
        rv, rz = [], []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (sign * z >= lo) & (sign * z < hi) & (ph > 0)
            ok = s.sum() >= MIN_PER_BIN
            rv.append(np.median(y[s]) if ok else np.nan)
            rz.append(np.median(sign * z[s]) if ok else np.nan)
        val.append(rv)
        zm.append(rz)
    return np.array(val), np.array(zm)


def scheme_phi(name, kind, zeta):
    """A standard scheme's phi in the towers' kappa convention (phi and zeta scale with kappa_scheme / KAPPA)."""
    s = sf.SCHEMES[name][2] / KAPPA
    return sf.phi(name, kind, zeta * s) / s


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
    xs = np.logspace(-2, np.log10(ZETA_MAX_PLOT), 300)
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for r, kind in enumerate("mh"):
        new = ref[kind] + [tuple(w) for w in winners[kind]]
        for c, sign in enumerate((-1, 1)):
            rows, zrow = per_site(data, kind, sign)
            med, zmed = np.nanmedian(rows, 0), np.nanmedian(zrow, 0)
            nsite = np.isfinite(rows).sum(0)
            ax, axr = axes[r, c], axes[r, c + 2]
            for row, zr in zip(rows, zrow):
                ax.plot(zr, row, color="0.8", lw=0.8)
            ok = np.isfinite(med)
            q25, q75 = np.nanpercentile(rows, [25, 75], 0)
            ax.errorbar(zmed[ok], med[ok], yerr=[med[ok] - q25[ok], q75[ok] - med[ok]], fmt="o", color="0.3", ms=6,
                        capsize=3, zorder=5, label="towers: median over sites at their median z/L (bar 25-75%; "
                                                   "number = sites in bin); thin: each site")
            for x_, y_, n_ in zip(zmed[ok], med[ok], nsite[ok]):
                ax.annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(5, -10), fontsize=7, color="0.3")
            others = [n for n in sf.SCHEMES if n != "dyer1974"]
            for j, name in enumerate(others):
                ax.plot(xs, scheme_phi(name, kind, sign * xs), color="#56B4E9", lw=0.8, alpha=0.6,
                        label=f"{len(others)} other standard schemes (kappa 0.4 convention)" if j == 0 else None)
            for i, (label, spec, p) in enumerate(new):
                col, ls = OKABE_ITO[[0, 2, 6, 5, 1, 7][i]], ["-", "--", "-", "-.", ":", "-"][i]
                ax.plot(xs, curve(spec, p, sign * xs), color=col, ls=ls, lw=2.2, label=label)
                ratio, zr = per_site(data, kind, sign, lambda z, spec=spec, p=p: curve(spec, p, z))
                rm, zrm = np.nanmedian(ratio, 0), np.nanmedian(zr, 0)
                axr.plot(zrm[ok], rm[ok], color=col, ls=ls, marker="o", lw=2, label=label)
            for x in (ax, axr):
                x.set_xscale("log")
                x.set_yscale("log")
                x.set_xlabel("-z/L (unstable)" if sign < 0 else "z/L (stable)")
                x.grid(color="0.9", which="both")
            ax.set_ylabel(f"phi_{kind}")
            ax.set_title(f"{'momentum' if kind == 'm' else 'heat'}, {'unstable' if sign < 0 else 'stable'}")
            axr.axhline(1, color="0.5", lw=1)
            axr.set_ylim(0.2, 2.5)
            axr.set_ylabel(f"median over sites of site-median phi_obs/phi_pred, phi_{kind}")
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
