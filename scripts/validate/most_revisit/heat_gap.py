#!/usr/bin/env python
"""phi_h of the towers against a reference curve (Dyer on the unstable side, Dyer or J3 on the stable side): is a gap in
the towers or in the instrument? Held-out (even-day) phi_h/phi_ref per site and bin, each sample at its own zeta,
under four checks:
  offset   : offset-corrected samples vs the same cache built without the neutral offset;
  split    : on the subset with a second sonic, zeta from the lowest flux (own) vs from the next sonic up (split);
  perm_h   : H permuted between timestamps (round2_selfcorr.permute), so phi and zeta share only the algebra;
  per site : every site's own ratio, to see whether a few sites carry the median.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import round2 as r2  # noqa: E402
import round2_selfcorr as rs  # noqa: E402
from plot_phi_panels import curve, per_site, scheme_phi  # noqa: E402
from plot_round2 import EDGES, OKABE_ITO, reference_curves  # noqa: E402

N_SEEDS = 200
FIXED_BINS = (0.1, 5.0)  # fixed-site curve: sites with a ratio in every bin between these |z/L| edges
PERM_FROM = 0.5  # permutation band drawn only from this |z/L| up (unstable: below it the algebra alone exceeds the data)


def dyer(z):
    return scheme_phi("dyer1974", "h", z)


def reference(name, bench=None):
    """phi_h reference: 'dyer', or 'j3' = the benchmark's J3 global fit (odd-day fit), both sides of zeta."""
    if name == "dyer":
        return dyer
    label, spec, p = reference_curves(bench)["h"][1]
    assert label.startswith("J3"), label
    return lambda z: curve(spec, p, z)


def ratio(data, sign=-1, fn=dyer):
    """(median over sites, per-site matrix, site-median |zeta| median over sites) of phi_h/fn on one side."""
    rows, zr = per_site(data, "h", sign, fn)
    return np.nanmedian(rows, 0), rows, np.nanmedian(zr, 0)


def oracle_scale_gain(data, sign=-1, fn=dyer, min_count=30):
    """Per site: fn times the site's own best constant (geometric-mean ratio on odd days), scored on even days, one side.
    Returns {site: even-day log-RMSE(scaled) - log-RMSE(fn)}: what a per-site constant could gain over fn."""
    out = {}
    for s, v in data.items():
        (zo, po), (ze, pe) = v["h"][1], v["h"][0]
        so, se = (sign * zo > 0) & (po > 0), (sign * ze > 0) & (pe > 0)
        if so.sum() < min_count or se.sum() < min_count:
            continue
        c = np.exp(np.mean(np.log(po[so] / fn(zo[so]))))
        out[s] = r2.log_rmse(pe[se], c * fn(ze[se])) - r2.log_rmse(pe[se], fn(ze[se]))
    return out


def paired(x, y):
    """Both ratio() results restricted to (site, bin) cells finite in both, so the arms compare like with like."""
    ok = np.isfinite(x[1]) & np.isfinite(y[1])
    out = []
    for med, rows, zm in (x, y):
        r = np.where(ok, rows, np.nan)
        out.append((np.nanmedian(r, 0), r, zm))
    return out


def fixed_site_set(sites, rows):
    """Sites with a finite ratio in every bin inside FIXED_BINS."""
    span = (EDGES[:-1] >= FIXED_BINS[0]) & (EDGES[1:] <= FIXED_BINS[1])
    return [s for s, row in zip(sites, rows) if np.isfinite(row[span]).all()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True, help="offset-corrected cache (round2_cache)")
    ap.add_argument("--samples-raw", required=True, help="same cache with --no-offset-correct")
    ap.add_argument("--out", required=True, help="json path; figure written next to it as .png")
    ap.add_argument("--side", required=True, choices=("unstable", "stable"))
    ap.add_argument("--ref", required=True, choices=("dyer", "j3"))
    ap.add_argument("--bench", help="benchmark json holding the J3 global fit (needed for --ref j3)")
    a = ap.parse_args()
    np.seterr(all="ignore")
    sign = -1 if a.side == "unstable" else 1
    if a.ref == "j3" and not a.bench:
        ap.error("--ref j3 needs --bench")
    ref = reference(a.ref, a.bench)

    def rat(d):
        return ratio(d, sign, ref)
    corr, raw_ = r2.load(a.samples), r2.load(a.samples_raw)
    sites = sorted(corr)
    res = {"edges": EDGES.tolist(), "sites": sites, "side": a.side, "ref": a.ref}
    arms = {"offset_corrected": rat(corr)}
    arms["uncorrected"], arms["corrected_same_sites"] = paired(rat({s: raw_[s] for s in sites}), arms["offset_corrected"])
    rows = arms["offset_corrected"][1]
    fixed = fixed_site_set(sites, rows)
    arms["fixed_sites"] = rat({s: corr[s] for s in fixed})
    res["fixed_sites"] = fixed
    own, split, _ = rs.split_second_level(rs.raw(a.samples))
    arms["split_subset_own_zeta"], arms["split_subset_split_zeta"] = paired(rat(own), rat(split))
    rawd = rs.raw(a.samples)
    perm = np.array([rat(rs.permute(rawd, "h", np.random.default_rng(s)))[0] for s in range(N_SEEDS)])
    for k, (med, rows, zm) in arms.items():
        res[k] = {"median": med.tolist(), "zeta": zm.tolist(), "n_sites": np.isfinite(rows).sum(0).tolist()}
    res["per_site"] = dict(zip(sites, arms["offset_corrected"][1].tolist()))
    res["per_site_uncorrected"] = dict(zip(sites, arms["uncorrected"][1].tolist()))
    res["perm_h"] = {"median_of_seeds": np.nanmedian(perm, 0).tolist(), "q05": np.nanpercentile(perm, 5, 0).tolist(),
                     "q95": np.nanpercentile(perm, 95, 0).tolist()}
    res["oracle_scale_gain"] = oracle_scale_gain(corr, sign, ref)
    res["oracle_scale_gain_median"] = float(np.median(list(res["oracle_scale_gain"].values())))
    json.dump(res, open(a.out, "w"), indent=1)

    fig, (ax, axs) = plt.subplots(1, 2, figsize=(15, 6))
    med, rows, zm = arms["offset_corrected"]
    for s, row in zip(sites, rows):
        ok = np.isfinite(row)
        if ok.any():
            ax.plot(zm[ok], row[ok], color="0.8", lw=0.8)
            ax.annotate(s, (zm[ok][-1], row[ok][-1]), fontsize=6, color="0.5")
    styles = {"offset_corrected": (0, "-"), "uncorrected": (2, "--"), "split_subset_own_zeta": (6, "-."),
              "split_subset_split_zeta": (5, ":"), "fixed_sites": (7, "-")}
    for k, (ci, ls) in styles.items():
        m, _, z = arms[k]
        ok = np.isfinite(m)
        ax.plot(z[ok], m[ok], color=OKABE_ITO[ci], ls=ls, marker="o", lw=2, label=k.replace("_", " ") + (f" ({len(fixed)} sites, all bins {FIXED_BINS[0]:g}-{FIXED_BINS[1]:g})" if k == "fixed_sites" else ""))
    ok = np.isfinite(med) & (EDGES[:-1] >= (PERM_FROM if sign < 0 else 0))
    pm = np.array(res["perm_h"]["median_of_seeds"])
    ax.fill_between(zm[ok], np.array(res["perm_h"]["q05"])[ok], np.array(res["perm_h"]["q95"])[ok], color=OKABE_ITO[1],
                    alpha=0.3, label=f"H permuted ({N_SEEDS} seeds, 5-95%)" + (f", -z/L >= {PERM_FROM:g}" if sign < 0 else ""))
    ax.plot(zm[ok], pm[ok], color=OKABE_ITO[1], lw=1.5)
    n = np.isfinite(rows).sum(0)
    axs.bar(np.arange(len(n)), n, color="0.6")
    axs.set_xticks(np.arange(len(n)), [f"{lo:g}-{hi:g}" for lo, hi in zip(EDGES[:-1], EDGES[1:])], rotation=45)
    zl = "-z/L" if sign < 0 else "z/L"
    axs.set_xlabel(f"{zl} bin")
    axs.set_title("sites per bin (offset-corrected)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1, color="0.5", lw=1)
    ax.set_xlabel(f"{zl} ({a.side})")
    ax.set_ylabel(f"median over sites of site-median phi_h,obs / phi_h,{a.ref}")
    ax.grid(color="0.9", which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"{a.side.capitalize()} heat: towers vs {a.ref} (even days); thin = each site")
    fig.tight_layout()
    fig.savefig(a.out[:-5] + ".png", dpi=120)
    print("WROTE", a.out)
    for k in arms:
        print(k, np.round(res[k]["median"], 2).tolist(), res[k]["n_sites"])
    print(f"oracle per-site scale gain vs {a.ref}, median", round(res["oracle_scale_gain_median"], 4), len(res["oracle_scale_gain"]))
    print("perm_h", np.round(res["perm_h"]["median_of_seeds"], 2).tolist())
    for s, row in res["per_site"].items():
        print(f"{s:20s}", np.round(row, 2).tolist())


if __name__ == "__main__":
    main()
