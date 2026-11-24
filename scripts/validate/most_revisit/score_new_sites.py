#!/usr/bin/env python
"""Out-of-sample check of the v5 benchmark on towers added after it (vote V12, coscientist/newdata/votes.md).

For each new site and panel: even-day log-RMSE of J3 with its frozen v5 global parameters, of J3 refitted in v6, and of
the standard schemes (from the v6 benchmark json); rank among the standards. Split-sample control: J3 minus Dyer with
zeta from the lowest flux (own) and from the next sonic up (split), on the same samples. Instrument checks: the v5
sites' standard-scheme scores are unchanged in v6, and the sample cache reproduces v5's frozen-J3 and Dyer scores.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import round2 as r2
import round2_selfcorr as rs

PANELS = ("m_stable", "m_unstable", "h_stable", "h_unstable")
CHECK_TOL = 1e-6  # same tolerance as round2.check


def j3_specs(p):
    """Benchmark J3 (candidate_formulations.make) at fixed parameters, in the round2 spec language."""
    m_par = {k: [p[k], 0, 99] for k in ("a0", "a_inf", "zc")}
    return {"m": {"stable": "1 + a_inf*z + (a0 - a_inf)*z/(1 + z/zc)", "unstable": "(1 + 16*x)**-0.25", "params": m_par},
            "h": {"stable": "pr0 + a_inf*z + (a0 - a_inf)*z/(1 + z/zc)", "unstable": "pr0*(1 + 16*x)**-0.5",
                  "params": {**m_par, "pr0": [p["pr0"], 0, 9]}}}


def fixed_score(spec, kind, data, min_count):
    return r2.score(spec, [v[0] for v in spec["params"].values()], kind, data, min_count)


def gradient(bench, site, panel, name):
    return bench["sites"][site]["offset_corrected"].get(panel, {}).get(name, {}).get("gradient", {}).get("log_rmse", np.nan)


def unchanged(v5, v6):
    """Worst relative change of any standard-scheme score of a v5 site between the two benchmark files; a score present
    in one file only counts as infinite change."""
    worst, n = 0.0, 0
    get = lambda b, s, arm, panel, name: b["sites"][s][arm].get(panel, {}).get(name, {}).get("gradient", {}).get("log_rmse", np.nan)  # noqa: E731
    for s, site in v5["sites"].items():
        if s not in v6["sites"]:
            return np.inf, n
        for arm in ("offset_corrected", "uncorrected"):
            for panel in set(site[arm]) | set(v6["sites"][s][arm]):
                for name in set(v5["standard"]) | set(v6["standard"]):
                    a, b = get(v5, s, arm, panel, name), get(v6, s, arm, panel, name)
                    if np.isfinite(a) or np.isfinite(b):
                        d = abs(a - b) / max(abs(a), 1e-12)
                        worst = max(worst, d) if np.isfinite(d) else np.inf
                        n += 1
    return worst, n


def missing_from_cache(data, bench, sites, min_count):
    """Benchmark Dyer scores (finite) of these sites that the sample cache cannot reproduce at all."""
    have = {(panel, s) for k in "mh" for panel, per in fixed_score(r2.DYER[k], k, subset(data, sites), min_count).items() for s in per}
    return sorted((panel, s) for s in sites for panel in PANELS if np.isfinite(gradient(bench, s, panel, "dyer1974"))
                  and (panel, s) not in have)


def rank_row(j3v5, j3v6, std):
    """Ranks of frozen and refitted J3 among the finite standard scores (1 = better than all)."""
    std = {k: v for k, v in std.items() if np.isfinite(v)}
    return {"J3_v5": j3v5, "J3_v6": j3v6, "standard": std, "n_standard": len(std),
            "rank_J3_v5": 1 + sum(x < j3v5 for x in std.values()),
            "rank_J3_v6": 1 + sum(x < j3v6 for x in std.values()) if np.isfinite(j3v6) else None,
            "best_standard": min(std, key=std.get) if std else None, "J3_v5_minus_dyer": j3v5 - std.get("dyer1974", np.nan)}


def same_sites(ref, v5):
    """The invariance reference must cover exactly the v5 sites, or some old site would go unchecked."""
    if set(ref["sites"]) != set(v5["sites"]):
        raise SystemExit(f"--unchanged-ref sites differ from v5: {sorted(set(ref['sites']) ^ set(v5['sites']))}")
    return ref


def subset(data, sites):
    return {s: v for s, v in data.items() if s in sites}


def bench_subset(bench, sites):
    return {**bench, "sites": {s: v for s, v in bench["sites"].items() if s in sites}}


def plot(out, path):
    """Per new site and panel: each standard scheme's held-out log-RMSE, and J3 frozen (v5) and refitted (v6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = [s for s in out["new_sites"] if out["sites"].get(s)]
    if not sites:
        print("nothing to plot")
        return
    fig, axes = plt.subplots(len(sites), len(PANELS), figsize=(3.2 * len(PANELS), 2.6 * len(sites)), squeeze=False)
    for i, s in enumerate(sites):
        for j, panel in enumerate(PANELS):
            ax = axes[i, j]
            r = out["sites"][s].get(panel)
            ax.set_title(f"{s} {panel.replace('_', ' ')}", fontsize=9)
            if r is None:
                ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            names = sorted(r["standard"], key=r["standard"].get)
            ax.scatter([r["standard"][k] for k in names], range(len(names)), color="#999999", s=14, label="standard")
            ax.set_yticks(range(len(names)), names, fontsize=6)
            if "dyer1974" in names:
                ax.scatter([r["standard"]["dyer1974"]], [names.index("dyer1974")], color="#000000", s=18, label="Dyer 1974")
            ax.axvline(r["J3_v5"], color="#E69F00", lw=2, label="J3 frozen v5")
            if np.isfinite(r["J3_v6"]):
                ax.axvline(r["J3_v6"], color="#0072B2", lw=1.2, ls="--", label="J3 refit v6")
            ax.set_xlabel("held-out log-RMSE of phi", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("WROTE", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True, help="round2_cache npz of the v6 tables")
    ap.add_argument("--bench-v5", required=True)
    ap.add_argument("--bench-v6", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--plot")
    ap.add_argument("--unchanged-ref", help="v5 tables rescored with the current code (default --bench-v5): the "
                    "old-site check then isolates the effect of adding sites from later scheme-code changes")
    a = ap.parse_args()
    np.seterr(all="ignore")
    v5, v6 = json.load(open(a.bench_v5)), json.load(open(a.bench_v6))
    data = r2.load(a.samples)
    new = sorted(set(v6["sites"]) - set(v5["sites"]))
    if a.unchanged_ref:
        with open(a.unchanged_ref) as fh:
            ref = same_sites(json.load(fh), v5)
    else:
        ref = v5
    worst, n = unchanged(ref, v6)
    print(f"CHECK v5 sites, standard schemes, v6 vs v5: worst relative change {worst:.2e} over {n} scores", flush=True)
    cache_worst = r2.check(subset(data, set(v5["sites"])), v5, a.min_count)
    new_worst = r2.check(subset(data, set(new)), bench_subset(v6, set(new)), a.min_count)  # new sites vs the v6 json
    for label, w in (("v5 sites unchanged", worst if n else np.inf), ("cache vs v5", cache_worst), ("new-site cache vs v6", new_worst)):
        if not w <= CHECK_TOL:
            raise SystemExit(f"instrument check failed: {label} worst relative difference {w}")
    for bench, sites in ((v5, set(v5["sites"])), (v6, set(new))):
        gone = missing_from_cache(data, bench, sites, a.min_count)
        if gone:
            raise SystemExit(f"instrument check failed: cache lacks benchmark scores {gone}")
    out = {"new_sites": new, "check": {"v5_sites_unchanged_worst": worst, "n": n, "cache_vs_v5_worst": cache_worst,
                                       "new_cache_vs_v6_worst": new_worst},
           "j3_v5": v5["global_fit"]["J3"], "j3_v6": v6["global_fit"]["J3"], "sites": {}, "split": {}}
    specs = {"J3_v5": j3_specs(v5["global_fit"]["J3"]), "dyer": r2.DYER}
    new_data = subset(data, set(new))
    frozen = {k: {name: fixed_score(sp[k], k, new_data, a.min_count) for name, sp in specs.items()} for k in "mh"}
    for s in new:
        out["sites"][s] = {}
        for panel in PANELS:
            k = panel[0]
            j3v5 = frozen[k]["J3_v5"].get(panel, {}).get(s, np.nan)
            if not np.isfinite(j3v5):
                continue
            out["sites"][s][panel] = rank_row(j3v5, gradient(v6, s, panel, "J3_global"),
                                              {name: gradient(v6, s, panel, name) for name in v6["standard"]})
            print(s, panel, {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                             for kk, vv in out["sites"][s][panel].items() if kk != "standard"}, flush=True)
    own, split, _ = rs.split_second_level({key: v for key, v in rs.raw(a.samples).items() if key[0] in new})
    for label, d in (("own", own), ("split", split)):
        for k in "mh":
            sc = {name: fixed_score(sp[k], k, d, a.min_count) for name, sp in specs.items()}
            for panel, per in sc["J3_v5"].items():
                for s, v in per.items():
                    out["split"].setdefault(s, {}).setdefault(panel, {})[f"J3_minus_dyer_{label}"] = v - sc["dyer"][panel].get(s, np.nan)
    for s, per in out["split"].items():
        for panel, v in per.items():
            z = split[s][panel[0]][0][0]
            v["n"] = int(((z > 0) if panel.endswith("_stable") else (z < 0)).sum())
            print("split", s, panel, {kk: round(vv, 4) for kk, vv in v.items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out)
    if a.plot:
        plot(out, a.plot)


if __name__ == "__main__":
    main()
