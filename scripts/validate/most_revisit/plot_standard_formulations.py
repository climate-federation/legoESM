#!/usr/bin/env python
"""League table (sites x schemes) and phi curves over site data for the stability-function benchmark."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from legoesm import constants  # noqa: E402

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
PANELS = ("m_unstable", "m_stable", "h_unstable", "h_stable")


def league(bench, out, space):
    sites, schemes = list(bench["sites"]), list(bench["schemes"])
    fig, axes = plt.subplots(1, 4, figsize=(22, 0.45 * len(sites) + 3), sharey=True)
    for ax, panel in zip(axes, PANELS):
        key = "log_rmse" if space == "gradient" else "rmse"
        grid = np.full((len(sites), len(schemes)), np.nan)
        for i, s in enumerate(sites):
            res = bench["sites"][s]["offset_corrected"].get(panel, {})
            for j, k in enumerate(schemes):
                grid[i, j] = res.get(k, {}).get(space, {}).get(key, np.nan)
        winners = [int(np.nanargmin(g)) if np.isfinite(g).any() else None for g in grid]
        if space == "integrated":  # relative to the site's best scheme, so sites with different units share a scale
            best = np.nanmin(np.where(grid > 0, grid, np.nan), axis=1, keepdims=True)
            grid = np.where(grid == 0, 0.0, grid / best)
        im = ax.imshow(grid, cmap="cividis_r", aspect="auto")
        for i, w in enumerate(winners):
            if w is not None:
                ax.text(w, i, "best", ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks(range(len(schemes)), schemes, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(sites)), sites, fontsize=8)
        ax.set_title(panel.replace("_", " ").replace("m ", "momentum ").replace("h ", "heat "))
        fig.colorbar(im, ax=ax, shrink=0.7, label="log-RMSE of phi" if space == "gradient" else "RMSE / best scheme")
    fig.suptitle(f"Model stability functions vs towers, {space} space (lower is better; sites never pooled)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def inversion(bench, out):
    """Fluxes predicted from level differences, shared samples, lowest wind pair per site; RMSE / best scheme."""
    sites = [s for s in bench["sites"] if bench["sites"][s].get("inversion")]
    schemes = list(bench["schemes"]) + list(bench.get("bulk_only", {}))
    fig, axes = plt.subplots(1, 4, figsize=(22, 0.45 * len(sites) + 3), sharey=True)
    for ax, (regime, q) in zip(axes, [(r, q) for r in ("unstable", "stable") for q in ("ustar", "h")]):
        grid = np.full((len(sites), len(schemes)), np.nan)
        for i, s in enumerate(sites):
            inv = bench["sites"][s]["inversion"]
            key = min(inv, key=lambda k: float(k.split(":")[1].split("-")[0]))
            res = inv[key].get(regime, {})
            for j, k in enumerate(schemes):
                grid[i, j] = res.get(k, {}).get("common", {}).get(q, {}).get("rmse", np.nan)
        winners = [int(np.nanargmin(g)) if np.isfinite(g).any() else None for g in grid]
        best = np.nanmin(np.where(grid > 0, grid, np.nan), axis=1, keepdims=True)
        grid = np.where(grid == 0, 0.0, grid / best)
        im = ax.imshow(grid, cmap="cividis_r", aspect="auto")
        for i, w in enumerate(winners):
            if w is not None:
                ax.text(w, i, "best", ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks(range(len(schemes)), schemes, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(sites)), sites, fontsize=8)
        ax.set_title(f"{'u*' if q == 'ustar' else 'H'}, {regime}")
        fig.colorbar(im, ax=ax, shrink=0.7, label="RMSE / best scheme")
    fig.suptitle("Fluxes predicted from measured level differences (models' direction), same samples for every scheme")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def curves(bench, sites_json, out):
    v4 = json.load(open(sites_json))
    z = np.array(bench["curves"]["zeta"])
    styles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for (kind, arm), row in (( ("m", "primary"), 0), (("h", "offset_corrected"), 1)):
        for col, sign in enumerate((-1, 1)):
            ax = axes[row, col]
            for s, site in v4["sites"].items():
                bins = site.get(arm, {}).get("site", {}).get(kind, {}).get("bins", [])
                pts = np.array([[b[3], b[5]] for b in bins if np.isfinite(b[5]) and sign * b[3] > 0]) if bins else np.empty((0, 2))
                if len(pts):
                    ax.plot(np.abs(pts[:, 0]), pts[:, 1], "o", color="0.6", ms=4, label="site bin medians" if s == next(iter(v4["sites"])) else None)
            for i, (name, meta) in enumerate((k, v) for k, v in bench["schemes"].items() if k in bench["curves"]):
                s = meta["kappa"] / constants.kappa_von_karman
                sel = (sign * z > 0) & (np.abs(z) < 9.9)  # legoESM psi clips at |z/L| = 10
                own = np.interp(z * s, z, np.array(bench["curves"][name][kind])) / s  # scheme's own kappa, on our axes
                ax.plot(np.abs(z[sel]), own[sel],
                        styles[i // len(OKABE_ITO)], color=OKABE_ITO[i % len(OKABE_ITO)], lw=2, label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-2, 10)
            ax.set_xlabel("|z/L|")
            ax.set_ylabel(f"phi_{kind}")
            ax.set_title(f"phi_{kind}, {'unstable' if sign < 0 else 'stable'}")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def dump_bins(bench, out):
    """Observed bin medians per site, same samples and arm as the benchmark (heat offset-corrected)."""
    import standard_formulations as sf
    from flux_gradient import binned_median
    a = bench["args"]
    np.seterr(all="ignore")
    res = {}
    for name in bench["sites"]:
        pairs_, _ = sf.site_pairs(sf.fs.open_site(f"{a['tables']}/{name}.nc"), a["min_ustar"], True, a["max_zg"])
        res[name] = {}
        for kind, plist in pairs_.items():
            if plist:
                z, p = (np.concatenate([q[k][q["keep"]] for q in plist]) for k in ("zeta", "phi"))
                res[name][kind] = binned_median(z, p, sf.fs.EDGES, a["min_count"])
    json.dump(res, open(out, "w"), default=float)


TOWER_LINES = (  # (scheme, side: -1 unstable, 1 stable, 0 both, kinds)
    ("J3_global", 1, "mh"), ("J17_global", -1, "m"), ("J5_global", 1, "m"),
    ("dyer1974", 0, "mh"), ("wrf_revmm5", 0, "mh"), ("mynn_sfc", 0, "mh"), ("gfs", 0, "mh"))


def towers(bench, bins, kind, out):
    """One panel per site: observed bin medians with quartiles, our fits and the model schemes."""
    z = np.array(bench["curves"]["zeta"])
    names = list(bins)
    ncol = 5
    fig, axes = plt.subplots(-(-len(names) // ncol), ncol, figsize=(4 * ncol, 3.2 * -(-len(names) // ncol)), sharex=True, sharey=True)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        b = np.array([r for r in bins[name].get(kind, []) if np.isfinite(r[5])]).reshape(-1, 7)
        if len(b):
            ax.errorbar(b[:, 3], b[:, 5], yerr=[b[:, 5] - b[:, 4], b[:, 6] - b[:, 5]], fmt="o", color="0.4", ms=4, capsize=2, zorder=5)
        for i, (scheme, side, kinds) in enumerate(TOWER_LINES):
            if kind not in kinds:
                continue
            sel = (np.abs(z) < 9.9) & ((side == 0) | (np.sign(z) == side))
            ax.plot(z[sel], np.array(bench["curves"][scheme][kind])[sel], ["-", "--"][i // 4], color=OKABE_ITO[[6, 5, 1, 0, 3, 7, 2][i]], lw=1.6)
        scores = bench["sites"][name]["offset_corrected"]
        txt = [f"{lab} {scores[f'{kind}_{reg}'][f]['gradient']['log_rmse']:.2f}"
               for lab, f, reg in (("J17", "J17_global", "unstable"), ("J3", "J3_global", "stable"))
               if (lab, kind) != ("J17", "h") and f in scores.get(f"{kind}_{reg}", {})]
        ax.text(0.03, 0.97, f"{name}\nn={int(b[:, 2].sum()) if len(b) else 0}", transform=ax.transAxes, va="top", fontsize=8)
        if txt:
            ax.text(0.97, 0.03, "log-RMSE\n" + "\n".join(txt), transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("log")
        ax.set_xlim(-10, 10)
        ax.set_xticks([-10, -1, -0.1, 0, 0.1, 1, 10], ["-10", "-1", "-0.1", "0", "0.1", "1", "10"])
        ax.axvline(0, color="0.8", lw=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("z/L")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"phi_{kind}")
    handles = [plt.Line2D([], [], ls=["-", "--"][i // 4], color=OKABE_ITO[[6, 5, 1, 0, 3, 7, 2][i]], lw=1.6, label=s)
               for i, (s, _, k) in enumerate(TOWER_LINES) if kind in k]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.4", label="site bin median, quartiles"))
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.04), fontsize=10)
    fig.suptitle(f"phi_{kind} per tower (heat offset-corrected, |z/L|>=0.1)" if kind == "h" else "phi_m per tower")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


OURS = ("J3_global", "J17_global", "J5_global", "J3_site", "J17_site", "J5_site")


def _strict_label(bench, scheme, kind):
    s = bench["strict"].get(scheme, {})
    if "_site" in scheme:
        ok = [v[kind]["pass"] for v in s.values()]
        return f"C1 {sum(ok)}/{len(ok)}"
    return "C1 pass" if s.get(kind, {}).get("pass") else "C1 fail"


def compare(bench, out):
    """Per-tower held-out gradient log-RMSE: our formulations vs Dyer and the best standard scheme."""
    standard = bench["standard"]
    marks = {s: (OKABE_ITO[c], m) for s, c, m in zip(OURS, (6, 5, 1, 6, 5, 1), "osDosD")}
    fig, axes = plt.subplots(1, 4, figsize=(24, 9), sharey=True)
    sites = [s for s in bench["sites"] if any(np.isfinite(v.get("dyer1974", {}).get("gradient", {}).get("log_rmse", np.nan))
                                               for v in bench["sites"][s]["offset_corrected"].values())]
    summary = {}
    for ax, panel in zip(axes, ("m_stable", "m_unstable", "h_stable", "h_unstable")):
        kind = panel[0]
        stats = {}
        for i, s in enumerate(sites):
            sc = {k: v["gradient"]["log_rmse"] for k, v in bench["sites"][s]["offset_corrected"].get(panel, {}).items()}
            if not np.isfinite(sc.get("dyer1974", np.nan)):
                ax.text(0.5, i, "no data", transform=ax.get_yaxis_transform(), fontsize=7, color="0.6", va="center")
                continue
            std = {k: sc[k] for k in standard if np.isfinite(sc.get(k, np.nan))}
            best = min(std, key=std.get)
            ax.plot(sc["dyer1974"], i, "|", color="0.2", ms=14, mew=2)
            ax.plot(std[best], i, "x", color="0.5", ms=8)
            ax.annotate(best, (std[best], i), xytext=(0, 6), textcoords="offset points", fontsize=6, color="0.4", ha="center")
            for f in OURS:
                v = sc.get(f, np.nan)
                if not np.isfinite(v):
                    continue
                c, m = marks[f]
                ax.plot(v, i, m, color=c, mfc=c if f.endswith("global") else "none", ms=6)
                rank = 1 + sum(x < v for x in std.values())
                st = stats.setdefault(f, {"rank": [], "best": 0, "dyer": [], "bstd": []})
                st["rank"].append(rank)
                st["best"] += rank == 1
                st["dyer"].append(v - sc["dyer1974"])
                st["bstd"].append(v - std[best])
        summary[panel] = {f: {"median_rank": float(np.median(st["rank"])), "best": st["best"], "n": len(st["rank"]),
                              "med_vs_dyer": float(np.median(st["dyer"])), "med_vs_best_std": float(np.median(st["bstd"]))}
                          for f, st in stats.items()}
        txt = "\n".join(f"{f}: rank {v['median_rank']:.0f}/{len(standard) + 1}, best {v['best']}/{v['n']}"
                        for f, v in summary[panel].items())
        ax.text(0.98, 0.01, "median rank vs standard\n" + txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(fc="white", ec="0.8"))
        ax.set_yticks(np.arange(len(sites)), sites)
        ax.set_xscale("log")
        ax.set_xlabel("gradient log-RMSE (lower is better)")
        ax.set_title(panel.replace("m_", "momentum ").replace("h_", "heat "))
        ax.grid(axis="x", color="0.9")
        ax.invert_yaxis()
    handles = [plt.Line2D([], [], ls="", marker=marks[f][1], color=marks[f][0], mfc=marks[f][0] if f.endswith("global") else "none",
                          label=f"{f} (m {_strict_label(bench, f, 'm')}, h {_strict_label(bench, f, 'h')})") for f in OURS]
    handles += [plt.Line2D([], [], ls="", marker="|", color="0.2", ms=12, mew=2, label=f"dyer1974, legoESM default (m {_strict_label(bench, 'dyer1974', 'm')})"),
                plt.Line2D([], [], ls="", marker="x", color="0.5", label="best standard scheme at that tower")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Held-out score per tower: fits on odd UTC days, scored on even days (offset-corrected arm); C1 = strict smoothness")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out, dpi=110)
    return summary


if __name__ == "__main__":
    bench = json.load(open(sys.argv[1]))
    if sys.argv[2] == "--compare":
        print(json.dumps(compare(bench, sys.argv[3]), indent=1))
        sys.exit()
    if sys.argv[2] == "--dump-bins":
        sys.exit(dump_bins(bench, sys.argv[3]))
    if sys.argv[2] == "--towers":
        bins = json.load(open(sys.argv[3]))
        for kind in "mh":
            towers(bench, bins, kind, f"{sys.argv[4]}_{kind}.png")
        sys.exit()
    stem = sys.argv[2]
    league(bench, f"{stem}.png", "gradient")
    league(bench, f"{stem}_integrated.png", "integrated")
    inversion(bench, f"{stem}_inversion.png")
    curves(bench, sys.argv[3], f"{stem}_curves.png")
