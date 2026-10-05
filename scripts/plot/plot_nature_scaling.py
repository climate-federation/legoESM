"""Nature-figure scaling panels from JSONL receipts: strong + weak, CPU + GPU.

Two figures (``--mode strong`` / ``--mode weak``), each 2 rows (atmosphere,
ocean) x 3 grids (lat-lon / MPAS / cubed-sphere; tripole / MPAS / FESOM2).
Every point is a measured receipt row (``steady_median_ms``); the loader keys
rows by (component, grid, backend, precision, mode, resolution, n_devices) and
keeps the FASTEST receipt per key (across partitionings / comm env — a
"best measured" figure), writing a provenance CSV (job id, timed steps, file)
next to the figure so any point can be traced.  Weak series are keyed by the
PER-DEVICE size (rows, cells or tile edge per device); mixed-precision rows
are not plotted.

Encoding (fixed across panels): GPU = blue, CPU = orange; float32 = thick
line, float64 = thin line; resolution = marker.  Ideal (dashed, grey) is
anchored at each series' first measured point: t0*n0/n for strong, flat for
weak.  Cube counts are 6*kt^2 and are ticked at their real values.

    python scripts/plot/plot_nature_scaling.py --mode strong \
        --receipts /work/bd1083/b309178/diffESM/scaling_receipts \
        /scratch/b/b381103/legoesm_scaling/nature_* --out fig_scaling_strong.pdf
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (row, col) -> (component, grid, title, sub)
PANELS = [
    ("atmosphere", "latlon", "lat–lon", "4096×8192 L26 (strong) · 32 / 8 rows per device (weak)"),
    ("atmosphere", "icosahedral", "MPAS icosahedral", "subdiv-9/10 L26 · 81k / 5k cells per device (weak)"),
    ("atmosphere", "cubed-sphere", "cubed-sphere", "C768 L26 · 6·kt² tiles"),
    ("ocean", "tripole", "tripole (ORCA fold)", "3072×4352 L75 (ORCA12-class) · GPU only — see note"),
    ("ocean", "mpas", "MPAS Voronoi", "subdiv-9 L40 (2.6M cells) · 20k / 5k cells per device (weak)"),
    ("ocean", "fesom", "FESOM2 (fesom_jax)", "forca20 2.1M nodes L70 · float64 only"),
]
BACKEND_COLOR = {"gpu": "#0072B2", "cpu": "#D55E00"}
PREC_LW = {"float32": 1.7, "float64": 0.8}
MARKERS = ["o", "s", "^", "D", "v", "P"]
# The SPMD benches labelled precision from JAX_ENABLE_X64 but built fp32 state
# until commit 4b763579a (2026-08-26); every float64 receipt from a job older
# than the first post-fix ladder is a mislabelled fp32 run and is refused.
FIRST_REAL_F64_JOB = 27253192
# Receipts without a SLURM job id (Derecho PBS/PALS, interactive runs) are
# dated instead: the fix reached the benches' branch on 2026-09-16.
FIRST_REAL_F64_UTC = "2026-09-17"


def _real_f64(r, job):
    if job:
        return int(job) >= FIRST_REAL_F64_JOB
    ts = r.get("metadata", {}).get("timestamp_utc")
    return isinstance(ts, str) and ts >= FIRST_REAL_F64_UTC
# canonical vertical levels per lane: receipts at other level counts are a
# different problem and are dropped (e.g. the 32-level MPAS probe rows)
#: The vertical level count each lane's curve is built from.  This is a
#: FILTER, not a label: a receipt at another count is refused rather than
#: drawn, because the level count changes the cost per cell and mixing two
#: of them in one curve would read as scaling.
#:
#: The atmosphere ladder moved from 26 levels to 40 on 2026-09-23 (26 was
#: the most expensive count in the repo's measured table, 1.6x the per-level
#: cost of 40 at subdivision 9, while production AMIP runs 40).  This still
#: defaults to 26 so the existing curves keep plotting; pass --atm-nlev 40
#: once enough receipts at the new count exist.  The two sets are NOT
#: comparable and must not share a figure.
ATM_NLEV_DEFAULT = 26


def _nlev_map(atm_nlev: int) -> dict:
    return {("atmosphere", "latlon"): atm_nlev,
            ("atmosphere", "icosahedral"): atm_nlev,
            ("atmosphere", "cubed-sphere"): atm_nlev,
            ("ocean", "tripole"): 75, ("ocean", "mpas"): 40}


NLEV = _nlev_map(ATM_NLEV_DEFAULT)


def _component(r):
    c = r.get("component") or r.get("metadata", {}).get("component") or ""
    if c in ("mpas_atm", "atmosphere") or r.get("metadata", {}).get("component") == "atmosphere":
        return "atmosphere"
    return "ocean"


def _grid(r, comp):
    g = r.get("grid_type") or r.get("metadata", {}).get("grid") or ""
    if comp == "ocean" and g in ("icosahedral", "mpas"):
        return "mpas"
    if comp == "atmosphere" and g in ("icosahedral", "mpas"):
        return "icosahedral"
    return g


def _backend(r):
    b = (r.get("platform") or r.get("backend") or r.get("metadata", {}).get("backend") or "").lower()
    return "gpu" if b in ("gpu", "cuda", "rocm") else "cpu"


def _res(r, grid, mode):
    """Series label: global size for strong, PER-DEVICE size for weak (the
    global size grows with the device count there, so it cannot be the key)."""
    nd = int(r["n_devices"])
    if grid in ("latlon", "tripole"):
        n_lat = r.get("n_lat")
        if n_lat is None:
            raise KeyError(f"receipt without n_lat: {r.get('resolution')}")
        n_lon = r.get("n_lon")
        if n_lon is None:
            raise KeyError(f"receipt without n_lon: {r.get('resolution')}")
        return (f"LL{n_lat}x{n_lon}" if mode == "strong"
                else f"{int(n_lat) // nd} rows/dev x{n_lon}")
    if grid in ("icosahedral", "mpas"):
        if mode == "strong":
            return f"s{r.get('subdivision', r.get('resolution'))}"
        return f"{round(int(r['n_cells']) / nd / 1e3)}k cells/dev"
    if grid == "cubed-sphere":
        n = int(r["resolution"])
        kt = r.get("kt")
        if mode == "weak" and kt is None:
            raise KeyError("cube receipt without kt")
        return f"C{n}" if mode == "strong" else f"C{n // int(kt)}/tile"
    return str(r.get("resolution", ""))


def _mode(r, path):
    """The MPAS-atm and cube benches have no --mode: their weak arms are
    strong-bench runs at a larger mesh per device count, tagged only by the
    ladder's FILENAME (nature_ladder.sbatch)."""
    return "weak" if "_weak_" in os.path.basename(path) else r.get("mode", "strong")


# MPASOceanConfig defaults the figure is measured at: fixed_iters, precond,
# poly_sweeps.  Rows solving anything else are a different model.
OCEAN_MPAS_PCG_ITERS = 20
OCEAN_MPAS_PCG_PRECOND = ("poly", 4)
# NCCL channel count both MPAS lanes pin (nature_ladder.sbatch): 8 -> 32 on
# 2026-09-21 (s9 atm: 5.57 -> 4.65 ms at 128 GPUs).  Multi-device MPAS GPU
# rows at any other count or chunk size, or without the stamp, are refused.
# The 32-channel gain is ATMOSPHERE evidence; the ocean lane (PCG-dominated,
# 2M+9 allreduces/step) is pinned by decision and A/B-checked separately.
MPAS_NCCL_CHANNELS = {"icosahedral": "64", "mpas": "8"}
# Atmosphere moved 32 -> 64 on 2026-09-25 (s9 at 128 GPUs: -7%, both run
# orders, jobs 27627022 / 27670098); 32-channel atmosphere rows are refused.
# Ocean MPAS pins 8 (ladder since 2026-09-24: 32 channels hang that lane at
# 128 GPUs); older 32-channel ocean rows are refused (owner, 2026-09-25).
# Multi-rank CPU rows need each rank's full core share (nature_ladder.sbatch
# passes --cpus-per-task = node threads / ranks-per-node = 64 since
# 2026-09-21); rows stamped below this, or unstamped, were 1-core ranks
# (7.4x slower per rank) and are refused -- single-rank rows included, the
# old launch bound a one-task step to one core just the same.  The floor is
# 4, not 16: the Derecho CPU ladder runs 16 ranks x 8 cores per node
# (affinity 8, 2026-09-25), while a one-core rank reads 1 or 2.
CPU_AFFINITY_MIN = 4
MPAS_NCCL_CHUNK = "131072"


def load(dirs):
    best = {}
    dropped = []
    dropped_nccl = []
    dropped_aff = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
            if f.endswith(".failed.jsonl"):     # quarantined by the ladder
                continue
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ms = r.get("steady_median_ms")
                # legacy receipts (pre-2026-09 benches) carry neither valid nor
                # finite_ok; an explicit False on either, or a non-positive /
                # non-finite timing, is refused.
                if (ms is None or not (0.0 < float(ms) < float("inf"))
                        or r.get("valid") is False or r.get("finite_ok") is False):
                    continue
                if r.get("metadata", {}).get("virtual_cpu_devices"):
                    continue
                # A row with no device count cannot sit on a scaling curve at
                # all; probe/census receipts in the same trees carry none.
                if r.get("n_devices") is None:
                    continue
                comp = _component(r)
                grid = _grid(r, comp)
                # Atmosphere receipts before the 2026-09-27 benchmark fixes
                # timed states that went non-finite (lat-lon: polar filter
                # off at dt 60 s; MPAS: a fixed del4 coefficient unstable
                # from level 7 up) and none was checked: only rows that
                # PROVED a finite state are data.
                if comp == "atmosphere" and r.get("finite_ok") is not True:
                    continue
                prec = r.get("precision") or r.get("metadata", {}).get("precision")
                if prec not in PREC_LW:
                    continue
                job = r.get("slurm_job_id") or r.get("metadata", {}).get("slurm_job_id")
                if prec == "float64" and grid != "fesom" and not _real_f64(r, job):
                    continue
                nlev = r.get("nlev", r.get("n_levels"))
                if (comp, grid) in NLEV and nlev != NLEV[(comp, grid)]:
                    continue
                # The ocean MPAS barotropic solver's iteration count sets most
                # of its step time, and it changed from 60 to 30 (commit
                # 4a208be8d).  Receipts from either setting are valid timings
                # of DIFFERENT models, so mixing them inside one curve would
                # attribute a solver change to parallel scaling.  Receipts
                # predating the field carry no count and are refused here.
                if comp == "ocean" and grid == "mpas":
                    extra = r.get("metadata", {}).get("extra", {})
                    iters = extra.get("pcg_fixed_iters")
                    # Single-device rows solve to a TOLERANCE (stock CG) and
                    # never run the fixed count, so their recorded count is
                    # inert and they stay comparable across the change.
                    solver = extra.get("pcg_solver_path")
                    pre = (extra.get("pcg_precond"), extra.get("pcg_poly_sweeps"))
                    # single_reduce is a different recurrence (and faster);
                    # a best-of key would silently pick it over the standard row.
                    variant = extra.get("pcg_variant", "standard")
                    if variant != "standard" or (
                            solver != "stock_cg_to_tol"
                            and (iters != OCEAN_MPAS_PCG_ITERS
                                 or pre != OCEAN_MPAS_PCG_PRECOND)):
                        dropped.append((f, int(r["n_devices"]),
                                        f"{iters}/{pre[0]}{pre[1]}/{variant}"))
                        continue
                if _backend(r) == "cpu":
                    aff = r.get("metadata", {}).get("cpu_affinity")
                    if aff is None or int(aff) < CPU_AFFINITY_MIN:
                        dropped_aff.append((f, int(r["n_devices"]), aff))
                        continue
                if (grid in ("icosahedral", "mpas") and _backend(r) == "gpu"
                        and int(r["n_devices"]) > 1):
                    env = r.get("metadata", {}).get("extra", {}).get("nccl_env") or {}
                    ch = (env.get("NCCL_MIN_NCHANNELS"), env.get("NCCL_MAX_NCHANNELS"),
                          env.get("NCCL_P2P_NET_CHUNKSIZE"))
                    pin = MPAS_NCCL_CHANNELS.get(grid)
                    if pin is None or ch != (pin, pin, MPAS_NCCL_CHUNK):
                        dropped_nccl.append((f, int(r["n_devices"]), ch))
                        continue
                mode = _mode(r, f)
                key = (comp, grid, _backend(r), prec, mode, _res(r, grid, mode),
                       int(r["n_devices"]))
                steps = r.get("steps") or r.get("metadata", {}).get("extra", {}).get("steps")
                if key not in best or ms < best[key][0]:
                    best[key] = (float(ms), job, f, steps)
    if dropped:
        # Never silent: a refused row and a node that died both look like a
        # missing point on the curve, and only one of them is the reader's
        # problem.
        counts = {}
        for _f, nd, it in dropped:
            counts[it] = counts.get(it, 0) + 1
        print(f"load: refused {len(dropped)} ocean-MPAS receipts solving "
              f"other than {OCEAN_MPAS_PCG_ITERS} iterations with "
              f"{OCEAN_MPAS_PCG_PRECOND[0]}{OCEAN_MPAS_PCG_PRECOND[1]} "
              f"(counts found: "
              + ", ".join(f"{k!r}x{v}" for k, v in sorted(
                  counts.items(), key=lambda kv: str(kv[0]))) + ")",
              file=sys.stderr)
        for _f, nd, it in sorted(dropped, key=lambda d: d[1])[:20]:
            print(f"  nd={nd:<5} iters={it!r}  {_f}", file=sys.stderr)
    if dropped_aff:
        print(f"load: refused {len(dropped_aff)} CPU receipts whose "
              f"ranks had fewer than {CPU_AFFINITY_MIN} hardware threads "
              f"(unstamped = pre-2026-09-21 one-core ranks): "
              + ", ".join(sorted({f"{d[0].split('/')[-2]}" for d in dropped_aff})),
              file=sys.stderr)
    if dropped_nccl:
        counts = {}
        for _f, nd, ch in dropped_nccl:
            counts[ch] = counts.get(ch, 0) + 1
        print(f"load: refused {len(dropped_nccl)} multi-device MPAS GPU receipts "
              f"not at {MPAS_NCCL_CHANNELS} NCCL channels / {MPAS_NCCL_CHUNK} chunk "
              f"(min/max/chunk found: "
              + ", ".join(f"{k!r}x{v}" for k, v in sorted(
                  counts.items(), key=lambda kv: str(kv[0]))) + ")",
              file=sys.stderr)
        for _f, nd, ch in sorted(dropped_nccl, key=lambda d: d[1])[:20]:
            print(f"  nd={nd:<5} channels={ch!r}  {_f}", file=sys.stderr)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["strong", "weak"], default="strong")
    ap.add_argument("--atm-nlev", type=int, default=ATM_NLEV_DEFAULT,
                    help="vertical level count the ATMOSPHERE curves are "
                         "built from. Receipts at any other count are "
                         "refused, because the level count changes the cost "
                         "per cell and mixing two of them in one curve reads "
                         "as scaling. The ladder moved 26 -> 40 on "
                         "2026-09-23; this still defaults to 26 so existing "
                         "curves keep plotting.")
    ap.add_argument("--receipts", nargs="+", required=True)
    ap.add_argument("--out", default="fig_scaling.pdf")
    ap.add_argument("--png", default=None)
    ap.add_argument("--max-res", type=int, default=2,
                    help="resolutions per panel (highest first)")
    ap.add_argument("--layout", choices=["main", "supp", "all"], default="main",
                    help="main = the five grids whose multi-device path is "
                         "performance-tuned (the cube slot carries the legend); "
                         "supp = the cubed-sphere panel alone; all = every panel")
    args = ap.parse_args()
    global NLEV
    NLEV = _nlev_map(args.atm_nlev)

    best = load(args.receipts)
    series = defaultdict(list)     # (comp,grid,backend,prec,res) -> [(nd, ms, job, file, steps)]
    for (comp, grid, be, prec, mode, res, nd), (ms, job, f, steps) in best.items():
        if mode == args.mode:
            series[(comp, grid, be, prec, res)].append((nd, ms, job, f, steps))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7.5,
        "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 5.8, "axes.linewidth": 0.6, "lines.markersize": 3.2,
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # The tiled cubed-sphere step is correctness-validated but not yet
    # performance-tuned above 6 devices (it anti-scales: 23 -> 41 -> 34 ms at
    # 24 -> 54 -> 96 GPUs on C768), so the main figure leaves its slot to the
    # legend and the panel moves to the supplement.
    cube_i = next(i for i, p in enumerate(PANELS) if p[1] == "cubed-sphere")
    if args.layout == "supp":
        fig, axs = plt.subplots(1, 1, figsize=(70 / 25.4, 60 / 25.4), squeeze=False)
        panels = [(cube_i, PANELS[cube_i])]
    else:
        fig, axs = plt.subplots(2, 3, figsize=(180 / 25.4, 105 / 25.4))
        panels = [(i, p) for i, p in enumerate(PANELS)
                  if args.layout == "all" or i != cube_i]
        if args.layout == "main":
            axs.ravel()[cube_i].set_axis_off()
    prov = []
    summary = []
    for slot, (i, (comp, grid, title, sub)) in enumerate(panels):
        ax = axs.ravel()[slot if args.layout == "supp" else i]
        keys = [k for k in series if k[0] == comp and k[1] == grid]
        # highest resolutions first, capped per panel
        ress = sorted({k[4] for k in keys},
                      key=lambda s: -int("".join(ch for ch in s if ch.isdigit()) or 0))[:args.max_res]
        allx = set()
        for k in sorted(keys, key=lambda k: (k[2], k[3], k[4])):
            _, _, be, prec, res = k
            if res not in ress:
                continue
            pts = sorted(series[k])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            allx |= set(xs)
            m = MARKERS[ress.index(res) % len(MARKERS)]
            c = BACKEND_COLOR[be]
            ax.plot(xs, ys, m + "-", color=c, lw=PREC_LW[prec], markerfacecolor="white",
                    markeredgewidth=0.8, clip_on=False, zorder=3,
                    label=f"{be.upper()} {prec[:1]}{prec[-2:]} {res}")
            n0, t0 = xs[0], ys[0]
            ideal = [t0 * n0 / n for n in xs] if args.mode == "strong" else [t0] * len(xs)
            ax.plot(xs, ideal, "--", color=c, lw=0.6, alpha=0.5, zorder=2)
            eff = (t0 * n0 / (xs[-1] * ys[-1]) if args.mode == "strong" else t0 / ys[-1])
            summary.append((comp, grid, be, prec, res, xs[0], xs[-1], ys[0], ys[-1], eff))
            for nd, ms, job, f, steps in pts:
                prov.append((comp, grid, be, prec, res, args.mode, nd, ms, steps, job, f))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        if allx:
            ticks = sorted(allx)
            if len(ticks) > 8:
                ticks = [x for j, x in enumerate(ticks) if j % 2 == 0 or x == ticks[-1]]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(x) for x in ticks])
        ax.minorticks_off()
        ax.set_title(title, pad=8, loc="left", fontweight="bold")
        ax.text(0, 1.01, sub, transform=ax.transAxes, fontsize=5.2, color="#555555", va="bottom")
        ax.tick_params(direction="out", length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if keys:
            ax.legend(frameon=False, loc="lower left" if args.mode == "strong" else "upper left",
                      handlelength=1.6, borderpad=0.2, labelspacing=0.25)
        else:
            ax.text(0.5, 0.5, "no receipts yet", transform=ax.transAxes, ha="center",
                    color="#999999", fontsize=7)
        if args.layout == "supp" or i % 3 == 0:
            ax.set_ylabel("time per step (ms)")
        if args.layout == "supp" or i >= 3 or (args.layout == "main" and i == 1):
            ax.set_xlabel("devices (GPUs or CPU ranks)")
        ax.text(-0.22, 1.18, chr(ord("a") + slot), transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")

    handles = [
        Line2D([], [], color=BACKEND_COLOR["gpu"], lw=1.7, label="A100 GPU (NCCL)"),
        Line2D([], [], color=BACKEND_COLOR["cpu"], lw=1.7, label="Milan CPU (gloo)"),
        Line2D([], [], color="#444444", lw=1.7, label="float32"),
        Line2D([], [], color="#444444", lw=0.8, label="float64"),
        Line2D([], [], color="#666666", ls="--", lw=0.6, label="ideal (anchored at first point)"),
    ]
    if args.layout == "main":
        axs.ravel()[cube_i].legend(handles=handles, frameon=False, loc="center left",
                                   handlelength=1.8, labelspacing=0.5, fontsize=6.2)
    else:
        fig.legend(handles=handles, frameon=False, loc="lower center", ncol=5,
                   bbox_to_anchor=(0.5, -0.01), handlelength=1.8)
    if args.layout != "supp":
        for row, name in ((0, "ATMOSPHERE"), (1, "OCEAN")):
            fig.text(0.008, 0.93 if row == 0 else 0.46, name, fontsize=7.5,
                     fontweight="bold", rotation=90, va="top", ha="left")
    fig.suptitle(f"{args.mode.capitalize()} scaling — Levante (4×A100-80/node, 2×AMD Milan 7763/node)",
                 fontsize=7.5, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.14, wspace=0.42, hspace=0.62)
    fig.savefig(args.out, bbox_inches="tight")
    if args.png:
        fig.savefig(args.png, bbox_inches="tight")

    base = os.path.splitext(args.out)[0]
    with open(base + "_provenance.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["component", "grid", "backend", "precision", "resolution", "mode",
                    "n_devices", "ms_per_step", "timed_steps", "slurm_job_id", "receipt"])
        w.writerows(sorted(prov))
    print(f"{args.out}  ({len(prov)} points)")
    print("component grid backend prec res  n0->n1   ms0->ms1   efficiency")
    for s in sorted(summary):
        print(f"{s[0][:3]} {s[1]:12s} {s[2]} {s[3]} {s[4]:7s} {s[5]:4d}->{s[6]:<4d} "
              f"{s[7]:9.2f}->{s[8]:<9.2f} {s[9]:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
