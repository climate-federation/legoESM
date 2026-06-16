"""Scaling-progress ledger for the baroclinic-wave campaign.

Tracks headline scaling metrics *as a function of campaign iteration* so each
implementation/measurement step can be plotted against the theoretical limit.

Two subcommands:

* ``snapshot`` — read the tidy CSV (``aggregate_bcw_scaling.py``), compute, for
  every (backend, grid, case, precision) strong-scaling series, the strong
  efficiency at the largest device count, the peak SYPD, the peak per-device
  throughput (Mcells/s), and the max device count reached; append one ledger row
  per series with an auto-incremented ``iteration`` index, a UTC timestamp, the
  git commit, and a free-text ``--note``.
* ``plot`` — render scaling metric vs iteration (strong efficiency -> 1.0 ideal;
  per-device throughput -> roofline), one line per icosahedral series.

Strong efficiency E = [SYPD(n_max)/SYPD(n_min)] / (n_max/n_min) on the resolution
with the most device points in the series (ideal E=1).  Pure stdlib +
matplotlib; no JAX -> runs on a login node.

The ledger is append-only at ``results/bcw_scaling/scaling_ledger.csv``; the
``iteration`` is ``max(existing)+1`` per snapshot call.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path

LEDGER_FIELDS = [
    "iteration", "timestamp", "commit", "note",
    "backend", "grid", "case", "precision",
    "max_ndev", "peak_sypd", "strong_eff", "weak_eff", "peak_mcells_per_s",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def series_metrics(rows: list[dict]) -> dict:
    """Headline metrics for one (backend,grid,case,precision) strong series."""
    byres: dict = defaultdict(list)
    for r in rows:
        byres[r["resolution"]].append(
            (int(r["n_devices"]), _f(r["sypd"]), _f(r.get("mcells_per_s"))))
    # Anchor efficiency on the LARGEST resolution that has >=2 device points.
    # Strong-scaling a *small* problem to many ranks is Amdahl/comm-bound by
    # construction (cells/rank -> 0); the theoretical-limit question is how
    # close to ideal we scale a *large* problem.  (MPAS/MOM6/E3SM report
    # strong scaling at fixed large size + weak scaling at constant work/rank.)
    best_res, best_pts = None, []
    for res in sorted(byres, key=lambda r: float(r), reverse=True):
        upts = sorted(set(byres[res]))
        if len(upts) >= 2:
            best_res, best_pts = res, upts
            break
    if best_res is None:  # nothing has >=2 points yet
        best_res = max(byres, key=lambda r: float(r))
        best_pts = sorted(set(byres[best_res]))
    strong_eff = float("nan")
    if len(best_pts) >= 2:
        n0, s0, _ = best_pts[0]
        nN, sN, _ = best_pts[-1]
        if nN > n0 and s0 > 0:
            strong_eff = (sN / s0) / (nN / n0)
    return {
        "max_ndev": max(int(r["n_devices"]) for r in rows),
        "peak_sypd": max(_f(r["sypd"]) for r in rows),
        "strong_eff": strong_eff,
        "peak_mcells_per_s": max(_f(r.get("mcells_per_s")) for r in rows),
    }


def weak_efficiency(rows: list[dict]) -> float:
    """Per-rank-throughput retention for a weak-scaling series (ideal=1).

    Weak ideal = constant per-rank throughput as devices+work both grow.  The
    icosahedral subdivision jumps cell count 4x per level, so cells/rank is NOT
    exactly constant; normalize by the ACTUAL per-rank throughput
    tput(n) = (total_cells/n) / (time_per_step_s), then
    weak_eff = tput(n_max)/tput(n_min).
    """
    pts = []
    for r in rows:
        n = int(r["n_devices"])
        tps_s = _f(r.get("time_per_step_ms")) / 1000.0
        tc = _f(r.get("total_cells"))
        if n > 0 and tps_s > 0 and tc > 0:
            pts.append((n, (tc / n) / tps_s))
    if len(pts) < 2:
        return float("nan")
    pts.sort()
    return pts[-1][1] / pts[0][1] if pts[0][1] > 0 else float("nan")


def compute_rows(tidy_csv: Path) -> list[dict]:
    with tidy_csv.open() as f:
        all_rows = list(csv.DictReader(f))
    strong: dict = defaultdict(list)
    weak: dict = defaultdict(list)
    for r in all_rows:
        key = (r["backend"], r["grid"], r["case"], r["precision"])
        (strong if r["mode"] == "strong" else weak)[key].append(r)
    out = []
    for key, grp in sorted(strong.items()):
        backend, grid, case, prec = key
        m = series_metrics(grp)
        m["weak_eff"] = weak_efficiency(weak.get(key, []))
        out.append({"backend": backend, "grid": grid, "case": case,
                    "precision": prec, **m})
    return out


def _next_iteration(ledger: Path) -> int:
    if not ledger.exists():
        return 1
    with ledger.open() as f:
        its = [int(r["iteration"]) for r in csv.DictReader(f) if r.get("iteration")]
    return (max(its) + 1) if its else 1


def snapshot(tidy_csv: Path, ledger: Path, note: str, timestamp: str) -> int:
    it = _next_iteration(ledger)
    commit = _git_commit()
    metric_rows = compute_rows(tidy_csv)
    new = [{"iteration": it, "timestamp": timestamp, "commit": commit,
            "note": note, **m} for m in metric_rows]
    exists = ledger.exists()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(new)
    print(f"iteration {it}: appended {len(new)} series rows to {ledger}")
    for m in metric_rows:
        print(f"  {m['backend']:3s} {m['grid']:12s} {m['case']:5s} "
              f"{m['precision']:8s}  ndev<= {m['max_ndev']:>3}  "
              f"peakSYPD={m['peak_sypd']:8.2f}  Estrong={m['strong_eff']:.3f}  "
              f"Eweak={m['weak_eff']:.3f}  peakMc/s={m['peak_mcells_per_s']:7.1f}")
    return it


def plot(ledger: Path, out: Path, grid: str = "icosahedral") -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with ledger.open() as f:
        rows = [r for r in csv.DictReader(f) if r["grid"] == grid]
    if not rows:
        return None
    series: dict = defaultdict(list)
    for r in rows:
        series[(r["backend"], r["case"], r["precision"])].append(
            (int(r["iteration"]), _f(r["strong_eff"]), _f(r["weak_eff"]),
             _f(r["peak_mcells_per_s"])))

    fig, (ax_e, ax_w, ax_t) = plt.subplots(3, 1, figsize=(8.0, 10.5), sharex=True)
    for (backend, case, prec), pts in sorted(series.items()):
        pts = sorted(pts)
        its = [p[0] for p in pts]
        lbl = f"{backend} {case} {prec}"
        ls = "-" if prec == "float64" else "--"
        mk = "o" if case == "dry" else "s"
        ax_e.plot(its, [p[1] for p in pts], ls=ls, marker=mk, lw=1.6, ms=5, label=lbl)
        ax_w.plot(its, [p[2] for p in pts], ls=ls, marker=mk, lw=1.6, ms=5, label=lbl)
        ax_t.plot(its, [p[3] for p in pts], ls=ls, marker=mk, lw=1.6, ms=5, label=lbl)

    ax_e.axhline(1.0, color="0.4", ls=":", lw=1.2, label="ideal (E=1)")
    ax_e.set_ylabel("strong-scaling efficiency")
    ax_e.set_title(f"{grid}: scaling vs campaign iteration", fontsize=10, loc="left")
    ax_e.grid(True, alpha=0.25)
    ax_e.legend(fontsize=7, ncol=2)
    ax_w.axhline(1.0, color="0.4", ls=":", lw=1.2, label="ideal (E=1)")
    ax_w.set_ylabel("weak-scaling efficiency")
    ax_w.grid(True, alpha=0.25)
    ax_t.set_ylabel("peak per-device throughput [Mcells/s]")
    ax_t.set_xlabel("campaign iteration")
    ax_t.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--csv", default="results/bcw_scaling/bcw_scaling_tidy.csv")
    s.add_argument("--ledger", default="results/bcw_scaling/scaling_ledger.csv")
    s.add_argument("--note", default="")
    s.add_argument("--timestamp", required=True,
                   help="UTC timestamp string (caller-supplied; no clock here).")
    pl = sub.add_parser("plot")
    pl.add_argument("--ledger", default="results/bcw_scaling/scaling_ledger.csv")
    pl.add_argument("--out", default="docs/scaling/bcw_scaling_vs_iteration.png")
    pl.add_argument("--grid", default="icosahedral")
    args = p.parse_args()

    if args.cmd == "snapshot":
        snapshot(Path(args.csv), Path(args.ledger), args.note, args.timestamp)
    elif args.cmd == "plot":
        out = plot(Path(args.ledger), Path(args.out), args.grid)
        print(f"Wrote {out}" if out else "No ledger rows for grid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
