"""Spectral level-shard STRONG-scaling cliff — steps/s and speedup vs devices.

Renders the EMPIRICAL evidence for spectral's documented single-device verdict
(``create_level_mesh``: ``_valid_gpu_counts -> [1]``).  The spectral PE state is
``(n_sh, nlev)``; sharding the level axis (``P(None, "level")``) forces the
semi-implicit ``(nlev, nlev)``-per-wavenumber solve to all-gather the level axis
while the SH transforms emit no per-shard collectives — so adding devices buys NO
throughput (flat / anti-scaling), unlike the cube/latlon/icosahedral grids that
domain-decompose the horizontal.  This is the spectral grid's THEORETICAL LIMIT
on a level-only sharding scheme; true scaling needs the transpose method
(all-to-all), which is comm-bound and HW-blocked on the Gloo/TCP fabric here.

Reads the CSV emitted by ``scripts/cluster/scaling_ginsburg/spectral_cliff_probe.sbatch``
(columns: ``n_max,nlev,n_devices,shard,steps_per_s``).

    python scripts/plot/plot_spectral_level_shard.py \\
        --csv results/scaling_ginsburg/spectral_cliff_<job>.csv --out docs/scaling
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

_REQUIRED_COLS = ("n_max", "nlev", "n_devices", "shard", "steps_per_s")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_COLOR = {21: "#1f77b4", 42: "#2ca02c", 85: "#9467bd", 170: "#d62728"}


def _read(csv_path: Path) -> list[dict]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def compute_curves(rows: list[dict]) -> dict[int, list[tuple[int, float, float]]]:
    """Per ``n_max`` -> sorted ``[(n_devices, steps_per_s, speedup)]`` for the
    LEVEL-sharded path, where speedup = steps_per_s(level, N) divided by the
    TRUE single-device baseline ``steps_per_s(shard=none, N=1)``.

    Falls back to the level N=1 point as baseline if no ``none`` row exists for
    that ``n_max`` (so a level-only sweep still plots a self-relative speedup).

    Robust by codex review: a missing required column raises ``ValueError``
    (malformed CSV is loud, not silently "no data"); non-finite / nonpositive
    ``steps_per_s`` (the probe writes ``NaN`` for a crashed run) and nonpositive
    ``n_max``/``n_devices`` are skipped; a CONFLICTING duplicate (same key,
    different value) for a baseline or a ``(T, N)`` level point raises (silent
    last-write-win would let CSV row order change the curve).
    """
    if rows:
        missing = [c for c in _REQUIRED_COLS if c not in rows[0]]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    base: dict[int, float] = {}          # T -> none@N=1 baseline sps
    level1: dict[int, float] = {}        # T -> level@N=1 sps (fallback baseline)
    level: dict[tuple[int, int], float] = {}   # (T, nd) -> level sps
    for r in rows:
        try:
            T = int(r["n_max"])
            nd = int(r["n_devices"])
            sps = float(r["steps_per_s"])
        except (ValueError, KeyError, TypeError):
            continue
        if T <= 0 or nd <= 0 or not math.isfinite(sps) or sps <= 0:
            continue
        shard = (r.get("shard") or "").strip()
        if shard == "none" and nd == 1:
            if T in base and base[T] != sps:
                raise ValueError(
                    f"conflicting none@N=1 baseline for T{T}: "
                    f"{base[T]} vs {sps}")
            base[T] = sps
        elif shard == "level":
            key = (T, nd)
            if key in level and level[key] != sps:
                raise ValueError(
                    f"conflicting level point for T{T} N={nd}: "
                    f"{level[key]} vs {sps}")
            level[key] = sps
            if nd == 1:
                level1[T] = sps

    by_T: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (T, nd), sps in level.items():
        by_T[T].append((nd, sps))

    curves: dict[int, list[tuple[int, float, float]]] = {}
    for T, pts in by_T.items():
        b = base.get(T, level1.get(T))
        if not b or b <= 0:
            continue
        rows_out = sorted(pts)
        curves[T] = [(nd, sps, sps / b) for nd, sps in rows_out]
    return curves


def make_figure(rows: list[dict], out_dir: Path) -> Path:
    curves = compute_curves(rows)
    fig, (ax_sps, ax_su) = plt.subplots(1, 2, figsize=(13, 5))

    max_nd = 1
    for T, pts in sorted(curves.items()):
        c = T_COLOR.get(T, "#777777")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        su = [p[2] for p in pts]
        max_nd = max(max_nd, max(xs) if xs else 1)
        ax_sps.plot(xs, ys, marker="o", color=c, label=f"T{T} level-shard")
        ax_su.plot(xs, su, marker="o", color=c, label=f"T{T} level-shard")

    # ideal-linear reference on the speedup panel.
    if curves:
        ideal = sorted({nd for pts in curves.values() for nd, _, _ in pts})
        ax_su.plot(ideal, ideal, "k--", alpha=0.5, label="ideal (linear)")
        ax_su.axhline(1.0, color="grey", lw=0.8, alpha=0.6)

    for ax in (ax_sps, ax_su):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("emulated devices (level-axis shards)")
        ax.grid(True, which="both", alpha=0.3)
        if curves:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
    ax_sps.set_yscale("log", base=2)
    ax_sps.set_ylabel("throughput (steps/s)")
    ax_sps.set_title("Spectral level-shard — raw throughput")
    ax_su.set_ylabel("strong-scaling speedup vs 1 device")
    ax_su.set_title("Spectral level-shard — speedup (cliff)")
    ax_su.annotate(
        "SI (nlev,nlev)/wavenumber solve all-gathers levels\n"
        "+ SH transforms emit no collectives -> no parallel win",
        xy=(0.5, 0.04), xycoords="axes fraction", ha="center", fontsize=8,
        color="#a00000",
        bbox=dict(boxstyle="round", fc="#fff0f0", ec="#a00000", alpha=0.9))

    fig.suptitle("Spectral grid theoretical limit: level-only sharding does NOT "
                 "scale (SI vertical coupling) — single-device by design",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "spectral_level_shard_cliff.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True,
                   help="spectral_cliff_<job>.csv from the probe sbatch")
    p.add_argument("--out", default="docs/scaling")
    args = p.parse_args()
    rows = _read(Path(args.csv))
    out = make_figure(rows, Path(args.out))
    print(f"wrote {out}")
    for T, pts in sorted(compute_curves(rows).items()):
        for nd, sps, su in pts:
            print(f"  T{T:<4d} N={nd:<2d}  {sps:8.2f} steps/s  speedup={su:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
