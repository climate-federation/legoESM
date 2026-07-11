"""Plot lat-band SPMD scaling for the lat-lon C-grid hydrostatic atm step
(bench_atm_latlon_spmd_scaling.py output). Strong: speedup vs ideal-linear;
weak: parallel efficiency (t1/tn) vs ideal-flat. Reads one or more JSONL files
(strong + weak; CPU + GPU)."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _row_time_ms(r):
    """Per-step time of a bench row.

    M1 measurement-contract rows carry ``fused_step_ms`` (and the canonical
    ``time_per_step_ms``); pre-M1 rows carried ``steady_min_ms`` — accepted
    as a legacy fallback so old JSONL ladders still plot (codex batch4:
    this plotter previously REQUIRED the removed ``steady_min_ms``).
    """
    for key in ("fused_step_ms", "time_per_step_ms", "steady_min_ms"):
        v = r.get(key)
        if v is not None:
            return float(v)
    raise KeyError(
        "bench row has none of fused_step_ms/time_per_step_ms/"
        f"steady_min_ms: {sorted(r)}")


def _by_dev(rows):
    rows = sorted(rows, key=lambda r: r["n_devices"])
    nd = [r["n_devices"] for r in rows]
    t = [_row_time_ms(r) for r in rows]
    return nd, t


def speedup_strong(rows):
    nd, t = _by_dev(rows)
    if not nd:
        return [], []
    t1 = t[0]
    return nd, [t1 / ti if ti else 0.0 for ti in t]


def efficiency_weak(rows):
    # weak: per-device work fixed -> ideal time CONSTANT; efficiency = t1/tn.
    nd, t = _by_dev(rows)
    if not nd:
        return [], []
    t1 = t[0]
    return nd, [t1 / ti if ti else 0.0 for ti in t]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strong", action="append", default=[],
                   help="strong-scaling JSONL (repeatable, e.g. CPU + GPU)")
    p.add_argument("--weak", action="append", default=[])
    p.add_argument("--label", action="append", default=[],
                   help="label per --strong file (same order)")
    p.add_argument("--out", default="docs/scaling/atm_latlon_spmd_scaling.png")
    args = p.parse_args()

    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(12, 5))

    # Strong: speedup vs ideal.
    max_nd = 1
    for i, path in enumerate(args.strong):
        rows = _load(path)
        nd, sp = speedup_strong(rows)
        if not nd:
            continue
        max_nd = max(max_nd, max(nd))
        lab = args.label[i] if i < len(args.label) else os.path.basename(path)
        plat = rows[0].get("platform", "")
        ax_s.plot(nd, sp, "o-", label=f"{lab} ({plat})")
    ax_s.plot([1, max_nd], [1, max_nd], "k--", alpha=0.5, label="ideal linear")
    ax_s.set_xlabel("devices"); ax_s.set_ylabel("speedup vs 1 device")
    ax_s.set_title("Strong scaling (fixed grid)")
    ax_s.legend(); ax_s.grid(alpha=0.3)

    # Weak: efficiency vs ideal-flat.
    for i, path in enumerate(args.weak):
        rows = _load(path)
        nd, eff = efficiency_weak(rows)
        if not nd:
            continue
        lab = args.label[i] if i < len(args.label) else os.path.basename(path)
        plat = rows[0].get("platform", "")
        ax_w.plot(nd, eff, "s-", label=f"{lab} ({plat})")
    ax_w.axhline(1.0, color="k", ls="--", alpha=0.5, label="ideal flat")
    ax_w.set_xlabel("devices"); ax_w.set_ylabel("parallel efficiency (t1/tn)")
    ax_w.set_title("Weak scaling (fixed rows/device)")
    ax_w.set_ylim(0, 1.15); ax_w.legend(); ax_w.grid(alpha=0.3)

    fig.suptitle("lat-band SPMD — lat-lon C-grid hydrostatic atm step")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
