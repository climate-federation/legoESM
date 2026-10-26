#!/usr/bin/env python
"""Plot a WeatherBench training run's loss curves and parameter trajectories.

Reads ``training_history.jsonl`` (one JSON object per epoch, written by
``scripts/run/train_weatherbench_scale.py``)::

    {"epoch": 0, "train_loss": 0.2458, "val_loss": 0.2491,
     "params": {"schemes.atm.clouds.CloudConfig.rh_crit": 0.85, ...}}

and writes two figures next to it:

* ``training_history_loss.png`` — training and held-out loss vs epoch. The
  held-out curve is absent for a run launched without ``--val-windows``, and
  the figure says so rather than drawing a training curve alone unlabelled.
* ``training_history_params.png`` — every parameter that MOVED, as a fraction
  of its epoch-0 value, so knobs whose units differ by ten orders of magnitude
  share one axis. Parameters that never moved are listed in the caption
  instead of drawing flat lines over the ones that did.

Import-light and JAX-free (login-node runnable): matplotlib is imported inside
``main``.

Usage::

    .venv/bin/python scripts/plot/plot_wb_training_history.py \\
        --history results/wb_amip/training_history.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_MOVED_REL = 1e-6      # a parameter counts as moved above this relative change


def load_history(path: Path) -> list[dict]:
    """Every epoch row, oldest first, de-duplicated by epoch.

    A resumed run APPENDS, so an epoch can appear twice (the chain re-ran it
    after a walltime kill). The LAST row for an epoch wins — it is the one the
    checkpoint chain kept.
    """
    rows: dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["epoch"])] = row
    if not rows:
        raise SystemExit(f"{path}: no epoch rows")
    return [rows[k] for k in sorted(rows)]


def split_params(rows: list[dict]) -> tuple[dict, list[str]]:
    """({name: [relative value per epoch]}, [names that never moved]).

    Relative to the FIRST epoch's value; a parameter whose first value is zero
    is reported as an absolute change instead (dividing would be infinite).
    """
    first = rows[0].get("params") or {}
    moved, flat = {}, []
    for name, v0 in first.items():
        if isinstance(v0, list):
            continue                      # array parameters: not plotted
        series = []
        for row in rows:
            v = (row.get("params") or {}).get(name)
            if v is None or isinstance(v, list):
                series = []
                break
            series.append((v / v0) if v0 not in (0.0, -0.0) else (v - v0))
        if not series:
            continue
        span = max(series) - min(series)
        if span > _MOVED_REL:
            moved[name] = series
        else:
            flat.append(name)
    return moved, flat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", required=True, type=Path,
                    help="training_history.jsonl written by the trainer")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the PNGs go (default: beside the history)")
    ap.add_argument("--top", type=int, default=12,
                    help="how many of the most-moved parameters to label")
    a = ap.parse_args()

    rows = load_history(a.history)
    out_dir = a.out_dir or a.history.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    train = [r["train_loss"] for r in rows]
    val = [r.get("val_loss") for r in rows]
    has_val = any(v is not None for v in val)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(epochs, train, "o-", label="training")
    if has_val:
        ve = [e for e, v in zip(epochs, val) if v is not None]
        vv = [v for v in val if v is not None]
        ax.plot(ve, vv, "s--", label="held out")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("WeatherBench training" + ("" if has_val
                 else " (no held-out pass: run without --val-windows)"))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    loss_png = out_dir / "training_history_loss.png"
    fig.savefig(loss_png, dpi=140)
    plt.close(fig)

    moved, flat = split_params(rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ranked = sorted(moved.items(),
                    key=lambda kv: max(kv[1]) - min(kv[1]), reverse=True)
    for name, series in ranked[:a.top]:
        ax.plot(epochs[:len(series)], series, label=name.split(".")[-1])
    for name, series in ranked[a.top:]:
        ax.plot(epochs[:len(series)], series, color="0.8", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value / value at epoch 0")
    ax.set_title(f"{len(moved)} parameters moved, {len(flat)} did not")
    ax.grid(alpha=0.3)
    if ranked:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    par_png = out_dir / "training_history_params.png"
    fig.savefig(par_png, dpi=140)
    plt.close(fig)

    print(f"wrote {loss_png}")
    print(f"wrote {par_png}")
    if flat:
        print(f"{len(flat)} parameters never moved: {', '.join(flat[:10])}"
              + (" ..." if len(flat) > 10 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
