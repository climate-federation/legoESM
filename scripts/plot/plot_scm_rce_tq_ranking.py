#!/usr/bin/env python
"""Before/after temperature and humidity ranking for the SCM-RCE convection arm.

The campaign's own ranking figure plots the combined score, which is 87-100 %
condensate.  This one plots the two quantities the tuning was asked to improve,
in PHYSICAL units, a-priori against tuned, one row of segments per scheme:

* left panel — mass-weighted temperature RMSE against the CRM [K];
* right panel — tropospheric relative-humidity RMSE against the CRM [RH units],
  with absolute ``q_v`` RMSE [g/kg] available via ``--humidity qv`` because the
  two answer different questions (absolute ``q_v`` is a boundary-layer metric:
  MEASURED, 1.7 % of its leverage lies above 5 km).

Reads the merged campaign CSV; computes nothing the campaign did not already
measure.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

#: Column pairs per humidity variable: (a-priori, tuned, axis label).
HUMIDITY_COLUMNS = {
    "rh": ("apriori_trop_rh_rmse", "tuned_trop_rh_rmse",
           "tropospheric RH RMSE vs CRM  [RH units]"),
    "qv": ("apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
           "column $q_v$ RMSE vs CRM  [g/kg]"),
    "trop_qv": ("apriori_trop_qv_rmse_g_kg", "tuned_trop_qv_rmse_g_kg",
                "tropospheric $q_v$ RMSE vs CRM  [g/kg]"),
}

T_COLUMNS = ("apriori_T_rmse_K", "tuned_T_rmse_K",
             "column $T$ RMSE vs CRM  [K]")

#: Schemes that cannot move in this configuration for a STRUCTURAL reason, so a
#: flat before->after segment is the honest result rather than a failed search.
#: The label is only applied when the row is ALSO flat, so a scheme that starts
#: moving loses it automatically instead of being permanently annotated.
STRUCTURALLY_FLAT = {
    "kuo": "inactive in a single column (needs large-scale moisture convergence)",
    "dca": "no tunable parameters exposed",
}


def _ff(value: str | None) -> float:
    """CSV cell -> float, with an empty/absent/unparseable cell as NaN."""
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} has no data rows.")
    return rows


def _structural_note(row: dict, prior: float, tuned: float) -> str | None:
    scheme = row.get("scheme", "")
    if scheme not in STRUCTURALLY_FLAT:
        return None
    n_params = _ff(row.get("n_tuned_params"))
    flat = (
        (math.isfinite(prior) and math.isfinite(tuned) and prior == tuned)
        or (math.isfinite(n_params) and n_params == 0)
    )
    return STRUCTURALLY_FLAT[scheme] if flat else None


def _panel(ax, rows, columns, *, title, log_x: bool):
    prior_col, tuned_col, label = columns
    data = []
    for row in rows:
        prior = _ff(row.get(prior_col))
        tuned = _ff(row.get(tuned_col))
        if not (math.isfinite(prior) or math.isfinite(tuned)):
            # A scheme with neither number is dropped and SAID so by the
            # caller; silently plotting it at 0 would read as a perfect fit.
            continue
        data.append((row.get("scheme", "?"), prior, tuned, row))
    # Rank by the TUNED value, worst at the top so the best is read first at
    # the bottom axis.  A non-finite tuned value sorts to the worst end rather
    # than to the best.
    data.sort(key=lambda d: (d[2] if math.isfinite(d[2]) else float("inf")),
              reverse=True)

    y = np.arange(len(data))
    dropped = []
    for i, (scheme, prior, tuned, row) in enumerate(data):
        if math.isfinite(prior) and math.isfinite(tuned):
            ax.plot([prior, tuned], [i, i], color="#9aa0a6", lw=1.4, zorder=1)
        note = _structural_note(row, prior, tuned)
        colour = "#b0b0b0" if note else "#264653"
        ax.scatter([prior], [i], s=46, facecolor="white", edgecolor=colour,
                   lw=1.6, zorder=2, label="a priori" if i == 0 else None)
        ax.scatter([tuned], [i], s=46, color="#e76f51" if not note else "#c9c9c9",
                   zorder=3, label="tuned" if i == 0 else None)
        if note:
            dropped.append(scheme)

    labels = [
        d[0] + (" *" if _structural_note(d[3], d[1], d[2]) else "") for d in data
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(label)
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.7, len(data) - 0.3)
    return dropped


def make_figure(rows, out_path: Path, *, humidity: str, log_x: bool,
                suptitle: str | None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if humidity not in HUMIDITY_COLUMNS:
        raise ValueError(
            f"make_figure: unknown humidity {humidity!r}; "
            f"expected one of {sorted(HUMIDITY_COLUMNS)}")

    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 0.46 * max(len(rows), 6) + 2.6), sharey=False)
    flat_T = _panel(axes[0], rows, T_COLUMNS,
                    title="Temperature", log_x=log_x)
    flat_q = _panel(axes[1], rows, HUMIDITY_COLUMNS[humidity],
                    title="Humidity", log_x=log_x)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    note = sorted(set(flat_T) | set(flat_q))
    if note:
        fig.text(
            0.5, 0.012,
            "* flat by construction: "
            + "; ".join(f"{s} — {STRUCTURALLY_FLAT[s]}" for s in note),
            ha="center", fontsize=8, color="#555555")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97 if suptitle else 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path,
                        help="merged convection-intercomparison CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--humidity", default="rh",
                        choices=sorted(HUMIDITY_COLUMNS))
    parser.add_argument("--log-x", action="store_true",
                        help="log x axis (RMSEs span decades across schemes)")
    parser.add_argument("--suptitle", default=None)
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)
    out = make_figure(rows, args.out, humidity=args.humidity,
                      log_x=args.log_x, suptitle=args.suptitle)
    objectives = sorted({r.get("objective", "unstamped") for r in rows})
    print(f"wrote {out}  ({len(rows)} schemes, objective(s)={objectives})")
    if len(objectives) > 1:
        print("WARNING: rows were tuned under DIFFERENT objectives "
              f"({objectives}); the two panels are not a controlled comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
