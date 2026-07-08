"""Map the per-column CORRECTED clubb_lite coefficient field(s) from a campaign output.

Companion to ``plot_campaign_bias_trajectory.py``: where that answers 'did the bias
fall?', this shows WHERE / HOW the LES-informed correction changed the coefficient —
the spatial structure that lets a user sanity-check the method physically (e.g. is a
raised ``C_K`` concentrated over stratocumulus decks / the storm tracks?).

Reads a campaign ``--out`` JSON: the single-coefficient form (a top-level
``C_K``/``Pr_t``/``C_eps`` per-column array) OR the multi-coefficient ``"fields"`` dict
(``{promotion_key: array}``), reshaped to the recorded ``grid.shape_2d``.  One panel per
coefficient — ``imshow`` for a 2-D (lat-lon / Gaussian) grid, else a flat per-column line
(cubed-sphere ``(6,n,n)`` / MPAS).  Only READS the stored field (no physics recomputed).
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COEFF_FIELDS = ("C_K", "Pr_t", "C_eps")


def extract_coefficient_fields(output: dict) -> dict:
    """``{label: flat list}`` for the single (top-level ``C_K``/``Pr_t``/``C_eps``) or
    multi (``"fields"`` keyed by promotion_key) output.  ``{}`` if none present."""
    fields = output.get("fields")
    if isinstance(fields, dict) and fields:
        return dict(fields)
    return {f: output[f] for f in _COEFF_FIELDS if f in output}


def _grid_caption(grid: dict) -> str:
    """A short 'which grid the coefficients live on' caption from the output's ``grid``
    provenance block, or ``""`` when absent.  The field-map shows the SPATIAL pattern of
    the correction, so naming the grid (type + shape/ncol) makes the panel self-describing
    — a per-column field is meaningless without knowing the grid it indexes."""
    if not grid:
        return ""
    gt = grid.get("grid_type")
    shape = grid.get("shape_2d")
    ncol = grid.get("ncol") or grid.get("n_columns")
    where = f"{tuple(shape)}" if shape else (f"{int(ncol)} cols" if ncol else "")
    parts = [p for p in (gt, where) if p]
    return ("  —  " + " ".join(str(p) for p in parts)) if parts else ""


def plot_corrected_coefficient_field(output: dict, out_png: str) -> str:
    """Render one panel per corrected coefficient; returns ``out_png``."""
    coeffs = extract_coefficient_fields(output)
    grid = output.get("grid") or {}
    shape_2d = grid.get("shape_2d")
    n = max(len(coeffs), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0), squeeze=False)
    if not coeffs:
        axes[0][0].text(0.5, 0.5, "no corrected coefficient\nfield in the output",
                        ha="center", va="center")
        axes[0][0].set_axis_off()
    for ax, (label, flat) in zip(axes[0], coeffs.items()):
        arr = np.asarray(flat, dtype=float).reshape(-1)
        if shape_2d is not None and len(tuple(shape_2d)) == 2 and arr.size == int(
                np.prod(shape_2d)):
            im = ax.imshow(arr.reshape(tuple(shape_2d)), origin="upper", aspect="auto",
                           cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("lon index")
            ax.set_ylabel("lat index")
        else:                                          # non-2-D grid → per-column line
            ax.plot(arr, ".")
            ax.set_xlabel("column (flat index)")
            ax.set_ylabel(label)
        ax.set_title(f"{label}  (mean {arr.mean():.4g})")
    fig.suptitle("Corrected per-column clubb_lite coefficient(s)" + _grid_caption(grid),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_json", help="campaign --out JSON file")
    p.add_argument("--png", default=None, help="output PNG (default: <output_json>.coeff.png)")
    args = p.parse_args(argv)
    with open(args.output_json) as f:
        output = json.load(f)
    png = args.png or f"{args.output_json}.coeff.png"
    plot_corrected_coefficient_field(output, png)
    print(f"[plot] wrote corrected-coefficient field map to {png}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
