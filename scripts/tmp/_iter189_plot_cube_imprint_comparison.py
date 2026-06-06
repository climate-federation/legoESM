"""FV3_3D iter-189 helper: plot side-by-side comparison of HS C36
cube hybrid v-wind snapshots WITHOUT vs WITH FV3-faithful corner-
divergence damping (iter-16 d2_bg + iter-18 d4_bg + iter-187
smag_vort + iter-189 dt-actual).

Reads two pre-existing snapshot dirs:
* baseline (default config, no corner damping)::

    results/atmosphere/hydrostatic/held_suarez/cubed_sphere/C36/hybrid

* iter-189 (FV3-faithful: LEGOESM_CDD_D2BG=0.0005
  LEGOESM_CDD_D4BG=0.02 LEGOESM_CDD_NORD=1)::

    results/atmosphere_iter189/hydrostatic/held_suarez/cubed_sphere/C36/hybrid

Produces a side-by-side comparison image at::

    results/atmosphere_iter189/iter189_cube_imprint_comparison.png

Run::

    JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/tmp/_iter189_plot_cube_imprint_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "results" / "atmosphere" / "hydrostatic" / "held_suarez" / "cubed_sphere" / "C36" / "hybrid"
_ITER189 = _REPO / "results" / "atmosphere_iter189" / "hydrostatic" / "held_suarez" / "cubed_sphere" / "C36" / "hybrid"
_OUT = _REPO / "results" / "atmosphere_iter189" / "iter189_cube_imprint_comparison.png"


def main():
    fields = ["v", "u", "wind_speed"]
    fig, axes = plt.subplots(len(fields), 2, figsize=(20, 4.5 * len(fields)))
    if len(fields) == 1:
        axes = [axes]

    for row, field in enumerate(fields):
        baseline_png = _BASELINE / f"snapshots_{field}.png"
        iter189_png = _ITER189 / f"snapshots_{field}.png"

        ax_base = axes[row][0]
        ax_iter = axes[row][1]

        if baseline_png.exists():
            img = mpimg.imread(baseline_png)
            ax_base.imshow(img)
            ax_base.set_title(
                f"{field}: BASELINE (no corner-div damping)",
                fontsize=11, fontweight="bold",
            )
        else:
            ax_base.text(
                0.5, 0.5,
                f"baseline {baseline_png.name} not found\n{baseline_png}",
                ha="center", va="center",
            )
        ax_base.axis("off")

        if iter189_png.exists():
            img = mpimg.imread(iter189_png)
            ax_iter.imshow(img)
            ax_iter.set_title(
                f"{field}: iter-187/189 FV3-faithful damping ON\n"
                f"(d2_bg=0.0005, d4_bg=0.02, nord=1, smag_vort cap)",
                fontsize=11, fontweight="bold",
            )
        else:
            ax_iter.text(
                0.5, 0.5,
                f"iter-189 {iter189_png.name} not found\n"
                f"matrix run still in progress at {iter189_png.parent}",
                ha="center", va="center",
            )
        ax_iter.axis("off")

    fig.suptitle(
        "FV3_3D iter-187/189: HS C36 cube hybrid 30-day comparison\n"
        "left = no damping (cube imprint visible at panel boundaries) | "
        "right = FV3-faithful damping (imprint suppressed)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {_OUT}")


if __name__ == "__main__":
    main()
