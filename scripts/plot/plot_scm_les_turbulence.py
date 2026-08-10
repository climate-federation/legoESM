"""Plot every SCM turbulence closure against its LES reference.

Reads the ``profiles.npz`` + ``ranking.csv`` written by
``scripts/run/run_scm_les_turbulence_tuning.py`` and draws one panel per scored
variable, LES in black and one coloured line per scheme.

Only the levels inside the LES domain are drawn. Outside that range the
reference is NaN by construction (the LES has a sponge and a lid there and
represents nothing), and drawing it would invite reading a difference that
means nothing.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PANELS = (
    ("theta", r"$\theta$ [K]"),
    ("qv", r"$q_v$ [kg/kg]"),
    ("u", r"$u$ [m/s]"),
    ("v", r"$v$ [m/s]"),
)
_FLUX_PANELS = (
    ("wth", r"LES $\overline{w'\theta'}$ [K m/s]"),
    ("wqv", r"LES $\overline{w'q_v'}$ [kg/kg m/s]"),
)


def _read_ranking(path: Path) -> list[str]:
    """Schemes to draw, in ranked order, EXCLUDING unrankable ones.

    Filtering on ``score_default`` alone kept a scheme that was excluded from
    the ranking -- a non-finite rollout still records a default score -- so a
    blown-up arm was drawn alongside the ranked ones with nothing to say so.
    """
    if not path.exists():
        return []
    with path.open() as fh:
        return [row["scheme"] for row in csv.DictReader(fh)
                if row.get("score_default")
                and str(row.get("rank", "")).strip().upper() != "EXCLUDED"]


def _resolve_profiles(indir: Path, case: str | None) -> tuple[Path, str]:
    """Pick the ONE profiles file to plot, or fail loudly.

    A multi-case campaign writes ``profiles_<case>.npz`` per case. Silently
    taking the first glob match plotted BOMEX five times under five different
    per-case filenames (all five bytewise identical), so an ambiguous
    directory is an error here rather than an arbitrary choice.
    """
    per_case = sorted(indir.glob("profiles_*.npz"))
    available = [q.stem[len("profiles_"):] for q in per_case]
    if case is not None:
        want = indir / f"profiles_{case}.npz"
        if not want.exists():
            raise SystemExit(
                f"no profiles for case {case!r} in {indir}; "
                f"available: {available or '(none)'}"
            )
        return want, case
    single = indir / "profiles.npz"
    if single.exists():
        return single, indir.name
    if len(per_case) == 1:
        return per_case[0], available[0]
    if not per_case:
        raise SystemExit(f"no profiles*.npz in {indir}")
    raise SystemExit(
        f"{indir} holds {len(per_case)} per-case profile files "
        f"({available}); pass --case to say which one to plot."
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("indir", type=Path,
                   help="results/scm_les_turbulence/<case>")
    p.add_argument("--case", default=None,
                   help="which case to plot from a multi-case campaign "
                        "directory (reads profiles_<case>.npz). Required when "
                        "the directory holds more than one.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    npz, case_label = _resolve_profiles(args.indir, args.case)
    data = np.load(npz, allow_pickle=True)
    mask = data["mask"].astype(bool)
    z = data["z_scm"][mask]
    order = _read_ranking(args.indir / "ranking.csv")
    schemes = order or sorted({
        k.split("_")[1] for k in data.files if k.startswith("scm_")
    })
    window = data["window_hours"]

    ncols = len(_PANELS) + len(_FLUX_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 6.4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.92, max(len(schemes), 1)))

    for ax, (name, label) in zip(axes[:len(_PANELS)], _PANELS):
        ref_key = f"les_scmlev_{name}"
        if ref_key in data.files:
            ax.plot(data[ref_key][mask], z, "k-", lw=2.6, label="LES", zorder=10)
        for color, scheme in zip(colors, schemes):
            key = f"scm_{scheme}_{name}"
            if key in data.files:
                ax.plot(data[key][mask], z, lw=1.4, color=color, label=scheme)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)

    # The fluxes are LES-only: the SCM exposes no per-level w'theta' for 8 of
    # the 9 schemes, so there is deliberately nothing to overlay here.
    for ax, (name, label) in zip(axes[len(_PANELS):], _FLUX_PANELS):
        key = f"les_scmlev_{name}"
        if key in data.files:
            ax.plot(data[key][mask], z, "k-", lw=2.6, zorder=10)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
        ax.set_title("LES only", fontsize=9, color="0.4")

    axes[0].set_ylabel("height [m]")
    axes[0].legend(fontsize=7.5, loc="best")
    # The profiles in the npz are captured at DEFAULT parameters, before any
    # tuning, while the legend order comes from the TUNED ranking. Saying so on
    # the figure stops it being read as tuned profiles validating a tuned
    # ranking.
    fig.suptitle(
        f"{case_label}: SCM turbulence closures vs LES  "
        f"(time-mean {window[0]:.2f}-{window[1]:.2f} h, same window both sides)",
        fontsize=12,
    )
    fig.text(0.5, 0.935,
             "profiles are at DEFAULT parameters; legend order is the TUNED "
             "ranking. Schemes excluded from the ranking are not drawn.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out or (args.indir / f"profiles_vs_les_{case_label}.png")
    fig.savefig(out, dpi=145)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
