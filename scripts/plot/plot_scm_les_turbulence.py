"""Plot every SCM turbulence closure against its LES reference.

Reads the ``profiles.npz`` + ``ranking.csv`` written by
``scripts/run/run_scm_les_turbulence_tuning.py`` and draws one panel per scored
variable, LES in black and one coloured line per scheme.

Each scheme is drawn TWICE where the data allows it: dashed at its DEFAULT
parameters and solid at the jointly TUNED ones, in the same colour, so the
figure shows what the fit moved rather than only where it started. The npz
carries the tuned family as ``scm_<scheme>_tuned_<var>``; a scheme that was
never tuned (excluded, failed, nothing spec'd) has no such key and is drawn
once, dashed. ``--which`` restricts the figure to one state when the overlay
is too busy to read.

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
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

# (npz key infix, line style, line width, alpha, legend word). The DEFAULT
# family has no infix -- that is the on-disk layout every earlier campaign
# wrote, so an old profiles npz still plots, it simply has nothing tuned to
# overlay.
_STATES = {
    "default": ("", "--", 0.9, 0.4, "default"),
    "tuned": ("tuned_", "-", 1.7, 1.0, "tuned"),
}

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


def _schemes_in(files) -> list[str]:
    """Scheme names present in an npz, derived from the key layout alone.

    Used only when ``ranking.csv`` is absent. ``k.split("_")[1]`` truncated
    every multi-word scheme -- ``scm_holtslag_boville_theta`` reported
    ``holtslag`` -- and the tuned family would additionally have invented a
    scheme called ``tuned``. Strip the trailing variable and the optional
    ``tuned`` infix instead; whatever sits between them IS the scheme.
    """
    varnames = {n for n, _ in _PANELS}
    out: set[str] = set()
    for key in files:
        if not key.startswith("scm_"):
            continue
        body = key[len("scm_"):]
        var = body.rsplit("_", 1)[-1]
        if var not in varnames:
            continue
        body = body[: -(len(var) + 1)]
        if body.endswith("_tuned"):
            body = body[: -len("_tuned")]
        if body:
            out.add(body)
    return sorted(out)


def _states_present(data, schemes: list[str], want: str) -> list[str]:
    """Which of default/tuned this npz can actually draw.

    Asking for ``both`` on a campaign that predates the tuned family must
    produce the default-only figure, not an empty one; asking for ``tuned``
    on it is an error rather than a blank page, because a silently empty
    panel reads as "the tuned profiles match nothing".
    """
    order = ["default", "tuned"] if want == "both" else [want]
    present = [
        s for s in order
        if any(f"scm_{sch}_{_STATES[s][0]}{var}" in data.files
               for sch in schemes for var, _ in _PANELS)
    ]
    if not present:
        raise SystemExit(
            f"no {want!r} profiles in this npz for schemes {schemes}; "
            "re-run the driver so it writes the scm_<scheme>_tuned_<var> keys."
        )
    return present


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
    p.add_argument("--which", choices=("both", "default", "tuned"),
                   default="both",
                   help="draw each scheme at its DEFAULT parameters (dashed), "
                        "at the jointly TUNED ones (solid), or both.")
    args = p.parse_args(argv)

    npz, case_label = _resolve_profiles(args.indir, args.case)
    data = np.load(npz, allow_pickle=True)
    mask = data["mask"].astype(bool)
    z = data["z_scm"][mask]
    order = _read_ranking(args.indir / "ranking.csv")
    schemes = order or _schemes_in(data.files)
    states = _states_present(data, schemes, args.which)
    window = data["window_hours"]

    ncols = len(_PANELS) + len(_FLUX_PANELS)
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 6.4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.92, max(len(schemes), 1)))

    for ax, (name, label) in zip(axes[:len(_PANELS)], _PANELS):
        ref_key = f"les_scmlev_{name}"
        if ref_key in data.files:
            ax.plot(data[ref_key][mask], z, "k-", lw=2.6, zorder=10)
        for color, scheme in zip(colors, schemes):
            for state in states:
                infix, ls, lw, alpha, _ = _STATES[state]
                key = f"scm_{scheme}_{infix}{name}"
                if key in data.files:
                    ax.plot(data[key][mask], z, ls=ls, lw=lw, alpha=alpha,
                            color=color)
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
    # Colour identifies the SCHEME, line style identifies the parameter STATE,
    # so the legend is built from proxies rather than from the drawn artists:
    # nine schemes times two states would otherwise print eighteen entries.
    handles = [Line2D([], [], color="k", lw=2.6, label="LES")]
    handles += [Line2D([], [], color=c, lw=1.6, label=s)
                for c, s in zip(colors, schemes)]
    if len(states) > 1:
        handles += [
            Line2D([], [], color="0.35", ls=_STATES[s][1], lw=_STATES[s][2],
                   alpha=_STATES[s][3], label=f"({_STATES[s][4]})")
            for s in states
        ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="best")
    fig.suptitle(
        f"{case_label}: SCM turbulence closures vs LES  "
        f"(time-mean {window[0]:.2f}-{window[1]:.2f} h, same window both sides)",
        fontsize=12,
    )
    # Say WHICH parameter state is drawn. The earlier figure carried default
    # profiles under a tuned legend order, which invited reading untuned curves
    # as a tuned result.
    drawn = (" and ".join(f"{_STATES[s][4]} ({_STATES[s][1]})" for s in states)
             if len(states) > 1 else _STATES[states[0]][4])
    fig.text(0.5, 0.935,
             f"one colour per scheme, drawn at {drawn} parameters; legend "
             "order is the TUNED ranking. Schemes excluded from the ranking "
             "are not drawn.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out or (args.indir / f"profiles_vs_les_{case_label}.png")
    fig.savefig(out, dpi=145)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
