"""Case x grid map overview for the standard ocean-grid benchmark suite.

One figure per CASE: a map per grid at the final saved time, showing the
field that case is actually about (SSH for the wave/adjustment cases, SST
for the tracer cases), plus a combined contact sheet across all cases.

Run AFTER the suite:

    sbatch --array=0-8 scripts/cluster/ocean_grid_benchmark_suite.sbatch
    .venv/bin/python scripts/plot/plot_ocean_grid_benchmark_maps.py

See docs/ocean/experiments/lock_exchange_benchmark.md.
"""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]

#: Print settings. The defaults here are for a figure that will be READ on
#: paper: 300 dpi raster plus a vector PDF, and type large enough to survive
#: a two-column reduction. The previous 130 dpi / 7-8 pt was fine on screen
#: and illegible in print.
PUB = {"title": 10, "label": 10, "tick": 9, "suptitle": 12, "dpi": 300}

# Fixed column order; a grid absent from a case leaves an annotated blank
# rather than silently shifting the others along.
# Ocean grids only; cubed_sphere removed 2026-08-11 (not an ocean grid).
GRIDS = ["latlon", "mpas", "fesom", "tripole"]

#: Cases that live on their OWN grid family rather than the four global
#: arms. The f-plane channel IGW is a Cartesian box, so drawing it in the
#: global columns would leave the row empty and read as "this case did not
#: run" -- which is how the case went unplotted entirely (2026-08-13).
CASE_GRIDS = {
    "inertia_gravity_wave_channel": ["latlon_channel"],
}


def _grids_for(case: str) -> list[str]:
    return CASE_GRIDS.get(case, GRIDS)

# Per case: (field, colormap, label). Sequential map for a magnitude-like
# tracer, diverging for a signed anomaly around zero.
CASES = {
    "rest_state_stratified_with_land": ("SST", "viridis", "SST [degC]"),
    "rest_state_uniform_with_land": ("SST", "viridis", "SST [degC]"),
    "rest_state_stratified_no_land": ("SST", "viridis", "SST [degC]"),
    "rest_state_uniform_no_land": ("SST", "viridis", "SST [degC]"),
    "barotropic_wave": ("eta", "RdBu_r", "SSH [m]"),
    "geostrophic_adjustment": ("SST", "viridis", "SST [degC]"),
    "phillips_two_layer": ("eta", "RdBu_r", "SSH [m]"),
    "inertia_gravity_wave": ("eta", "RdBu_r", "SSH [m]"),
    "lock_exchange": ("SST", "viridis", "SST [degC]"),
    # The one case with an EXACT solution to be scored against; its axes
    # are Cartesian channel coordinates carried as pseudo-degrees.
    "inertia_gravity_wave_channel": ("eta", "RdBu_r", "SSH [m]"),
}


def _find(root: Path, case: str, grid: str) -> Path | None:
    hits = sorted(root.glob(f"{case}/**/{grid}/*/snapshots_latlon.npz"))
    # A case dir can hold a regional sibling (latlon_regional) whose name
    # contains the global one; match the grid directory EXACTLY.
    hits = [h for h in hits if h.parent.parent.name == grid]
    return hits[0] if hits else None


def _status(npz: Path) -> str:
    res = npz.parent / "results.txt"
    if not res.exists():
        return "?"
    for line in res.read_text().splitlines():
        if line.startswith("status:"):
            return line.split(":", 1)[1].strip()
    return "?"


def _load_field(npz: Path, field: str):
    """(lat, lon, masked 2-D field at the final saved time, t_days)."""
    z = np.load(npz)
    a = np.asarray(z[field])
    a = a[-1] if a.ndim == 3 else a
    if "land_mask" in z:
        m = np.asarray(z["land_mask"])
        m = m[-1] if m.ndim == 3 else m
        a = np.where(m > 0.5, a, np.nan)
    return (np.asarray(z["lat"]), np.asarray(z["lon"]), a,
            float(np.asarray(z["times_days"])[-1]))


def _shared_scale(fields, cmap: str):
    """One colour scale for a whole CASE, so the grids are comparable.

    Also refuses to amplify round-off: if the spread across every arm is a
    negligible fraction of the field magnitude (the rest-state cases hold
    T constant to ~1e-14), autoscaling would render machine noise as
    vivid structure and the panel would read as a broken model. In that
    case the scale is widened to a physically meaningful band and the
    caller annotates the true spread.
    """
    finite = np.concatenate([f[np.isfinite(f)].ravel() for f in fields
                             if np.isfinite(f).any()]) if fields else \
        np.array([0.0])
    lo, hi = float(np.min(finite)), float(np.max(finite))
    scale = max(abs(lo), abs(hi), 1e-30)
    degenerate = (hi - lo) < 1e-9 * scale
    if cmap == "RdBu_r":
        v = max(abs(lo), abs(hi)) or 1.0
        return dict(vmin=-v, vmax=v), degenerate, (lo, hi)
    if degenerate:
        mid = 0.5 * (lo + hi)
        pad = max(0.5 * abs(mid), 1.0)       # a real physical band
        return dict(vmin=mid - pad, vmax=mid + pad), degenerate, (lo, hi)
    return dict(vmin=lo, vmax=hi), degenerate, (lo, hi)


def _panel(ax, lat, lon, a, cmap: str, kw):
    return ax.pcolormesh(lon, lat, a, cmap=cmap, shading="auto", **kw)


def _provenance(root: Path) -> str:
    """One line naming WHICH run each figure was built from.

    A missing panel is now obvious (hatched, labelled). A panel from a
    PREVIOUS run is not: it is present, plausible, and silently wrong, so a
    figure mixing two runs looks perfectly fine. The plotter reads whatever is
    on disk, so the defence is to stamp what it read -- the oldest and newest
    artifact it drew from. A wide span means mixed runs.

    The commit is the PLOTTING checkout's, and is labelled as such: nothing on
    disk records which commit produced the arms, so a stamp reading "commit X"
    beside the run times would assert a provenance this script cannot know --
    check out another branch and replot, and the figures would claim the arms
    came from it.
    """
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30).stdout.strip() or "?"
    except Exception:
        sha = "?"
    # WHAT PRODUCED EACH ARM, read from the arms themselves. Every arm records
    # the commit it ran at (and whether that tree was dirty); differing commits
    # mean the figure mixes runs, which is the thing the time span was standing
    # in for and could not actually detect -- rerun one case hours later
    # against different code and the span stays narrow.
    commits = set()
    for res in root.rglob("results.txt"):
        for line in res.read_text(errors="replace").splitlines():
            if line.startswith("provenance_commit:"):
                commits.add(line.split(":", 1)[1].strip())
    if len(commits) > 1:
        mixed = "  ** MIXED RUNS: arms came from " + ", ".join(sorted(commits))
    elif commits and any("dirty" in c for c in commits):
        mixed = "  ** the tree was DIRTY when these arms ran **"
    elif commits:
        mixed = f"  arms ran at {commits.pop()}"
    else:
        mixed = "  ** arms record no commit — rerun to stamp them **"

    stamps = sorted(p.stat().st_mtime
                    for p in root.rglob("snapshots_latlon.npz"))
    if not stamps:
        return f"plotted at commit {sha} — no run artifacts found{mixed}"
    import datetime as _dt
    fmt = lambda t: _dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    span_h = (stamps[-1] - stamps[0]) / 3600.0
    warn = ("  ** artifacts span %.1f h — check this is ONE run **" % span_h
            if span_h > 6.0 else "")
    return (f"plotted at commit {sha} — {len(stamps)} arms, written "
            f"{fmt(stamps[0])} to {fmt(stamps[-1])}{warn}{mixed}")


def build_case_figure(root: Path, case: str, out_dir: Path) -> Path | None:
    field, cmap, label = CASES[case]
    grids = _grids_for(case)
    found = {g: _find(root, case, g) for g in grids}
    if not any(found.values()):
        return None
    loaded = {g: _load_field(p, field) for g, p in found.items() if p}
    kw, degenerate, (lo, hi) = _shared_scale(
        [v[2] for v in loaded.values()], cmap)
    fig, axes = plt.subplots(1, len(grids), figsize=(4.0 * len(grids), 3.2),
                             constrained_layout=True)
    im = None
    for ax, g in zip(np.atleast_1d(axes), grids):
        if g not in loaded:
            ax.text(0.5, 0.5, f"{g}\n(not registered)", ha="center",
                    va="center", fontsize=9, color="0.4")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        lat, lon, a, t = loaded[g]
        im = _panel(ax, lat, lon, a, cmap, kw)
        st = _status(found[g])
        colour = {"PASS": "#228833", "FAIL": "#EE6677"}.get(st, "0.3")
        # The panel's OWN range, always. On a shared scale a genuinely
        # weaker arm renders as blank white, and a reader cannot tell that
        # from a broken run -- which matters here because the arms are not
        # all the same experiment (some grids carry realistic continents,
        # some an idealised domain), so one arm routinely sets a scale the
        # others never approach.
        fin = a[np.isfinite(a)]
        rng = (f"  [{fin.min():.3g}, {fin.max():.3g}]" if fin.size
               else "  (no finite data)")
        ax.set_title(f"{g}  [{st}]  day {t:g}\nown range{rng}",
                     fontsize=PUB["title"], color=colour)
        ax.set_xlabel("lon [deg]", fontsize=PUB["label"])
        ax.tick_params(labelsize=PUB["tick"])
    np.atleast_1d(axes)[0].set_ylabel("lat [deg]", fontsize=PUB["label"])
    if im is not None:
        fig.colorbar(im, ax=fig.axes, shrink=0.8, label=label, location="right")
    sub = (f"{case} — {label}, final saved time "
           f"(shared colour scale [{lo:.4g}, {hi:.4g}])")
    if degenerate:
        sub += (f"\nFIELD IS UNIFORM across every arm: full spread "
                f"{hi - lo:.2e} — the scale is widened so machine-precision "
                f"round-off is not drawn as structure")
    fig.suptitle(sub, fontsize=PUB["suptitle"])
    fig.text(0.005, 0.005, _provenance(root), fontsize=PUB["tick"] - 3,
             color="0.45", ha="left", va="bottom")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"maps_{case}.png"
    fig.savefig(out, dpi=PUB["dpi"], bbox_inches="tight")
    # Vector alongside the raster: a reviewer zooms, a typesetter rescales.
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def build_contact_sheet(root: Path, out: Path,
                        per_panel: bool = False) -> Path:
    """One page: rows = case, columns = grid.

    ``per_panel`` scales EVERY panel to its own range instead of sharing one
    scale down the row. Both sheets are emitted, because neither is honest
    alone: colour encodes magnitude, so a shared scale is a visual claim that
    the arms are comparable -- and here they are not, since some grids carry
    realistic continents and some an idealised domain. On the shared sheet a
    20x weaker arm renders as blank white; a printed range asks the reader to
    mentally undo the encoding, which is caption-level honesty doing
    figure-level work (GLM-5.2). So: the shared sheet for amplitude, the
    per-panel sheet for structure, each saying which it is.

    The sheet keeps the four global columns; a case that runs on its own grid
    family gets its per-case figure and is listed as such rather than drawn
    as four blanks.
    """
    cases = [c for c in CASES
             if _grids_for(c) is GRIDS and any(_find(root, c, g)
                                               for g in GRIDS)]
    fig, axes = plt.subplots(len(cases), len(GRIDS),
                             figsize=(2.6 * len(GRIDS), 1.9 * len(cases)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, case in enumerate(cases):
        field, cmap, _label = CASES[case]
        found = {g: _find(root, case, g) for g in GRIDS}
        loaded = {g: _load_field(p, field) for g, p in found.items() if p}
        kw, degenerate, (lo, hi) = _shared_scale(
            [v[2] for v in loaded.values()], cmap)
        for j, g in enumerate(GRIDS):
            if per_panel and g in loaded:
                kw, degenerate, (lo, hi) = _shared_scale([loaded[g][2]], cmap)
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if g not in loaded:
                # A bare dash reads as a broken run. Say which it is, and
                # hatch the panel so it cannot be mistaken for data.
                ax.set_facecolor("0.94")
                ax.patch.set_edgecolor("0.75")
                ax.patch.set_hatch("///")
                ax.text(0.5, 0.5, f"{g}\nnot run\nfor this case",
                        ha="center", va="center", fontsize=6, color="0.45",
                        linespacing=1.4)
            else:
                lat, lon, a, _t = loaded[g]
                _panel(ax, lat, lon, a, cmap, kw)
                st = _status(found[g])
                ax.text(0.02, 0.05, st, transform=ax.transAxes,
                        fontsize=PUB["tick"] - 2, fontweight="bold",
                        color={"PASS": "#228833",
                               "FAIL": "#EE6677"}.get(st, "0.3"),
                        bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="none", pad=1.2))
                if degenerate and j == 0:
                    ax.text(0.02, 0.86,
                            f"at rest: spread {hi - lo:.1e}",
                            transform=ax.transAxes,
                            fontsize=PUB["tick"] - 2, color="0.15",
                            bbox=dict(facecolor="white", alpha=0.85,
                                      edgecolor="none", pad=1.2))
            if i == 0:
                ax.set_title(g, fontsize=PUB["tick"])
            if j == 0:
                # ALWAYS show the colour range next to the row name. A
                # shared scale still stretches a physically negligible
                # spread across the whole colormap -- the rest-state rows
                # span 0.002 degC across arms, which without this label
                # reads as structure instead of "flat to 4 decimals".
                # Names are WRAPPED, never truncated: the old [:22] slice
                # produced "rest_stratified_with_l", which is not a caption.
                _nm = case.replace("rest_state_", "rest ").replace("_", " ")
                _wrapped = "\n".join(textwrap.wrap(_nm, 16))
                ax.set_ylabel(
                    f"{_wrapped}\n[{lo:.4g}, {hi:.4g}]",
                    fontsize=PUB["tick"] - 1, rotation=0, ha="right",
                    va="center")
    fig.suptitle(
        "legoESM ocean benchmark suite — every case on every grid\n"
        "sea-surface height for the wave and adjustment cases, sea-surface "
        "temperature for the tracer cases.\n"
        + ("EACH PANEL ON ITS OWN COLOUR SCALE — read this sheet for spatial "
           "STRUCTURE.\nMagnitudes are NOT comparable between panels here; "
           "use the shared-scale sheet for that."
           if per_panel else
           "One shared colour scale per ROW, its range printed beside the row "
           "name — read this sheet for AMPLITUDE.\nThe arms are NOT all the "
           "same experiment (some grids carry realistic continents, some an "
           "idealised domain), so a pale panel is a weaker signal, not a "
           "failed run; the per-panel sheet shows its structure."),
        fontsize=PUB["tick"])
    fig.text(0.005, 0.002, _provenance(root), fontsize=PUB["tick"] - 3,
             color="0.45", ha="left", va="bottom")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=PUB["dpi"], bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path,
                    default=_REPO / "results" / "ocean_grid_benchmark")
    ap.add_argument("--out-dir", type=Path,
                    default=_REPO / "results" / "ocean_grid_benchmark" / "maps")
    a = ap.parse_args()
    if not a.runs_root.is_dir():
        raise SystemExit(f"no suite output under {a.runs_root}; run the "
                         f"benchmark suite first (see module docstring).")
    made, missing = [], []
    for c in CASES:
        out = build_case_figure(a.runs_root, c, a.out_dir)
        (made if out else missing).append(c)
    sheet = build_contact_sheet(a.runs_root, a.out_dir / "maps_all_cases.png")
    build_contact_sheet(a.runs_root,
                        a.out_dir / "maps_all_cases_per_panel_scale.png",
                        per_panel=True)
    print(f"COMPLETED: {len(made)} case figures + {sheet}")
    # A case that produces nothing is NAMED. Silently returning 9 of 10
    # figures reads as "the tenth case did not run" when in fact it ran and
    # writes its snapshots in a different form (2026-08-13: the f-plane
    # channel IGW writes field_snapshots.png per arm and no regridded
    # snapshots_latlon.npz, so it can never appear on this map).
    for c in missing:
        where = a.runs_root / c
        print(f"  NO MAP for {c}: no regridded snapshot under {where} "
              f"(the case may write its own figures there instead -- look "
              f"for field_snapshots.png)")


if __name__ == "__main__":
    main()
