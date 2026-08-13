"""Paired before/after summary of the SCM-RCE convection-tuning campaign.

Reads the per-scheme checkpoints ``scheme_<name>.json`` written by
``scripts/run/run_scm_rce_convection_intercomparison.py`` and draws two
figures:

``tuning_before_after``
    One box per condition (default / tuned) of the across-scheme
    distribution, with every scheme's own pair drawn on top and joined.
    Panels are the combined score plus each of its four components.

``profiles_overlay``
    Every scheme's a-posteriori profile against the CRM reference, one
    panel per scored variable, with the legend in tuned-ranking order.

Two things this plotter refuses to hide, because both make a flat pair look
like a tuning failure when it is not:

* a scheme with **zero** tunable parameters (``dca``) cannot move, and
* a scheme that is **inactive** in a single column (``kuo``, which needs a
  large-scale moisture-convergence operator and is deliberately off on a
  state with no ``v`` — ``convection/integration.py``) is not being scored
  as itself at all; its row IS the no-convection baseline.

Both are drawn in grey and labelled rather than silently plotted as
ordinary flat segments.  ``--no-convection-score`` supplies the measured
baseline so the "worse than running no convection at all" line can be
drawn; without it the line is omitted rather than guessed.

KNOWN LIMITATION, stated because a reader will otherwise assume more than
is true (codex round 2, findings 1-2).  Both labels are **inferences from
the checkpoint**, not facts the runner recorded:

* "no tunable parameters" is read from an EMPTY ``records`` list, which is
  also what a search that aborted before recording a trial would leave; and
* "scores the no-convection baseline" is score equality, which is evidence
  of inactivity rather than proof — an active scheme could in principle
  land on the same float, and a genuinely inactive one could drift off it
  if the diagnostics changed.

Closing this properly means persisting ``n_tunable_parameters`` and an
activity diagnostic in the checkpoint itself, which is a change to
``run_scm_rce_convection_intercomparison.py`` and a re-run of the arms.
Until then, treat a grey row as "very probably structural, verify before
citing".

The combined score is ``sqrt((T^2 + qv^2 + cloud^2 + w*precip^2)/(3+w))``
with ``w = 1`` (``training/scm_rce_metrics.score_profiles_precip_jax``),
each component normalized by the reference profile's own mass-weighted
standard deviation.  Because the condensate term is one to two orders of
magnitude larger than the thermodynamic ones, the combined score is very
nearly ``cloud_rmse / 2`` for every scheme -- which is exactly why the
per-component panels are drawn rather than the combined score alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Validated categorical slots 1 and 2 (shared with
# ``plot_scm_les_tuning_summary.py`` so the two campaigns read as one
# figure family): the pair passes the lightness, chroma, CVD-separation and
# contrast checks against the light surface.  The tritan separation sits in
# the floor band, so it is carried with secondary encoding -- the two
# conditions also differ by x position and are directly labelled.
BEFORE = "#2a78d6"
AFTER = "#008300"
WORSE = "#e34948"
INERT = "#9a9994"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a9994"
REF_COLOR = "#0b0b0b"

# Panels: the combined score plus every component that enters it.  Keys are
# (json key, axis title, whether lower is better is trivially true).
COMPONENTS: tuple[tuple[str, str], ...] = (
    ("score", "combined score"),
    ("T_rmse", "temperature"),
    ("qv_rmse", "water vapour"),
    ("cloud_rmse", "condensate"),
    ("precip_rmse", "surface precipitation"),
)

# Bit-identity tolerance for "this scheme scored the no-convection baseline".
# Exact equality is the honest test -- the campaign's kuo checkpoint carries
# 5.736693549527354, the same float the standalone convection=none probe
# printed -- but a hair of slack keeps the label from vanishing on a rerun
# that differs in the last ulp.
BASELINE_RTOL = 1.0e-12


def _ff(x: Any) -> float:
    """Float or NaN.  ``None`` is a legitimate value in these checkpoints
    (a diagnostic that was not computed); it must not raise and must not
    silently become 0.0, which would plot as a real measurement."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _load(indir: Path, schemes: list[str] | None) -> list[dict[str, Any]]:
    paths = sorted(indir.glob("scheme_*.json"))
    if not paths:
        raise SystemExit(f"no scheme_*.json under {indir}")
    recs = []
    for p in paths:
        rec = json.loads(p.read_text())
        if schemes and rec["scheme"] not in schemes:
            continue
        if "prior" not in rec or "tuned" not in rec:
            raise SystemExit(
                f"{p.name}: expected both a 'prior' and a 'tuned' arm, got "
                f"{sorted(rec)}"
            )
        recs.append(rec)
    if not recs:
        raise SystemExit("no scheme matched the requested list")
    # NaN is unordered, so sorting on it directly leaves a scheme with a
    # missing score wherever the input filename order happened to put it,
    # while the profile legend advertises "best first" (codex finding 3).
    # Non-finite scores sort last, deterministically.
    def _key(r):
        s = _ff(r["tuned"].get("score"))
        return (not np.isfinite(s), s if np.isfinite(s) else np.inf,
                r["scheme"])

    recs.sort(key=_key)
    return recs


def _structural_note(rec: dict[str, Any],
                     no_conv_score: float | None = None) -> str | None:
    """Reason this scheme's pair is flat BY CONSTRUCTION, or None.

    Derived from MEASUREMENTS in the checkpoint, never from a table of scheme
    names (codex finding 1).  Two independent reasons:

    * the runner recorded zero tunable parameters, so no search was possible;
    * the scheme scored the measured no-convection baseline, which means it
      contributed nothing -- the evidence for `kuo` being inactive in a single
      column, and evidence that survives `kuo` later being wired up, because
      then it stops matching the baseline and the label disappears by itself.

    Requires the pair to be genuinely flat, with both scores finite: a
    non-finite score is a failure to be shown, not a structural constant, and
    `NaN != NaN` would otherwise silently read as "moved".
    """
    prior = _ff(rec["prior"].get("score"))
    tuned = _ff(rec["tuned"].get("score"))
    if not (np.isfinite(prior) and np.isfinite(tuned)) or prior != tuned:
        return None
    if len(rec.get("records") or ()) == 0:
        return "no tunable parameters"
    if (no_conv_score is not None and np.isfinite(no_conv_score)
            and abs(tuned - no_conv_score) <= BASELINE_RTOL * abs(no_conv_score)):
        return "scores the no-convection baseline"
    return None


def _panel(
    ax,
    names: list[str],
    before: np.ndarray,
    after: np.ndarray,
    notes: list[str | None],
    title: str,
    ylabel: str | None,
    baseline: float | None = None,
) -> tuple[float, float]:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)

    finite = np.isfinite(before) & np.isfinite(after)
    if not finite.any():
        ax.set_title(f"{title}\n(no finite pair)", fontsize=11, color=INK)
        return float("nan"), float("nan")
    # Complete-case filtering silently shrinks the cohort: a scheme missing
    # either arm vanishes from both boxes AND both medians, so the published
    # median can describe fewer schemes than the caption claims. Disclose it
    # on the panel rather than dropping it quietly (codex round 2, finding 3).
    n_drop = int((~finite).sum())
    if n_drop:
        dropped = ", ".join(n for n, ok in zip(names, finite) if not ok)
        ax.text(0.02, 0.02,
                f"{int(finite.sum())}/{finite.size} schemes\nmissing: {dropped}",
                transform=ax.transAxes, fontsize=7.5, color=WORSE,
                va="bottom", ha="left")

    bp = ax.boxplot(
        [before[finite], after[finite]], positions=[0, 1], widths=0.45,
        patch_artist=True,
        medianprops=dict(color=INK, lw=2.0),
        whiskerprops=dict(color=MUTED, lw=1.2),
        capprops=dict(color=MUTED, lw=1.2),
        flierprops=dict(marker="", ls="none"),   # points drawn individually
        zorder=1,
    )
    for patch, colour in zip(bp["boxes"], (BEFORE, AFTER)):
        patch.set_facecolor(colour)
        patch.set_alpha(0.16)
        patch.set_edgecolor(colour)
        patch.set_linewidth(1.6)

    if baseline is not None and np.isfinite(baseline):
        ax.axhline(baseline, color=WORSE, lw=1.0, ls=(0, (5, 3)), alpha=0.75,
                   zorder=0)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.055, 0.055, size=before.size)
    for i, name in enumerate(names):
        if not finite[i]:
            continue
        x0, x1 = 0.0 + jitter[i], 1.0 + jitter[i]
        # Grey means "this component did not move AND the scheme is
        # structurally unable to move".  The structural status is decided on
        # the combined score, so a scheme can be structurally flat overall
        # while a component moved (offsetting changes); that component must
        # be coloured by its OWN direction rather than greyed, or the figure
        # hides a real change behind a "flat by construction" legend entry
        # (codex finding 2).
        this_flat = before[i] == after[i]
        inert = notes[i] is not None and this_flat
        if inert:
            seg, mark = INERT, INERT
        elif this_flat:
            seg, mark = INERT, None      # unchanged is not "degraded"
        else:
            seg = AFTER if after[i] < before[i] else WORSE
            mark = None
        # gid tags the per-scheme artists so a test can select THEM rather
        # than whatever else is on the axis -- the boxplot's whiskers, caps
        # and medians are also two-point lines, and picking lines by point
        # count silently returned those instead.
        ax.plot([x0, x1], [before[i], after[i]], "-",
                color=seg, alpha=0.45 if inert else 0.55,
                lw=1.4, ls=(0, (3, 2)) if inert else "-", zorder=2,
                gid=f"seg:{name}")
        ax.plot(x0, before[i], "o", ms=8, color=mark or BEFORE,
                mec=SURFACE, mew=2.0, zorder=3, gid=f"before:{name}")
        # The tuned endpoint takes the DIRECTION colour, not a fixed "after"
        # green. A green dot on a degraded scheme is the strongest visual
        # encoding on the panel and contradicts the red segment joining it
        # (codex round 2, finding 7).
        ax.plot(x1, after[i], "o", ms=8, color=mark or seg,
                mec=SURFACE, mew=2.0, zorder=3, gid=f"after:{name}")

    # Label the extremes plus every structurally-flat scheme: those are the
    # rows a reader must not mistake for "tuning did nothing".
    idx_lab = {int(np.nanargmin(np.where(finite, after, np.nan))),
               int(np.nanargmax(np.where(finite, after, np.nan)))}
    idx_lab |= {i for i, n in enumerate(notes)
                if n is not None and finite[i] and before[i] == after[i]}
    for i in sorted(idx_lab):
        flat_here = before[i] == after[i]
        label = (names[i] if (notes[i] is None or not flat_here)
                 else f"{names[i]} ({notes[i]})")
        ax.annotate(label, (1.0 + jitter[i] + 0.09, after[i]),
                    fontsize=8.0,
                    color=INERT if (notes[i] is not None and flat_here)
                    else INK_2,
                    va="center", ha="left")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["default", "tuned"], fontsize=10, color=INK)
    ax.set_xlim(-0.45, 2.05)
    ax.set_title(title, fontsize=11, color=INK, pad=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
    ax.grid(axis="y", alpha=0.22, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)

    return (float(np.median(before[finite])), float(np.median(after[finite])))


def figure_before_after(
    recs: list[dict[str, Any]],
    meta: dict[str, Any],
    out: Path,
    no_conv_score: float | None,
) -> dict[str, tuple[float, float]]:
    names = [r["scheme"] for r in recs]
    notes = [_structural_note(r, no_conv_score) for r in recs]

    fig, axes = plt.subplots(1, len(COMPONENTS),
                             figsize=(4.0 * len(COMPONENTS), 5.6))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    medians: dict[str, tuple[float, float]] = {}
    for ax, (key, title) in zip(axes, COMPONENTS):
        ax.set_facecolor(SURFACE)
        before = [_ff(r["prior"].get(key)) for r in recs]
        after = [_ff(r["tuned"].get(key)) for r in recs]
        medians[key] = _panel(
            ax, names, before, after, notes, title,
            "normalized profile RMSE  (lower = better)"
            if key == "score" else None,
            baseline=no_conv_score if key == "score" else None,
        )

    mb, ma = medians["score"]
    if not (np.isfinite(mb) and np.isfinite(ma) and mb != 0.0):
        raise SystemExit(
            f"the combined-score median is not finite (before={mb!r}, "
            f"after={ma!r}); refusing to publish a figure titled "
            f"'median nan -> nan (nan%)' (codex round 2, finding 4)"
        )
    fig.suptitle(
        f"SCM convection schemes before and after tuning against a CRM in "
        f"RCEMIP1 —  median {mb:.4f} → {ma:.4f} "
        f"({100.0 * (ma - mb) / mb:+.1f}%)",
        fontsize=12.5, color=INK, y=0.975,
    )
    sub = (
        f"{len(recs)} schemes, one column, {_ff(meta.get('days')):.0f} d at "
        f"dt {_ff(meta.get('dt')):.0f} s, last {_ff(meta.get('analysis_days')):.0f} d "
        f"averaged (same window both sides); reference "
        f"{Path(str(meta.get('reference_dir', '?'))).name}.  "
        f"Segments join a scheme's own pair; grey dashed = flat by "
        f"construction, not a failed search."
    )
    if no_conv_score is not None and np.isfinite(no_conv_score):
        sub += f"  Red line = no convection at all ({no_conv_score:.3f})."
    fig.text(0.5, 0.925, sub, ha="center", fontsize=9.5, color=INK_2)

    handles = [
        Line2D([0], [0], color=AFTER, lw=1.6, label="improved"),
        Line2D([0], [0], color=WORSE, lw=1.6, label="degraded"),
        Line2D([0], [0], color=INERT, lw=1.6, label="unchanged"),
        Line2D([0], [0], color=INERT, lw=1.6, ls=(0, (3, 2)),
               label="flat by construction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 0.90))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return medians


def figure_profiles(
    recs: list[dict[str, Any]],
    meta: dict[str, Any],
    out: Path,
    ref,
    pressure_hpa: np.ndarray,
) -> None:
    """Overlay every scheme's a-posteriori profile on the CRM reference.

    Legend order is the TUNED ranking, so the reader can map a curve to its
    rank without a second figure.
    """
    panels = (
        ("T_profile", np.asarray(ref.T_ref, float), 1.0, r"$T$ [K]",
         "temperature"),
        ("qv_profile", np.asarray(ref.qv_ref, float), 1.0e3,
         r"$q_v$ [g/kg]", "water vapour"),
        ("qcond_profile", np.asarray(ref.qcond_ref, float), 1.0e3,
         r"$q_{cond}$ [g/kg]", "condensate"),
    )
    fig, axes = plt.subplots(1, len(panels) + 1,
                             figsize=(4.1 * (len(panels) + 1), 6.0),
                             sharey=False)
    fig.patch.set_facecolor(SURFACE)
    cmap = plt.get_cmap("viridis")
    colours = [cmap(v) for v in np.linspace(0.0, 0.92, len(recs))]

    p_top, p_bot = float(np.min(pressure_hpa)), float(np.max(pressure_hpa))
    for ax, (key, ref_prof, scale, xlabel, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for rec, colour in zip(recs, colours):
            prof = rec["tuned"].get(key)
            if not prof:
                continue
            ax.plot(np.asarray(prof, float) * scale, pressure_hpa,
                    color=colour, lw=1.25, label=rec["scheme"], zorder=4)
        ax.plot(ref_prof * scale, pressure_hpa, color=REF_COLOR, lw=2.4,
                label="CRM reference", zorder=5)
        ax.set_ylim(p_bot, p_top)
        ax.set_xlabel(xlabel, fontsize=10, color=INK_2)
        ax.set_title(title, fontsize=11, color=INK, pad=9)
        ax.grid(alpha=0.22, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK_2, labelsize=9)
    axes[0].set_ylabel("pressure [hPa]", fontsize=10, color=INK_2)

    # Water budget panel: P and E against the CRM's precipitation.  A scheme
    # whose P and E disagree is not in steady state, and its profile score is
    # therefore not a statement about an equilibrated column.
    ax = axes[-1]
    ax.set_facecolor(SURFACE)
    y = np.arange(len(recs))[::-1]
    P = np.array([_ff(r["tuned"].get("precip_mm_day")) for r in recs])
    E = np.array([_ff(r["tuned"].get("evap_mm_day")) for r in recs])
    # Matplotlib silently skips a NaN bar, leaving a blank that reads as zero
    # precipitation rather than as missing data (codex round 2, finding 5).
    missing = [r["scheme"] for r, p, e in zip(recs, P, E)
               if not (np.isfinite(p) and np.isfinite(e))]
    if missing:
        raise SystemExit(
            f"missing precip/evap for {', '.join(missing)}; a blank bar is "
            f"indistinguishable from a measured zero"
        )
    ax.barh(y + 0.18, P, height=0.34, color=AFTER, alpha=0.75, label="P")
    ax.barh(y - 0.18, E, height=0.34, color=BEFORE, alpha=0.75, label="E")
    # Every checkpoint carries its own copy of the reference precipitation.
    # Reading it off recs[0] alone would silently make the plotted CRM line
    # depend on which scheme happened to sort first (codex finding 5), so the
    # copies are required to AGREE before one is drawn.
    p_refs = [_ff(r["tuned"].get("precip_ref_mm_day")) for r in recs]
    # EVERY checkpoint must carry the value, not just one: one finite copy
    # beside nine NaNs would pass an agreement test vacuously and draw the
    # lone value as if all schemes had confirmed it (codex round 2, finding 6).
    absent = [r["scheme"] for r, v in zip(recs, p_refs) if not np.isfinite(v)]
    if absent:
        raise SystemExit(
            f"no finite precip_ref_mm_day in checkpoints for "
            f"{', '.join(absent)}; cannot confirm they share one reference"
        )
    if (max(p_refs) - min(p_refs)) > 1e-9 * abs(p_refs[0]):
        raise SystemExit(
            f"checkpoints disagree on precip_ref_mm_day "
            f"({min(p_refs)!r} .. {max(p_refs)!r}); they were scored "
            f"against different references and must not share a figure"
        )
    p_ref = p_refs[0]
    ax.axvline(p_ref, color=REF_COLOR, lw=2.0, label=f"CRM P = {p_ref:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels([r["scheme"] for r in recs], fontsize=9, color=INK_2)
    ax.set_xlabel("water flux [mm/day]", fontsize=10, color=INK_2)
    ax.set_title("water budget (tuned)", fontsize=11, color=INK, pad=9)
    ax.grid(axis="x", alpha=0.22, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
               frameon=False, fontsize=9.0, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "SCM convection schemes vs CRM in RCEMIP1 — a-posteriori profiles",
        fontsize=12.5, color=INK, y=0.975,
    )
    fig.text(
        0.5, 0.930,
        f"legend order is the TUNED ranking (best first); last "
        f"{_ff(meta.get('analysis_days')):.0f} d of a "
        f"{_ff(meta.get('days')):.0f} d run, same window as the reference.",
        ha="center", fontsize=9.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.90))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("indir", type=Path,
                   help="directory holding scheme_<name>.json + run_meta.json")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: indir")
    p.add_argument("--schemes", nargs="*", default=None,
                   help="restrict to these schemes (default: all found)")
    p.add_argument("--no-convection-score", type=float, default=None,
                   help="measured combined score of convection=none, drawn as "
                        "a reference line; omitted if not supplied")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="CRM reference; default: read from run_meta.json")
    p.add_argument("--last-reference-files", type=int, default=5)
    p.add_argument("--skip-profiles", action="store_true",
                   help="draw only the before/after figure (does not need the "
                        "reference snapshots)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    indir: Path = args.indir
    outdir: Path = args.out_dir or indir
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path = indir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    recs = _load(indir, args.schemes)

    out1 = outdir / "tuning_before_after.png"
    medians = figure_before_after(recs, meta, out1, args.no_convection_score)
    print(f"wrote {out1}")
    for k, (b, a) in medians.items():
        if np.isfinite(b) and np.isfinite(a) and b != 0:
            print(f"  {k:14s} median {b:.5f} -> {a:.5f}  "
                  f"({100 * (a - b) / b:+.1f}%)")

    if not args.skip_profiles:
        ref_dir = args.reference_dir or Path(str(meta.get("reference_dir", "")))
        if not ref_dir or not Path(ref_dir).exists():
            raise SystemExit(
                f"reference dir {ref_dir!r} not found; pass --reference-dir "
                f"or --skip-profiles"
            )
        from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
            WING_P_SFC,
        )
        from scripts.run.run_scm_rce_campaign import build_reference_profiles

        ref = build_reference_profiles(Path(ref_dir),
                                       args.last_reference_files)
        pressure_hpa = np.asarray(ref.sigma_full, float) * WING_P_SFC / 100.0
        out2 = outdir / "profiles_overlay.png"
        figure_profiles(recs, meta, out2, ref, pressure_hpa)
        print(f"wrote {out2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
