#!/usr/bin/env python
"""Plot the legoESM WeatherBench-2 forecast scorecard vs published SOTA models.

Reads one or more ``scorecard.json`` files (as written by
``scripts/validate/run_weatherbench_eval.py::main`` — one per legoESM model
family, e.g. ``physics``/``neural_gcm``/``sfno``) and, optionally, a CSV of
PUBLISHED WB2 headline RMSE for reference models (parsed by
``evaluations.baselines.load_sota_headline``). Produces a multi-panel figure of
metric (RMSE by default, ACC where present) vs forecast lead time — one subplot
per headline field. Lower RMSE = better; higher ACC = better.

Each scorecard JSON has the structure written by the eval driver::

    {
      "meta":        {...},
      "model":       {field_key: {str(lead_hours): {"rmse", "acc", "bias"}}},
      "persistence": {field_key: {...}},   # must-beat floor
      "climatology": {field_key: {...}},   # must-beat floor
    }

with ``field_key`` drawn from ``evaluations.wb_forecast.HEADLINE_FIELD_KEYS``.
The SOTA CSV keys forecasts by WB2 ``(variable, level, lead_hours)``; this module
reconciles that naming with the scorecard ``field_key`` via
:data:`FIELD_KEY_TO_SOTA` (documented, explicit; an unmapped field is a LOUD
skip-with-warning for the SOTA overlay only — the legoESM curves still plot).

The arg-parse + plotting-builder layer is import-light and JAX-free (login-node
testable, mirroring ``run_weatherbench_eval.py``): matplotlib is imported inside
``main`` / the builder, and there are no legoESM-physics or JAX imports at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Repo root on sys.path so ``import evaluations`` resolves (this file lives at
# scripts/plot/; evaluations/ is at the repo root). Mirrors run_weatherbench_eval.py.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_OUT = "results/wb_eval/wb_scorecard.png"
_VALID_METRICS = ("rmse", "acc", "bias")

# --- scorecard field_key  <->  SOTA (variable, level) reconciliation ---
# The scorecard uses short field_keys (evaluations.wb_forecast.HEADLINE_FIELD_KEYS);
# the SOTA CSV (evaluations.baselines.load_sota_headline) keys forecasts by the WB2
# canonical (variable, level) pair with an INTEGER level in hPa (surface fields use
# level 0 by convention). Only fields with a published WB2 headline analogue are
# mapped; the rest (surface-wind proxies, etc.) simply have no SOTA overlay.
#   field_key -> (wb2_variable, level_hPa)
FIELD_KEY_TO_SOTA: dict[str, tuple[str, int]] = {
    "z500": ("geopotential", 500),
    "t850": ("temperature", 850),
    "q700": ("specific_humidity", 700),
    "u850": ("u_component_of_wind", 850),
    "v850": ("v_component_of_wind", 850),
    "u700": ("u_component_of_wind", 700),
    "v700": ("v_component_of_wind", 700),
    "u500": ("u_component_of_wind", 500),
    "v500": ("v_component_of_wind", 500),
    "u250": ("u_component_of_wind", 250),
    "v250": ("v_component_of_wind", 250),
    "mslp": ("mean_sea_level_pressure", 0),
    "t2m": ("2m_temperature", 0),
    "u10": ("10m_u_component_of_wind", 0),
    "v10": ("10m_v_component_of_wind", 0),
    "wind_speed_10m": ("10m_wind_speed", 0),
}

# --- SOTA unit reconciliation (RMSE is unit-carrying; a mismatch is a silent
# --- factor error on the most-quoted headline field) ---------------------------
# WB2 publishes ``geopotential`` in m^2/s^2. Our ``z500`` is geopotential HEIGHT
# in metres (``wb_forecast.diagnose_headline_fields`` calls
# ``geopotential_height_at``, which already divides by g), so the SOTA series
# must be divided by g before it shares an axis with ours. Every other mapped
# field already agrees (K, m/s, kg/kg, Pa) and carries no factor.
# Only relevant for RMSE/bias: ACC is dimensionless, so the factor is skipped.
_SOTA_UNIT_DIVISOR_FIELDS = ("z500",)


def sota_unit_divisor(field_key: str, metric: str = "rmse") -> float:
    """Divisor putting a SOTA series into OUR units for ``field_key``.

    ``1.0`` when the units already agree. ACC is dimensionless -> always 1.0.

    ``legoesm.constants`` is imported function-scope (it pulls jax) so this
    module's top level stays import-light, per the module docstring.
    """
    if str(metric).lower() == "acc" or field_key not in _SOTA_UNIT_DIVISOR_FIELDS:
        return 1.0
    from legoesm import constants
    return float(constants.g)

# Human-readable per-field axis labels + units (RMSE units in comments). Z500 is
# geopotential HEIGHT in metres here (geopotential_height_at returns m), NOT
# m^2/s^2 — the diagnostic already divides by g. Kept explicit so the y-axis is
# labelled with the right unit per field.
_FIELD_LABELS: dict[str, tuple[str, str]] = {
    "z500": ("Z500 geopotential height", "m"),
    "t850": ("T850 temperature", "K"),
    "q700": ("Q700 specific humidity", "kg/kg"),
    "u850": ("U850 zonal wind", "m/s"),
    "v850": ("V850 meridional wind", "m/s"),
    "u700": ("U700 zonal wind", "m/s"),
    "v700": ("V700 meridional wind", "m/s"),
    "u500": ("U500 zonal wind", "m/s"),
    "v500": ("V500 meridional wind", "m/s"),
    "u250": ("U250 zonal wind", "m/s"),
    "v250": ("V250 meridional wind", "m/s"),
    "mslp": ("Mean sea-level pressure", "Pa"),
    "t2m": ("2 m temperature", "K"),
    "u10": ("10 m zonal wind", "m/s"),
    "v10": ("10 m meridional wind", "m/s"),
    "wind_speed_10m": ("10 m wind speed", "m/s"),
}

# Distinct colors/markers for the legoESM families (solid lines). Cycled if a run
# has more families than entries.
_FAMILY_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2")
_FAMILY_MARKERS = ("o", "s", "^", "D", "v", "P")
# Greyed reference colors for the SOTA models (dashed).
_SOTA_COLORS = ("#555555", "#777777", "#999999", "#333333", "#aaaaaa", "#666666")


def sota_key_for_field(field_key: str):
    """Return ``(wb2_variable, level_hPa)`` for a scorecard ``field_key``.

    Returns ``None`` (a loud skip is the caller's job) if the field has no mapped
    SOTA analogue — never guesses a mapping.
    """
    return FIELD_KEY_TO_SOTA.get(field_key)


def _series_from_section(section: dict, field_key: str, metric: str):
    """Extract ``(sorted lead_hours, values)`` for one field+metric from a
    scorecard section (``model``/``persistence``/``climatology``).

    Missing field, or a lead lacking the metric, is simply absent from the
    returned series (no fabricated point). Returns ``([], [])`` if nothing present.
    """
    per_lead = section.get(field_key)
    if not per_lead:
        return [], []
    leads, vals = [], []
    for lead_str, metrics in per_lead.items():
        if not isinstance(metrics, dict) or metric not in metrics:
            continue
        v = metrics[metric]
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:                      # NaN -> absent, not a fake point
            continue
        leads.append(int(lead_str))
        vals.append(fv)
    order = sorted(range(len(leads)), key=lambda i: leads[i])
    return [leads[i] for i in order], [vals[i] for i in order]


def _sota_series_for_field(sota: dict, model: str, field_key: str,
                           metric: str = "rmse"):
    """``(sorted lead_hours, values)`` for one SOTA model + field, IN OUR UNITS.

    ``sota`` is ``{model: {(variable, level, lead_hours): rmse}}`` from
    ``load_sota_headline``. Unmapped field -> empty (caller warns once).
    The series is divided by :func:`sota_unit_divisor` so it shares an axis
    with ours (Z500: WB2 publishes m^2/s^2, we diagnose geopotential height
    in m — plotting them raw is a silent factor-g error).
    """
    mapped = sota_key_for_field(field_key)
    if mapped is None:
        return [], []
    var, level = mapped
    div = sota_unit_divisor(field_key, metric)
    # Accept BOTH CSV schemas. The committed
    # config/wb/sota/wb2_headline_rmse.csv keys rows by the WB2 long name
    # ("geopotential"), but scripts/data/fetch_wb2_sota.py writes the legoESM
    # headline key instead (HeadlineVar.csv_variable, i.e. "z500"). Matching
    # only the long name meant a REGENERATED CSV silently produced an empty
    # SOTA overlay — no warning, because the unmapped-field warning only fires
    # for keys absent from FIELD_KEY_TO_SOTA (codex review 2026-07-28).
    # Units are identical either way: both carry WB2's published RMSE, so the
    # /g divisor applies unchanged.
    accepted = {var, field_key}
    pts = []
    for (v, lev, lead), rmse_val in sota[model].items():
        if v in accepted and int(lev) == level:
            pts.append((int(lead), float(rmse_val) / div))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _resolve_fields(scorecards: dict, requested):
    """Ordered list of field_keys to plot.

    If ``requested`` is given, keep only those present in >=1 scorecard's ``model``
    section (a requested-but-absent field warns and is dropped). Otherwise use
    every field appearing in any ``model`` section, in HEADLINE_FIELD_KEYS order
    where possible (unknown extras appended, sorted).
    """
    try:
        from evaluations.wb_forecast import HEADLINE_FIELD_KEYS
        canonical = list(HEADLINE_FIELD_KEYS)
    except Exception:                      # keep the plotter usable without the pkg
        canonical = list(_FIELD_LABELS.keys())

    present: set = set()
    for sc in scorecards.values():
        present.update((sc.get("model") or {}).keys())

    if requested:
        out = []
        for f in requested:
            if f in present:
                out.append(f)
            else:
                warnings.warn(
                    f"plot_wb_scorecard: requested field {f!r} not in any "
                    "scorecard 'model' section; skipping.")
        return out

    ordered = [f for f in canonical if f in present]
    ordered += sorted(present - set(ordered))
    return ordered


def build_scorecard_figure(scorecards: dict, sota: dict | None = None, *,
                           metric: str = "rmse", fields=None, ncols: int = 3):
    """Build (do not save) the WB2 scorecard comparison figure.

    Parameters
    ----------
    scorecards : dict[str, dict]
        ``{family_name: scorecard_dict}`` — each value has the JSON structure
        written by ``run_weatherbench_eval.main`` (``model``/``persistence``/
        ``climatology`` sections, each ``{field_key: {str(lead): {metric: v}}}``).
    sota : dict, optional
        ``{model: {(variable, level, lead_hours): rmse}}`` from
        ``evaluations.baselines.load_sota_headline``. SOTA overlays are RMSE-only
        (the published CSV carries no ACC), so they are drawn only for
        ``metric == "rmse"``.
    metric : str
        One of ``rmse``/``acc``/``bias``. A scorecard lacking the metric for a
        field contributes no line there (graceful skip).
    fields : sequence of str, optional
        Subset of headline field_keys to plot; default = all present.
    ncols : int
        Subplot columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")                  # headless / login-node safe
    import matplotlib.pyplot as plt

    if metric not in _VALID_METRICS:
        raise ValueError(
            f"metric must be one of {_VALID_METRICS}, got {metric!r}")
    if not scorecards:
        raise ValueError("build_scorecard_figure: no scorecards supplied")

    field_keys = _resolve_fields(scorecards, fields)
    if not field_keys:
        raise ValueError(
            "build_scorecard_figure: no headline fields to plot (none present "
            "in the scorecards' 'model' sections, or all requested fields absent)")

    n = len(field_keys)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    better = "lower is better" if metric in ("rmse",) else (
        "higher is better" if metric == "acc" else "closer to 0 is better")

    # Warn once per unmapped field for the SOTA overlay (RMSE only).
    warned_unmapped: set = set()
    warned_empty: set = set()

    for idx, field_key in enumerate(field_keys):
        ax = flat_axes[idx]

        # --- legoESM families (solid, distinct colors + markers) ---
        for fi, (fam_name, sc) in enumerate(scorecards.items()):
            leads, vals = _series_from_section(
                sc.get("model") or {}, field_key, metric)
            if not leads:
                continue                    # family lacks this field/metric -> absent
            ax.plot(leads, vals,
                    color=_FAMILY_COLORS[fi % len(_FAMILY_COLORS)],
                    marker=_FAMILY_MARKERS[fi % len(_FAMILY_MARKERS)],
                    linestyle="-", linewidth=2.0, markersize=5,
                    label=f"legoESM:{fam_name}")

        # --- persistence + climatology floors (dotted), from the FIRST family
        # that has them (they are eval-window baselines, family-independent). ---
        for floor_name, style, color in (("persistence", ":", "#b0b0b0"),
                                          ("climatology", "-.", "#c8a0a0")):
            for sc in scorecards.values():
                leads, vals = _series_from_section(
                    sc.get(floor_name) or {}, field_key, metric)
                if leads:
                    ax.plot(leads, vals, linestyle=style, color=color,
                            linewidth=1.4, alpha=0.9, label=floor_name)
                    break

        # --- SOTA reference models (dashed, greyed) — RMSE only ---
        if sota and metric == "rmse":
            mapped = sota_key_for_field(field_key)
            if mapped is None:
                if field_key not in warned_unmapped:
                    warnings.warn(
                        f"plot_wb_scorecard: field {field_key!r} has no SOTA "
                        "(variable, level) mapping in FIELD_KEY_TO_SOTA; "
                        "plotting legoESM curves only for this panel.")
                    warned_unmapped.add(field_key)
            else:
                for si, model in enumerate(sorted(sota)):
                    leads, vals = _sota_series_for_field(
                        sota, model, field_key, metric)
                    if not leads:
                        # Mapped but zero matching rows: a schema mismatch or a
                        # model that simply does not publish this field. Warn
                        # once per field so an empty overlay is never mistaken
                        # for "SOTA happens to be off-scale here".
                        if field_key not in warned_empty:
                            warnings.warn(
                                f"plot_wb_scorecard: no SOTA rows for "
                                f"{field_key!r} (looked for variable "
                                f"{mapped[0]!r} or {field_key!r} at level "
                                f"{mapped[1]}); overlay omitted for this panel.")
                            warned_empty.add(field_key)
                        continue
                    ax.plot(leads, vals, linestyle="--",
                            color=_SOTA_COLORS[si % len(_SOTA_COLORS)],
                            linewidth=1.5, marker="", alpha=0.85,
                            label=f"SOTA:{model}")

        label, unit = _FIELD_LABELS.get(field_key, (field_key, ""))
        ax.set_xlabel("Forecast lead time [h]")
        ylab = f"{metric.upper()}" + (f" [{unit}]" if unit else "")
        ax.set_ylabel(ylab)
        ax.set_title(f"{label}  ({metric.upper()}, {better})")
        ax.grid(True, alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="best")

    # blank any unused panels
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(
        f"legoESM WeatherBench-2 scorecard vs SOTA — {metric.upper()} vs lead time",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def _load_scorecards(pairs):
    """``["name=path", ...]`` -> ``{name: scorecard_dict}`` (JSON-loaded)."""
    out: dict = {}
    for spec in pairs:
        if "=" not in spec:
            raise SystemExit(
                f"--scorecard expects NAME=PATH, got {spec!r} (missing '=').")
        name, _, path = spec.partition("=")
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"--scorecard NAME=PATH has empty part: {spec!r}")
        if name in out:
            raise SystemExit(f"--scorecard duplicate family name {name!r}")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"--scorecard {name!r}: file not found: {path}")
        with p.open() as fh:
            out[name] = json.load(fh)
    return out


def parse_args(argv=None):
    """Parse CLI. Import-light + JAX-free (login-node testable)."""
    p = argparse.ArgumentParser(
        description="Plot the legoESM WB2 forecast scorecard vs SOTA models.")
    p.add_argument(
        "--scorecard", action="append", default=[], metavar="NAME=PATH",
        required=True,
        help="legoESM family scorecard as NAME=PATH (repeatable), e.g. "
             "physics=out/physics/scorecard.json neural_gcm=out/ngcm/scorecard.json")
    p.add_argument("--sota-csv", default=None, dest="sota_csv",
                   help="Optional published-SOTA headline RMSE CSV "
                        "(evaluations.baselines.load_sota_headline schema).")
    p.add_argument("--out", default=_DEFAULT_OUT,
                   help=f"Output PNG (default {_DEFAULT_OUT}).")
    p.add_argument("--metric", default="rmse", choices=_VALID_METRICS,
                   help="Metric to plot (default rmse).")
    p.add_argument("--fields", default=None,
                   help="Optional comma-separated field_key subset "
                        "(default = all headline fields present).")
    p.add_argument("--ncols", type=int, default=3, help="Subplot columns.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    scorecards = _load_scorecards(args.scorecard)

    sota = None
    if args.sota_csv:
        from evaluations.baselines import load_sota_headline
        sota = load_sota_headline(args.sota_csv)

    fields = None
    if args.fields:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    fig = build_scorecard_figure(
        scorecards, sota, metric=args.metric, fields=fields, ncols=args.ncols)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"wrote WB2 scorecard plot -> {out}")
    return str(out)


if __name__ == "__main__":
    main()
