"""LES-suite scorecard assembly — Q2 ranking + Q3 coefficient spread.

The one generated scorecard (LES_SUITE.md §5/§7): it consumes the per-(closure,
regime) tuned-result records written by ``tune_scm_to_les.py`` and assembles the
science answers:

* **Q2** — per-regime ranking of the closures by their best tuned LES loss (lower =
  better), so "does closure order buy skill once everything is tuned?" is answered
  per regime.
* **Q3** — for each closure, the inter-regime SPREAD of every tuned coefficient
  (max − min across regimes) alongside its value per regime — the raw material for
  "how far do optimal coefficients travel between regimes?". The LES error bar
  ``σ_LES`` gate (report a spread only when it exceeds σ_LES) is applied by the CLI
  when the D7 SGS-spread runs exist; this module produces the spreads themselves.

Pure assembly over already-computed tuned records — no LES, no SCM, no tuning. Each
record is the dict ``tune_scm_to_les`` writes (``case``, ``scheme``, ``best_loss``,
``default_loss``, ``best_overrides``) plus a ``regime`` (looked up from the registry
by the CLI).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as _dc_field  # aliased: ``field`` is a loop var below


class ScorecardError(ValueError):
    """Raised when tuned records are malformed for the scorecard."""


@dataclass(frozen=True)
class ClosureRanking:
    """Closures ranked by best tuned loss (ascending) for one (regime, subcase).

    ``subcase`` labels the slice this ranking covers: a per-surface-flux point
    (e.g. ``"Q0=0.06"``) for the primary per-flux tables, or ``"(flux mean)"`` for
    the regime-level flux-averaged summary. ``q0`` is the surface kinematic heat
    flux [K m/s] when the slice is a single flux point, else ``None``. ``counts``
    maps scheme -> number of flux slices averaged (populated only for the flux-mean
    summary, so the rendered denominator is honest when a closure is missing at some
    flux — a controlled-comparison guard, not a silent partial mean).
    """

    regime: str
    ranked: tuple  # tuple of (scheme, best_loss, default_loss), best first
    subcase: str = "(flux mean)"
    q0: float | None = None
    counts: dict = _dc_field(default_factory=dict)


@dataclass(frozen=True)
class CoefficientSpread:
    """One closure's inter-regime spread of one tuned coefficient."""

    scheme: str
    field: str
    per_regime: dict           # regime -> tuned value
    spread: float              # max - min across regimes
    n_regimes: int


@dataclass(frozen=True)
class Scorecard:
    rankings: tuple            # ClosureRanking per regime, flux-mean summary (Q2)
    spreads: tuple             # CoefficientSpread per (scheme, field) (Q3)
    per_flux: tuple = ()       # ClosureRanking per (regime, flux) — primary Q2 view


_REQUIRED = ("case", "scheme", "regime", "best_loss")


def _validate(records: list[dict]) -> None:
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise ScorecardError(f"record {i} is not a dict")
        missing = [k for k in _REQUIRED if k not in r]
        if missing:
            raise ScorecardError(f"record {i} ({r.get('case')}) missing {missing}")
        # ``best_loss`` MUST be finite: a NaN/inf would defeat the "lowest loss wins"
        # dedup comparison and could sort ahead of real losses. The tuner already
        # writes only finite best_loss (+inf-on-diverge is caught there and never
        # selected as best), so a non-finite value here means a corrupt record.
        bl = r["best_loss"]
        if bl is None or not math.isfinite(float(bl)):
            raise ScorecardError(
                f"record {i} ({r.get('case')}) has non-finite best_loss {bl!r}")


# Canonical flux precision: coarser than float32 artifact noise (~1e-8, e.g.
# float32(0.06)=0.059999998…) yet far finer than any physically distinct campaign
# flux (≥1e-3 K m/s). Rounding to this many decimals is what lets an f32-stamped q0
# and an exact/filename-parsed q0 of the SAME flux merge to one slice while distinct
# fluxes (0.02, 0.04, …) never do. The canonical value is used for the key, the
# stored q0, sorting, AND the display label, so none of those can disagree.
_Q0_DECIMALS = 6


def _flux_group(r: dict) -> tuple[tuple, str, float | None]:
    """Return ``(group_key, display_label, q0)`` for one record's flux slice.

    ``group_key`` is a TYPED tuple (``("q0", <canonical>)`` or ``("label", <str>)``)
    so a numeric flux and a string label can never collide. The canonical (rounded)
    flux drives key/label/sort together — no insertion-order-dependent labels.
    """
    q0 = r.get("q0")
    if q0 is not None:
        q0c = round(float(q0), _Q0_DECIMALS)
        return ("q0", q0c), f"Q0={q0c:g}", q0c
    label = str(r.get("artifact") or r.get("subcase") or r.get("case") or "?")
    return ("label", label), label, None


def _dedup_by_scheme(
    rows: list[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    """Collapse repeated closures within one slice to the LOWEST-loss record.

    Guards against the same (flux, closure) being tuned twice (e.g. a stale
    pre-rename tuned JSON alongside its replacement) silently double-counting.
    """
    best_of: dict[str, tuple[float, float]] = {}
    for scheme, best, default in rows:
        if scheme not in best_of or best < best_of[scheme][0]:
            best_of[scheme] = (best, default)
    return [(s, b, d) for s, (b, d) in best_of.items()]


def rank_closures_per_flux(records: list[dict]) -> list[ClosureRanking]:
    """Q2 (primary): within each (regime, surface-flux) slice, rank closures by best
    tuned loss (ascending). One ranking per flux point — the honest per-flux view
    that a single regime-mean hides. Ordered by regime then ascending flux."""
    groups: dict[tuple[str, tuple], dict] = {}
    for r in records:
        # a null/absent default_loss (the default config diverged) falls back to
        # best_loss so the ranking column stays finite.
        default = r.get("default_loss")
        default = float(default) if default is not None else float(r["best_loss"])
        gkey, label, q0 = _flux_group(r)
        g = groups.setdefault(
            (r["regime"], gkey), {"label": label, "q0": q0, "rows": []})
        g["rows"].append((r["scheme"], float(r["best_loss"]), default))
    rankings: list[ClosureRanking] = []
    for (regime, _gkey), g in groups.items():
        rows = _dedup_by_scheme(g["rows"])
        rows.sort(key=lambda t: t[1])  # ascending best_loss = best first
        rankings.append(ClosureRanking(
            regime=regime, ranked=tuple(rows), subcase=g["label"], q0=g["q0"]))
    # regime, then ascending flux (unlabelled slices, q0=None, sort last by label)
    rankings.sort(key=lambda rk: (
        rk.regime, rk.q0 is None, rk.q0 if rk.q0 is not None else 0.0, rk.subcase))
    return rankings


def rank_closures_per_regime(records: list[dict]) -> list[ClosureRanking]:
    """Q2 (summary): per regime, rank closures by their flux-MEAN best tuned loss.

    The mean is taken over the distinct flux slices from
    :func:`rank_closures_per_flux` (which already collapses duplicate (flux, closure)
    records), so a repeated flux point is counted once — not double-weighted. Each
    closure's mean is over the slices where it appears; ``counts[scheme]`` records
    that denominator so an incomplete closure×flux matrix is reported, not hidden.
    """
    per_flux = rank_closures_per_flux(records)
    by_regime: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for rk in per_flux:
        for scheme, best, default in rk.ranked:
            by_regime.setdefault(rk.regime, {}).setdefault(scheme, []).append(
                (best, default))
    rankings: list[ClosureRanking] = []
    for regime in sorted(by_regime):
        rows = []
        counts: dict[str, int] = {}
        for scheme, vals in by_regime[regime].items():
            best = sum(v[0] for v in vals) / len(vals)
            default = sum(v[1] for v in vals) / len(vals)
            rows.append((scheme, best, default))
            counts[scheme] = len(vals)
        rows.sort(key=lambda t: t[1])  # ascending best_loss = best first
        rankings.append(ClosureRanking(
            regime=regime, ranked=tuple(rows), subcase="(flux mean)", q0=None,
            counts=counts))
    return rankings


def coefficient_spreads(records: list[dict]) -> list[CoefficientSpread]:
    """Q3: per closure+coefficient, the inter-regime spread of the tuned value.

    Uses one tuned value per (scheme, field, regime) — the value from the record
    with the LOWEST best_loss in that regime (the regime's chosen optimum). A field
    tuned in only one regime has spread 0 (reported, with n_regimes=1).
    """
    # pick each regime's best record per scheme
    best_rec: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["scheme"], r["regime"])
        if key not in best_rec or float(r["best_loss"]) < float(best_rec[key]["best_loss"]):
            best_rec[key] = r
    # collect tuned coefficient values per (scheme, field) across regimes
    values: dict[tuple[str, str], dict[str, float]] = {}
    for (scheme, regime), r in best_rec.items():
        for field, val in (r.get("best_overrides") or {}).items():
            values.setdefault((scheme, field), {})[regime] = float(val)
    spreads: list[CoefficientSpread] = []
    for (scheme, field), per_regime in sorted(values.items()):
        vals = list(per_regime.values())
        spreads.append(CoefficientSpread(
            scheme=scheme, field=field, per_regime=dict(per_regime),
            spread=(max(vals) - min(vals)) if vals else 0.0,
            n_regimes=len(per_regime),
        ))
    return spreads


def assemble_scorecard(records: list[dict]) -> Scorecard:
    """Assemble the Q2 ranking + Q3 coefficient-spread scorecard from tuned records."""
    _validate(records)
    return Scorecard(
        rankings=tuple(rank_closures_per_regime(records)),
        spreads=tuple(coefficient_spreads(records)),
        per_flux=tuple(rank_closures_per_flux(records)),
    )


def _ranking_table(ranked: tuple, best_col: str, default_col: str) -> list[str]:
    rows = [f"| rank | closure | {best_col} | {default_col} |", "|---|---|---|---|"]
    for i, (scheme, best, default) in enumerate(ranked, 1):
        rows.append(f"| {i} | {scheme} | {best:.4f} | {default:.4f} |")
    return rows


def _mean_table(ranked: tuple, counts: dict, n_total: int) -> list[str]:
    """Flux-mean table with an explicit per-closure denominator (``n fluxes``): a
    closure averaged over fewer than ``n_total`` slices is flagged ⚠ so an
    incomplete matrix is never read as an apples-to-apples mean."""
    rows = ["| rank | closure | mean best loss | mean default loss | n fluxes |",
            "|---|---|---|---|---|"]
    for i, (scheme, best, default) in enumerate(ranked, 1):
        n = counts.get(scheme, n_total)
        mark = "" if n == n_total else " ⚠"
        rows.append(f"| {i} | {scheme} | {best:.4f} | {default:.4f} | {n}{mark} |")
    return rows


def render_markdown(card: Scorecard) -> str:
    """Human-readable scorecard (the generated artifact)."""
    lines = ["# LES-suite scorecard", "", "## Q2 — per-regime closure ranking",
             "(lower tuned LES loss = better)", ""]
    # group the per-flux rankings by regime, preserving their (regime, q0) order
    per_flux_by_regime: dict[str, list[ClosureRanking]] = {}
    for rk in card.per_flux:
        per_flux_by_regime.setdefault(rk.regime, []).append(rk)
    for rk in card.rankings:
        lines.append(f"### {rk.regime}")
        flux_rankings = per_flux_by_regime.get(rk.regime, [])
        if flux_rankings:
            lines += ["#### per surface-flux ranking", ""]
            for fr in flux_rankings:
                label = (f"Q0 = {fr.q0:g} K m/s" if fr.q0 is not None
                         else fr.subcase)
                lines.append(f"##### {label}")
                lines += _ranking_table(fr.ranked, "best loss", "default loss")
                lines.append("")
            n = len(flux_rankings)
            lines.append(f"#### flux-mean ranking (mean over {n} flux point"
                         f"{'s' if n != 1 else ''})")
            lines += _mean_table(rk.ranked, rk.counts, n)
        else:
            lines += _ranking_table(
                rk.ranked, "mean best loss", "mean default loss")
        lines.append("")
    lines += ["## Q3 — inter-regime tuned-coefficient spread", ""]
    if card.spreads:
        lines.append("| closure | coeff | n regimes | spread (max−min) | per-regime |")
        lines.append("|---|---|---|---|---|")
        for s in card.spreads:
            pr = ", ".join(f"{k}={v:.3g}" for k, v in sorted(s.per_regime.items()))
            lines.append(
                f"| {s.scheme} | {s.field} | {s.n_regimes} | {s.spread:.4g} | {pr} |")
    else:
        lines.append("(no tuned coefficients yet)")
    lines.append("")
    return "\n".join(lines)
