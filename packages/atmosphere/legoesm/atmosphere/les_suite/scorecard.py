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

from dataclasses import dataclass


class ScorecardError(ValueError):
    """Raised when tuned records are malformed for the scorecard."""


@dataclass(frozen=True)
class ClosureRanking:
    """One regime's closures ranked by best tuned loss (ascending)."""

    regime: str
    ranked: tuple  # tuple of (scheme, best_loss, default_loss), best first


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
    rankings: tuple            # ClosureRanking per regime (Q2)
    spreads: tuple             # CoefficientSpread per (scheme, field) (Q3)


_REQUIRED = ("case", "scheme", "regime", "best_loss")


def _validate(records: list[dict]) -> None:
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise ScorecardError(f"record {i} is not a dict")
        missing = [k for k in _REQUIRED if k not in r]
        if missing:
            raise ScorecardError(f"record {i} ({r.get('case')}) missing {missing}")


def rank_closures_per_regime(records: list[dict]) -> list[ClosureRanking]:
    """Q2: within each regime, rank closures by best tuned loss (ascending).

    When a (regime, scheme) pair appears more than once (e.g. several cases in a
    regime), the closure's score is the MEAN best_loss over those cases — a single
    per-regime number per closure, computed identically for every closure.
    """
    by_regime: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for r in records:
        # a null/absent default_loss (the default config diverged) falls back to
        # best_loss so the ranking column stays finite.
        default = r.get("default_loss")
        default = float(default) if default is not None else float(r["best_loss"])
        by_regime.setdefault(r["regime"], {}).setdefault(r["scheme"], []).append(
            (float(r["best_loss"]), default)
        )
    rankings: list[ClosureRanking] = []
    for regime in sorted(by_regime):
        rows = []
        for scheme, vals in by_regime[regime].items():
            best = sum(v[0] for v in vals) / len(vals)
            default = sum(v[1] for v in vals) / len(vals)
            rows.append((scheme, best, default))
        rows.sort(key=lambda t: t[1])  # ascending best_loss = best first
        rankings.append(ClosureRanking(regime=regime, ranked=tuple(rows)))
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
    )


def render_markdown(card: Scorecard) -> str:
    """Human-readable scorecard (the generated artifact)."""
    lines = ["# LES-suite scorecard", "", "## Q2 — per-regime closure ranking",
             "(lower tuned LES loss = better)", ""]
    for rk in card.rankings:
        lines.append(f"### {rk.regime}")
        lines.append("| rank | closure | best loss | default loss |")
        lines.append("|---|---|---|---|")
        for i, (scheme, best, default) in enumerate(rk.ranked, 1):
            lines.append(f"| {i} | {scheme} | {best:.4f} | {default:.4f} |")
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
