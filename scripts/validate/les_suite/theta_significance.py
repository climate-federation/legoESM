"""θ-consistent D7 significance for the dry-CBL closure ranking (LES_SUITE.md §7/D7).

Rank the tuned closures by their θ-ONLY profile RMSE (not the combined θ/u/v loss) and
gate the closure-to-closure margins on σ_LES(θ) — the robust θ floor (≈0.0074 for the free
CBL). Motivation: σ_LES(combined)≈0.319 is WIND-dominated and ill-conditioned for a Ug=0
free-convective CBL (the Ekman σ_v is huge), so the combined gate declares every closure
margin "not significant". θ is the quantity a convective-BL closure is actually judged on;
a θ-only gate asks the honest question "are the closures distinguishable on the field that
matters?".

Method: re-score each tuned ``best_overrides`` with ``scm_les_final_score`` — the SAME
build + normalization the tuner used (nlev/dt read from the record, so the re-score matches
the tuning; no protocol drift). PRECISION-GATE self-check (CLAUDE.md): the re-scored
``.combined`` MUST reproduce the stored ``best_loss`` (within tol) before the θ component is
trusted; a mismatch means the re-score does not reproduce the harness and is flagged, not
silently ranked. Reuses the tuner's exact config apply-site (no duplicated scheme roster or
nested-CLUBB apply logic).

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/les_suite/theta_significance.py \
        --tuned-dir results/les_suite/tuned --artifacts-dir results/les_suite/artifacts \
        --sigma-theta 0.0074 --artifact-prefix cbl_nieuwstadt__lasd
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Reuse the tuner's EXACT config apply-site (public ``apply_overrides_to_base``) and base
# builder so this analysis and the tuner cannot diverge on the scheme roster or the
# nested-CLUBB descend/rewrap. Script-to-script reuse (both under scripts/); no duplication.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "run"))
from tune_scm_to_les import _base_turbulence, apply_overrides_to_base  # noqa: E402

_Q0_DECIMALS = 6
# Fallback resolution/timestep for pre-nlev/dt-record tuned JSONs (the campaign's config).
# The self-check catches a wrong fallback: a bad nlev/dt makes .combined != best_loss.
_DEFAULT_NLEV = 24
_DEFAULT_DT = 10.0


@dataclass(frozen=True)
class ThetaRanking:
    """θ-only ranking for one (regime, flux) slice + the significance of the top margin."""
    regime: str
    q0: float | None
    label: str
    ranked: tuple[tuple[str, float], ...]      # (scheme, theta_rmse), ascending
    sigma_theta: float | None
    # margin between best and 2nd-best θ_rmse (None if <2 closures); significant if
    # sigma_theta is set AND margin > sigma_theta (the top pair is distinguishable on θ).
    top_margin: float | None
    top_significant: bool | None


def _flux_key(r: dict) -> tuple[tuple, str, float | None]:
    """(group_key, label, q0) — typed key so a numeric flux and a string label never
    collide (mirrors the scorecard's convention)."""
    q0 = r.get("q0")
    if q0 is not None:
        q0c = round(float(q0), _Q0_DECIMALS)
        return ("q0", q0c), f"Q0={q0c:g}", q0c
    label = str(r.get("artifact") or r.get("case") or "?")
    return ("label", label), label, None


def rank_theta_significance(
    records: list[dict], sigma_theta: float | None
) -> list[ThetaRanking]:
    """Pure ranking/gating logic: within each (regime, flux) slice rank closures by
    ``theta_rmse`` ascending and flag whether the best-vs-2nd margin exceeds σ_LES(θ).

    ``records`` need ``regime``, ``scheme``, ``theta_rmse`` (+ optional ``q0``/``artifact``).
    Duplicate (slice, scheme) collapses to the LOWEST θ_rmse (a stale re-tune can't
    double-count). Ordered by regime then ascending flux.
    """
    groups: dict[tuple[str, tuple], dict] = {}
    for r in records:
        if r.get("theta_rmse") is None:
            continue
        gkey, label, q0 = _flux_key(r)
        g = groups.setdefault(
            (r["regime"], gkey), {"label": label, "q0": q0, "best": {}})
        s = r["scheme"]
        tr = float(r["theta_rmse"])
        if s not in g["best"] or tr < g["best"][s]:
            g["best"][s] = tr
    out: list[ThetaRanking] = []
    for (regime, _gk), g in groups.items():
        ranked = tuple(sorted(g["best"].items(), key=lambda kv: kv[1]))
        margin = ranked[1][1] - ranked[0][1] if len(ranked) >= 2 else None
        sig = (margin > sigma_theta) if (margin is not None
                                         and sigma_theta is not None) else None
        out.append(ThetaRanking(
            regime=regime, q0=g["q0"], label=g["label"], ranked=ranked,
            sigma_theta=sigma_theta, top_margin=margin, top_significant=sig))
    out.sort(key=lambda t: (t.regime, t.q0 is None,
                            t.q0 if t.q0 is not None else 0.0, t.label))
    return out


def split_trusted(enriched: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition re-scored records into (trusted, untrusted) by the self-check GATE.

    Trusted = the re-score reproduced the tuner's stored loss (``self_check_ok is True``).
    Untrusted = a mismatch (``False``) OR unverifiable (``None``, no stored best_loss) —
    these are EXCLUDED from the θ ranking (their θ_rmse is not to be trusted) but REPORTED,
    never silently dropped. This is the precision-gate the codex review required: a warning
    alone let an untrusted record still be ranked ``SIGNIFICANT``.
    """
    trusted = [e for e in enriched if e.get("self_check_ok") is True]
    untrusted = [e for e in enriched if e.get("self_check_ok") is not True]
    return trusted, untrusted


def rescore_theta(
    record: dict, artifacts_dir: Path, *, self_check_tol: float = 1e-3
) -> dict | None:
    """Re-score one tuned record's ``best_overrides`` and attach ``theta_rmse`` (+ u/v/
    combined + the self-check). Returns an enriched copy, or ``None`` if the artifact is
    missing or the SCM diverged. The re-score uses the record's nlev/dt (fallback to the
    campaign default) so it matches the tuning protocol."""
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    from legoesm.atmosphere.les_suite.scm_runner import scm_les_final_score

    stem = record.get("artifact") or record.get("case")
    art_path = artifacts_dir / f"{stem}.npz"
    if not art_path.exists():
        print(f"[skip] artifact {art_path.name} not found", file=sys.stderr)
        return None
    artifact = load_artifact(art_path)
    base, _key = _base_turbulence(record["scheme"])
    cfg = apply_overrides_to_base(base, record.get("best_overrides") or {})
    nlev = int(record.get("nlev") or _DEFAULT_NLEV)
    dt = float(record.get("dt") or _DEFAULT_DT)
    score = scm_les_final_score(artifact, cfg, nlev=nlev, dt=dt)
    if score is None:
        print(f"[skip] {record['scheme']} @ {stem}: SCM diverged on re-score",
              file=sys.stderr)
        return None
    combined = float(score.combined)
    best_loss = record.get("best_loss")
    # Precision-gate: the re-score MUST reproduce the tuner's stored loss for the θ
    # component to be trusted. ``ok`` is a TRISTATE — True (reproduced), False (mismatch:
    # wrong nlev/dt or a non-reproducing re-score), or None (UNVERIFIABLE: no stored
    # best_loss). Only ``True`` records are ranked (the gate lives in main); False/None are
    # reported + excluded, never silently trusted (a missing best_loss defaulting to True
    # was the codex-flagged hole).
    ok: bool | None = None
    if best_loss is not None:
        ref = max(1.0, abs(float(best_loss)))
        ok = abs(combined - float(best_loss)) <= self_check_tol * ref
        if not ok:
            print(f"[WARN] {record['scheme']} @ {stem}: re-scored combined={combined:.5f} "
                  f"!= stored best_loss={float(best_loss):.5f} (nlev={nlev} dt={dt}); "
                  "θ component NOT trusted — check the recorded nlev/dt.", file=sys.stderr)
    enriched = dict(record)
    enriched.update(
        theta_rmse=float(score.theta_rmse), u_rmse=float(score.u_rmse),
        v_rmse=float(score.v_rmse), rescored_combined=combined, self_check_ok=ok)
    return enriched


def _load_records(tuned_dir: Path, prefix: str) -> list[dict]:
    """Load tuned JSONs matching ``{prefix}*__df.json``; stamp a ``regime`` (all dry CBL
    here) so the ranker groups them. A record needs artifact/scheme/best_overrides."""
    records = []
    for p in sorted(tuned_dir.glob(f"{prefix}*__df.json")):
        try:
            r = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {p.name}: {e}", file=sys.stderr)
            continue
        r.setdefault("regime", "dry_convective")
        records.append(r)
    return records


def _format(rankings: list[ThetaRanking], untrusted: list[dict] | None = None) -> str:
    lines = ["# θ-consistent D7 significance (closures ranked by θ-only RMSE)\n"]
    if untrusted:
        lines.append(f"> ⚠ {len(untrusted)} record(s) EXCLUDED (self-check failed — the "
                     "re-score did not reproduce the tuner's stored loss, so their θ_rmse "
                     "is not trusted):")
        for e in untrusted:
            bl = e.get("best_loss")
            rc = e.get("rescored_combined")
            lines.append(f">   - {e.get('scheme')} @ {e.get('artifact')}: "
                         f"rescored_combined={rc if rc is None else f'{rc:.5f}'} vs "
                         f"stored best_loss={bl if bl is None else f'{float(bl):.5f}'}")
        lines.append("")
    for rk in rankings:
        st = rk.sigma_theta
        lines.append(f"## {rk.regime} — {rk.label}"
                     + (f"  (σ_LES(θ)={st:.4f})" if st is not None else ""))
        for i, (scheme, tr) in enumerate(rk.ranked):
            mark = " ← best" if i == 0 else ""
            lines.append(f"  {i+1}. {scheme:<18} θ_rmse={tr:.4f}{mark}")
        if rk.top_margin is not None:
            verdict = ("SIGNIFICANT" if rk.top_significant else "not significant"
                       if rk.top_significant is not None else "no σ_LES(θ) given")
            lines.append(f"  top margin = {rk.top_margin:.4f}  → {verdict} "
                         f"(vs σ_LES(θ))")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tuned-dir", type=Path, default=Path("results/les_suite/tuned"))
    p.add_argument("--artifacts-dir", type=Path,
                   default=Path("results/les_suite/artifacts"))
    p.add_argument("--artifact-prefix", default="cbl_nieuwstadt__lasd",
                   help="tuned-JSON prefix to include (default: the dry-CBL anchor+sweep)")
    p.add_argument("--sigma-theta", type=float, default=None,
                   help="σ_LES(θ) from compute_sigma_les.py (the θ significance floor)")
    p.add_argument("--self-check-tol", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=None,
                   help="write the markdown report here (default: stdout)")
    args = p.parse_args(argv)

    records = _load_records(args.tuned_dir, args.artifact_prefix)
    if not records:
        print(f"error: no tuned JSONs matching {args.artifact_prefix}*__df.json in "
              f"{args.tuned_dir}", file=sys.stderr)
        return 2
    enriched = [e for r in records
                if (e := rescore_theta(r, args.artifacts_dir,
                                       self_check_tol=args.self_check_tol)) is not None]
    if not enriched:
        print("error: no records could be re-scored (missing artifacts / all diverged)",
              file=sys.stderr)
        return 2
    trusted, untrusted = split_trusted(enriched)
    if untrusted:
        print(f"[WARN] {len(untrusted)}/{len(enriched)} records EXCLUDED from the θ ranking "
              "— the re-score did not reproduce the tuner's stored loss (or there was no "
              "best_loss to verify); their θ_rmse is not trusted.", file=sys.stderr)
    if not trusted:
        print("error: no records passed the self-check; cannot produce a trusted θ ranking "
              "(check the recorded nlev/dt against the tuning config).", file=sys.stderr)
        return 2
    rankings = rank_theta_significance(trusted, args.sigma_theta)
    report = _format(rankings, untrusted=untrusted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"-> {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
