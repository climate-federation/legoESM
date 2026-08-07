#!/usr/bin/env python
"""Broad AMIP scorecard: one scalar objective from TOA radiation + tas + pr.

Thin CLI over :mod:`legoesm.training.amip_scorecard`.  It does NOT compute a
single metric of its own: the numbers come from the canonical scorer,
``scripts/plot/plot_amip_pattern_eval.py``, which is the script that produced
every published ``bias`` / ``r_pattern`` figure for this model.  This one only
weights them, and says whether a candidate beat a baseline WITHOUT selling off
spatial skill.

Two input modes, so the same weights apply whether you are scoring a finished
archive or a run inside a tuning loop:

* ``--metrics-json`` — consume a JSON already written by
  ``plot_amip_pattern_eval.py --json``.  No netCDF is opened; instant.
* ``<run_dir>`` — parse the ``*_metrics.txt`` the scorer left in the run
  directory.  Convenience for archives scored before ``--json`` existed.

Usage::

    # score one run
    python scripts/validate/amip_scorecard.py --metrics-json pe.json

    # candidate vs baseline, with the pattern-regression constraint
    python scripts/validate/amip_scorecard.py --metrics-json cand.json \
        --baseline-json base.json --gate

    # re-weight without recomputing anything
    python scripts/validate/amip_scorecard.py --metrics-json pe.json \
        --weights rsut=0.4,rlut=0.2,tas=0.2,pr=0.2 --bias-fraction 0.6

With ``--gate`` the exit code is 0 only when the candidate improved the
objective AND no scored field lost more than ``--max-pattern-drop`` of centred
pattern correlation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from legoesm.training.amip_scorecard import (  # noqa: E402
    DEFAULT_MAX_PATTERN_DROP,
    SCORED_FIELDS,
    ScorecardWeights,
    compare_scorecards,
    evaluate_scorecard,
)

# Row layout written by plot_amip_pattern_eval.py, e.g.
#   rsut       bias=  +59.326  rmse_c=  40.799  r_pattern=-0.120   vs ...
_METRIC_ROW = re.compile(
    r"^(?P<field>\S+)\s+bias=\s*(?P<bias>[-+0-9.eE]+)\s+"
    r"rmse_c=\s*(?P<rmse>[-+0-9.eE]+)\s+"
    r"r_pattern=\s*(?P<corr>[-+0-9.eE]+)")


def parse_metrics_txt(text: str) -> dict[str, dict[str, float]]:
    """Parse a ``*_metrics.txt`` report back into the metrics table.

    Only the per-field rows are read; the TOA/E-P/Hadley trailer is ignored.
    Returns the same structure ``pattern_stats`` produces, so both input modes
    feed byte-identical numbers into the objective.
    """
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        m = _METRIC_ROW.match(line.strip())
        if m is None:
            continue
        out[m.group("field")] = {
            "bias": float(m.group("bias")),
            "rmse_centered": float(m.group("rmse")),
            "pattern_corr": float(m.group("corr")),
        }
    return out


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Load a metrics table from a pattern-eval ``--json`` file or a run dir."""
    if path.is_dir():
        hits = sorted(path.glob("*_metrics.txt"))
        if not hits:
            raise SystemExit(
                f"no *_metrics.txt in {path} — run "
                f"scripts/plot/plot_amip_pattern_eval.py on it first")
        if len(hits) > 1:
            raise SystemExit(
                f"{len(hits)} *_metrics.txt files in {path}; pass the one you "
                f"mean explicitly: {[h.name for h in hits]}")
        return parse_metrics_txt(hits[0].read_text())
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        fields = payload.get("fields", payload)
        return {k: dict(v) for k, v in fields.items()}
    return parse_metrics_txt(path.read_text())


def parse_weights(spec: str, bias_fraction: float) -> ScorecardWeights:
    """Parse ``rsut=0.4,rlut=0.2,...`` into validated weights.

    An empty spec keeps the documented defaults.  An unknown or duplicated
    field is a hard error — a typo must never silently leave a term at its
    default while the user believes it was re-weighted.
    """
    kwargs: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "=" not in part:
            raise SystemExit(f"--weights entry {part!r} is not 'field=value'")
        key, _, val = part.partition("=")
        key = key.strip()
        if key not in SCORED_FIELDS:
            raise SystemExit(
                f"--weights field {key!r} unknown; scored fields are "
                f"{list(SCORED_FIELDS)}")
        if key in kwargs:
            raise SystemExit(f"--weights field {key!r} given twice")
        try:
            kwargs[key] = float(val)
        except ValueError:
            raise SystemExit(f"--weights value for {key!r} is not a number: "
                             f"{val!r}") from None
    return ScorecardWeights(bias_fraction=bias_fraction, **kwargs).validate()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", type=Path,
                    help="run directory containing a *_metrics.txt "
                         "(alternative to --metrics-json)")
    ap.add_argument("--metrics-json", type=Path, default=None,
                    help="pattern_eval --json output for the candidate")
    ap.add_argument("--baseline-json", type=Path, default=None,
                    help="pattern_eval --json (or run dir / _metrics.txt) for "
                         "the baseline; enables the comparison verdict")
    ap.add_argument("--weights", default="",
                    help="override field weights, e.g. "
                         "'rsut=0.4,rlut=0.2,tas=0.2,pr=0.2'; must sum to 1")
    ap.add_argument("--bias-fraction", type=float, default=0.5,
                    help="within-field split: bias share of each field score, "
                         "pattern gets the rest (default 0.5)")
    ap.add_argument("--max-pattern-drop", type=float,
                    default=DEFAULT_MAX_PATTERN_DROP,
                    help="tolerated fall in centred pattern correlation vs "
                         f"baseline (default {DEFAULT_MAX_PATTERN_DROP})")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="write the scorecard result as JSON")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 unless the candidate improved without any "
                         "pattern regression (requires --baseline-json)")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    source = args.metrics_json or args.run_dir
    if source is None:
        raise SystemExit("give a run_dir or --metrics-json")
    weights = parse_weights(args.weights, args.bias_fraction)

    metrics = load_metrics(Path(source))
    result = evaluate_scorecard(metrics, weights)

    print(f"# AMIP broad scorecard — {source}")
    print(f"# weights: rsut={weights.rsut} rlut={weights.rlut} "
          f"tas={weights.tas} pr={weights.pr} "
          f"bias_fraction={weights.bias_fraction}")
    print(result.format_table())

    payload: dict = {
        "source": str(source),
        "objective": result.objective,
        "weights": weights._asdict(),
        "fields": {fs.field: fs._asdict() for fs in result.fields},
    }
    status = 0
    if args.baseline_json is not None:
        base_metrics = load_metrics(Path(args.baseline_json))
        cmp_ = compare_scorecards(
            base_metrics, metrics, weights, args.max_pattern_drop)
        print()
        print(f"# vs baseline {args.baseline_json}")
        print(cmp_.format_report())
        payload["baseline"] = {
            "source": str(args.baseline_json),
            "objective": cmp_.baseline.objective,
        }
        payload["delta_objective"] = cmp_.delta_objective
        payload["improved"] = cmp_.improved
        payload["pattern_regressions"] = [
            r._asdict() for r in cmp_.regressions]
        if args.gate and not cmp_.improved:
            status = 1
    elif args.gate:
        raise SystemExit("--gate needs --baseline-json to compare against")

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
