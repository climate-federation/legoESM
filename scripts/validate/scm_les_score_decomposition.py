"""Decompose an SCM-vs-LES joint turbulence score into what actually drives it.

``run_scm_les_turbulence_tuning.py`` fits ONE parameter set per scheme to
several cases at once and aggregates with an UNWEIGHTED MEAN of per-case
scores. Each per-case score is a mass-weighted RMSE normalized by the LES
profile's own mass-weighted VERTICAL spread, and the driver's docstring reads
that normalization as making "a plain mean weight every regime equally".

That is a claim about the optimizer's effective objective, and it is testable:
the mean weights every regime's *absolute normalized error* equally, so a case
on which every scheme scores badly contributes proportionally more gradient
than a case every scheme already fits. This script measures the difference.

For each case and each scored variable it reports, on the SAME masked levels
and with the SAME mass weights the scorer uses:

* ``sigma_ref``    the LES profile's mass-weighted vertical spread, in PHYSICAL
                   units — the normalization's denominator.
* ``rmse_abs``     the mass-weighted RMSE in PHYSICAL units — the error a
                   reader can judge against the variable's own scale.
* ``score``        ``rmse_abs / sigma_ref``, the normalized per-variable score
                   the driver combines in quadrature.
* ``share``        that case's percentage of the scheme's joint loss.

A case with a large ``score`` but a small ``rmse_abs`` is dominating the fit
because its reference has little vertical structure to normalize by, NOT
because the SCM is far from the LES there. Those two are indistinguishable in
the joint number alone, which is the reason this script exists.

The metric helpers are IMPORTED from :mod:`legoesm.training.scm_rce_metrics`
rather than re-derived, so a change to the score definition cannot silently
desynchronise this diagnostic from the thing it is diagnosing.

Usage::

    python scripts/validate/scm_les_score_decomposition.py \\
        results/scm_les_turbulence/all_cases
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The scorer's own helpers. Re-deriving a mass-weighted std here is exactly the
# way a diagnostic ends up measuring something other than the quantity it
# claims to.
from legoesm.training.scm_rce_metrics import weighted_rmse, weighted_std

# Physical units of every variable the LES reference can carry, for the report.
_UNITS = {"theta": "K", "qv": "kg/kg", "u": "m/s", "v": "m/s"}


def _git_sha(repo) -> str:
    from legoesm.io.git_provenance import git_provenance
    return git_provenance(repo).commit or "unknown"


def decompose_case(npz_path: Path, scheme: str) -> dict[str, dict[str, float]]:
    """Per-variable sigma_ref / absolute RMSE / normalized score for one case.

    Reproduces the scorer's arithmetic exactly: mask the levels outside the LES
    domain, apply the stored mass weights, normalize by the reference's own
    weighted vertical std.
    """
    data = np.load(npz_path, allow_pickle=True)
    mask = data["mask"].astype(bool)
    weights = np.asarray(data["weights"], dtype=np.float64)
    scored = [str(s) for s in np.atleast_1d(data["scored"])]

    out: dict[str, dict[str, float]] = {}
    for name in scored:
        ref_key, scm_key = f"les_scmlev_{name}", f"scm_{scheme}_{name}"
        if ref_key not in data.files or scm_key not in data.files:
            continue
        # nan_to_num matches score_against_les: the reference is NaN outside
        # the LES domain and the weights are zero there, so the product is a
        # no-op -- but a raw NaN would still poison the sum.
        ref = np.nan_to_num(np.asarray(data[ref_key], dtype=np.float64), nan=0.0)
        raw = np.asarray(data[scm_key], dtype=np.float64)
        # A non-finite SCM profile must NOT be reported as a number. safe_sqrt
        # (which weighted_rmse uses) returns 0 for a NaN argument, because
        # `NaN > 0` is False -- and 0 is the PERFECT score. So a blown-up arm
        # silently decomposed to "rmse_abs 0.000, score 0.000", the single most
        # dangerous output this probe could produce. The tuner maps such an arm
        # to NONFINITE_PENALTY instead; here it is flagged, never scored.
        finite = bool(np.all(np.isfinite(raw[mask])))
        if not finite:
            out[name] = {
                "sigma_ref": float(weighted_std(ref, weights)),
                "rmse_abs": float("nan"),
                "score": float("nan"),
                "nonfinite": True,
                "units": _UNITS.get(name, "?"),
            }
            continue
        scm = np.where(mask, raw, 0.0)
        sigma = float(weighted_std(ref, weights))
        rmse_abs = float(weighted_rmse(scm - ref, weights))
        # The scorer floors sigma at PROFILE_FLOOR (1e-8); report the RAW sigma
        # so a floored -- i.e. structureless -- reference is visible as such
        # rather than hidden behind the floor.
        score = rmse_abs / sigma if sigma > 0.0 else float("inf")
        out[name] = {
            "sigma_ref": sigma,
            "rmse_abs": rmse_abs,
            "score": score,
            "nonfinite": False,
            "units": _UNITS.get(name, "?"),
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("indir", type=Path,
                   help="a multi-case run_scm_les_turbulence_tuning output dir")
    p.add_argument("--scheme", default=None,
                   help="scheme to decompose (default: every ranked scheme)")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    tuned = args.indir / "tuned_parameters.json"
    if not tuned.exists():
        raise SystemExit(f"no tuned_parameters.json in {args.indir}")
    payload = json.loads(tuned.read_text())
    cases = list(payload["protocol"]["cases"])

    schemes = ([args.scheme] if args.scheme else
               [s["scheme"] for s in payload["schemes"]
                if s.get("per_case_default")])
    if not schemes:
        raise SystemExit("no schemes with per-case scores in this directory")

    missing = [c for c in cases if not (args.indir / f"profiles_{c}.npz").exists()]
    if missing:
        raise SystemExit(
            f"missing profiles for {missing}; this decomposition needs the "
            "per-case profile files the multi-case driver writes."
        )

    by_scheme = {s["scheme"]: s for s in payload["schemes"]}
    report: dict[str, dict] = {}

    for scheme in schemes:
        entry = by_scheme.get(scheme)
        if entry is None or not entry.get("per_case_default"):
            continue
        per_case = entry["per_case_default"]
        total = sum(per_case[c] for c in cases)
        rows = {}
        for case in cases:
            rows[case] = {
                "joint_share_pct": 100.0 * per_case[case] / total,
                "case_score": per_case[case],
                "variables": decompose_case(
                    args.indir / f"profiles_{case}.npz", scheme),
            }
        report[scheme] = rows

    hdr = (f"{'case':<9}{'var':<7}{'sigma_ref':>13}{'rmse_abs':>13}"
           f"{'score':>9}{'share%':>9}  units")
    for scheme, rows in report.items():
        print(f"\n=== {scheme} ===")
        print(hdr)
        for case, r in rows.items():
            for var, v in r["variables"].items():
                if v.get("nonfinite"):
                    print(f"{case:<9}{var:<7}{v['sigma_ref']:>13.4g}"
                          f"{'NONFINITE':>13}{'--':>9}"
                          f"{r['joint_share_pct']:>9.1f}  {v['units']}")
                    continue
                print(f"{case:<9}{var:<7}{v['sigma_ref']:>13.4g}"
                      f"{v['rmse_abs']:>13.4g}{v['score']:>9.3f}"
                      f"{r['joint_share_pct']:>9.1f}  {v['units']}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"git_sha": _git_sha(Path(__file__).resolve().parents[2]),
             "indir": str(args.indir), "cases": cases, "report": report},
            indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
