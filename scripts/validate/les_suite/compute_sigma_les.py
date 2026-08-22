#!/usr/bin/env python
"""Compute σ_LES (D7) for a case from its emitted SGS-spread artifacts.

σ_LES is the LES's OWN uncertainty in the SCM tuner's loss units — the spread of the
same case run under {lasd, smagorinsky, vreman} (emit with ``run_les_suite --sgs``).
A closure's tuned-loss advantage or a Q1b margin below σ_LES is NOT a result (D7). This
driver loads the variant artifacts and prints σ_LES via
``legoesm.atmosphere.les_suite.sigma_les_prognostic`` (no LES, no SCM — pure assembly).

Usage::

    python scripts/validate/les_suite/compute_sigma_les.py \\
        --artifacts-dir results/les_suite/artifacts --case cbl_nieuwstadt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legoesm.atmosphere.les_suite.bridge import load_artifact
from legoesm.atmosphere.les_suite.sigma_les import SigmaLESError, sigma_les_prognostic

# The D7 SGS-spread closures (catalog `_SGS_SPREAD`). This is an EXPLICIT tag list, not
# a glob: a bare `{case}__*.npz` glob would wrongly pull in the flux-sweep artifacts
# (`{case}__lasd__q0_0.02.npz`), which are DIFFERENT physics, not the σ_LES spread. Pass
# resolution variants (e.g. `lasd__2x`, written by `--sgs lasd --label 2x`) via --sgs.
_DEFAULT_SGS = ("lasd", "smagorinsky", "vreman")


def _load_variants(artifacts_dir: Path, case: str, sgs: tuple[str, ...]) -> list:
    arts = []
    for s in sgs:
        path = artifacts_dir / f"{case}__{s}.npz"
        if path.exists():
            arts.append(load_artifact(path))
        else:
            print(f"[skip] {path.name} not found", file=sys.stderr)
    return arts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", type=Path,
                   default=Path("results/les_suite/artifacts"))
    p.add_argument("--case", required=True, help="registry LESCase name")
    p.add_argument("--sgs", nargs="+", default=list(_DEFAULT_SGS),
                   help="SGS variants to include (artifact tags)")
    args = p.parse_args(argv)

    arts = _load_variants(args.artifacts_dir, args.case, tuple(args.sgs))
    if len(arts) < 2:
        print(f"error: need >=2 SGS-variant artifacts for {args.case} in "
              f"{args.artifacts_dir} (found {len(arts)}); emit them with "
              "run_les_suite --sgs <lasd|smagorinsky|vreman>", file=sys.stderr)
        return 2
    try:
        s = sigma_les_prognostic(arts)
    except SigmaLESError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"σ_LES ({s.case_name}, variants={list(s.variants)}, {s.n_pairs} pairs) "
          "— in the SCM tuner's normalized-RMSE loss units:")
    print(f"  σ_LES(combined) = {s.sigma_combined:.4f}")
    print(f"  σ_LES(θ) = {s.sigma_theta:.4f}   σ_LES(u) = {s.sigma_u:.4f}   "
          f"σ_LES(v) = {s.sigma_v:.4f}")
    print("  per-pair combined distances (truth → candidate; score normalized by "
          "truth's std):")
    for truth, cand, d in s.per_pair:
        print(f"    truth={truth:>12}  candidate={cand:<12} = {d:.4f}")
    print("\nInterpretation (D7): a closure's tuned-loss advantage or a Q1b margin "
          f"below {s.sigma_combined:.4f} is within the LES's own spread — NOT a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
