"""Phase 6 validation: 120-day Eady Redi-only at high kappa.

This is the test the GM/Redi plan doc specifies for confirming that the
triad scheme's per-step cancellation actually translates to *zero T
divergence* under the pressure→velocity→advection feedback that
amplified the centered scheme's tiny residual into a 1.17 K drift over
120 days at kappa_Redi = 50000.

We run three cases at 20×10 (latlon_channel, ~100 km), 120 days, with
``kappa_Redi`` boosted to 50000 (50× production):

    1. baseline           — kappa_GM = kappa_Redi = 0  (dynamics + sponge only)
    2. redi-only-centered — kappa_GM = 0, kappa_Redi = 5e4, slope_scheme="centered"
    3. redi-only-triads   — kappa_GM = 0, kappa_Redi = 5e4, slope_scheme="triads"

For an algebraically-correct Redi tensor with q = f(rho), case 2 should
match case 1 to round-off (the residual was the exact failure mode the
plan diagnosed); the centered scheme should diverge.

Run with:
    JAX_ENABLE_X64=1 python scripts/validate_triad_redi_120day.py
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path

# Make the sibling test-matrix package importable.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from ocean_test_matrix.experiments import run_eady_gm_redi
from ocean_test_matrix.testcase import TestCase


def _make_tc(name: str, gm_mode: str, slope_scheme: str,
             kappa_Redi: float = 5.0e4,
             beta_S: float = 0.0) -> TestCase:
    return TestCase(
        case=name,
        grid_type="latlon_channel",
        resolution="20x10",
        duration_days=120.0,
        quick_days=10.0,
        run_kwargs={
            "gm_mode": gm_mode,
            "slope_scheme": slope_scheme,
            "kappa_GM_override": 0.0,
            "kappa_Redi_override": kappa_Redi,
            # Force β_S = 0 in the linear EOS so that ρ depends *only*
            # on T.  Without this, even a 1e-5 PSU evolution of S from
            # numerical noise breaks the q = f(ρ) assumption that the
            # Redi-cancellation property relies on.
            "beta_S_override": beta_S,
        },
    )


def _run(tc: TestCase, out_root: Path) -> dict:
    out = out_root / tc.case
    out.mkdir(parents=True, exist_ok=True)
    status, wall, notes = run_eady_gm_redi(tc, out, days=tc.duration_days)
    # Read back the diagnostic dump that run_eady_gm_redi writes via
    # _save_case_diagnostics — easier to just re-extract via the txt.
    return {
        "case": tc.case,
        "status": status,
        "wall": wall,
        "notes": notes,
    }


def main() -> None:
    out_root = Path("results/triad_phase6_validation")
    out_root.mkdir(parents=True, exist_ok=True)

    cases = [
        _make_tc("baseline_no_gmredi", "baseline", "centered"),
        _make_tc("redi_only_centered_k5e4", "redi_only", "centered"),
        _make_tc("redi_only_triads_k5e4", "redi_only", "triads"),
    ]

    print("=" * 78)
    print("  Phase 6 validation: 120-day Eady, kappa_Redi = 5e4")
    print("=" * 78)

    results = []
    for tc in cases:
        print(f"\n>>> Running {tc.case} ({tc.duration_days:.0f} days, "
              f"slope_scheme={tc.run_kwargs['slope_scheme']}, "
              f"gm_mode={tc.run_kwargs['gm_mode']})")
        r = _run(tc, out_root)
        results.append(r)
        print(f"  {r['status']:6s} | {r['wall']:6.1f}s | {r['notes']}")

    print("\n" + "=" * 78)
    print("  SUMMARY — T_drift (K) over 120 days")
    print("=" * 78)
    for r in results:
        print(f"  {r['case']:35s}  {r['notes']}")

    print("\nThe plan's failure mode was: centered Redi-only diverges from")
    print("baseline by ~1.17 K at 120 days (centered amplification of the")
    print("~1e-11 K/s single-step residual through pressure-velocity")
    print("feedback).  If triads are working, redi-only-triads should match")
    print("baseline; redi-only-centered should diverge.")


if __name__ == "__main__":
    main()
