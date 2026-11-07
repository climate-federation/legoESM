#!/usr/bin/env python3
"""Which card runs NEMO's UP3 momentum advection, and how big a row move is.

Round 61 found that four consecutive rounds had walked a statement belonging
to the OVERFLOW tank while reporting its improvement as ORCA2's.  The two
questions that settle that, and the two this probe answers, are:

* **which card resolves which momentum advection** -- the ORCA2 and GYRE decks
  take ``ln_dynadv_vec``, the OVERFLOW and LOCK decks ``ln_dynadv_up3``, and
  legoESM's flux-form routine (the only home of the UP3 face flux) is reached
  only from the ``flux_form`` arm, so a UP3 edit can move the tanks and
  nothing else;
* **how a trajectory row move compares with the error that row already
  carries** -- as a row-maximum ratio AND cell by cell, because the maximum of
  the worsening and the maximum of the residual are in general at DIFFERENT
  cells and the row statistic alone overstates the case.

Committed because round 61's numbers were produced by a throwaway derivation
and an adversarial review correctly refused them as unreproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The deck path is an external immutable input, not a repository artifact.
ORCA2_DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")

#: card -> (momentum_advection, momentum_flux_scheme-or-None-if-inert).
#: ``momentum_flux_scheme`` is read only under ``flux_form``, so on the
#: vector-invariant cards its value selects nothing and is not pinned here.
EXPECTED_CARD_SCOPE = {
    "ORCA2-zps": ("vector_invariant", None),
    "GYRE-zco": ("vector_invariant", None),
    "OVERFLOW-zps": ("flux_form", "nemo_up3"),
    "LOCK-zco": ("flux_form", "nemo_up3"),
}


def card_scope(*, orca2_deck: Path = ORCA2_DECK) -> dict[str, tuple[str, str]]:
    """Build the four cards and read back what each selects."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_card,
        build_lock_exchange_zco_card,
        build_orca2_zps_card,
        build_overflow_zps_card,
    )

    builders = {
        "ORCA2-zps": lambda: build_orca2_zps_card(orca2_deck),
        "GYRE-zco": build_gyre_zco_card,
        "OVERFLOW-zps": build_overflow_zps_card,
        "LOCK-zco": build_lock_exchange_zco_card,
    }
    scope = {}
    for name, builder in builders.items():
        config = builder().recipe.model_config
        scope[name] = (config.momentum_advection, config.momentum_flux_scheme)
    return scope


def check_card_scope(scope: dict[str, tuple[str, str]]) -> list[str]:
    """Names of cards whose selection disagrees with EXPECTED_CARD_SCOPE."""
    wrong = []
    for name, (expected_adv, expected_flux) in EXPECTED_CARD_SCOPE.items():
        if name not in scope:
            wrong.append(f"{name}: not built")
            continue
        advection, flux = scope[name]
        if advection != expected_adv:
            wrong.append(f"{name}: momentum_advection {advection!r} != "
                         f"{expected_adv!r}")
        elif expected_flux is not None and flux != expected_flux:
            wrong.append(f"{name}: momentum_flux_scheme {flux!r} != "
                         f"{expected_flux!r}")
    return wrong


def row_move_ratios(oracle, reference_candidate, candidate) -> dict:
    """Row and cellwise size of a move, against the residual already there.

    ``residual`` follows the gate's own sign convention: the non-negative
    distance from the oracle.  ``degradation`` is the candidate's residual
    minus the reference's, so a positive value is a move AWAY from NEMO.
    """
    oracle = np.asarray(oracle, dtype=np.float64)
    before = np.abs(np.asarray(reference_candidate, dtype=np.float64) - oracle)
    after = np.abs(np.asarray(candidate, dtype=np.float64) - oracle)
    degradation = after - before
    worst = int(np.argmax(degradation))
    return {
        "n_cells": int(oracle.size),
        "max_worsening": float(degradation[worst]),
        "max_reference_residual": float(np.max(before, initial=0.0)),
        # The row statistic: two maxima that need not share a cell.
        "row_ratio": (float(np.max(degradation, initial=0.0)
                            / np.max(before, initial=0.0))
                      if np.max(before, initial=0.0) > 0.0 else float("inf")),
        # The honest statistic: the same cell on both sides.
        "cellwise_ratio_at_worst_cell": (
            float(degradation[worst] / before[worst])
            if before[worst] > 0.0 else float("inf")),
        "n_cells_move_exceeds_own_residual": int(
            np.count_nonzero(degradation > before)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orca2-deck", type=Path, default=ORCA2_DECK)
    args = parser.parse_args(argv)
    scope = card_scope(orca2_deck=args.orca2_deck)
    wrong = check_card_scope(scope)
    print(json.dumps({
        "format": "nemo-testcase-up3-card-scope-v1",
        "card_scope": {k: list(v) for k, v in scope.items()},
        "disagreements": wrong,
        "status": "PASS" if not wrong else "FAIL",
    }, indent=2, sort_keys=True))
    return 0 if not wrong else 1


if __name__ == "__main__":
    raise SystemExit(main())
