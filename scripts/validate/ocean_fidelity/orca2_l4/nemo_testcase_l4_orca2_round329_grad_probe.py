#!/usr/bin/env python3
"""Round 329: round 324's rung-10 N = 1 gradient probe from the comparison
card's own initial state (DECISION 157: the record's SI3 initial ice).

Round 324's probe steps ``probe.build``'s state, the shipped card's; here that
state is the comparison card's, whose SSH carries the record's initial ice.
Usage: nemo_testcase_l4_orca2_round329_grad_probe.py <repo> <out.json> on
"""
import runpy
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
for p in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    sys.path.insert(0, str(p))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10_gate,
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
)

if sys.argv[3] != "on":
    raise SystemExit("round 329 measures the comparison card only (arg 'on')")
_build = probe.build


def build(rung):
    card, state, forcing = _build(rung)
    if rung != "rung10" or state is not card.recipe.initial_state:
        raise SystemExit("round 329 expects rung 10 stepped from its card state")
    state = rung10_gate.comparison_card(card).recipe.initial_state
    print("comparison initial SSH (wet):", sorted({float(x) for x in
          state.eta.data[card.recipe.z_coord.is_active[..., 0]]}), flush=True)
    return card, state, forcing


probe.build = build
runpy.run_path(str(Path(__file__).with_name(
    "nemo_testcase_l4_orca2_round324_grad_probe.py")), run_name="__main__")
