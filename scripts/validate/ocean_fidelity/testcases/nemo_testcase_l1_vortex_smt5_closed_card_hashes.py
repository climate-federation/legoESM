#!/usr/bin/env python
"""SHA-256 of each closed card's prognostic state after N production steps.

Run once at the parent commit and once after a card-selected option lands;
identical digests prove the option leaves every card that does not select it
bit-identical.  fp64, scalar libm, the production ``model.step`` closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CLOSED_CARDS = (
    "GYRE-zco", "VORTEX-zco", "VORTEX_VEC-zco",
    "VORTEX_SMT-zps", "VORTEX_SMT_VEC-zps",
    "VORTEX_SMT1_VEC-zps", "VORTEX_SMT2_VEC-zps",
    "VORTEX_SMT3_VEC-zps", "VORTEX_SMT4_VEC-zps",
)
STATE_FIELDS = ("u", "v", "T", "S", "eta")


def state_digest(state) -> str:
    handle = hashlib.sha256()
    for name in STATE_FIELDS:
        data = np.ascontiguousarray(np.asarray(getattr(state, name).data,
                                               dtype=np.float64))
        if not np.all(np.isfinite(data)):
            raise SystemExit(f"non-finite {name}")
        handle.update(name.encode())
        handle.update(repr(data.shape).encode())
        handle.update(data.tobytes())
    return handle.hexdigest()


def run_case(case: str, steps: int) -> str:
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    if not jax.config.jax_enable_x64:
        raise SystemExit("JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(case)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    for _ in range(steps):
        state = model.step(state, dt=card.dt_s)
    return state_digest(state)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CLOSED_CARDS)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path,
                        help="baseline json; exit 1 on any differing digest")
    args = parser.parse_args(argv)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True,
                                check=True).stdout.strip())
    result = {"git_sha": sha, "worktree_dirty": dirty, "steps": args.steps,
              "digests": {}}
    for case in args.case or CLOSED_CARDS:
        result["digests"][case] = run_case(case, args.steps)
        print(case, result["digests"][case], flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.compare is not None:
        base = json.loads(args.compare.read_text())["digests"]
        bad = [c for c, d in result["digests"].items() if base.get(c) != d]
        print("CLOSED_CARDS_" + ("DIFFER " + ",".join(bad) if bad
                                 else "BIT_IDENTICAL"))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
