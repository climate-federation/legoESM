#!/usr/bin/env python3
"""Round 316: forward-bit fingerprint of one card (measurement only).

Runs the production ``model.step`` ``--steps`` times from the card's start
state and writes, per kt, the SHA-256 of every state leaf's raw bytes (NaN
payloads and signed zeros included), plus the SHA-256 of the step's jaxpr.
Two fingerprints from two commits are compared with ``--compare A B``: any
leaf whose bytes differ at any kt is listed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

ORCA2 = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
IDEALIZED = ("gyre", "vortex_zco_flux", "vortex_zco_vector", "vortex_smt_flux",
             "vortex_smt_vector", "overflow", "lock_exchange")


def _card(case: str):
    from legoesm.ocean.fidelity import nemo_testcase_recipe as rec

    if case.startswith("vortex"):
        kind, mom = case.split("_")[1:]
        return (rec.build_vortex_zco_card(momentum=mom) if kind == "zco"
                else rec.build_vortex_smt_zps_card(momentum=mom))
    return {"gyre": rec.build_gyre_zco_card, "overflow": rec.build_overflow_zps_card,
            "lock_exchange": rec.build_lock_exchange_zco_card}[case]()


def leaf_hashes(state) -> dict:
    import jax
    import numpy as np

    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        a = np.ascontiguousarray(np.asarray(leaf))
        out[jax.tree_util.keystr(path)] = (
            f"{a.dtype}{a.shape}:" + hashlib.sha256(a.tobytes()).hexdigest())
    return out


def fingerprint(case: str, steps: int) -> dict:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if case in ORCA2:
        from scripts.validate.ocean_fidelity.orca2_l4 import (
            nemo_testcase_l4_orca2_round315_grad_probe as probe,
        )
        card, state, forcing = probe.build(case)
        steps = min(steps, len(forcing))
    else:
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        card = _card(case)
        state, forcing = card.recipe.initial_state, [(None, None)] * steps
    r = card.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                  iwm_forcing=getattr(r, "iwm_forcing", None))

    def call(s, fw, sf):
        if fw is None and sf is None:
            return model.step(s, card.dt_s)
        return model.step(s, card.dt_s, freshwater=fw, surface_forcing=sf)

    text = str(jax.make_jaxpr(lambda s: call(s, *forcing[0]))(state))
    text = re.sub(r" at 0x[0-9a-f]+", " at 0xADDR", text)
    out = {"case": case, "worktree": worktree_stamp(allow_dirty=True),
           "jaxpr_lines": len(text.splitlines()),
           "jaxpr_sha256": hashlib.sha256(text.encode()).hexdigest(), "kt": {}}
    for kt in range(1, steps + 1):
        state = call(state, *forcing[kt - 1])
        out["kt"][str(kt)] = leaf_hashes(state)
        print(f"{case} kt={kt}", flush=True)
    return out


def compare(a: Path, b: Path) -> dict:
    fa, fb = (json.loads(p.read_text()) for p in (a, b))
    if fa["case"] != fb["case"] or set(fa["kt"]) != set(fb["kt"]):
        raise SystemExit(f"not comparable: {a} vs {b}")
    moved = [f"kt{kt}:{leaf}" for kt in fa["kt"]
             for leaf in sorted(set(fa["kt"][kt]) | set(fb["kt"][kt]))
             if fa["kt"][kt].get(leaf) != fb["kt"][kt].get(leaf)]
    n = sum(len(v) for v in fa["kt"].values())
    return {"case": fa["case"], "steps": len(fa["kt"]), "leaf_steps": n,
            "moved": moved, "jaxpr_identical": fa["jaxpr_sha256"] == fb["jaxpr_sha256"],
            "verdict": "BIT-IDENTICAL" if not moved else f"MOVED {len(moved)}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=ORCA2 + IDEALIZED)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        result = compare(*args.compare)
        print(f"{result['case']}: {result['verdict']} over {result['leaf_steps']} "
              f"leaf-steps (jaxpr identical: {result['jaxpr_identical']})")
        for m in result["moved"][:20]:
            print("  ", m)
        return 0 if not result["moved"] else 1
    import jax
    if not (jax.config.jax_enable_x64 and jax.default_backend() == "cpu"):
        raise SystemExit("needs JAX_ENABLE_X64=1 on CPU")
    args.output.write_text(json.dumps(fingerprint(args.case, args.steps), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
