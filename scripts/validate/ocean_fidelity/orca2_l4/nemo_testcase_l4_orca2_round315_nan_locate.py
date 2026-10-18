#!/usr/bin/env python3
"""Round 315: name the first statement that makes a one-step cotangent non-finite.

MEASUREMENT ONLY.  Pulls ``d L_T / d (state leaf)`` back through ONE production
step (the round-315 probe's traced step at s = 1) with ``jax_debug_nans`` or
``jax_debug_infs`` on: JAX re-runs the failing computation op by op and raises
at the first primitive whose output is non-finite, with the source line of the
forward statement that primitive belongs to (a transpose carries its forward
statement's source location).  The repository frames of that traceback are
written out; the forward value is never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
)


def repo_frames(error: BaseException) -> list[str]:
    return [f"{f.filename.split('/packages/')[1]}:{f.lineno} {f.name}: {f.line}"
            for f in traceback.extract_tb(error.__traceback__)
            if "/packages/" in f.filename]


def locate(rung: str, leaf: str, mode: str) -> dict:
    """``leaf="state"`` pulls back to the whole state with every traceable
    scale traced, exactly as the probe's per-step pullback does."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    card, state0, forcing = probe.build(rung)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config,
                                  iwm_forcing=card.recipe.iwm_forcing)
    fw, sf = forcing[0]
    model.step(state0, card.dt_s, freshwater=fw, surface_forcing=sf)  # prime caches
    losses, _ = probe.make_losses(model, card.recipe.z_coord)
    step = probe.traced_step(model)
    trace = probe.traceability(model, state0, forcing, card.dt_s)
    ones = {k: jnp.float64(1.0) for k in probe.SCALARS if trace[k]["traceable"]}
    if leaf == "state":
        def loss_of(st):
            return losses["T"](step(st, ones, card.dt_s, fw, sf))

        arg = state0
    else:
        field = getattr(state0, leaf)

        def loss_of(data):
            st = state0._replace(**{leaf: field.replace(data=data)})
            return losses["T"](step(st, ones, card.dt_s, fw, sf))

        arg = field.data
    grad = jax.jit(jax.grad(loss_of, allow_int=True))
    plain = grad(arg)
    bad = [jax.tree_util.keystr(path) for path, x in jax.tree_util.tree_leaves_with_path(plain)
           if jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)
           and not bool(jnp.all(jnp.isfinite(x)))]
    out = {"rung": rung, "leaf": leaf, "mode": mode, "traced_scales": sorted(ones),
           "worktree": probe_stamp(), "nonfinite_leaves_jit": bad,
           "nonfinite_count_jit": len(bad)}
    # The debug flag must be on BEFORE a computation is dispatched: a cached
    # executable's fast path skips the check (round 315's first locator run
    # returned raised=False on an all-NaN gradient for exactly this reason).
    # No outer jit: the inner jitted forward and its transpose are checked
    # separately, and the failing one is re-run op by op.
    jax.clear_caches()
    jax.config.update(f"jax_debug_{mode}", True)
    try:
        jax.grad(loss_of, allow_int=True)(arg)
        out["raised"] = False
    except FloatingPointError as error:
        out.update(raised=True, message=str(error)[:2000],
                   frames=repo_frames(error))
    finally:
        jax.config.update(f"jax_debug_{mode}", False)
    return out


def probe_stamp() -> dict:
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    return worktree_stamp(allow_dirty=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=tuple(probe.ROOTS), required=True)
    parser.add_argument("--leaf", default="land_mask")
    parser.add_argument("--mode", choices=("nans", "infs"), default="nans")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = locate(args.rung, args.leaf, args.mode)
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"STATUS LOCATED raised={result.get('raised')} "
          f"nonfinite={result['nonfinite_count_jit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
