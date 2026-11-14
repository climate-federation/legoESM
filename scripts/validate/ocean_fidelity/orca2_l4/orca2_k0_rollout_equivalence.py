#!/usr/bin/env python3
"""K0 rollout == step-by-step composed K0 run, and its gradient is finite (MEASUREMENT).

Step-by-step: ``nemo_orca2_k0_ocean_step`` for kt = 1..N, carrying s0/s2 as the
round-343 benchmark does.  Loop: the same for kt = 1, 2, then
``legoesm.training.orca2_k0_rollout`` over kt = 3..N in pairs.  Reports the
max |difference| of every state leaf and of the s0/s2 carries at kt = N, and
the finiteness and norm of d(volume-mean T^2 at kt = N)/d(T at kt = 2) through
the loop.  ``--card exact`` (CPU, libm) or ``normal`` (native maths).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean",
           REPO_ROOT / "packages/ml"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def build(card_mode: str, steps: int):
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean import bulk_flux_omip as bulk
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round343_speed_benchmark as r343,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm" if card_mode == "exact" else "native"))
    base = bulk.build_nemo_orca2_k0_ocean_card(r343.DEFAULT_DECK_ROOT)
    card = base if card_mode == "exact" else r343.build_normal_mode_orca2_k0_card(base)
    controls = r343.precompute_step_controls(card, r343.DEFAULT_DECK_ROOT,
                                             r343.DEFAULT_CORE_MANIFEST,
                                             r343.DEFAULT_ICE_COVER_PATH, steps)
    r = card.ocean.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config, iwm_forcing=r.iwm_forcing)
    return card, model, controls


def eager(card, model, controls, state):
    """kt = 1..len(controls), round 343's carry hand-off; returns the result per step."""
    from legoesm.ocean import bulk_flux_omip as bulk
    from legoesm.training.orca2_k0_rollout import S2_CARRY

    results, carry_s0, carry_s2, prior = [], None, None, None
    for kt, ctrl in enumerate(controls, start=1):
        s0, s2 = ctrl["s0_base"], ctrl["s2_base"]
        if carry_s0 is not None:
            s0 = s0._replace(prior=carry_s0.prior,
                             shortwave_fraction_top=carry_s0.shortwave_fraction_top)
            s2 = s2._replace(**{f: getattr(carry_s2, f) for f in S2_CARRY})
        res = bulk.nemo_orca2_k0_ocean_step(
            card, state, model=model, nemo_kt=kt, s0f=ctrl["s0f"], ice_if=ctrl["ice_if"],
            s0=s0, s2=s2, chl=ctrl["chl"],
            geothermal_heat_flux_mwm2=ctrl["geothermal_heat_flux_mwm2"],
            prior_surface=prior, sponge=ctrl["sponge"])
        state = res.ocean.state_after if hasattr(res.ocean, "state_after") else res.ocean
        carry_s0, carry_s2, prior = res.next_s0, res.next_s2, res.surface
        results.append(res)
    return results


def max_diffs(a, b) -> dict:
    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    if ta != tb:
        raise ValueError("pytree structures differ")
    out = {}
    for (path, x), (_, y) in zip(la, lb):
        x, y = np.asarray(x), np.asarray(y)
        if np.issubdtype(x.dtype, np.inexact):
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0.0, np.abs(x - y))
            out[jax.tree_util.keystr(path)] = float(np.max(d)) if d.size else 0.0
    return out


def main(argv=None) -> int:
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.training.orca2_k0_rollout import (
        carry_from_result, make_k0_rollout, rollout_operands, split_controls)

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--card", choices=("normal", "exact"), required=True)
    p.add_argument("--steps", type=int, default=6, help="even N >= 4")
    p.add_argument("--schedule", default="uniform")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.steps < 4 or args.steps % 2:
        raise SystemExit("--steps must be even and >= 4")
    if not jax.config.jax_enable_x64:
        raise SystemExit("needs JAX_ENABLE_X64=1")
    stamp = worktree_stamp(repo=REPO_ROOT)
    card, model, controls = build(args.card, args.steps)
    ref = eager(card, model, controls, card.ocean.recipe.initial_state)
    start = carry_from_result(ref[1])                       # after kt = 2
    split = split_controls(controls[2:], kt0=3)
    kinds, values = split.kinds, rollout_operands(split)
    rollout = make_k0_rollout(card, model, split, schedule=args.schedule)
    looped = jax.jit(rollout)(start, values)
    final = carry_from_result(ref[-1])
    diffs = max_diffs(final, looped)
    worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:8]

    z = card.ocean.recipe.z_coord
    w = (jnp.asarray(model.grid.area)[..., None] * jnp.asarray(z.h_partial)
         * jnp.asarray(z.is_active, dtype=jnp.float64))

    def loss(dT, vals, c0):
        st = c0.state
        c = c0._replace(state=st._replace(T=st.T.replace(data=st.T.data + dT)))
        out = rollout(c, vals).state
        return jnp.sum(w * out.T.data ** 2) / jnp.sum(w)

    g = np.asarray(jax.jit(jax.grad(loss))(jnp.zeros_like(start.state.T.data), values, start))
    result = {
        "card": args.card, "steps": args.steps, "looped_steps": args.steps - 2,
        "schedule": args.schedule, "kinds": {k: kinds.count(k) for k in set(kinds)},
        "bitwise_equal_leaves": sum(v == 0.0 for v in diffs.values()),
        "leaves": len(diffs), "worst_leaves": worst,
        "grad_nonfinite": int((~np.isfinite(g)).sum()),
        "grad_l2": float(np.sqrt(np.nansum(g * g))),
        "provenance": stamp, "backend": jax.default_backend(),
    }
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("card", "bitwise_equal_leaves", "leaves",
                                             "worst_leaves", "grad_nonfinite", "grad_l2")},
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
