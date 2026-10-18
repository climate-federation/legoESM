#!/usr/bin/env python3
"""Round 316: one fused multi-step gradient of an ORCA2 card (measurement only).

ONE ``jax.jit(jax.grad(L o step^N))`` with respect to the initial T field:
step 1 (the cold-start state structure) followed by a ``lax.scan`` of
per-step ``jax.checkpoint`` bodies over steps 2..N (one compiled body, so
the reverse executable stays under the per-process memory-map limit).
Checked against forward mode over the SAME N-step function along the round-315
direction T_dir0, and the forward value against the production trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
    nemo_testcase_l4_orca2_round315_nan_locate as locate,
)


def fused(rung: str, n: int, losses_wanted: tuple[str, ...]) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    probe.require(bool(jax.config.jax_enable_x64) and jax.default_backend() == "cpu",
                  "needs x64 on CPU")
    card, state0, forcing = probe.build(rung)
    n = min(n, len(forcing))
    r = card.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                  iwm_forcing=r.iwm_forcing)
    losses, tmask = probe.make_losses(model, r.z_coord)
    raw = type(model).__dict__["_step_jitted"].__wrapped__
    dt = card.dt_s

    def body(state, f):
        return raw(model, model._seed_tke_preclosure_carry(state), dt, *f)

    rest = jax.tree.map(lambda *xs: jnp.stack(xs), *forcing[1:n]) if n > 1 else None

    def run(t0):
        st = body(probe._with_t(state0, t0), forcing[0])
        if n > 1:
            st, _ = jax.lax.scan(lambda s, f: (jax.checkpoint(body)(s, f), None), st, rest)
        return st

    # control: the fused forward equals the production trajectory, every leaf
    st = state0
    for k in range(n):
        st = model.step(st, dt, freshwater=forcing[k][0], surface_forcing=forcing[k][1])
    mine = jax.jit(run)(state0.T.data)
    pairs = zip(jax.tree.leaves(st), jax.tree.leaves(mine))
    same = all(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) for a, b in pairs)
    d = locate.probe_direction(state0, tmask)
    out = {"rung": rung, "n": n, "worktree": worktree_stamp(allow_dirty=True),
           "forward_equals_production_every_leaf": same, "losses": {}}
    for name in losses_wanted:
        f = lambda t, name=name: losses[name](run(t))  # noqa: E731
        t0 = time.perf_counter()
        g = jax.jit(jax.grad(f))(state0.T.data)
        jax.block_until_ready(g)
        t_rev = time.perf_counter() - t0
        t0 = time.perf_counter()
        fwd = float(jax.jit(lambda t, v: jax.jvp(f, (t,), (v,))[1])(state0.T.data, d))
        t_fwd = time.perf_counter() - t0
        g = np.asarray(g)
        rev = float(np.vdot(g, np.asarray(d)))
        out["losses"][name] = {
            "grad_finite": bool(np.all(np.isfinite(g))),
            "nonfinite_entries": int(np.sum(~np.isfinite(g))),
            "grad_l2": float(np.linalg.norm(np.nan_to_num(g))),
            "rev_dot_dir0": rev, "fwd_dir0": fwd,
            "rev_vs_fwd_rel": abs(rev - fwd) / max(abs(fwd), 1e-300),
            "compile_plus_run_s": {"reverse": round(t_rev, 1), "forward": round(t_fwd, 1)}}
        print(name, out["losses"][name], flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=tuple(probe.ROOTS), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--losses", nargs="+", default=["T", "ssh"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fused(args.rung, args.steps, tuple(args.losses))
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
