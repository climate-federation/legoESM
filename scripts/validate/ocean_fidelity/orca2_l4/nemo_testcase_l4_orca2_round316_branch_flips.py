#!/usr/bin/env python3
"""Round 316: count branch flips between x+eps*d and x-eps*d (measurement only).

Evaluates ONE production step's jaxpr equation by equation (round 315's
walker, no cotangent) at T0 + eps*T_dir0 and T0 - eps*T_dir0, and records the
predicate of every branch-like primitive: comparisons (lt/gt/le/ge/eq/ne),
max/min (which operand wins), sign, floor/ceil/round, clamp.  Each occurrence
(scan iterations included) is keyed by (equation, call count).  A flip = a
lane whose predicate differs between the two runs.  Also writes the central
FD of the T and SSH losses from the compiled step and forward mode, so the
FD miss and the flip count are reported at the same eps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

COMPARE = {"lt", "gt", "le", "ge", "eq", "ne"}
ROUNDING = {"sign", "floor", "ceil", "round"}
MAXMIN = {"max", "min"}


def _predicate(name, ins, outs):
    import jax.numpy as jnp
    if name in COMPARE:
        return outs[0]
    if name in MAXMIN:
        return jnp.asarray(ins[0]) >= jnp.asarray(ins[1])
    if name in ROUNDING:
        return outs[0]
    if name == "cond":  # the branch index itself
        return jnp.asarray(ins[0])
    if name == "clamp":  # clamp(lo, x, hi): which side binds
        return jnp.stack([jnp.asarray(ins[1]) < ins[0], jnp.asarray(ins[1]) > ins[2]])
    return None


class PredicateRecorder(locate.BackwardWalker):
    def __init__(self):
        super().__init__(sanitize=False)
        self.calls, self.preds, self.frames = Counter(), {}, {}

    def _eqn(self, e, ins, tin, fr):
        outs = super()._eqn(e, ins, tin, fr)
        name = e.primitive.name
        pred = _predicate(name, ins, outs) if (
            name in COMPARE | ROUNDING | MAXMIN | {"clamp", "cond"}) else None
        if pred is not None:
            k = id(e)
            n = self.calls[k]
            self.calls[k] += 1
            pred = np.asarray(pred)
            # booleans packed 8x (a full-step record is otherwise tens of GB)
            self.preds[(k, n)] = (np.packbits(pred) if pred.dtype == bool else pred, pred.size)
            self.frames.setdefault(k, (name, fr[0] if fr else "?"))
        return outs


def record(jaxpr, t):
    w = PredicateRecorder()
    w.run(jaxpr.jaxpr, list(jaxpr.consts), [t], [False])
    return w


def flips(rung: str, eps: float) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card, state0, forcing = probe.build(rung)
    r = card.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                  iwm_forcing=r.iwm_forcing)
    losses, tmask = probe.make_losses(model, r.z_coord)
    raw = type(model).__dict__["_step_jitted"].__wrapped__
    fw, sf = forcing[0]

    def step_t(t):
        s = probe._with_t(state0, t)
        return raw(model, model._seed_tke_preclosure_carry(s), card.dt_s, fw, sf)

    d = locate.probe_direction(state0, tmask)
    t0 = state0.T.data
    out = {"rung": rung, "eps": eps, "worktree": worktree_stamp(allow_dirty=True), "losses": {}}
    for name in ("T", "ssh"):
        f = jax.jit(lambda t, name=name: losses[name](step_t(t)))
        fd = (float(f(t0 + eps * d)) - float(f(t0 - eps * d))) / (2 * eps)
        fwd = float(jax.jit(lambda t, v, name=name: jax.jvp(
            lambda x: losses[name](step_t(x)), (t,), (v,))[1])(t0, d))
        out["losses"][name] = {"fd": fd, "fwd": fwd, "rel_err": abs(fd - fwd) / abs(fwd)}
    closed = jax.make_jaxpr(step_t)(t0)
    plus = record(closed, t0 + eps * d)
    minus = record(closed, t0 - eps * d)
    by_stmt, total, n_keys = Counter(), 0, 0
    for key, p in plus.preds.items():
        q = minus.preds.get(key)
        if q is None:  # a cond took the other branch: count the whole record
            n = p[1]
        elif p[0].dtype == np.uint8 and q[0].dtype == np.uint8:
            n = int(np.unpackbits(p[0] ^ q[0]).sum())
        else:
            n = int(np.count_nonzero(p[0] != q[0]))
        n_keys += 1
        if n:
            total += n
            by_stmt["%s %s" % plus.frames[key[0]]] += n
    missing = len(set(minus.preds) - set(plus.preds))
    out.update(predicate_occurrences=n_keys, occurrences_only_in_minus=missing, flipped_lanes=total,
               statements_with_flips=len(by_stmt), top_statements=by_stmt.most_common(25))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=tuple(probe.ROOTS), required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(flips(args.rung, args.eps), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
