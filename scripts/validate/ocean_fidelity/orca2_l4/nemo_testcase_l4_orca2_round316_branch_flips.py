#!/usr/bin/env python3
"""Round 316: count branch flips between x+eps*d and x-eps*d (measurement only).

Evaluates ONE production step's jaxpr equation by equation on T0 + eps*T_dir0
and T0 - eps*T_dir0 in lockstep, and compares the decision of every
branch-like primitive: comparisons (lt/gt/le/ge/eq/ne), max/min (which
operand wins), abs/sign, floor/ceil/round, clamp, cond index, while guard.
A flip = a lane whose decision differs between the two evaluations.  Also writes the central
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
KINKS = {"max", "min", "abs", "sign", "floor", "ceil", "round", "clamp"}


def _pred(name, ins, out):
    """The branch decision an equation took (a boolean or integer array)."""
    import jax.numpy as jnp
    if name in COMPARE or name in ("sign", "floor", "ceil", "round"):
        return out
    if name in ("max", "min"):
        return jnp.asarray(ins[0]) >= jnp.asarray(ins[1])
    if name == "abs":
        return jnp.asarray(ins[0]) >= 0
    if name == "clamp":
        return (jnp.asarray(ins[1]) < ins[0]) | (jnp.asarray(ins[1]) > ins[2])
    return None


class Lockstep:
    """Evaluate a jaxpr on two inputs at once; count lanes whose branch
    decision differs.  Values are freed after their last use."""

    def __init__(self):
        self.by_stmt, self.occurrences, self.flipped = Counter(), 0, 0

    def run(self, jaxpr, consts, args, frames=()):
        from jax._src import core
        last = {}
        for i, e in enumerate(jaxpr.eqns):
            for v in e.invars:
                if not isinstance(v, core.Literal):
                    last[v] = i
        keep = {v for v in jaxpr.outvars if not isinstance(v, core.Literal)}
        env = dict(zip(jaxpr.constvars, [(c, c) for c in consts]))
        env.update(zip(jaxpr.invars, args))

        def read(v):
            return (v.val, v.val) if isinstance(v, core.Literal) else env[v]

        for i, e in enumerate(jaxpr.eqns):
            ins = [read(v) for v in e.invars]
            fr = locate.BackwardWalker._frames(e, frames)
            for v, o in zip(e.outvars, self.eqn(e, ins, fr)):
                env[v] = o
            for v in e.invars:
                if (not isinstance(v, core.Literal) and last.get(v) == i
                        and v not in keep):
                    env.pop(v, None)
        return [read(v) for v in jaxpr.outvars]

    def _count(self, name, fr, pa, pb):
        n = int(np.count_nonzero(np.asarray(pa) != np.asarray(pb)))
        self.occurrences += 1
        if n:
            self.flipped += n
            self.by_stmt[f"{name} {fr[0] if fr else '?'}"] += n

    def eqn(self, e, ins, fr):
        name, p = e.primitive.name, e.params
        if name in locate._CALLS:
            j, c = locate._open(p.get("jaxpr") or p.get("call_jaxpr") or p.get("fun_jaxpr"))
            return self.run(j, c, ins, fr)
        if name == "scan":
            return self._scan(p, ins, fr)
        if name == "while":
            return self._while(p, ins, fr)
        if name == "cond":
            ia, ib = int(ins[0][0]), int(ins[0][1])
            self._count("cond", fr, ia, ib)
            outs = []
            for side, idx in ((0, ia), (1, ib)):
                j, c = locate._open(p["branches"][idx])
                outs.append(self.run(j, c, [(x[side], x[side]) for x in ins[1:]], fr))
            return [(a[0], b[1]) for a, b in zip(*outs)]
        res = []
        for side in (0, 1):
            o = e.primitive.bind(*[x[side] for x in ins], **p)
            res.append(list(o) if e.primitive.multiple_results else [o])
        if name in COMPARE or name in KINKS:
            self._count(name, fr, _pred(name, [x[0] for x in ins], res[0][0]),
                        _pred(name, [x[1] for x in ins], res[1][0]))
        return list(zip(res[0], res[1]))

    def _scan(self, p, ins, fr):
        import jax.numpy as jnp
        nc, nk, n = p["num_consts"], p["num_carry"], p["length"]
        j, c = locate._open(p["jaxpr"])
        consts, carry, xs = ins[:nc], ins[nc:nc + nk], ins[nc + nk:]
        ys = [None] * n
        for i in range(n):
            k = n - 1 - i if p["reverse"] else i
            out = self.run(j, c, consts + carry + [(x[0][k], x[1][k]) for x in xs], fr)
            carry, ys[k] = out[:nk], out[nk:]
        return carry + [(jnp.stack([y[m][0] for y in ys]), jnp.stack([y[m][1] for y in ys]))
                        for m in range(len(j.outvars) - nk)]

    def _while(self, p, ins, fr):
        cn, bn = p["cond_nconsts"], p["body_nconsts"]
        cj, cc = locate._open(p["cond_jaxpr"])
        bj, bc = locate._open(p["body_jaxpr"])
        cconsts, bconsts, carry = ins[:cn], ins[cn:cn + bn], ins[cn + bn:]
        while True:
            ga, gb = self.run(cj, cc, cconsts + carry, fr)[0]
            ga, gb = bool(ga), bool(gb)
            self._count("while", fr, ga, gb)
            if not (ga or gb):
                return carry
            new = self.run(bj, bc, bconsts + carry, fr)
            carry = [(nv[0] if ga else ov[0], nv[1] if gb else ov[1])
                     for nv, ov in zip(new, carry)]


def flips(rung: str, eps: float) -> dict:
    import jax
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
    w = Lockstep()
    got = w.run(closed.jaxpr, list(closed.consts), [(t0 + eps * d, t0 - eps * d)])
    # control: the interpreter reproduces the compiled step on both sides
    ref = [jax.tree.leaves(jax.jit(step_t)(t0 + s * eps * d)) for s in (1, -1)]
    out["interp_vs_jit_max_abs"] = max(
        float(np.nanmax(np.abs(np.asarray(g[side], np.float64) - np.asarray(r, np.float64))))
        if np.asarray(r).size else 0.0
        for side in (0, 1) for g, r in zip(got, ref[side]))
    out.update(predicate_occurrences=w.occurrences, flipped_lanes=w.flipped,
               statements_with_flips=len(w.by_stmt), top_statements=w.by_stmt.most_common(25))
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
