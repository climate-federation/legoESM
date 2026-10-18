#!/usr/bin/env python3
"""Round 315: name the first statement that makes a one-step cotangent non-finite.

MEASUREMENT ONLY.  Pulls ``d L_T / d (state leaf)`` back through ONE production
step (the round-315 probe's traced step at s = 1) with ``jax_debug_nans`` or
``jax_debug_infs`` on: JAX re-runs the failing computation op by op and raises
at the first primitive whose output is non-finite, with the source line of the
forward statement that primitive belongs to (a transpose carries its forward
statement's source location).  The repository frames of that traceback are
written out; the forward value is never changed.

``--backward`` (resume of round 315): the debug flag above stops at the
FORWARD 0/0 (a forward residual), not where the reverse pass first goes
non-finite.  ``BackwardWalker`` instead evaluates the transpose jaxpr of one
step's pullback equation by equation (recursing into jit, remat, scan, while,
cond) and records every equation whose cotangent output is non-finite while
its cotangent inputs are finite: class ``generated`` (all inputs finite, e.g.
0/0 on a masked lane) or ``residual`` (a non-finite FORWARD residual meets the
cotangent, e.g. 0 * NaN).  With ``sanitize`` each recorded output is zeroed so
later, independent sources are not hidden behind the first.  A cotangent input
passed as ``ZERO`` is a symbolic zero, propagated through every equation that
is linear in it, which reproduces a fused ``jax.grad`` (JAX prunes zero
cotangents) from the same pullback the hand-chained probe uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
)

ZERO = object()  # symbolic-zero cotangent
_CALLS = ("jit", "pjit", "closed_call", "core_call", "remat2", "checkpoint",
          "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr")


def _open(j):
    from jax._src import core

    return (j.jaxpr, list(j.consts)) if isinstance(j, core.ClosedJaxpr) else (j, [])


def _inexact(x) -> bool:
    import jax.numpy as jnp

    return x is not ZERO and jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)


def _bad(x) -> int:
    return int(np.count_nonzero(~np.isfinite(np.asarray(x)))) if _inexact(x) else 0


def _taint_fixpoint(jaxpr, const_t, carry_t, x_t, n_carry_out):
    """Loop-carried taint: a carry is cotangent-dependent once any path makes it so."""

    carry_t = list(carry_t)
    while True:
        t = {v: tv for v, tv in zip(jaxpr.invars, list(const_t) + carry_t + list(x_t))}
        for e in jaxpr.eqns:
            o = any(t.get(v, False) for v in e.invars if not hasattr(v, "val"))
            for v in e.outvars:
                t[v] = o
        out = [t.get(v, False) if not hasattr(v, "val") else False
               for v in jaxpr.outvars[:n_carry_out]]
        new = [a or b for a, b in zip(carry_t, out)]
        if new == carry_t:
            return carry_t
        carry_t = new


class BackwardWalker:
    def __init__(self, sanitize: bool):
        self.sanitize, self.events, self.order = sanitize, {}, 0

    def run(self, jaxpr, consts, args, taints, frames=()):
        from jax._src import core

        env, tenv = {}, {}

        def read(v):
            return v.val if isinstance(v, core.Literal) else env[v]

        def rt(v):
            return False if isinstance(v, core.Literal) else tenv[v]

        for v, c in zip(jaxpr.constvars, consts):
            env[v], tenv[v] = c, False
        for v, a, t in zip(jaxpr.invars, args, taints):
            env[v], tenv[v] = a, t
        for e in jaxpr.eqns:
            ins, tin = [read(v) for v in e.invars], [rt(v) for v in e.invars]
            outs = self._eqn(e, ins, tin, self._frames(e, frames))
            for v, o in zip(e.outvars, outs):
                env[v], tenv[v] = o, any(tin)
        return [read(v) for v in jaxpr.outvars]

    @staticmethod
    def _frames(e, parent):
        from jax._src import source_info_util

        tb = e.source_info.traceback
        fr = [f"{f.file_name.split('/packages/')[1]}:{f.start_line} {f.function_name}"
              for f in (source_info_util.user_frames(tb) if tb else ())
              if "/packages/" in f.file_name]
        return tuple(fr) or tuple(parent)

    def _eqn(self, e, ins, tin, fr):
        import jax.numpy as jnp

        name, p = e.primitive.name, e.params
        if any(tin) and all(x is ZERO for x, t in zip(ins, tin) if t):
            return [ZERO] * len(e.outvars)  # linear in a symbolic-zero cotangent
        if name in _CALLS:
            j, c = _open(p.get("jaxpr") or p.get("call_jaxpr") or p.get("fun_jaxpr"))
            return self.run(j, c, ins, tin, fr)
        ins = [jnp.zeros(v.aval.shape, v.aval.dtype) if x is ZERO else x
               for x, v in zip(ins, e.invars)]
        if name == "scan":
            return self._scan(p, ins, tin, fr)
        if name == "while":
            return self._while(p, ins, tin, fr)
        if name == "cond":
            j, c = _open(p["branches"][int(ins[0])])
            return self.run(j, c, ins[1:], tin[1:], fr)
        outs = e.primitive.bind(*ins, **p)
        outs = list(outs) if e.primitive.multiple_results else [outs]
        self.order += 1
        if any(tin) and any(_bad(o) for o in outs):
            bad_in = [(i, "ct" if tin[i] else "res", _bad(x)) for i, x in enumerate(ins) if _bad(x)]
            if not any(kind == "ct" for _, kind, _ in bad_in):
                self._record(name, fr, "residual" if bad_in else "generated", outs, bad_in)
                if self.sanitize:
                    outs = [jnp.where(jnp.isfinite(o), o, 0.0) if _inexact(o) else o
                            for o in outs]
        return outs

    def _record(self, name, fr, cls, outs, bad_in):
        key = (name, fr[0] if fr else "?", cls)
        ev = self.events.get(key)
        if ev is None:
            o = np.concatenate([np.ravel(np.asarray(x)) for x in outs if _inexact(x)])
            ev = self.events[key] = {
                "primitive": name, "class": cls, "first_order": self.order,
                "frames": list(fr[:8]), "count": 0,
                "first_nan": int(np.count_nonzero(np.isnan(o))),
                "first_inf": int(np.count_nonzero(np.isinf(o))),
                "first_nonfinite_inputs": bad_in}
        ev["count"] += 1

    def _scan(self, p, ins, tin, fr):
        import jax.numpy as jnp

        nc, nk, n = p["num_consts"], p["num_carry"], p["length"]
        j, c = _open(p["jaxpr"])
        consts, carry, xs = ins[:nc], ins[nc:nc + nk], ins[nc + nk:]
        ct = _taint_fixpoint(j, tin[:nc], tin[nc:nc + nk], tin[nc + nk:], nk)
        taints = list(tin[:nc]) + ct + list(tin[nc + nk:])
        ys = [None] * n
        for i in range(n):
            k = n - 1 - i if p["reverse"] else i
            out = self.run(j, c, consts + carry + [x[k] for x in xs], taints, fr)
            out = [jnp.zeros(v.aval.shape, v.aval.dtype) if o is ZERO else o
                   for o, v in zip(out, j.outvars)]
            carry, ys[k] = out[:nk], out[nk:]
        stacked = [jnp.stack([y[m] for y in ys]) for m in range(len(j.outvars) - nk)]
        return carry + stacked

    def _while(self, p, ins, tin, fr):
        cn, bn = p["cond_nconsts"], p["body_nconsts"]
        cj, cc = _open(p["cond_jaxpr"])
        bj, bc = _open(p["body_jaxpr"])
        cconsts, bconsts, carry = ins[:cn], ins[cn:cn + bn], ins[cn + bn:]
        ct = _taint_fixpoint(bj, tin[cn:cn + bn], tin[cn + bn:], (), len(carry))
        while bool(self.run(cj, cc, cconsts + carry, list(tin[:cn]) + ct, fr)[0]):
            carry = self.run(bj, bc, bconsts + carry, list(tin[cn:cn + bn]) + ct, fr)
        return carry


def walk_pullback(vjp_fn, ct, symbolic_zero_paths=(), sanitize=False):
    """Run ``vjp_fn(ct)`` through ``BackwardWalker``; ct leaves whose keystr is
    in ``symbolic_zero_paths`` enter as symbolic zeros.  Returns (cotangent
    pytree with ZERO instantiated, ordered event list)."""

    import jax
    import jax.numpy as jnp

    closed = jax.make_jaxpr(vjp_fn)(ct)
    leaves = jax.tree_util.tree_leaves_with_path(ct)
    args = [ZERO if jax.tree_util.keystr(k) in symbolic_zero_paths else x for k, x in leaves]
    walker = BackwardWalker(sanitize)
    outs = walker.run(closed.jaxpr, list(closed.consts), args, [True] * len(args))
    shapes = jax.eval_shape(vjp_fn, ct)
    flat, tree = jax.tree_util.tree_flatten(shapes)
    outs = [jnp.zeros(s.shape, s.dtype) if o is ZERO else o for o, s in zip(outs, flat)]
    events = sorted(walker.events.values(), key=lambda ev: ev["first_order"])
    return jax.tree_util.tree_unflatten(tree, outs), events


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


LOSS_LEAF = {"T": ".T", "ssh": ".eta"}


def probe_direction(state0, tmask):
    """The probe's T_dir0 (first draw of default_rng(315), unit RMS on wet cells)."""

    import jax.numpy as jnp

    wet = np.asarray(tmask) > 0.5
    d = np.random.default_rng(315).standard_normal(np.shape(state0.T.data)) * np.asarray(tmask)
    return jnp.asarray(d / np.sqrt(np.mean(d[wet] ** 2)))


def backward_sources(rung: str, output: Path, compare_jit: bool = True) -> dict:
    """One step: every backward non-finite source, hand-chained (dense) and fused
    (symbolic-zero) cotangents, raw and sanitized; controls in the docstring of
    the module."""

    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    card, state0, forcing = probe.build(rung)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config,
                                  iwm_forcing=card.recipe.iwm_forcing)
    fw, sf = forcing[0]
    losses, tmask = probe.make_losses(model, card.recipe.z_coord)
    step = probe.traced_step(model)
    trace = probe.traceability(model, state0, forcing, card.dt_s)
    ones = {k: jnp.float64(1.0) for k in probe.SCALARS if trace[k]["traceable"]}
    out = {"rung": rung, "traced_scales": sorted(ones), "worktree": probe_stamp(),
           "runs": {}, "log": []}
    clock = time.perf_counter()

    def save(phase):
        out["log"].append({"phase": phase, "t_s": round(time.perf_counter() - clock, 1)})
        print(f"[{time.perf_counter() - clock:8.1f}s] {phase}", flush=True)
        output.write_text(json.dumps(out, indent=1, default=str) + "\n")

    st1, vjp_fn = jax.vjp(lambda st, sc: step(st, sc, card.dt_s, fw, sf), state0, ones)
    jax.block_until_ready(st1)
    save("forward vjp")
    d = probe_direction(state0, tmask)
    paths = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_leaves_with_path(st1)]
    for name in ("T", "ssh"):
        ct = jax.vjp(losses[name], st1)[1](jnp.float64(1.0))[0]
        ct_leaf = [q for q in paths
                   if q.startswith(LOSS_LEAF[name] + "[") or q == LOSS_LEAF[name]]
        if compare_jit:
            ref = vjp_fn(ct)
            save(f"{name}: jitted pullback")
        for mode in ("chained", "fused"):
            zeros = () if mode == "chained" else tuple(q for q in paths if q not in ct_leaf)
            for sanitize in (False, True):
                g, events = walk_pullback(vjp_fn, ct, zeros, sanitize)
                leaves = jax.tree_util.tree_leaves_with_path(g)
                row = {
                    "symbolic_zero_ct_leaves": len(zeros),
                    "nonfinite_cotangent_leaves": {
                        jax.tree_util.keystr(k): _bad(x) for k, x in leaves if _bad(x)},
                    "dT0_l2": float(np.linalg.norm(
                        np.nan_to_num(np.asarray(g[0].T.data), nan=0.0))),
                    "dT0_finite": bool(np.all(np.isfinite(np.asarray(g[0].T.data)))),
                    "dT0_dot_dir0": float(jnp.vdot(g[0].T.data, d)),
                    "dscales": {k: float(v) for k, v in g[1].items()},
                    "events": events}
                if compare_jit and mode == "chained" and not sanitize:
                    # control: the walker reproduces the compiled pullback
                    pairs = [(np.asarray(a), np.asarray(b)) for a, b in zip(
                        jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(ref))
                        if _inexact(a)]
                    rel = [float(np.max(np.abs(a - b)[np.isfinite(b)])
                                 / max(float(np.max(np.abs(b)[np.isfinite(b)])), 1e-300))
                           for a, b in pairs if np.isfinite(b).any()]
                    row["control_vs_jit"] = {
                        "finite_mask_equal_all": all(np.array_equal(np.isfinite(a), np.isfinite(b))
                                                     for a, b in pairs),
                        "max_rel_diff_finite": max(rel) if rel else None}
                out["runs"][f"{name}/{mode}/{'sanitized' if sanitize else 'raw'}"] = row
                save(f"{name}/{mode}/{'sanitized' if sanitize else 'raw'}: "
                     f"{len(events)} sources, nonfinite leaves "
                     f"{len(row['nonfinite_cotangent_leaves'])}")
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
    parser.add_argument("--backward", action="store_true",
                        help="walk one step's pullback (BackwardWalker) instead")
    args = parser.parse_args()
    if args.backward:
        backward_sources(args.rung, args.output)
        print("STATUS WALKED")
        return 0
    result = locate(args.rung, args.leaf, args.mode)
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"STATUS LOCATED raised={result.get('raised')} "
          f"nonfinite={result['nonfinite_count_jit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
