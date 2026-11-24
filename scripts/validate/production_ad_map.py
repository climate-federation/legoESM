#!/usr/bin/env python
"""Gradient map of the PRODUCTION AMIP model, component by component (review F9).

Builds the production deck (``config/amip/amip_production.yaml``: MPAS L32,
CAM6 suite, two-leaf multilayer land) through run_amip's own argument path at
a SMALL mesh, exports the run loop's step closures
(``ModelDriver.export_mpas_step``) and measures, per stage:

  components  each CAM6 module exactly as the combined physics calls it
              (radiation, ZM convection, CLUBB, Morrison, McFarlane GWD),
              recorded by spies on the module factories during one eager
              production physics call
  physics     the combined CAM6 physics (macmic loop, production cadence)
  params      the same physics REBUILT by the run loop's own builder from the
              production config with the active schemes' tier-1 (core) spec'd
              parameters spliced in as traced leaves (raw values at the spec
              defaults): per-parameter |dL/dtheta| and one FD check
  dycore      one MPAS dynamics step (step_pure, no physics)
  step        mpas_segment: N production steps (dycore + physics + post-step)
  land        the two-leaf multilayer land step (inputs: the marshalled
              atmosphere-to-surface forcing of the first step)
  eager       eager run loop vs mpas_segment over N steps (forward max|diff|;
              with land + ice skin on the segment holds their feedback)

Each gradient check: a scalar loss L = sum_leaf <w_leaf, out_leaf> with a fixed
random projection w (a norm would vanish where the output vanishes), the
reverse-mode directional derivative <dL/dx, v> against a central difference
(L(x+hv) - L(x-hv)) / 2h along a fixed random direction v.  Reported:
finite?, |grad|max, AD, FD, FD/AD.  x = (T, q_v) of the state unless noted.

Usage (sbatch; needs the Ginsburg forcing files, see amip_production.ginsburg.sh):
  production_ad_map.py --stage components -- <run_amip path flags>
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DECK = REPO / "config" / "amip" / "amip_production.yaml"


def _run_amip():
    spec = importlib.util.spec_from_file_location(
        "run_amip_ad_map", REPO / "scripts" / "run" / "run_amip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_driver(argv, resolution, n_steps):
    """run_amip.main's argument path (deck -> argv -> postprocess -> config),
    then a ModelDriver with ``days`` set to ``n_steps`` model steps."""
    ra = _run_amip()
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = ra.build_arg_parser()
    keys = load_yaml_config(str(DECK), parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    argv = list(argv) + ["--resolution", str(resolution)]
    args = parser.parse_args(argv)
    ra._resolve_land_model(args, parser, argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    args = ra._apply_aimip_classical_overrides(args)
    ra._require_full_physics_for_amip(args, parser)
    config = ra.build_config_from_args(args)
    dt = config.dycore.dt
    # +0.5 step: int(days*86400/dt) must not round 16 down to 15
    config = config._replace(days=(n_steps + 0.5) * dt / 86400.0)
    driver = ModelDriver(config)
    driver.setup()
    return driver


def _proj_loss(out, seed=0):
    leaves = [x for x in jax.tree_util.tree_leaves(out)
              if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact)]
    key = jax.random.PRNGKey(seed)
    tot = 0.0
    for i, x in enumerate(leaves):
        w = jax.random.normal(jax.random.fold_in(key, i), x.shape, x.dtype)
        tot = tot + jnp.sum(w * x)
    return tot


H_SWEEP = (1.0,)          # multiples of --h; --h-sweep widens it
RAD_FP64 = False          # --rad-fp64


def nonfinite_census(g, batched=False):
    """Census, not verdict: which leaf, how many entries, which columns carry
    a non-finite cotangent (columns = axis 0, or axis 1 with ``batched``,
    where axis 0 is the window and the windows hit are listed too)."""
    for path, a in jax.tree_util.tree_leaves_with_path(g):
        bad = np.asarray(~jnp.isfinite(a))
        if not bad.any():
            continue
        idx = np.nonzero(bad)
        cols = np.unique(idx[1 if batched else 0])
        win = (f" windows={np.unique(idx[0]).tolist()}" if batched else "")
        print(f"NONFINITE {jax.tree_util.keystr(path)} shape={a.shape} "
              f"count={int(bad.sum())}/{bad.size}{win} ncols={cols.size} "
              f"cols={cols[:12].tolist()}", flush=True)


def grad_check(f, x, v, h, name):
    """f: pytree -> scalar.  AD directional derivative vs central FD (at each
    step in ``H_SWEEP * h``: an FD that converges to AD as h shrinks is a
    smooth map; one that wanders is a kink / sampling noise, not a severed
    gradient)."""
    t0 = time.time()
    # jit both: op-by-op dispatch of a full physics graph maps one executable
    # per op and exhausts LLVM's code-section memory on CPU
    f = jax.jit(f)
    g = jax.jit(jax.grad(f))(x)
    t_ad = time.time() - t0
    gl = jax.tree_util.tree_leaves(g)
    finite = all(bool(jnp.all(jnp.isfinite(a))) for a in gl)
    if not finite:
        nonfinite_census(g)
    gmax = max(float(jnp.max(jnp.abs(a))) for a in gl)
    ad = float(sum(jnp.vdot(a, b) for a, b in zip(gl, jax.tree_util.tree_leaves(v))))
    fds = {}
    for m in H_SWEEP:
        hh = h * m
        xp = jax.tree_util.tree_map(lambda a, b, _h=hh: a + _h * b, x, v)
        xm = jax.tree_util.tree_map(lambda a, b, _h=hh: a - _h * b, x, v)
        fds[f"{hh:.0e}"] = float((f(xp) - f(xm)) / (2.0 * hh))
    fd = fds[f"{h:.0e}"]
    rec = dict(name=name, finite=finite, grad_absmax=gmax, ad=ad, fd=fd,
               ratio=(fd / ad if ad != 0.0 else float("nan")), t_grad_s=t_ad)
    if len(fds) > 1:
        rec["fd_sweep"] = fds
    print("GRADCHECK " + json.dumps(rec), flush=True)
    return rec


_X_TRACERS = ("q_v",)     # --x-tracers: tracers differentiated alongside T


def _state_x(state):
    x = {"T": state.T.data}
    for k in _X_TRACERS:
        x[k] = state.tracers[k].data
    return x


def _with_x(state, x):
    tr = dict(state.tracers)
    for k in _X_TRACERS:
        tr[k] = tr[k].replace(data=x[k])
    return state._replace(T=state.T.replace(data=x["T"]), tracers=tr)


def make_grad_stash(n_steps, example, step0=0):
    """Per-step adjoint recorder for ``mpas_segment(tap=...)``: an identity
    ``custom_vjp`` whose backward pass writes max|cotangent| of every inexact
    leaf of ``example`` (the ``(state, phys_state)`` pytree) into row ``step``
    of a ``jax.ref`` (the JAX docs' ``stash_grads`` pattern).  Row k is
    d loss / d(state entering window step k, i.e. absolute step step0 + k),
    so a jump between rows k+1 and k lies inside that step.  Returns ``(tap, log_ref, leaf_names)``; read the ref
    after the gradient ran (``log_ref[...]``)."""
    paths, _ = jax.tree_util.tree_flatten_with_path(example)
    names = [jax.tree_util.keystr(kp) for kp, _ in paths]
    shapes = [tuple(np.shape(x)) for _, x in paths]
    n = len(names)
    rec = dict(log=jax.new_ref(jnp.full((n_steps, n), jnp.nan, jnp.float64)),
               loc=jax.new_ref(jnp.zeros((n_steps, n), jnp.int64)),
               val=jax.new_ref(jnp.full((n_steps, n), jnp.nan, jnp.float64)),
               shapes=shapes)

    @jax.custom_vjp
    def stash(step, s, p):
        return s, p

    def _fwd(step, s, p):
        # the primal leaves ride as residuals so the backward pass can read
        # the state value at the cotangent's argmax (where the derivative blew)
        return (s, p), (step, jax.tree_util.tree_leaves((s, p)))

    def _bwd(res, g):
        step, prim = res
        row, loc, val = [], [], []
        for x, x0 in zip(jax.tree_util.tree_leaves(g), prim):
            if jnp.issubdtype(x.dtype, jnp.inexact):
                a = jnp.abs(x).ravel()
                i = jnp.argmax(a)
                row.append(a[i].astype(jnp.float64))
                loc.append(i.astype(jnp.int64))
                val.append(jnp.ravel(x0)[i].astype(jnp.float64))
            else:
                row.append(jnp.nan); loc.append(jnp.int64(0)); val.append(jnp.nan)
        k = step - step0
        rec["log"][k] = jnp.stack(row)
        rec["loc"][k] = jnp.stack(loc)
        rec["val"][k] = jnp.stack(val)
        return None, g[0], g[1]
    stash.defvjp(_fwd, _bwd)
    return stash, rec, names


def print_grad_stash(rec, names, label):
    """One ``ADJOINT`` line per step: largest leaf, its (cell, level) location
    and the primal value there, then T / q_v / u; and the step whose reverse
    pass grew the adjoint most (``ADJOINT_JUMP``)."""
    rows = np.asarray(rec["log"][...])
    locs = np.asarray(rec["loc"][...])
    vals = np.asarray(rec["val"][...])
    with np.errstate(all="ignore"):
        tot = np.array([np.nanmax(r) if np.any(np.isfinite(r)) else np.nan
                        for r in rows])
    pick = {k: next((i for i, n in enumerate(names) if n.endswith(k)), None)
            for k in ("['T'].data", "['q_v'].data", "['u'].data")}
    for k in range(rows.shape[0]):
        if not np.any(np.isfinite(rows[k])):
            print(f"ADJOINT {label} step={k} no cotangent recorded", flush=True)
            continue
        top = int(np.nanargmax(rows[k]))
        extra = " ".join(f"{key.split(chr(39))[1]}={rows[k, i]:.3e}"
                         for key, i in pick.items() if i is not None)
        shp = rec["shapes"][top]
        at = np.unravel_index(int(locs[k, top]), shp) if shp else ()
        print(f"ADJOINT {label} step={k} max={rows[k, top]:.3e} "
              f"leaf={names[top]} at={tuple(int(i) for i in at)} "
              f"primal={vals[k, top]:.6e} {extra}", flush=True)
    if rows.shape[0] > 1:
        # row k is the cotangent entering step k, row k+1 the one leaving it
        ratio = tot[:-1] / tot[1:]
        if not np.any(np.isfinite(ratio)):
            print(f"ADJOINT_JUMP {label} none (rows not recorded)", flush=True)
            return
        k = int(np.nanargmax(ratio))
        print(f"ADJOINT_JUMP {label} step={k} ratio={ratio[k]:.3e} "
              f"(entering {tot[k]:.3e}, leaving {tot[k + 1]:.3e})", flush=True)


_DIRECTION = "both"


def _direction(x, seed=1):
    key = jax.random.PRNGKey(seed)
    # 0.1 K in T, 1e-3 relative in each tracer: well inside the linear range
    v = {"T": 0.1 * jax.random.normal(key, x["T"].shape, x["T"].dtype)}
    for i, k in enumerate(_X_TRACERS):
        v[k] = 1e-3 * x[k] * jax.random.normal(
            jax.random.fold_in(key, i + 1), x[k].shape, x[k].dtype)
    for k in v:
        if _DIRECTION not in ("both", k):
            v[k] = jnp.zeros_like(v[k])
    return v


_SPIED = ("make_radiation_physics", "make_convection_physics",
          "make_turbulence_physics", "make_microphysics_physics",
          "make_gwd_physics")


CALL_TAP = None   # --stash-modules: identity tap on every module call's in/out


def make_call_stash(max_calls=128, max_leaves=64):
    """Per-CALL adjoint recorder for the module spies: ``tap(label, x)`` is an
    identity ``custom_vjp`` whose backward pass writes max|cotangent| per leaf
    of ``x`` (and the argmax and the primal there) into its own row of a
    ``jax.ref``.  Rows are assigned in trace order; ``reset()`` restarts the
    numbering at the top of each trace so the forward-only and the gradient
    trace label the same calls alike.  Leaf names are taken from each call's
    own pytree at trace time."""
    rec = dict(log=jax.new_ref(jnp.full((max_calls, max_leaves), jnp.nan,
                                        jnp.float64)),
               loc=jax.new_ref(jnp.zeros((max_calls, max_leaves), jnp.int64)),
               val=jax.new_ref(jnp.full((max_calls, max_leaves), jnp.nan,
                                        jnp.float64)),
               labels={}, names={}, shapes={}, i=0)

    def reset():
        rec["i"] = 0

    def tap(label, x):
        i = rec["i"]
        rec["i"] += 1
        if i >= max_calls:
            if i == max_calls:
                print(f"MODTAP_WARN more than {max_calls} module calls; the "
                      "rest are not recorded", flush=True)
            return x
        paths, _ = jax.tree_util.tree_flatten_with_path(x)
        if len(paths) > max_leaves:
            print(f"MODTAP_WARN {label}: {len(paths)} leaves, only the first "
                  f"{max_leaves} recorded", flush=True)
        rec["labels"][i] = label
        rec["names"][i] = [jax.tree_util.keystr(kp) for kp, _ in paths][:max_leaves]
        rec["shapes"][i] = [tuple(np.shape(v)) for _, v in paths][:max_leaves]

        @jax.custom_vjp
        def ident(y):
            return y

        def _fwd(y):
            return y, jax.tree_util.tree_leaves(y)

        def _bwd(prim, g):
            row, loc, val = [], [], []
            for c, x0 in list(zip(jax.tree_util.tree_leaves(g), prim))[:max_leaves]:
                if jnp.issubdtype(jnp.result_type(x0), jnp.inexact):
                    a_ = jnp.abs(jnp.asarray(c)).ravel()
                    j = jnp.argmax(a_)
                    row.append(a_[j].astype(jnp.float64))
                    loc.append(j.astype(jnp.int64))
                    val.append(jnp.ravel(jnp.asarray(x0))[j].astype(jnp.float64))
                else:
                    row.append(jnp.float64(jnp.nan))
                    loc.append(jnp.int64(0))
                    val.append(jnp.float64(jnp.nan))
            pad = max_leaves - len(row)
            rec["log"][i] = jnp.concatenate(
                [jnp.stack(row), jnp.full((pad,), jnp.nan, jnp.float64)])
            rec["loc"][i] = jnp.concatenate(
                [jnp.stack(loc), jnp.zeros((pad,), jnp.int64)])
            rec["val"][i] = jnp.concatenate(
                [jnp.stack(val), jnp.full((pad,), jnp.nan, jnp.float64)])
            return (g,)
        ident.defvjp(_fwd, _bwd)
        return ident(x)
    return tap, rec, reset


def print_call_stash(rec, label):
    """One ``MODTAP`` line per tapped call, in call order: the largest
    cotangent leaf, its location and primal value."""
    rows = np.asarray(rec["log"][...])
    locs = np.asarray(rec["loc"][...])
    vals = np.asarray(rec["val"][...])
    for i in sorted(rec["labels"]):
        r = rows[i]
        if not np.any(np.isfinite(r)):
            print(f"MODTAP {label} call={i} {rec['labels'][i]} "
                  "no finite cotangent recorded", flush=True)
            continue
        top = int(np.nanargmax(r))
        shp = rec["shapes"][i][top]
        at = np.unravel_index(int(locs[i, top]), shp) if shp else ()
        print(f"MODTAP {label} call={i} {rec['labels'][i]} max={r[top]:.3e} "
              f"leaf={rec['names'][i][top]} at={tuple(int(k) for k in at)} "
              f"primal={vals[i, top]:.6e}", flush=True)


def state_digest(tree):
    """sha256 over every leaf's dtype, shape and bytes: equal digests mean a
    bitwise-identical state."""
    import hashlib
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a = np.asarray(x)
        h.update(str((a.dtype, a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def save_state(path, tree, meta):
    leaves = jax.tree_util.tree_leaves(tree)
    np.savez(path, **{f"leaf{i}": np.asarray(x) for i, x in enumerate(leaves)},
             meta=np.asarray(json.dumps(meta)))


def load_state(path, like):
    """Leaves of ``path`` into the structure of ``like``; the leaf count,
    shapes and dtypes must match exactly (same code, same deck)."""
    leaves, tdef = jax.tree_util.tree_flatten(like)
    with np.load(path) as z:
        n = len([k for k in z.files if k.startswith("leaf")])
        if n != len(leaves):
            raise SystemExit(f"--load-state: {n} leaves in {path}, "
                             f"{len(leaves)} in this build")
        new = []
        for i, x in enumerate(leaves):
            a = z[f"leaf{i}"]
            if a.shape != np.shape(x) or a.dtype != np.asarray(x).dtype:
                raise SystemExit(f"--load-state: leaf {i} is {a.dtype}{a.shape}, "
                                 f"build has {np.asarray(x).dtype}{np.shape(x)}")
            new.append(jnp.asarray(a))
        meta = json.loads(str(z["meta"]))
    return jax.tree_util.tree_unflatten(tdef, new), meta


_INLINE_PRIMS = ("pjit", "jit", "closed_call", "core_call", "checkpoint",
                 "remat")


def eqn_trace(f, x):
    """Reverse pass of scalar ``f`` at ``x`` one jaxpr equation at a time.

    Jitted / closed-call / remat sub-jaxprs are recursed into (so their
    equations are reported with their own source lines); scan, while, cond and
    custom-derivative calls stay single opaque equations.  Forward keeps every
    intermediate; backward applies ``jax.vjp`` of each single equation to its
    accumulated output cotangent.  Returns ``(grad_x, rows)``; ``grad_x`` must
    equal ``jax.grad(f)(x)`` (the self-check), and each row is ``dict(i, prim,
    src, ct_out, ct_in, gain, primal_at)``: max|cotangent| over the equation's
    inexact outputs and inputs, their ratio, and the primal input value at the
    largest input cotangent (the operand the derivative blew up on)."""
    from jax._src import core as jcore
    from jax._src import source_info_util as siu
    flat_x, tdef = jax.tree_util.tree_flatten(x)
    closed = jax.make_jaxpr(
        lambda *a: f(jax.tree_util.tree_unflatten(tdef, a)))(*flat_x)

    def inexact(a):
        return jnp.issubdtype(jnp.result_type(a), jnp.inexact)

    def sub_of(eqn):
        if eqn.primitive.name not in _INLINE_PRIMS:
            return None
        j = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
        if j is None:
            return None
        return (j.jaxpr, j.consts) if hasattr(j, "consts") else (j, [])

    def fwd(jaxpr, consts, args):
        env = {}

        def read(v):
            return v.val if isinstance(v, jcore.Literal) else env[v]
        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        for v, a in zip(jaxpr.invars, args):
            env[v] = a
        saved = []
        for eqn in jaxpr.eqns:
            inv = [read(v) for v in eqn.invars]
            sub = sub_of(eqn)
            if sub is not None:
                out, inner = fwd(sub[0], sub[1], inv)
            else:
                bp = eqn.primitive.get_bind_params(eqn.params)
                out = eqn.primitive.bind(*inv, **bp)
                out = out if eqn.primitive.multiple_results else [out]
                inner = None
            for v, o in zip(eqn.outvars, out):
                if not isinstance(v, jcore.DropVar):
                    env[v] = o
            saved.append((inv, list(out), inner))
        return [read(v) for v in jaxpr.outvars], (env, saved)

    rows = []
    counter = [0]

    def bwd(jaxpr, rec, cts_out):
        env, saved = rec
        ct = {}
        for v, c in zip(jaxpr.outvars, cts_out):
            if c is not None and not isinstance(v, jcore.Literal):
                ct[v] = c if v not in ct else ct[v] + c
        for i in range(len(jaxpr.eqns) - 1, -1, -1):
            eqn = jaxpr.eqns[i]
            inv, outs, inner = saved[i]
            o_idx = [j for j, v in enumerate(eqn.outvars)
                     if not isinstance(v, jcore.DropVar) and v in ct
                     and inexact(outs[j])]
            d_idx = [j for j, v in enumerate(eqn.invars)
                     if inexact(inv[j]) and not isinstance(v, jcore.Literal)]
            if not o_idx or not d_idx:
                continue
            if inner is not None:
                sub = sub_of(eqn)
                c_out = [ct.get(v) if j in o_idx else None
                         for j, v in enumerate(eqn.outvars)]
                c_in_all = bwd(sub[0], inner, c_out)
                cts_in = [c_in_all[j] for j in d_idx]
                cts_in = [jnp.zeros_like(inv[j]) if c is None else c
                          for c, j in zip(cts_in, d_idx)]
            else:
                bp = eqn.primitive.get_bind_params(eqn.params)

                def one(*d, _inv=inv, _eqn=eqn, _bp=bp, _d=d_idx, _o=o_idx):
                    full = list(_inv)
                    for k, j in enumerate(_d):
                        full[j] = d[k]
                    r = _eqn.primitive.bind(*full, **_bp)
                    r = r if _eqn.primitive.multiple_results else [r]
                    return [r[j] for j in _o]
                _, vjp = jax.vjp(one, *[inv[j] for j in d_idx])
                cts_in = list(vjp([ct[eqn.outvars[j]] for j in o_idx]))
                mo = max(float(jnp.max(jnp.abs(ct[eqn.outvars[j]])))
                         for j in o_idx)
                best, best_at = -1.0, float("nan")
                for k, j in enumerate(d_idx):
                    c = jnp.asarray(cts_in[k])
                    if c.size == 0:
                        continue
                    a_ = jnp.abs(c).ravel()
                    m = int(jnp.argmax(a_))
                    if float(a_[m]) > best:
                        best = float(a_[m])
                        best_at = float(jnp.ravel(
                            jnp.broadcast_to(inv[j], c.shape))[m])
                counter[0] += 1
                rows.append(dict(i=counter[0], prim=eqn.primitive.name,
                                 src=siu.summarize(eqn.source_info),
                                 ct_out=mo, ct_in=best,
                                 gain=(best / mo if mo > 0 else float("inf")),
                                 primal_at=best_at))
            for k, j in enumerate(d_idx):
                v = eqn.invars[j]
                ct[v] = cts_in[k] if v not in ct else ct[v] + cts_in[k]
        return [ct.get(v) if not isinstance(v, jcore.Literal) else None
                for v in jaxpr.invars]

    out, rec = fwd(closed.jaxpr, closed.consts, flat_x)
    g = bwd(closed.jaxpr, rec, [jnp.ones_like(out[0])])
    grad = [jnp.zeros_like(a) if c is None else c for c, a in zip(g, flat_x)]
    return jax.tree_util.tree_unflatten(tdef, grad), rows


def loop_census(f, x, top=20):
    """Every scan / while loop in the jaxpr of ``f(x)`` (recursing into every
    sub-jaxpr: jit, cond branches, loop bodies, remat, custom derivatives),
    ranked by the bytes of its STACKED outputs -- a scan's ``ys`` with the
    loop length as leading axis, i.e. what a traced ``jax.grad(g)`` stores per
    iteration for the reverse pass (a while loop stacks nothing).  Prints ``LOOP {prim, length, stacked_mib,
    stacked, n_seen, src}`` (identical loops reached through several call sites
    are merged, ``n_seen`` counts them); returns the rows."""
    from jax._src import core as jcore
    from jax._src import source_info_util as siu
    rows = {}

    def subs(eqn):
        for v in eqn.params.values():
            for j in (v if isinstance(v, (tuple, list)) else (v,)):
                if isinstance(j, jcore.ClosedJaxpr):
                    yield j.jaxpr
                elif isinstance(j, jcore.Jaxpr):
                    yield j

    def walk(jaxpr):
        for e in jaxpr.eqns:
            if e.primitive.name in ("scan", "while"):
                n = e.params.get("length")
                st = ([v.aval for v in e.outvars[e.params["num_carry"]:]]
                      if e.primitive.name == "scan" else [])
                key = (e.primitive.name, n, siu.summarize(e.source_info),
                       tuple(a.str_short() for a in st))
                r = rows.setdefault(key, dict(
                    prim=e.primitive.name, length=n, src=key[2], n_seen=0,
                    stacked_mib=sum(a.size * a.dtype.itemsize
                                    for a in st) / 2**20,
                    stacked=[a.str_short() for a in st][:6]))
                r["n_seen"] += 1
            for j in subs(e):
                walk(j)
    walk(jax.make_jaxpr(f)(x).jaxpr)
    out = sorted(rows.values(), key=lambda r: -r["stacked_mib"])
    for r in out[:top]:
        print("LOOP " + json.dumps(r), flush=True)
    return out


def print_eqn_trace(rows, label, top=25):
    """The equations that multiplied the cotangent most (``EQN_GAIN``) and
    the ones whose input cotangent is largest (``EQN_MAX``)."""
    print(f"EQN_COUNT {label} {len(rows)} equations on the reverse path",
          flush=True)
    live = [r for r in rows if r["ct_out"] > 0.0]   # a zero cotangent has no gain
    for key, tag in (("gain", "EQN_GAIN"), ("ct_in", "EQN_MAX")):
        for r in sorted(live, key=lambda r: -r[key])[:top]:
            print(f"{tag} {label} eqn={r['i']} {r['prim']} gain={r['gain']:.3e} "
                  f"ct_out={r['ct_out']:.3e} ct_in={r['ct_in']:.3e} "
                  f"primal={r['primal_at']:.6e} src={r['src']}", flush=True)


ALL_CALLS = None   # dict name -> list of every concrete (fn, args, kwargs)


def install_spies(record):
    """Wrap combined's module factories so the FIRST call of each module
    records (fn, args, kwargs).  functools.wraps keeps every marker the
    combined dispatcher reads (_wants_forcing, ...)."""
    import legoesm.atmosphere.physics.combined as comb
    for name in _SPIED:
        orig = getattr(comb, name)

        def factory(*a, _orig=orig, _name=name, **k):
            if _name == "make_radiation_physics" and RAD_FP64:
                # diagnostic: undo the MPAS driver's fp32 RRTMGP solve so FD
                # is not swamped by single-precision roundoff
                a = (a[0]._replace(rrtmgp=a[0].rrtmgp._replace(
                    compute_fp32=False)),) + a[1:]
            fn = _orig(*a, **k)

            @functools.wraps(fn)
            def spy(*fa, **fk):
                if not any(isinstance(x, jax.core.Tracer)
                           for x in jax.tree_util.tree_leaves((fa, fk))):
                    record.setdefault(_name, (fn, fa, fk))
                    if ALL_CALLS is not None:
                        ALL_CALLS.setdefault(_name, []).append((fn, fa, fk))
                if CALL_TAP is None:
                    return fn(*fa, **fk)
                short = _name.replace("make_", "").replace("_physics", "")
                st, ps = CALL_TAP(f"{short}:in", (fa[0], fk.get("phys_state")))
                fa = (st,) + tuple(fa[1:])
                if "phys_state" in fk:
                    fk = dict(fk, phys_state=ps)
                return CALL_TAP(f"{short}:out", fn(*fa, **fk))
            return spy
        setattr(comb, name, factory)


def _git_sha():
    """Short HEAD SHA for the receipt header ('unknown' outside a checkout or
    without git on PATH -- sbatch nodes)."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except Exception:
        pass
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            head = (REPO / ".git" / head.split(None, 1)[1]).read_text().strip()
        return head[:12]
    except Exception:
        return "unknown"


def code_provenance(modules=("legoesm.driver.mpas_diff_step",
                             "legoesm.atmosphere.physics.microphysics.morrison")):
    """Where the code under test was actually imported from (PYTHONPATH may
    point at another worktree than this script's) and that checkout's HEAD
    plus its count of uncommitted paths, one ``name=path@sha+dirtyN`` per
    module."""
    import importlib
    import subprocess
    out = []
    for name in modules:
        try:
            path = Path(importlib.import_module(name).__file__).resolve()
        except Exception as e:      # report, do not hide
            out.append(f"{name}=UNIMPORTABLE({type(e).__name__})")
            continue
        sha, dirty = "unknown", "?"
        try:
            run = dict(cwd=path.parent, capture_output=True, text=True,
                       timeout=20, check=True)
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 **run).stdout.strip()
            dirty = str(len(subprocess.run(["git", "status", "--porcelain"],
                                           **run).stdout.splitlines()))
        except Exception:
            pass
        out.append(f"{name.rsplit('.', 1)[-1]}={path}@{sha}+dirty{dirty}")
    return " ".join(out)


def time_compiled(fn, args, repeats):
    """Compile ``fn`` once (``jax.jit(...).lower(*args).compile()``), then run
    it ``1 + repeats`` times, blocking on the result.  Returns ``(compile_s,
    first_s, [repeat_s, ...], out)``: compilation alone, the first execution,
    the steady-state executions a training loop would pay, and the last
    result (so a caller checks it without compiling again)."""
    t0 = time.time()
    c = jax.jit(fn).lower(*args).compile()
    t_c = time.time() - t0
    t0 = time.time()
    jax.block_until_ready(c(*args))
    t_1 = time.time() - t0
    reps, out = [], None
    for _ in range(repeats):
        t0 = time.time()
        out = jax.block_until_ready(c(*args))
        reps.append(time.time() - t0)
    return t_c, t_1, reps, out


def grad_diff(got, ref):
    """Per leaf: max|got-ref|/max|ref|, ||got-ref||/||ref|| and the cosine
    of the two (fp64 on host; NaN where either side is non-finite)."""
    out = {}
    for k in ref:
        r = np.asarray(ref[k], np.float64).ravel()
        g = np.asarray(got[k], np.float64).ravel()
        d = g - r
        nr, ng = np.linalg.norm(r), np.linalg.norm(g)
        out[k] = dict(max_rel=float(np.max(np.abs(d)) / np.max(np.abs(r))),
                      norm_rel=float(np.linalg.norm(d) / nr),
                      cos=float(r @ g / (nr * ng)))
    return out


def _tdot(a, b):
    return float(sum(np.vdot(np.asarray(a[k], np.float64),
                             np.asarray(b[k], np.float64)) for k in b))


def check_directions(x):
    """The probe's random direction, whole and one variable at a time."""
    v = _direction(x)
    return dict({"both": v}, **{
        k: {j: (a if j == k else jnp.zeros_like(a)) for j, a in v.items()}
        for k in v})


def gap_direction(x, gap):
    """Along a gradient gap, each variable on :func:`_direction`'s scale (0.1
    K in T, 1e-3 relative in a tracer) and at most that scale in any cell:
    two gradients that differ predict different derivatives here, so a
    central difference can tell them apart wherever its rounding error
    (``fd_rounding``) is below the difference of the two predictions.  None
    without a gap."""
    s = {k: (0.1 if k == "T" else 1e-3 * jnp.abs(x[k])) for k in gap}
    top = max(float(jnp.max(jnp.abs(s[k] * gap[k]))) for k in gap)
    if not (np.isfinite(top) and top > 0.0):
        return None
    return {k: s[k] * s[k] * gap[k] / top for k in gap}


def fd_sweep(f, x, v, h):
    """Central differences of ``f`` along ``v`` at each step of ``H_SWEEP *
    h``; ``f`` returns one loss or one loss per window."""
    def at(hh):
        return np.asarray(f(jax.tree_util.tree_map(
            lambda a, b: a + hh * b, x, v)), np.float64)
    return {f"{h * m:.0e}": (at(h * m) - at(-h * m)) / (2.0 * h * m)
            for m in H_SWEEP}


def one_ulp(x):
    """``x`` with every entry moved one rounding unit, random sign."""
    return {k: a * (1.0 + jnp.finfo(a.dtype).eps * jax.random.rademacher(
        jax.random.PRNGKey(j), a.shape, a.dtype))
        for j, (k, a) in enumerate(sorted(x.items()))}


def print_fd_check(path, window, direction, ad, ad_other, fds, loss_rounding):
    """``loss_rounding``: the path's loss change at one rounding unit in x;
    over 2h it is the resolution of that step's central difference."""
    print("FD_CHECK " + json.dumps(dict(
        path=path, window=window, direction=direction, ad=ad,
        ad_other_path=ad_other, fd={h: float(x) for h, x in fds.items()},
        fd_rounding={h: float(loss_rounding) / (2.0 * float(h)) for h in fds},
        fd_over_ad_minus_1={h: (float(x) / ad - 1.0 if ad else float("nan"))
                            for h, x in fds.items()})), flush=True)


def gap_concentration(gap, k=10):
    """Per leaf: the share of the squared gap held by its ``k`` largest
    entries and the largest one's index."""
    out = {}
    for name, d in gap.items():
        a = np.square(np.asarray(d, np.float64))
        tot = float(a.sum())
        out[name] = dict(
            top_share=(float(np.sort(a, axis=None)[-k:].sum()) / tot
                       if tot else 0.0),
            at=[int(i) for i in np.unravel_index(int(np.argmax(a)), a.shape)])
    return out


def window_check(exps, g_batch, losses_batch, n, kw, ids, fwd_batch=None,
                 h=1e-3):
    """Windows ``ids`` from a true standalone run (each its OWN export: own
    physics closures, forcing, clock built inside; no vmap; one compile per
    window) against the batched run: the forward loss (batched per-window
    ``losses_batch``) and the gradient (its slice of ``g_batch``).  The
    first id's standalone gradient is also compiled a second way (jax.vmap
    over a batch of one, same export): the gap between the two is the
    rounding noise of the gradient itself, the floor any batched-vs-alone
    difference is read against.  Each standalone gradient is also repeated
    (same executable: determinism) and taken again one rounding unit away in
    x (same executable: conditioning).  Numbers only, no tolerance.

    With ``fwd_batch`` (stacked x -> one loss per window) each path is held
    against its OWN central difference (FD_CHECK lines, steps ``H_SWEEP *
    h``): every window of the batch and each standalone window along the
    probe's random directions, and each checked window of both paths along
    the batched-minus-standalone gradient gap (:func:`gap_direction`)."""
    from legoesm.driver.mpas_diff_step import mpas_segment
    tm = jax.tree_util.tree_map
    if fwd_batch is not None:
        xs = [_state_x(e["state"]) for e in exps]
        dirs = [check_directions(x_) for x_ in xs]
        xb = tm(lambda *t: jnp.stack(t), *xs)
        noise_b = np.abs(np.asarray(fwd_batch(one_ulp(xb)), np.float64)
                         - np.asarray(fwd_batch(xb), np.float64))
        for name in dirs[0]:
            fds = fd_sweep(fwd_batch, xb,
                           tm(lambda *t: jnp.stack(t), *[d[name] for d in dirs]),
                           h)
            for b, d in enumerate(dirs):
                print_fd_check(
                    "batched", b, name,
                    _tdot({k: v[b] for k, v in g_batch.items()}, d[name]),
                    None, {m: f[b] for m, f in fds.items()}, noise_b[b])
    for i, b in enumerate(ids):
        e = exps[b]

        def one(x_, _e=e):
            out, _ = mpas_segment(_e, _with_x(_e["state"], x_),
                                  _e["phys_state"], n, **kw)
            return _proj_loss(out)
        x = _state_x(e["state"])
        t0 = time.time()
        vg = jax.jit(jax.value_and_grad(one))
        loss, gb = jax.block_until_ready(vg(x))
        row = dict(window=b, seconds=time.time() - t0,
                   loss_alone=float(loss), loss_batched=float(losses_batch[b]),
                   alone_finite=bool(all(np.all(np.isfinite(np.asarray(v)))
                                         for v in gb.values())),
                   batched_vs_alone=grad_diff({k: g_batch[k][b] for k in gb}, gb))
        g_b = {k: g_batch[k][b] for k in gb}
        gap = {k: g_b[k] - gb[k] for k in gb}
        row["gap_top10"] = gap_concentration(gap)
        row["repeat_vs_alone"] = grad_diff(vg(x)[1], gb)
        x_ulp = one_ulp(x)
        row["ulp_vs_alone"] = grad_diff(vg(x_ulp)[1], gb)
        if fwd_batch is not None:
            f1 = jax.jit(one)
            row["loss_alone_fwd"] = float(f1(x))
            noise = abs(float(f1(x_ulp)) - row["loss_alone_fwd"])
            todo = dict(dirs[b])
            w = gap_direction(x, gap)
            if w is not None:
                todo["gap"] = w
            for name, d in todo.items():
                print_fd_check("alone", b, name, _tdot(gb, d), _tdot(g_b, d),
                               fd_sweep(f1, x, d, h), noise)
            if w is not None:
                wb = {k: jnp.zeros_like(xb[k]).at[b].set(w[k]) for k in w}
                fds = fd_sweep(fwd_batch, xb, wb, h)
                print_fd_check("batched", b, "gap", _tdot(g_b, w),
                               _tdot(gb, w), {m: f[b] for m, f in fds.items()},
                               noise_b[b])
        if i == 0:
            x1 = jax.tree_util.tree_map(lambda t: t[None], x)
            g1 = jax.block_until_ready(jax.jit(jax.grad(
                lambda z: jnp.sum(jax.vmap(one)(z))))(x1))
            row["vmap1_vs_alone"] = grad_diff(
                {k: v[0] for k, v in g1.items()}, gb)
        print("WINDOW_CHECK " + json.dumps(row), flush=True)


def memory_report(label, fn, args, ncol):
    """Compile ``fn`` (named ``fn.__name__``, so an ``--xla_dump_hlo_module_re``
    filter can select it) WITHOUT running it and print XLA's own buffer
    accounting: ``MEMORY {label, temp/argument/output/alias/code GiB, temp
    MB per column}``.  Compile-only, so a window that would OOM is still
    measured."""
    t0 = time.time()
    c = jax.jit(fn).lower(*args).compile()
    m = c.memory_analysis()
    gib = {k: getattr(m, f"{k}_size_in_bytes") / 2**30
           for k in ("temp", "argument", "output", "alias", "generated_code")}
    print("MEMORY " + json.dumps(dict(
        label=label, ncol=ncol, compile_s=time.time() - t0,
        temp_mb_per_col=gib["temp"] * 2**30 / 1e6 / ncol,
        **{f"{k}_gib": v for k, v in gib.items()})), flush=True)


def peak_gib():
    """Peak memory so far, GiB: the device's peak_bytes_in_use where the
    backend reports it (GPU/TPU), else the process RSS high-water mark (CPU).
    Both are process-cumulative: compare schedules across separate runs."""
    try:
        st = jax.devices()[0].memory_stats()
        if st and "peak_bytes_in_use" in st:
            return st["peak_bytes_in_use"] / 2**30
    except Exception:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("components", "physics", "params", "dycore",
                             "step", "land", "eager", "eqns", "timing",
                             "memory", "loops", "modmem"))
    ap.add_argument("--module", default="microphysics",
                    help="eqns stage: the module whose calls are traced")
    ap.add_argument("--calls", default="1,2,3",
                    help="eqns stage: which calls of --module (1-based, in "
                         "the physics call's order) to trace")
    ap.add_argument("--resolution", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--h-sweep", action="store_true",
                    help="FD at h x {100, 10, 1, 0.1, 0.01}")
    ap.add_argument("--direction", default="both",
                    help="perturb 'both' (all of T and the --x-tracers) or "
                         "one of them by name")
    ap.add_argument("--x-tracers", default="q_v",
                    help="comma list of tracers differentiated alongside T")
    ap.add_argument("--at-step", type=int, default=0,
                    help="advance the exported state this many production "
                         "steps (eager segment) before the stage runs on it")
    ap.add_argument("--physics-variant", choices=("full", "norad"),
                    default="full",
                    help="components stage: which physics variant the module "
                         "spies record (norad = the cadence's no-radiation step)")
    ap.add_argument("--rad-fp64", action="store_true",
                    help="RRTMGP solve in fp64 (diagnostic: FD vs fp32 roundoff)")
    ap.add_argument("--per-knob", action="store_true",
                    help="params stage: AD vs FD per knob instead of one direction")
    ap.add_argument("--only", default="",
                    help="components stage: comma list of modules to check")
    ap.add_argument("--schedule", choices=("none", "uniform", "binomial"),
                    default="none",
                    help="step stage: mpas_segment loop form (none = the "
                         "unrolled loop with per-step remat; uniform / binomial "
                         "= the fixed-length checkpoint-scheduled loop)")
    ap.add_argument("--checkpoints", type=int, default=None,
                    help="binomial schedule: checkpoint budget K "
                         "(default ceil(sqrt(n_steps)))")
    ap.add_argument("--storage", choices=("recompute", "host"),
                    default="recompute",
                    help="scheduled loop: residual storage (host offloads "
                         "matmul residuals; recompute on a CPU backend)")
    ap.add_argument("--stash-grads", action="store_true",
                    help="step stage, unrolled loop: record max|cotangent| "
                         "per leaf per step in the reverse pass (ADJOINT "
                         "lines) to locate where the adjoint grows")
    ap.add_argument("--save-state", default="",
                    help="write the (state, phys_state) the stage runs on "
                         "(after --at-step) to this .npz")
    ap.add_argument("--load-state", default="",
                    help="run the stage on the (state, phys_state) of this "
                         ".npz instead of advancing; --at-step still sets "
                         "the forcing clock")
    ap.add_argument("--timing-steps", default="64,256",
                    help="timing stage: comma list of window lengths [steps]")
    ap.add_argument("--batch", type=int, default=1,
                    help="timing stage: B windows, each its own driver "
                         "export (see --window-spacing-days), in one jax.vmap "
                         "call; needs --schedule")
    ap.add_argument("--window-spacing-days", type=float, default=None,
                    help="timing stage with --batch: window b is the driver "
                         "started this many days * b after the deck's start "
                         "day (required with --batch)")
    ap.add_argument("--check-window-ids", default="",
                    help="--check-windows: comma-separated window indices "
                         "(default: first and last)")
    ap.add_argument("--check-windows", action="store_true",
                    help="timing stage with --batch: also take the first and "
                         "last window's gradient from a standalone run and "
                         "print its difference from the batched slice; "
                         "hold both paths against their own central "
                         "differences (--h, --h-sweep)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="timing stage: steady-state calls timed per window")
    ap.add_argument("--stash-modules", action="store_true",
                    help="physics stage: record max|cotangent| at the input "
                         "and output of every physics-module call inside the "
                         "call (MODTAP lines, call order)")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    rest = [x for x in a.rest if x != "--"]
    global H_SWEEP, _DIRECTION, RAD_FP64, _X_TRACERS
    _DIRECTION = a.direction
    _X_TRACERS = tuple(t.strip() for t in a.x_tracers.split(",") if t.strip())
    if _DIRECTION not in ("both", "T", *_X_TRACERS):
        raise SystemExit(f"--direction {a.direction!r} is not T or an --x-tracers")
    RAD_FP64 = a.rad_fp64
    if a.h_sweep:
        H_SWEEP = (100.0, 10.0, 1.0, 0.1, 0.01)
    print(f"stage={a.stage} res={a.resolution} n_steps={a.n_steps} "
          f"x64={jax.config.jax_enable_x64} backend={jax.default_backend()} "
          f"schedule={a.schedule} checkpoints={a.checkpoints} "
          f"storage={a.storage} sha={_git_sha()}", flush=True)
    print(f"CODE {code_provenance()}", flush=True)
    if a.stash_modules and a.stage != "physics":
        raise SystemExit("--stash-modules applies to --stage physics only")
    if a.stash_grads and a.stage != "step":
        raise SystemExit("--stash-grads applies to --stage step only")
    rec = {}
    if a.stage == "eqns":
        global ALL_CALLS
        ALL_CALLS = {}
    if (a.stage in ("components", "eqns", "modmem") or a.rad_fp64
            or a.stash_modules):
        install_spies(rec)
    if a.stage in ("components", "eqns", "modmem"):
        # the spies record only concrete calls, and jax.checkpoint traces its
        # arguments even when called eagerly: without this no module records.
        # Forward values are the same with or without the module remat.
        rest = rest + ["--no-physics-module-checkpoint"]
    t0 = time.time()
    drv = build_driver(rest, a.resolution, a.n_steps)
    exp = drv.export_mpas_step()
    print(f"built+exported in {time.time() - t0:.1f} s; ncol="
          f"{exp['state'].T.data.shape}, physics every "
          f"{exp['phys_update_steps']} steps, radiation every "
          f"{exp['rad_update_steps']}, land_k={exp['land_k']}", flush=True)
    model, state0, ps0 = exp["model"], exp["state"], exp["phys_state"]
    if a.load_state:
        (state0, ps0), meta = load_state(a.load_state, (state0, ps0))
        print(f"LOADED_STATE {a.load_state} from {meta}", flush=True)
        if meta.get("at_step") != a.at_step:
            raise SystemExit(f"--load-state: the file holds the state at step "
                             f"{meta.get('at_step')}, but --at-step is "
                             f"{a.at_step} (clock and cadence phase)")
        if meta.get("digest") not in (None, state_digest((state0, ps0))):
            raise SystemExit("--load-state: loaded state does not match the "
                             "digest recorded when it was saved")
    elif a.at_step:
        from legoesm.driver.mpas_diff_step import mpas_segment as _seg
        t1 = time.time()
        state0, ps0 = _seg(exp, state0, ps0, a.at_step)
        qi = state0.tracers["q_i"].data if "q_i" in state0.tracers else None
        print(f"AT_STEP {a.at_step} advanced in {time.time() - t1:.1f} s; "
              f"T [{float(jnp.min(state0.T.data)):.2f}, "
              f"{float(jnp.max(state0.T.data)):.2f}]"
              + (f" q_i [{float(jnp.min(qi)):.3e}, {float(jnp.max(qi)):.3e}]"
                 f" zero={int(jnp.sum(qi == 0.0))}/{qi.size}"
                 if qi is not None else ""), flush=True)
    print(f"STATE_DIGEST {state_digest((state0, ps0))} at_step={a.at_step} "
          f"backend={jax.default_backend()}", flush=True)
    if a.save_state:
        save_state(a.save_state, (state0, ps0),
                   dict(at_step=a.at_step, backend=jax.default_backend(),
                        sha=_git_sha(), digest=state_digest((state0, ps0))))
        print(f"SAVED_STATE {a.save_state}", flush=True)
    x0 = _state_x(state0)
    v = _direction(x0)
    forcing0 = dict(exp["forcing"])
    doy, sod = exp["calendar"](a.at_step)
    forcing0.update(day_of_year=jnp.asarray(doy), seconds_of_day=jnp.asarray(sod))
    mesh, sigma = model.mesh, model.sigma_coord

    if a.stage == "components":
        # one eager production physics call populates the spies
        pfn = exp["physics_fn" if a.physics_variant == "full" else
                  "physics_fn_norad"]
        if pfn is None:
            raise SystemExit("--physics-variant norad: this cadence has no "
                             "no-radiation variant")
        _ = pfn(state0, mesh, sigma, phys_state=ps0, forcing=forcing0)
        if not rec:
            raise SystemExit("no physics module call was recorded")
        print("recorded modules: " + ", ".join(sorted(rec)), flush=True)
        only = {x.strip() for x in a.only.split(",") if x.strip()}
        for name, (fn, fa, fk) in sorted(rec.items()):
            short = name.replace("make_", "").replace("_physics", "")
            if only and short not in only:
                continue
            st = fa[0]

            def f(x, _fn=fn, _fa=fa, _fk=fk, _st=st):
                return _proj_loss(_fn(_with_x(_st, x), *_fa[1:], **_fk))
            try:
                grad_check(f, _state_x(st), _direction(_state_x(st)), a.h,
                           short)
            except Exception as e:   # report, do not hide: a non-traceable module
                print(f"GRADCHECK_FAIL {name}: {type(e).__name__}: {e}"[:2000],
                      flush=True)
    elif a.stage == "timing":
        from legoesm.driver.mpas_diff_step import mpas_segment, stack_windows
        sched = None if a.schedule == "none" else a.schedule
        nb = a.batch
        if nb > 1:
            if sched is None:
                raise SystemExit("--batch needs --schedule uniform|binomial "
                                 "(the clock is vmapped into the scheduled loop)")
            if a.window_spacing_days is None:
                raise SystemExit("--batch needs --window-spacing-days")
            if a.at_step or a.load_state:
                raise SystemExit("--batch: each window starts at its own "
                                 "export; --at-step/--load-state set window "
                                 "0 only")
            # window b: the driver started window_spacing_days * b after the
            # deck's start day and exported there (own state, forcing, clock)
            day0 = float(drv.config.start_day)
            exps = [exp] + [build_driver(
                rest + ["--start-day", repr(day0 + b * a.window_spacing_days)],
                a.resolution, a.n_steps).export_mpas_step()
                for b in range(1, nb)]
            print("WINDOWS start days " + ", ".join(
                repr(day0 + b * a.window_spacing_days) for b in range(nb)),
                flush=True)
        for n in (int(t) for t in a.timing_steps.split(",") if t.strip()):
            kw = (dict(checkpoint=True) if sched is None else
                  dict(schedule=sched, checkpoints=a.checkpoints,
                       storage=a.storage))
            if nb == 1:
                def fwd(x, _n=n):
                    s, _ = mpas_segment(exp, _with_x(state0, x), ps0, _n, **kw)
                    return _proj_loss(s)
                args = (x0,)
            else:
                sb, pb, fb, cb = stack_windows(exps, n)

                def one_b(x_, s_, p_, f_, c_, _n=n):
                    out, _ = mpas_segment(exp, _with_x(s_, x_), p_, _n,
                                          forcing=f_, clock=c_, **kw)
                    return _proj_loss(out)

                def fwd(x, s, p, f, c):
                    return jnp.sum(jax.vmap(one_b)(x, s, p, f, c))
                args = (_state_x(sb), sb, pb, fb, cb)
            tf_c, tf_1, tf_r, _ = time_compiled(fwd, args, a.repeats)
            tg_c, tg_1, tg_r, (loss_g, g) = time_compiled(
                jax.value_and_grad(fwd), args, a.repeats)
            fin = all(bool(jnp.all(jnp.isfinite(t)))
                      for t in jax.tree_util.tree_leaves(g))
            if not fin:
                nonfinite_census(g, batched=nb > 1)
            med_f, med_g = float(np.median(tf_r)), float(np.median(tg_r))
            print("TIMING " + json.dumps(dict(
                n_steps=n, schedule=a.schedule, checkpoints=a.checkpoints,
                ncol=int(state0.T.data.shape[0]), batch=nb,
                grad_per_window_s=med_g / nb, fwd_compile_s=tf_c,
                fwd_first_s=tf_1, fwd_s=tf_r, grad_compile_s=tg_c,
                grad_first_s=tg_1, grad_s=tg_r, grad_per_step_s=med_g / n,
                grad_over_fwd=med_g / med_f, grad_finite=fin,
                loss_grad_exec=float(loss_g),
                peak_gib=peak_gib())), flush=True)
            if a.check_windows and nb > 1:
                losses_fn = jax.jit(jax.vmap(one_b))
                losses = np.asarray(losses_fn(*args))
                ids = ([int(t) for t in a.check_window_ids.split(",")]
                       if a.check_window_ids else sorted({0, nb - 1}))
                window_check(exps, g, losses, n, kw, ids,
                             fwd_batch=lambda x: losses_fn(x, *args[1:]),
                             h=a.h)
    elif a.stage == "loops":
        # trace-only: which loops stack per-iteration residuals in the
        # gradient of one full-physics step (no compile, no device run)
        tm = model.mass_target(state0)
        pfn = exp["physics_fn" if a.physics_variant == "full" else
                  "physics_fn_norad"]

        def step_loss(x):
            s = model.step_pure(_with_x(state0, x), exp["dt"], physics_fn=pfn,
                                forcing=forcing0, phys_state=ps0,
                                target_mass=tm)[0]
            return _proj_loss(s)
        loop_census(jax.grad(step_loss), x0)
    elif a.stage == "modmem":
        # compile-only gradient memory of each CAM6 module call, as the
        # combined physics calls it (spies record the first concrete call)
        pfn = exp["physics_fn" if a.physics_variant == "full" else
                  "physics_fn_norad"]
        _ = pfn(state0, mesh, sigma, phys_state=ps0, forcing=forcing0)
        ncol = int(state0.T.data.shape[0])
        for name, (fn, fa, fk) in sorted(rec.items()):
            st = fa[0]

            def mem_module(x, _fn=fn, _fa=fa, _fk=fk, _st=st):
                return _proj_loss(_fn(_with_x(_st, x), *_fa[1:], **_fk))
            try:
                memory_report(name.replace("make_", "").replace("_physics", ""),
                              jax.grad(mem_module), (_state_x(st),), ncol)
            except Exception as e:   # report, do not hide
                print(f"MEMORY_FAIL {name}: {type(e).__name__}: {e}"[:2000],
                      flush=True)
    elif a.stage == "memory":
        # Compile-only buffer accounting of each piece of one production
        # step, then the scheduled segment over --n-steps (forward), so the
        # per-column forward buffer is attributed to a piece.
        from legoesm.driver.mpas_diff_step import mpas_segment
        ncol = int(state0.T.data.shape[0])
        tm = model.mass_target(state0)

        def mem_dycore(s):
            return model.step_pure(s, exp["dt"], target_mass=tm)[0]
        memory_report("dycore_step", mem_dycore, (state0,), ncol)
        for key in ("physics_fn", "physics_fn_norad", "physics_fn_held"):
            pfn = exp[key]
            if pfn is None:
                continue

            def mem_physics(s, p, f, _p=pfn):
                return _p(s, mesh, sigma, phys_state=p, forcing=f)

            def mem_step(s, p, f, _p=pfn):
                return model.step_pure(s, exp["dt"], physics_fn=_p, forcing=f,
                                       phys_state=p, target_mass=tm)[:2]
            memory_report(key, mem_physics, (state0, ps0, forcing0), ncol)
            memory_report(f"step_{key}", mem_step, (state0, ps0, forcing0),
                          ncol)
        sched = "uniform" if a.schedule == "none" else a.schedule

        def mem_segment(s, p):
            return mpas_segment(exp, s, p, a.n_steps, step0=a.at_step,
                                schedule=sched, checkpoints=a.checkpoints,
                                storage=a.storage)
        memory_report(f"segment_{sched}_{a.n_steps}_fwd", mem_segment,
                      (state0, ps0), ncol)

        def mem_segment_grad(x):
            s, _ = mem_segment(_with_x(state0, x), ps0)
            return _proj_loss(s)
        memory_report(f"segment_{sched}_{a.n_steps}_grad",
                      jax.grad(mem_segment_grad), (x0,), ncol)
    elif a.stage == "eqns":
        pfn = exp["physics_fn" if a.physics_variant == "full" else
                  "physics_fn_norad"]
        _ = pfn(state0, mesh, sigma, phys_state=ps0, forcing=forcing0)
        calls = ALL_CALLS.get(f"make_{a.module}_physics", [])
        print(f"EQN_CALLS {a.module}: {len(calls)} concrete calls recorded",
              flush=True)
        for c in (int(t) for t in a.calls.split(",") if t.strip()):
            if c > len(calls):
                print(f"EQN_SKIP call {c}: only {len(calls)}", flush=True)
                continue
            fn, fa, fk = calls[c - 1]
            st = fa[0]

            def f(x, _fn=fn, _fa=fa, _fk=fk, _st=st):
                return _proj_loss(_fn(_with_x(_st, x), *_fa[1:], **_fk))
            name = f"{a.module}_call{c}"
            xs = _state_x(st)
            grad_check(f, xs, _direction(xs), a.h, name)
            t1 = time.time()
            g_eq, rows = eqn_trace(f, xs)
            g_ad = jax.jit(jax.grad(f))(xs)
            for k in xs:
                d = float(jnp.max(jnp.abs(g_eq[k] - g_ad[k])))
                sc = float(jnp.max(jnp.abs(g_ad[k])))
                print(f"EQN_SELFCHECK {name} {k} max|eqn-grad|={d:.3e} "
                      f"(scale {sc:.3e})", flush=True)
            print(f"EQN_TIME {name} {time.time() - t1:.1f} s", flush=True)
            print_eqn_trace(rows, name)
    elif a.stage == "physics":
        for label, pfn in (("physics_full", exp["physics_fn"]),
                           ("physics_norad", exp["physics_fn_norad"])):
            if pfn is None:
                continue

            global CALL_TAP
            if a.stash_modules:
                CALL_TAP, mrec, mreset = make_call_stash()

            def f(x, _p=pfn):
                if CALL_TAP is not None:
                    mreset()
                return _proj_loss(_p(_with_x(state0, x), mesh, sigma,
                                     phys_state=ps0, forcing=forcing0))
            grad_check(f, x0, v, a.h, label)
            if a.stash_modules:
                print_call_stash(mrec, label)
                CALL_TAP = None
    elif a.stage == "params":
        from legoesm.training.aimip_params import (
            _splice_scheme_overrides,
            aimip_active_scheme_keys,
            resolved_inactive_fields,
        )
        from legoesm.training.inert_params import assert_no_inert
        from legoesm.training.param_collector import build_trainable_params
        keys = aimip_active_scheme_keys(exp["phys_cfg"])
        # the production trainable set: knobs the resolved options never read
        # are frozen out by name, the rest must carry gradient (gate below)
        frozen = resolved_inactive_fields(
            exp["phys_cfg"], subgrid_orography=getattr(
                drv.grid, "subgrid_topo_stddev", None) is not None)
        tp = build_trainable_params(active_scheme_keys=keys, tier="core",
                                    dtype=jnp.float64,
                                    exclude=tuple(sorted(frozen)))
        print(f"active scheme keys: {sorted(keys)}; {len(tp.raw_values)} "
              f"core parameters; frozen out: {sorted(frozen)}", flush=True)

        def f(raw):
            trial = type(tp)(raw_values=raw, constraints=tp.constraints)
            cfg_ = _splice_scheme_overrides(exp["phys_cfg"], trial.to_overrides())
            pfn = exp["build_physics"](cfg_)
            return _proj_loss(pfn(state0, mesh, sigma, phys_state=ps0,
                                  forcing=forcing0))
        raw0 = dict(tp.raw_values)
        g = jax.jit(jax.grad(f))(raw0)
        for k in sorted(g):
            gk = np.asarray(g[k])
            print(f"PARAMGRAD {k} |g|={float(np.max(np.abs(gk))):.3e} "
                  f"finite={bool(np.all(np.isfinite(gk)))}", flush=True)
        dv = {k: jnp.asarray(0.1 * np.random.default_rng(5).standard_normal(
            np.shape(x))) for k, x in raw0.items()}
        if not a.per_knob:
            grad_check(f, raw0, dv, 1e-3, "physics_wrt_core_params")
        # per knob: reverse AD vs central FD at two steps (forward-mode jvp
        # is unavailable: the shared Thomas solver is a custom_vjp).  FD
        # stable in h and != AD -> the graph drops a dependence (stop_gradient,
        # degenerate branch, custom_vjp); FD that moves with h -> a kink.
        fj = jax.jit(f)
        for k in (sorted(raw0) if a.per_knob else ()):
            fds = []
            for hk in (1e-2, 1e-4):
                up = dict(raw0, **{k: raw0[k] + hk})
                dn = dict(raw0, **{k: raw0[k] - hk})
                fds.append(float((fj(up) - fj(dn)) / (2 * hk)))
            ad = float(np.sum(np.asarray(g[k])))
            print(f"PARAMFD {k} ad={ad:.6e} fd1e-2={fds[0]:.6e} "
                  f"fd1e-4={fds[1]:.6e}", flush=True)
        # gate LAST, so the diagnostics above are printed for a dead knob too
        assert_no_inert(g)
        print("NO_INERT_GATE passed", flush=True)
    elif a.stage == "dycore":
        tm = model.mass_target(state0)

        def f(x):
            return _proj_loss(model.step_pure(_with_x(state0, x), exp["dt"],
                                              target_mass=tm)[0])
        grad_check(f, x0, v, a.h, "dycore_step")
    elif a.stage == "step":
        from legoesm.driver.mpas_diff_step import mpas_segment
        sched = None if a.schedule == "none" else a.schedule
        seg_kw = (dict(checkpoint=True) if sched is None else
                  dict(schedule=sched, checkpoints=a.checkpoints,
                       storage=a.storage))
        if a.stash_grads and sched is not None:
            raise SystemExit("--stash-grads needs --schedule none")
        for n in (1, a.n_steps):
            kw = dict(seg_kw, step0=a.at_step)
            if a.stash_grads:
                tap, rec, names = make_grad_stash(n, (state0, ps0),
                                                  step0=a.at_step)
                kw["tap"] = tap

            def f(x, _n=n, _kw=kw):
                s, _ = mpas_segment(exp, _with_x(state0, x), ps0, _n, **_kw)
                return _proj_loss(s)
            grad_check(f, x0, v, a.h, f"segment_{n}_steps")
            print(f"PEAK_GIB segment_{n}_steps {peak_gib():.2f}", flush=True)
            if a.stash_grads:
                print_grad_stash(rec, names, f"segment_{n}_steps")
        if sched is not None:
            # forward of the scheduled loop vs the unrolled segment (eager
            # per-step jit, as the eager stage's reference)
            s_ref, _ = mpas_segment(exp, state0, ps0, a.n_steps,
                                    step0=a.at_step)
            s_sch, _ = jax.jit(lambda s, p: mpas_segment(
                exp, s, p, a.n_steps, step0=a.at_step, **seg_kw))(state0, ps0)
            for k in ("T", "u", "p_s"):
                ref_k, sch_k = getattr(s_ref, k).data, getattr(s_sch, k).data
                print(f"SCHED_VS_UNROLLED {k} max|diff| = "
                      f"{float(jnp.max(jnp.abs(sch_k - ref_k))):.3e} "
                      f"(scale {float(jnp.max(jnp.abs(ref_k))):.3e})", flush=True)
            dq = float(jnp.max(jnp.abs(s_sch.tracers["q_v"].data
                                       - s_ref.tracers["q_v"].data)))
            print(f"SCHED_VS_UNROLLED q_v max|diff| = {dq:.3e}", flush=True)
    elif a.stage == "land":
        if exp["land_step"] is None:
            print("land: OFF in this config", flush=True)
            return 0
        # one eager production step populates the surface export the land
        # marshalling reads (sfc_diag), exactly as the loop's first land call
        model.step(state0, exp["dt"], physics_fn=exp["physics_fn"],
                   forcing=forcing0, phys_state=ps0)
        a2s = exp["marshal_land"]()
        doy64 = jnp.asarray(doy, dtype=jnp.float64)
        land0 = exp["land_state"]

        def f(z):
            out = exp["land_step"](land0, z, doy64, doy64 - 1.0,
                                   jnp.asarray(1979.0, jnp.float64))
            return _proj_loss(out[1:9])   # skin T, albedo, q_sfc, fluxes, stress
        fl = {k: getattr(a2s, k) for k in ("T_lowest", "q_lowest", "sw_down",
                                           "lw_down", "u_lowest", "v_lowest")}
        key = jax.random.PRNGKey(3)
        dv = {k: 1e-3 * (jnp.abs(x) + 1e-6) * jax.random.normal(
            jax.random.fold_in(key, i), x.shape, x.dtype)
            for i, (k, x) in enumerate(fl.items())}
        grad_check(lambda z: f(a2s._replace(**z)), fl, dv, 1.0, "land_step_wrt_a2s")
    elif a.stage == "eager":
        from legoesm.driver.mpas_diff_step import mpas_segment
        s_seg, _ = mpas_segment(exp, state0, ps0, a.n_steps)
        drv2 = build_driver(rest, a.resolution, a.n_steps)
        print("eager status: " + drv2.run(), flush=True)
        for k in ("T", "u", "p_s"):
            d = float(jnp.max(jnp.abs(getattr(drv2.state, k).data
                                      - getattr(s_seg, k).data)))
            print(f"EAGER_VS_SEGMENT {k} max|diff| = {d:.3e} "
                  f"(scale {float(jnp.max(jnp.abs(getattr(s_seg, k).data))):.3e})",
                  flush=True)
        dq = float(jnp.max(jnp.abs(drv2.state.tracers['q_v'].data
                                   - s_seg.tracers['q_v'].data)))
        print(f"EAGER_VS_SEGMENT q_v max|diff| = {dq:.3e}", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
