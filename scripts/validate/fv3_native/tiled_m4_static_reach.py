"""STATIC stencil-reach analysis of the acoustic substep -- the tile pad
depth derived from the PROGRAM, not from probing values.

WHY THIS EXISTS.  Every value-probing measurement of reach can UNDER-report:
a flux limiter, a monotonicity constraint or a min/max tie can annihilate a
perturbation, so the footprint stops short of the true data dependence.  That
is not hypothetical here -- the perturbation ladder measured reach 5 at
eps=1e-8 and reach 7 at eps=1e-3 on the SAME configuration (job 9610385),
which is exactly that failure.  A probe of that kind can only ever report a
FLOOR.  This analyser reads the traced program instead, so a limiter cannot
hide a dependence: the limiter is itself just an operation whose output
depends on its inputs.

METHOD.  Trace one acoustic substep to a jaxpr and interpret it in a TAINT
abstraction: every array is replaced by a boolean array of the same shape,
arithmetic is replaced by logical OR over the operands, and index-moving
operations (slice, pad, concatenate, transpose, reshape, reverse, gather,
reductions, scans) are applied to the booleans with their own semantics.
Taint one interior line of the inputs; whatever comes out tainted is what
genuinely reads that line.  Values never enter, so nothing can be cancelled,
clipped or swallowed.

FAIL-CLOSED.  An unknown primitive RAISES with its name rather than being
skipped -- skipping one would silently under-report reach, which is the very
failure this instrument exists to remove.  Extend ``_RULES`` deliberately.

CONTRACT WITH THE PROBE.  The static number is the SOURCE; the perturbation
ladder is its VALIDATION, and they may disagree in one direction only: the
probe may report LESS than this bound (a swallow), never more.  A probe
reading HIGHER means this analyser missed an operation -- a defect in the
analyser, and worth catching.  ``--compare`` prints that check explicitly.
"""
from __future__ import annotations

import argparse

import numpy as np


class UnknownPrimitive(Exception):
    """A jaxpr primitive with no taint rule.  Fail-closed: skipping it
    would drop a data dependence and UNDER-report the reach."""


def _or_reduce(vals, out_shape, xp=np):
    """Elementwise OR of operands broadcast to ``out_shape``."""
    acc = xp.zeros(out_shape, bool)
    for v in vals:
        if not hasattr(v, "shape"):
            continue                     # python scalar constant: untainted
        b = xp.asarray(v, bool)
        if b.shape != out_shape:
            b = xp.broadcast_to(_align(b, len(out_shape), xp), out_shape)
        acc = acc | b
    return acc


def _align(b, ndim, xp=np):
    while b.ndim < ndim:
        b = xp.expand_dims(b, 0)
    return b


def _rule_elementwise(eqn, invals, jnp):
    shape = eqn.outvars[0].aval.shape
    return [_or_reduce(invals, shape, jnp)]


def _rule_identity(eqn, invals, jnp):
    return [jnp.asarray(invals[0], bool)]


def _rule_reshape(eqn, invals, jnp):
    return [jnp.reshape(jnp.asarray(invals[0], bool),
                        eqn.outvars[0].aval.shape)]


def _rule_transpose(eqn, invals, jnp):
    return [jnp.transpose(jnp.asarray(invals[0], bool),
                          eqn.params["permutation"])]


def _rule_broadcast(eqn, invals, jnp):
    p = eqn.params
    x = jnp.asarray(invals[0], bool)
    return [jnp.broadcast_to(
        jnp.expand_dims(x, [i for i in range(len(p["shape"]))
                            if i not in p["broadcast_dimensions"]]),
        p["shape"])]


def _rule_slice(eqn, invals, jnp):
    p = eqn.params
    x = jnp.asarray(invals[0], bool)
    sl = tuple(slice(s, l, st) for s, l, st in
               zip(p["start_indices"], p["limit_indices"],
                   p["strides"] or [1] * x.ndim))
    return [x[sl]]


def _rule_pad(eqn, invals, jnp):
    p = eqn.params
    x = jnp.asarray(invals[0], bool)
    # padding value is invals[1]; a padded cell carries NO taint
    lo_hi = [(lo, hi) for lo, hi, _interior in p["padding_config"]]
    if any(i != 0 for _, _, i in p["padding_config"]):
        raise UnknownPrimitive("pad with interior padding")
    return [jnp.pad(x, lo_hi, constant_values=False)]


def _rule_concat(eqn, invals, jnp):
    return [jnp.concatenate([jnp.asarray(v, bool) for v in invals],
                            axis=eqn.params["dimension"])]


def _rule_rev(eqn, invals, jnp):
    return [jnp.flip(jnp.asarray(invals[0], bool),
                     axis=tuple(eqn.params["dimensions"]))]


def _rule_squeeze(eqn, invals, jnp):
    return [jnp.squeeze(jnp.asarray(invals[0], bool),
                        axis=tuple(eqn.params["dimensions"]))]


def _rule_reduce(eqn, invals, jnp):
    """Any reduction: the output depends on EVERY reduced element."""
    return [jnp.any(jnp.asarray(invals[0], bool),
                    axis=tuple(eqn.params["axes"]))]


def _rule_argminmax(eqn, invals, jnp):
    return [jnp.any(jnp.asarray(invals[0], bool),
                    axis=tuple(eqn.params["axes"]))]


def _rule_select(eqn, invals, jnp):
    # select_n(pred, *cases): the result depends on the predicate AND on
    # every case -- a limiter cannot hide a dependence here, which is the
    # whole point of doing this statically.
    return _rule_elementwise(eqn, invals, jnp)


def _rule_dot(eqn, invals, jnp):
    """Conservative: contract by OR over the contracting dimensions."""
    p = eqn.params["dimension_numbers"]
    (lc, rc), (lb, rb) = p
    a = jnp.asarray(invals[0], bool)
    b = jnp.asarray(invals[1], bool)
    a2 = jnp.any(a, axis=tuple(lc)) if lc else a
    b2 = jnp.any(b, axis=tuple(rc)) if rc else b
    out_shape = eqn.outvars[0].aval.shape
    return [jnp.broadcast_to(
        jnp.zeros((), bool) | jnp.any(a2) | jnp.any(b2), out_shape)]


def _index_tainted(invals, first):
    """True if any INDEX operand (positions ``first:``) carries taint: the
    read/write POSITION then depends on seam data, so every output cell
    does -- a concrete index value does not remove that dependence
    (codex 2026-09-04)."""
    return any(hasattr(v, "shape") and bool(np.asarray(v, bool).any())
               for v in invals[first:])


def _rule_gather(eqn, invals, jnp):
    """SOUND and axis-aware: a gather with traced indices can read
    anywhere along the axes its start indices move (``start_index_map``),
    the collapsed and batching axes, and any axis whose slice is
    narrower than the operand.  Along every OTHER axis the slice starts
    at 0 and spans the operand, so output position == operand position
    (``x[:, :, k]`` with traced ``k`` keeps its horizontal footprint).
    Unknown axes are reduced with ``any`` and broadcast; nothing is
    dropped, so this over-reports along the moving axes only."""
    x = np.asarray(invals[0], bool)
    out_shape = tuple(eqn.outvars[0].aval.shape)
    if _index_tainted(invals, 1):
        return [np.ones(out_shape, bool)]
    dn = eqn.params["dimension_numbers"]
    sizes = tuple(eqn.params["slice_sizes"])
    collapsed = set(dn.collapsed_slice_dims)
    obatch = set(getattr(dn, "operand_batching_dims", ()))
    odims = [d for d in range(x.ndim) if d not in collapsed and d not in obatch]
    offs = list(dn.offset_dims)
    if len(odims) != len(offs):
        return [np.broadcast_to(np.any(x), out_shape)]
    unknown = set(dn.start_index_map) | collapsed | obatch
    unknown |= {d for d in range(x.ndim) if sizes[d] != x.shape[d]}
    kept = [(d, o) for d, o in zip(odims, offs) if d not in unknown]
    red = tuple(d for d in range(x.ndim) if d not in {d for d, _ in kept})
    reduced = np.any(x, axis=red) if red else x
    shape = [1] * len(out_shape)
    for d, o in kept:
        shape[o] = x.shape[d]
    return [np.broadcast_to(reduced.reshape(shape), out_shape)]


def _rule_dynamic_slice(eqn, invals, jnp):
    """Same axis rule as gather: a start index is clamped so a full-extent
    slice always starts at 0 -- those axes keep position; the rest are
    reduced with ``any`` and broadcast over the slice."""
    x = np.asarray(invals[0], bool)
    out_shape = tuple(eqn.outvars[0].aval.shape)
    if _index_tainted(invals, 1):
        return [np.ones(out_shape, bool)]
    sizes = tuple(eqn.params["slice_sizes"])
    unknown = tuple(d for d in range(x.ndim) if sizes[d] != x.shape[d])
    reduced = np.any(x, axis=unknown, keepdims=True) if unknown else x
    return [np.broadcast_to(reduced, out_shape)]


def _rule_dynamic_update_slice(eqn, invals, jnp):
    """operand with ``update`` written at a traced offset: full-extent
    axes keep position (offset clamps to 0), narrower axes are unknown
    and take the update's ``any`` along them."""
    x = np.asarray(invals[0], bool)
    u = np.asarray(invals[1], bool)
    if _index_tainted(invals, 2):
        return [np.ones(x.shape, bool)]
    unknown = tuple(d for d in range(x.ndim) if u.shape[d] != x.shape[d])
    reduced = np.any(u, axis=unknown, keepdims=True) if unknown else u
    return [x | np.broadcast_to(reduced, x.shape)]


_INDEXED = ("gather", "dynamic_slice", "dynamic_update_slice")


def _rule_indexed_exact(eqn, invals, inv_vals):
    """gather / dynamic_slice / dynamic_update_slice with CONCRETE index
    operands: run the primitive itself on the taint (as int8) with those
    indices, which is exact -- the very positions the program reads or
    writes.  Falls back to the axis-aware rule on any binding failure."""
    import jax

    first = 2 if eqn.primitive.name == "dynamic_update_slice" else 1
    if _index_tainted(invals, first):
        return [np.ones(tuple(eqn.outvars[0].aval.shape), bool)]
    try:
        if eqn.primitive.name == "dynamic_update_slice":
            op = np.asarray(invals[0], bool).astype(np.int8)
            up = np.asarray(invals[1], bool).astype(np.int8)
            idx = [np.asarray(v) for v in inv_vals[2:]]
            out = jax.lax.dynamic_update_slice(op, up, idx)
        else:
            op = np.asarray(invals[0], bool).astype(np.int8)
            idx = [np.asarray(v) for v in inv_vals[1:]]
            out = eqn.primitive.bind(op, *idx, **eqn.params)
        return [np.asarray(out).astype(bool)]
    except Exception:
        return _RULES[eqn.primitive.name](eqn, invals, np)


def _rule_convert(eqn, invals, jnp):
    return [jnp.asarray(invals[0], bool)]


def _rule_iota(eqn, invals, jnp):
    return [jnp.zeros(eqn.outvars[0].aval.shape, bool)]


def _rule_scatter(eqn, invals, jnp, idx=None):
    """scatter(operand, indices, updates): the output is the operand with
    updates written at ``indices``.  Taint = operand taint, with the
    updates' taint OR-ed in AT THE WRITTEN POSITIONS.

    The written positions are INDEX VALUES, which a taint-only
    interpretation does not have -- hence the concrete side-channel
    (integer subgraphs are evaluated for real; see ``_concrete``).  With
    the indices unavailable the rule falls back to tainting the WHOLE
    output, which over-reports rather than under-reports: safe for a pad
    bound, and it says so out loud so a saturated answer is never read as
    a real reach."""
    op_t = np.asarray(invals[0], bool)
    upd_t = np.asarray(invals[2], bool) if len(invals) > 2 else None
    if len(invals) > 1 and _index_tainted(invals[:2], 1):
        return [np.ones(op_t.shape, bool)]     # write POSITION is tainted
    if upd_t is None or not upd_t.any():
        return [op_t]
    dn = eqn.params["dimension_numbers"]
    if idx is None:
        _SCATTER_FALLBACKS.append(eqn.primitive.name)
        return [_scatter_unknown_index(op_t, upd_t, dn)]
    # Concrete indices: let the scatter itself place the update taint.
    # A replace-scatter of int8 taint is EXACT at every written position;
    # for scatter-add/mul/min/max only the written SET is taken from it
    # (mul/min/max of int8 could zero a tainted operand cell, so the
    # operand taint is kept by OR).  The earlier rule OR-ed ``upd.any()``
    # over the whole index ROW, which for ``x.at[:, k].set(u)`` tainted
    # all of column k from one tainted cell -- a smear that read 20 on
    # the fori selftest (job 9630872) and 15/27 on the FV3 substep.
    import jax

    try:
        kw = dict(indices_are_sorted=bool(eqn.params.get(
                      "indices_are_sorted", False)),
                  unique_indices=bool(eqn.params.get("unique_indices",
                                                     False)),
                  mode=eqn.params.get("mode"))
        i8 = np.asarray(upd_t, np.int8)
        zeros = np.zeros(op_t.shape, np.int8)
        # scatter_ADD so duplicate indices union instead of last-wins
        written = np.asarray(jax.lax.scatter_add(zeros, np.asarray(idx),
                                                 i8, dn, **kw)) > 0
        if eqn.primitive.name == "scatter" and kw["unique_indices"]:
            # replace with unique targets: written cells take the
            # update's taint exactly, everything else keeps the operand's
            mask = np.asarray(jax.lax.scatter_add(
                zeros, np.asarray(idx), np.ones_like(i8), dn, **kw)) > 0
            return [np.where(mask, written, op_t)]
        return [op_t | written]
    except Exception:
        _SCATTER_FALLBACKS.append(eqn.primitive.name)
        return [_scatter_unknown_index(op_t, upd_t, dn)]


_SCATTER_FALLBACKS = []


def _scatter_unknown_index(op_t, upd_t, dn):
    """Scatter with UNKNOWN write positions: taint only the operand axes
    the index can move along.  XLA's start index for operand dim ``d`` is
    the index value if ``d`` is in ``scatter_dims_to_operand_dims`` and 0
    otherwise, so a window dim whose update extent equals the operand
    extent maps position-to-position -- e.g. ``x.at[:, :, k].set(u)``
    with a traced ``k`` (a level recurrence) writes ``u[i, j]`` at
    ``(i, j)`` for SOME k.  Tainting every (i, j, k) for that case (the
    previous fallback) smeared a vertical write across the whole face and
    made the horizontal bound grow with the face size (15 at C24, 27 at
    C48 in job 9621158).  Axes the index CAN move along, inserted dims,
    batching dims and partial windows are still tainted end to end, so
    the result over-reports along those axes and never under-reports."""
    nd = op_t.ndim
    inserted = set(dn.inserted_window_dims)
    obatch = set(getattr(dn, "operand_batching_dims", ()))
    wdims = [d for d in range(nd) if d not in inserted and d not in obatch]
    uwin = list(dn.update_window_dims)
    if len(wdims) != len(uwin):
        return np.ones(op_t.shape, bool)
    unknown = set(dn.scatter_dims_to_operand_dims) | inserted | obatch
    for ud, wd in zip(uwin, wdims):
        if upd_t.shape[ud] != op_t.shape[wd]:
            unknown.add(wd)
    keep_ud = [ud for ud, wd in zip(uwin, wdims) if wd not in unknown]
    red_axes = tuple(a for a in range(upd_t.ndim) if a not in keep_ud)
    reduced = np.any(upd_t, axis=red_axes) if red_axes else upd_t
    # ``reduced`` axes are the kept window dims in operand order
    # (update_window_dims is sorted, and the pairing preserves order).
    shape = [1] * nd
    for ud, wd in zip(uwin, wdims):
        if wd not in unknown:
            shape[wd] = op_t.shape[wd]
    return op_t | np.broadcast_to(reduced.reshape(shape), op_t.shape)


_ELEMENTWISE = (
    "add", "sub", "mul", "div", "max", "min", "pow", "integer_pow", "rem",
    "exp", "log", "log1p", "expm1", "sqrt", "rsqrt", "cbrt", "tanh", "sin",
    "cos", "tan", "atan", "atan2", "abs", "sign", "neg", "floor", "ceil",
    "round", "erf", "logistic", "and", "or", "xor", "not", "eq", "ne",
    "ge", "gt", "le", "lt", "nextafter", "clamp", "sub_any", "square",
    "is_finite", "select_and_scatter_add", "atan2", "copy",
)

_RULES = {
    **{k: _rule_elementwise for k in _ELEMENTWISE},
    "select_n": _rule_select,
    "reshape": _rule_reshape,
    "transpose": _rule_transpose,
    "broadcast_in_dim": _rule_broadcast,
    "slice": _rule_slice,
    "pad": _rule_pad,
    "concatenate": _rule_concat,
    "rev": _rule_rev,
    "squeeze": _rule_squeeze,
    "reduce_sum": _rule_reduce,
    "reduce_max": _rule_reduce,
    "reduce_min": _rule_reduce,
    "reduce_prod": _rule_reduce,
    "reduce_and": _rule_reduce,
    "reduce_or": _rule_reduce,
    "argmax": _rule_argminmax,
    "argmin": _rule_argminmax,
    "dot_general": _rule_dot,
    "gather": _rule_gather,
    "dynamic_slice": _rule_dynamic_slice,
    "dynamic_update_slice": _rule_dynamic_update_slice,
    "convert_element_type": _rule_convert,
    "stop_gradient": _rule_identity,
    "copy": _rule_identity,
    "iota": _rule_iota,
    "reduce_precision": _rule_identity,
}

_SCATTER_PRIMS = ("scatter", "scatter-add", "scatter_add", "scatter-mul",
                  "scatter_mul", "scatter-min", "scatter_min",
                  "scatter-max", "scatter_max")


def _concrete(eqn, vals):
    """Evaluate an equation FOR REAL when every input is concrete and
    INTEGRAL -- i.e. index arithmetic.  Float subgraphs are never
    evaluated (cost, and taint does not need them).  This is what gives
    the scatter rule its write positions."""
    if any(v is None for v in vals):
        return None
    try:
        arrs = [np.asarray(v) for v in vals]
    except Exception:
        return None
    if not all(a.dtype.kind in "iub" for a in arrs):
        return None
    if sum(a.size for a in arrs) > 4_000_000:
        return None
    try:
        out = eqn.primitive.bind(*arrs, **eqn.params)
    except Exception:
        return None
    return out if eqn.primitive.multiple_results else [out]


def _interp(jaxpr, consts, args, jnp, depth=0, stats=None,
            const_taints=None, arg_vals=None, const_vals=None,
            out_vals=None):
    """Taint-interpret one jaxpr.  ``stats`` accumulates, per primitive,
    the largest horizontal reach its output achieved -- so the reported
    number arrives with the operation that produced it.

    ``const_taints`` covers the case where a caller hoisted the
    sub-jaxpr's constants into leading operands (see ``_bind_call``):
    the constvars then carry the CALLER'S taint rather than being clean
    literals, and zeroing them would under-report.

    ``arg_vals`` / ``const_vals`` carry the caller's CONCRETE integer
    values (None = unknown) into this sub-jaxpr, and ``out_vals`` (a
    list, filled in place) hands the outvars' concrete values back.  This
    is what lets a scan's level counter stay concrete inside its body, so
    a level-indexed scatter/gather writes and reads ONE level instead of
    being smeared over all of them (the smear chained across iterations
    and made the C24/C48 bound read 15/27, job 9630774)."""
    env = {}
    env_val = {}          # concrete values for INTEGER subgraphs only

    def read(v):
        # jax.core.Literal is GONE in this version; the supported export is
        # jax.extend.core.Literal (jax/extend/core/__init__.py:28). Read the
        # API, never infer it -- the first draft of this file inferred it and
        # died on every arm (job 9610391).
        from jax.extend.core import Literal

        if isinstance(v, Literal):
            return False
        return env[v]

    def write(v, val):
        env[v] = val

    def read_val(v):
        from jax.extend.core import Literal

        if isinstance(v, Literal):
            return v.val
        return env_val.get(v)

    if const_taints is not None:
        if len(const_taints) != len(jaxpr.constvars):
            raise UnknownPrimitive(
                f"const_taints arity {len(const_taints)} != "
                f"{len(jaxpr.constvars)} constvars")
        for i, (cv, t) in enumerate(zip(jaxpr.constvars, const_taints)):
            write(cv, np.asarray(t, bool))
            if const_vals is not None and const_vals[i] is not None:
                env_val[cv] = np.asarray(const_vals[i])
    else:
        if len(consts) != len(jaxpr.constvars):
            raise UnknownPrimitive(
                f"consts arity {len(consts)} != {len(jaxpr.constvars)} "
                f"constvars -- refusing to zip-truncate, which would leave "
                f"constvars unbound and drop a dependence")
        for cv, c in zip(jaxpr.constvars, consts):
            write(cv, jnp.zeros(np.shape(c), bool))
            env_val[cv] = np.asarray(c)  # constants keep their VALUES
    if len(args) != len(jaxpr.invars):
        raise UnknownPrimitive(
            f"operand arity {len(args)} != {len(jaxpr.invars)} invars -- "
            f"refusing to zip-truncate, which would silently drop inputs")
    for i, (iv, a) in enumerate(zip(jaxpr.invars, args)):
        write(iv, a)
        if arg_vals is not None and arg_vals[i] is not None:
            env_val[iv] = np.asarray(arg_vals[i])

    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        invals = [read(v) for v in eqn.invars]
        inv_vals = [read_val(v) for v in eqn.invars]
        ho_vals = None
        if name in _HIGHER_ORDER:
            ho_vals = []
            outs = _interp_higher_order(eqn, invals, jnp, depth, stats,
                                        inv_vals, ho_vals)
        elif name in _SCATTER_PRIMS:
            idx = read_val(eqn.invars[1]) if len(eqn.invars) > 1 else None
            outs = _rule_scatter(eqn, invals, jnp, idx)
        elif name in _INDEXED and all(v is not None for v in inv_vals[1:]) \
                and (name != "dynamic_update_slice"
                     or all(v is not None for v in inv_vals[2:])):
            outs = _rule_indexed_exact(eqn, invals, inv_vals)
        else:
            rule = _RULES.get(name)
            if rule is None:
                raise UnknownPrimitive(
                    f"no taint rule for primitive {name!r} (depth {depth}) "
                    f"-- refusing to skip it: an unhandled operation drops a "
                    f"data dependence and UNDER-reports the reach, which is "
                    f"the exact failure this analyser exists to remove")
            outs = rule(eqn, invals, jnp)
        cval = _concrete(eqn, inv_vals)
        if cval is None and ho_vals:
            cval = ho_vals
        for i, (v, o) in enumerate(zip(eqn.outvars, outs)):
            write(v, o)
            if cval is not None and i < len(cval) and cval[i] is not None:
                env_val[v] = np.asarray(cval[i])
        if stats is not None:
            ext = stats.get("extent")

            m_a = max(ext) - 1 if ext else None

            def _seam_axis(x):
                # (axis, seam) for the array's layout: face-stacked
                # (6, i, j, ...) -> axis 1; per-face (i, j, ...) -> axis
                # 0.  A symmetric sub-window of extent e (interior n,
                # is-1:ie+1, ...) has its seam shifted by (m_a - e) // 2.
                # Anything else (flattened, transposed) has no seam to
                # measure and would read saturation by construction.
                if not hasattr(x, "ndim") or ext is None:
                    return (stats["axis"], stats["seam"]) if hasattr(
                        x, "ndim") else None
                lo, hi = m_a - 2 * ng_stats, m_a + 1
                if x.ndim >= 3 and x.shape[0] == 6 and lo <= x.shape[1] <= hi:
                    return 1, stats["seam"] - (m_a - x.shape[1]) // 2
                if x.ndim >= 2 and lo <= x.shape[0] <= hi \
                        and lo <= x.shape[1] <= hi:
                    return 0, stats["seam"] - (m_a - x.shape[0]) // 2
                return None
            ng_stats = stats.get("ng", 3)
            rin = max((_reach_of(x, *_seam_axis(x))
                       for x in invals if _seam_axis(x) is not None),
                      default=-1)
            for o in outs:
                ax = _seam_axis(o)
                if ax is None:
                    continue
                r = _reach_of(o, *ax)
                if r > stats["per_prim"].get(name, -1):
                    stats["per_prim"][name] = r
                # a reach that grows by more than a stencil width in ONE
                # operation is where a bound saturates; name it
                # rin == -1: NO input had a seam layout, yet the output
                # carries reach -- taint arrived through a reduced /
                # flattened / transposed intermediate (the reduce ->
                # broadcast signature of a nonlocal operator)
                layout = name in ("reshape", "transpose", "squeeze",
                                  "concatenate", "slice", "broadcast_in_dim",
                                  "convert_element_type", "copy", "copy_p")
                if "jumps" in stats and ((rin >= 0 and r - rin > 2) or
                                         (rin < 0 and r >= 3
                                          and not layout)):
                    stats["jumps"].append((r - rin, rin, r, name, depth,
                                           tuple(np.shape(o)),
                                           [tuple(np.shape(x)) for x in
                                            invals if hasattr(x, "shape")],
                                           list(stats.get("hist", []))))
            if "leak" in stats:
                # taint on faces 1-5 at ANY point (not only at the end --
                # it could leave face 0, return, and be overwritten later,
                # codex 2026-09-04): a nonzero maximum means the measured
                # reach may include an out-and-back path through the
                # halo tables and is then CONFOUNDED
                for o in outs:
                    # face-stacked AND padded on both horizontal axes: a
                    # level-major per-face array (km+1 = 6, i, j) also has
                    # shape[0] == 6 at km=5 and false-alarmed job 9631126
                    if (hasattr(o, "ndim") and o.ndim >= 3
                            and o.shape[0] == 6 and ext is not None
                            and o.shape[1] in ext and o.shape[2] in ext):
                        c = int(np.asarray(o[1:], bool).sum())
                        if c and stats.get("leak_first") is None:
                            stats["leak_first"] = (
                                c, name, depth, tuple(np.shape(o)),
                                [tuple(np.shape(x)) for x in invals
                                 if hasattr(x, "shape")],
                                list(stats.get("hist", [])))
                        if c > stats["leak"][0]:
                            stats["leak"] = (c, name, tuple(np.shape(o)))
            if "hist" in stats:
                # short trail of the eqns that fed a jump: name, depth,
                # output shape, and how many output cells are tainted
                stats["hist"].append((name, depth, tuple(np.shape(outs[0]))
                                      if outs and hasattr(outs[0], "shape")
                                      else None,
                                      int(np.asarray(outs[0], bool).sum())
                                      if outs and hasattr(outs[0], "shape")
                                      else -1))
                del stats["hist"][:-8]
    if out_vals is not None:
        del out_vals[:]
        out_vals.extend(read_val(v) for v in jaxpr.outvars)
    return [read(v) for v in jaxpr.outvars]


# The nested-call primitives, with the source each NAME was read from.
# Read, never recalled: the previous draft of this file guessed 'pjit'
# (the primitive is 'jit') and 'remat'/'checkpoint' (it is 'remat2'), and
# every arm refused at depth 0 on job 9610403 as a result.
_HIGHER_ORDER = (
    "jit",                    # jax/_src/pjit.py:915
    "closed_call",            # jax/_src/core.py:3026
    "call",                   # jax/_src/core.py:3013
    "scan",                   # jax/_src/lax/control_flow/loops.py:1359
    "while",                  # jax/_src/lax/control_flow/loops.py:2227
    "cond",                   # jax/_src/lax/control_flow/conditionals.py:941
    "custom_jvp_call",        # jax/_src/custom_derivatives.py:427
    "custom_vjp_call",        # jax/_src/custom_derivatives.py:1034
    "custom_jvp_call_jaxpr",  # jax/_src/custom_derivatives.py:1624
    "remat2",                 # jax/_src/ad_checkpoint.py:573
)

# Sub-jaxpr param key per primitive, each read from the source cited.
#   jit          params['jaxpr']       jax/_src/pjit.py:916, :1372
#   closed_call  params['call_jaxpr']  jax/_src/core.py:3028
#   call         params['call_jaxpr']  jax/_src/core.py:2990 (CallPrimitive)
#   scan         params['jaxpr']       jax/_src/lax/control_flow/loops.py:324
#   cond         params['branches']    .../conditionals.py:317
#   while        cond_jaxpr/body_jaxpr .../loops.py:1560
#   custom_*     params['call_jaxpr']  jax/_src/custom_derivatives.py:1039
#   remat2       params['jaxpr']       jax/_src/ad_checkpoint.py:583
_SUBJAXPR_KEYS = ("jaxpr", "call_jaxpr", "branches")


def _open(sub):
    """(jaxpr, consts) from either a ClosedJaxpr or a bare Jaxpr.
    remat2 binds a BARE Jaxpr (jax/_src/ad_checkpoint.py:583 calls
    ``core.eval_jaxpr(jaxpr, (), *args)``), jit/scan/cond bind ClosedJaxprs."""
    if hasattr(sub, "jaxpr"):
        return sub.jaxpr, list(sub.consts)
    return sub, []


def _bind_call(name, sub, invals):
    """Bind an eqn's operands onto a sub-jaxpr, returning
    ``(body, consts, const_taints, args)``.

    Two conventions occur in this jax and they differ only in ARITY:
      * the ClosedJaxpr carries its own ``.consts`` and the operands map
        1:1 onto ``jaxpr.invars`` -- e.g. closed_call binds
        ``partial(eval_jaxpr, jaxpr.jaxpr, jaxpr.consts)``
        (jax/_src/core.py:3021);
      * the constants are HOISTED into leading operands ('const args'),
        asserted at jax/_src/pjit.py:1384 as
        ``len(in_avals) == num_const_args + len(jaxpr.in_avals)``.
        Off by default here (jax_use_simplified_jaxpr_constants=False,
        jax/_src/config.py:1342) but handled so a flag flip cannot
        silently corrupt the bound.
    Choose by arity and REFUSE if neither fits: a mis-binding shuffles
    taint between variables and moves the answer in an unknown
    direction, which is precisely what fail-closed exists to prevent."""
    body, consts = _open(sub)
    n_in, n_cv = len(body.invars), len(body.constvars)
    if len(invals) == n_in and len(consts) == n_cv:
        return body, consts, None, list(invals)
    if len(invals) == n_in + n_cv and not consts:
        return body, [], list(invals[:n_cv]), list(invals[n_cv:])
    raise UnknownPrimitive(
        f"{name}: cannot bind {len(invals)} operands onto a sub-jaxpr with "
        f"{n_cv} constvars ({len(consts)} closed-over consts) and {n_in} "
        f"invars -- refusing to guess a calling convention, because a "
        f"mis-binding silently corrupts the taint")


def _call_sub(name, sub, invals, jnp, depth, stats, invals_v=None,
              out_vals=None):
    body, consts, ctaints, args = _bind_call(name, sub, invals)
    cvals = avals = None
    if invals_v is not None:
        n_cv = len(body.constvars)
        if ctaints is None:
            avals = list(invals_v)
        else:
            cvals, avals = list(invals_v[:n_cv]), list(invals_v[n_cv:])
    return _interp(body, consts, args, jnp, depth + 1, stats,
                   arg_vals=avals, const_vals=cvals, out_vals=out_vals,
                   const_taints=ctaints)


def _ctrl_or(outs, pred, jnp):
    """OR a control predicate's taint into every output.

    If the branch taken (cond) or the trip count (while) depends on
    seam data, then EVERY output depends on it.  Ignoring that would
    under-report, which this analyser may never do; over-reporting is
    the permitted direction."""
    if pred is None or not hasattr(pred, "shape"):
        return outs
    if not bool(np.asarray(pred, bool).any()):
        return outs
    _CONTROL_TAINTS.append(True)
    return [np.ones(np.shape(o), bool) if hasattr(o, "shape") else o
            for o in outs]


_CONTROL_TAINTS = []
_WHILE_LOOPS = []
_LONG_SCANS = []
_MAX_UNROLL = 512


def _interp_higher_order(eqn, invals, jnp, depth, stats, invals_v=None,
                         out_vals=None):
    """Recurse into a nested call's sub-jaxpr rather than treating it as
    opaque.  ``scan`` and ``while`` iterate their carry to a FIXPOINT
    (taint only ever grows, so this terminates); a single pass would
    under-report a recurrence.  ``cond`` takes the OR over ALL branches,
    never one -- at trace time either branch may run."""
    p = eqn.params
    name = eqn.primitive.name

    if name == "while":
        # while_p.bind(*cond_consts, *body_consts, *init_vals,
        #              cond_nconsts=, cond_jaxpr=, body_nconsts=,
        #              body_jaxpr=)   -- jax/_src/lax/control_flow/loops.py:1559
        cn, bn = p["cond_nconsts"], p["body_nconsts"]
        cconst = list(invals[:cn])
        bconst = list(invals[cn:cn + bn])
        carry = list(invals[cn + bn:])
        pred = None
        # Unlike scan, a while_loop's trip count is NOT known at trace
        # time, so a fixpoint is the only sound treatment -- and for a
        # stencil in the body that means the carry saturates.  That is a
        # true statement about an unbounded loop, but it makes the bound
        # useless, so every occurrence is counted and reported.
        _WHILE_LOOPS.append(depth)
        for _ in range(64):                 # fixpoint; taint only grows
            nc = len(carry)
            cv = (list(invals_v[:cn]) + [None] * nc) if invals_v else None
            bv = (list(invals_v[cn:cn + bn]) + [None] * nc) if invals_v \
                else None
            pred = _call_sub("while.cond", p["cond_jaxpr"],
                             cconst + carry, jnp, depth, stats, cv)[0]
            new = _call_sub("while.body", p["body_jaxpr"],
                            bconst + carry, jnp, depth, stats, bv)
            if all(bool(np.array_equal(a, b)) for a, b in zip(new, carry)):
                break
            carry = [np.asarray(a, bool) | np.asarray(b, bool)
                     for a, b in zip(new, carry)]
        return _ctrl_or(carry, pred, jnp)

    sub = None
    for key in _SUBJAXPR_KEYS:
        if key in p:
            sub = p[key]
            break
    if sub is None:
        raise UnknownPrimitive(f"{name}: no sub-jaxpr in params {list(p)}")

    if name == "cond":
        # cond_p.bind(index, *consts, *args, branches=...)
        # -- jax/_src/lax/control_flow/conditionals.py:317
        outs = None
        for br in sub:
            o = _call_sub(f"{name}.branch", br, invals[1:], jnp, depth, stats,
                          list(invals_v[1:]) if invals_v else None)
            outs = o if outs is None else [
                np.asarray(x, bool) | np.asarray(y, bool)
                for x, y in zip(outs, o)]
        return _ctrl_or(outs, invals[0], jnp)

    if name != "scan":
        return _call_sub(name, sub, invals, jnp, depth, stats, invals_v,
                         out_vals)

    # scan_p.bind(*consts, *args_flat, num_consts=, num_carry=, length=,
    #             jaxpr=)  -- jax/_src/lax/control_flow/loops.py:323
    n_consts = p["num_consts"]
    n_carry = p["num_carry"]
    cs = list(invals[:n_consts])
    carry = list(invals[n_consts:n_consts + n_carry])
    xs = invals[n_consts + n_carry:]
    xs_slices = [jnp.any(jnp.asarray(x, bool), axis=0) if hasattr(x, "ndim")
                 and x.ndim else x for x in xs]
    # A scan's trip count is STATIC (``length``), so unroll exactly that
    # many times.  Iterating to a fixpoint instead -- as an earlier draft
    # did -- keeps growing the carry past the number of steps the program
    # actually takes and saturates the array: the selftest's "scan k=5"
    # case reported 20 on a 41-cell line whose true reach is 5.
    length = int(p["length"])
    ys = None
    # Concrete values ride the carry (a fori_loop's counter is a carry),
    # so an indexed write in the body lands on ONE level per iteration.
    # With values in play the body is NOT iteration-invariant, so the old
    # "taint stopped growing -> break" shortcut is unsound (iteration k+1
    # may write a level k never touched); run the full static length.
    cs_v = list(invals_v[:n_consts]) if invals_v else [None] * n_consts
    carry_v = (list(invals_v[n_consts:n_consts + n_carry]) if invals_v
               else [None] * n_carry)
    xs_v = [None] * len(xs)
    have_vals = any(v is not None for v in cs_v + carry_v)
    for _ in range(min(length, _MAX_UNROLL)):
        body_out_v = []
        outs = _call_sub(name, sub, cs + carry + xs_slices, jnp, depth, stats,
                         cs_v + carry_v + xs_v, body_out_v)
        new_carry = outs[:n_carry]
        ys = outs[n_carry:]
        if body_out_v:
            carry_v = list(body_out_v[:n_carry])
        if not have_vals and all(bool(jnp.array_equal(a, b))
                                 for a, b in zip(new_carry, carry)):
            break                            # taint stopped growing early
        carry = [a | b for a, b in zip(new_carry, carry)]
    else:
        if length > _MAX_UNROLL:
            # Stopping short of ``length`` without reaching a fixpoint
            # would UNDER-report, so saturate instead and say so.
            _LONG_SCANS.append(length)
            carry = [np.ones(np.shape(c), bool) for c in carry]
    stacked = []
    for y in ys:
        yb = jnp.asarray(y, bool)
        stacked.append(jnp.broadcast_to(yb[None], (p["length"],) + yb.shape))
    return list(carry) + stacked


def _reach_of(b, axis, seam):
    """Largest |index - seam| along ``axis`` among tainted elements."""
    b = np.asarray(b, bool)
    if b.ndim <= axis or b.shape[axis] < 2:
        return -1
    idx = np.nonzero(np.any(
        b, axis=tuple(i for i in range(b.ndim) if i != axis)))[0]
    if idx.size == 0:
        return -1
    return int(max(abs(int(idx.min()) - seam), abs(int(idx.max()) - seam)))


def _selftest():
    """Non-vacuity: analyse toy programs whose reach is known by
    construction, so a rule that silently drops a dependence is caught
    here rather than in a 80k-equation jaxpr where nobody can see it.

    Each case nests the stencil inside a different higher-order
    primitive.  ``k`` 3-point stencils reach exactly ``k`` cells, so the
    analyser must return ``k`` -- not less (a dropped dependence) and,
    for these limiter-free programs, not more (a saturated bound)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    n, seam = 41, 20

    def stencil(x):
        # pad+slice, NOT jnp.roll: roll lowers to a gather, whose rule is
        # deliberately saturating, which would mask a real mistake here.
        xp = jnp.pad(x, 1)
        return (xp[:-2] + xp[1:-1] + xp[2:]) / 3.0

    def k_stencils(x, k):
        for _ in range(k):
            x = stencil(x)
        return x

    # Each case is (fn(x, flag) -> array, expected reach).  ``flag`` is an
    # UNTAINTED scalar, so a control predicate built from it exercises the
    # branch-OR without tripping the (separately tested) control-taint
    # saturation.
    cases = {}
    cases["plain k=3"] = (lambda x, f: k_stencils(x, 3), 3)
    cases["jit k=4"] = (lambda x, f: jax.jit(lambda y: k_stencils(y, 4))(x), 4)
    cases["nested jit k=2+3"] = (
        lambda x, f: jax.jit(lambda y: k_stencils(y, 3))(k_stencils(x, 2)), 5)
    cases["scan k=5 (carry recurrence)"] = (
        lambda x, f: jax.lax.scan(
            lambda c, _: (stencil(c), 0.0), x, None, length=5)[0], 5)
    cases["cond OR over branches (2|6)"] = (
        lambda x, f: jax.lax.cond(f > 0.0,
                                  lambda y: k_stencils(y, 2),
                                  lambda y: k_stencils(y, 6), x), 6)
    cases["remat k=3"] = (
        lambda x, f: jax.checkpoint(lambda y: k_stencils(y, 3))(x), 3)
    # A while_loop's trip count is not a trace-time constant, so the only
    # sound answer is the fixpoint -- which saturates.  Expect saturation,
    # not 4: claiming 4 would mean the analyser read a trip count it
    # cannot know.
    cases["while (unbounded -> saturates)"] = (
        lambda x, f: jax.lax.while_loop(lambda s: s[0] < 4,
                                        lambda s: (s[0] + 1, stencil(s[1])),
                                        (0, x))[1], max(seam, n - 1 - seam))
    # A predicate built FROM the tainted array is a control dependence:
    # every output must saturate (reach = farthest cell), never report 2.
    cases["cond on tainted pred (saturates)"] = (
        lambda x, f: jax.lax.cond(x.sum() > 0.0,
                                  lambda y: k_stencils(y, 2),
                                  lambda y: k_stencils(y, 2), x),
        max(seam, n - 1 - seam))

    # A level recurrence: fori_loop over a TRACED k writing column k of a
    # 2-D array with a 3-point stencil along axis 0.  The index is not
    # concrete inside the loop body, so the scatter rule cannot know k;
    # the axis-aware fallback must still report the stencil's reach (3),
    # not saturate axis 0 (which the whole-output fallback did).  Under
    # the whole-output rule this case reads max(seam, n-1-seam).
    def level_recurrence(x, f):
        x2 = jnp.broadcast_to(x[:, None], (n, 4))

        def body(k, a):
            return a.at[:, k].set(k_stencils(a[:, k], 3))
        return jax.lax.fori_loop(0, 4, body, x2)
    cases["fori scatter at traced k (axis-aware)"] = (level_recurrence, 3)

    # Same recurrence with k arriving as a scanned xs value, which the
    # analyser does NOT track: the axis-aware fallback cannot tell level
    # k from level k+1, so every iteration may read the previous write
    # and the chain composes to 4 x 3 = 12.  Over-report, never under.
    def level_recurrence_xs(x, f):
        x2 = jnp.broadcast_to(x[:, None], (n, 4))

        def body(a, k):
            return a.at[:, k].set(k_stencils(a[:, k], 3)), 0.0
        return jax.lax.scan(body, x2, jnp.arange(4))[0]
    cases["scan scatter at xs k (fallback chains)"] = (level_recurrence_xs,
                                                      12)

    # An INDEX computed from seam data: the read position depends on the
    # seam, so every output cell does (codex 2026-09-04) -- must saturate,
    # even though the table itself is clean and the index is one scalar.
    cases["gather on tainted index (saturates)"] = (
        lambda x, f: jnp.arange(n, dtype=jnp.float64)[
            (x.sum() > 0.0).astype(jnp.int32)] * jnp.ones(n),
        max(seam, n - 1 - seam))

    x0 = np.zeros(n)
    bad = []
    for label, (fn, expect) in cases.items():
        closed = jax.make_jaxpr(fn)(jnp.asarray(x0), jnp.float64(1.0))
        t = np.zeros(n, bool)
        t[seam] = True
        del _SCATTER_FALLBACKS[:], _CONTROL_TAINTS[:]
        del _WHILE_LOOPS[:], _LONG_SCANS[:]
        try:
            outs = _interp(closed.jaxpr, closed.consts,
                           [t, np.zeros((), bool)], np, 0,
                           {"axis": 0, "seam": seam, "per_prim": {}})
        except UnknownPrimitive as e:
            print(f"  {label:34s} REFUSED: {e}")
            bad.append(label)
            continue
        got = max((_reach_of(o, 0, seam) for o in outs
                   if hasattr(o, "ndim") and o.ndim), default=-1)
        ok = got == expect
        note = "" if ok else f"  <-- expected {expect}"
        print(f"  {label:34s} reach={got:3d} "
              f"{'ok' if ok else 'MISMATCH'}{note}")
        if not ok:
            bad.append(label)

    # The instrument must also FAIL when it should: a program with reach
    # 0 must not report 3, or the checks above prove nothing.
    closed = jax.make_jaxpr(lambda x: x * 2.0)(jnp.asarray(x0))
    t = np.zeros(n, bool)
    t[seam] = True
    del _SCATTER_FALLBACKS[:], _CONTROL_TAINTS[:]
    outs = _interp(closed.jaxpr, closed.consts, [t], np, 0,
                   {"axis": 0, "seam": seam, "per_prim": {}})
    got0 = max(_reach_of(o, 0, seam) for o in outs)
    print(f"  {'control: elementwise, reach 0':34s} reach={got0:3d} "
          f"{'ok' if got0 == 0 else 'MISMATCH'}")
    if got0 != 0:
        bad.append("control")

    print(f"[selftest] {len(cases) + 1 - len(bad)}/{len(cases) + 1} passed")
    return 1 if bad else 0


def _window_mode(args, model, ctx, state, nh, dp0, fields, order):
    """Certify a window pad for one deck: with the seam pad of ONE
    interior window seeded as garbage (every cell the tile does not own
    on its west side, all levels), run the taint through one substep on
    the real window program (window comm attached, refresh DISABLED so
    the seed is not overwritten at entry, exchanges ON so the ~23
    cross-face firings move their write-sets) and read how far the taint
    enters the tile's owned cells.  Required pad = pad + overshoot."""
    import jax
    import jax.numpy as jnp

    from legoesm.core.fv3_acoustic_3d import acoustic_substep_3d
    from legoesm.grids.fv3_duo_windows import (
        attach_window_comm, gather_windows, _owner_mask, horizontal_axes)
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

    parts = [int(v) for v in args.window.split(":")]
    kt, pad = parts[0], parts[1]
    band = parts[2] if len(parts) > 2 else ctx.ng   # seeded band depth
    wctx, comm = attach_window_comm(ctx, kt, pad)
    lay = comm.lay
    comm.refresh = lambda bundle: bundle      # keep the seeded pad
    ng = lay.ng
    # the seeded window: an INTERIOR tile when kt >= 3 (both sides seams),
    # else tile (0, 1) whose west side is the face edge but whose j-sides
    # are seams; on face 0
    ti = 1 if kt >= 3 else 0
    tj = 1 if kt >= 2 else 0
    w0 = ti * kt + tj
    print(f"[static:window] {lay}; scored window {w0} = (face 0, ti={ti}, "
          f"tj={tj}), origin {lay.origins[w0][1:]}; seeded: the outer "
          f"{band}-cell band on ALL four sides of EVERY window, state AND "
          f"NH carry (owned starts at {pad}); refresh DISABLED, "
          f"exchanges ON")

    wstate = {k: gather_windows(lay, v) for k, v in state.items()}
    wnh = None
    if nh is not None:
        wnh = {k: (gather_windows(lay, v)
                   if hasattr(v, "ndim") and v.ndim >= 3 and v.shape[0] == 6
                   else v) for k, v in nh.items()}

    nh_keys = [k for k in (wnh or {}) if hasattr(wnh[k], "ndim")
               and wnh[k].ndim >= 3 and wnh[k].shape[0] == lay.nb]
    nh_const = {k: v for k, v in (wnh or {}).items() if k not in nh_keys}

    def fn(*vals):
        st = dict(zip(order, vals[:len(order)]))
        nhd = None
        if wnh is not None:
            nhd = dict(nh_const)
            nhd.update(zip(nh_keys, vals[len(order):]))
        return acoustic_substep_3d(
            wctx, st, args.dt / 3.0, args.km, first_substep=True,
            ptop=float(model._ptop), akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
            exchange=True, hydrostatic=not args.nh, nh=nhd, dp0=dp0,
            remap_step=False, remap_follows=True, batched=True)

    try:
        closed = jax.make_jaxpr(fn)(*[wstate[k] for k in order],
                                    *[wnh[k] for k in nh_keys])
    finally:
        ctx.tab.window_comm = None
    print(f"[static:window] jaxpr: {len(closed.jaxpr.eqns)} equations")

    # THE SEED (codex round 2, item 6): at entry every window cell is a
    # correct copy of the flat state.  What goes wrong inside a substep is
    # confined to each window's outer band: the kernel never updates its
    # halo (only exchanges do, and at a seam none fires), so it goes
    # STALE, and the cube-edge/corner treatments rewrite the boundary as
    # if it were a face edge.  So the seed is the outer ``band`` cells on
    # ALL FOUR sides of EVERY window (not one side of one window: a
    # corrupted band elsewhere reaches this window through the write-set
    # gathers), for the state AND the NH carry (gz/zh/pk3/zs ride the
    # substep with their own stale halos).  ``band`` defaults to ng; a
    # fake-edge treatment that writes compute cells deeper than the halo
    # is covered by raising it (--window KT:PAD:BAND) -- the reported
    # requirement then grows one-for-one, which is the check that the
    # band models the corruption depth.  (Seeding the whole pad answers a
    # different question -- "garbage pad" -- and reads pad + reach for
    # every pad, job 9631834.)
    def _band_seed(a):
        t = np.zeros(a.shape, bool)
        axes = horizontal_axes(lay, a.shape, lay.nb)
        if axes != (1, 2):
            return t
        e1, e2 = a.shape[1], a.shape[2]
        t[:, :band, :, ...] = True
        t[:, e1 - band:, :, ...] = True
        t[:, :, :band, ...] = True
        t[:, :, e2 - band:, ...] = True
        # an edge window OWNS its real ring: that ring is stale between
        # firings in the flat program too, so it is not a window-induced
        # error and is not seeded (only NON-owned band cells are)
        for w in range(lay.nb):
            own = _owner_mask(lay, w, e1 + (lay.m_a - lay.W),
                              e2 + (lay.m_a - lay.W))
            t[w] &= ~own.reshape(own.shape + (1,) * (a.ndim - 3))
        return t

    taints = []
    for k in order:
        a = np.asarray(wstate[k])
        taints.append(jnp.asarray(_band_seed(a) if k in fields
                                  else np.zeros(a.shape, bool)))

    for k in nh_keys:                     # the NH carry's bands too
        taints.append(jnp.asarray(_band_seed(np.asarray(wnh[k]))))
    print(f"[static:window] seeded {len(fields)} state fields + "
          f"{len(nh_keys)} NH-carry arrays {nh_keys}")

    del _SCATTER_FALLBACKS[:], _CONTROL_TAINTS[:]
    del _WHILE_LOOPS[:], _LONG_SCANS[:]
    stats = {"axis": 1, "seam": pad, "per_prim": {}}
    try:
        outs = _interp(closed.jaxpr, closed.consts,
                       [np.asarray(t) for t in taints], np, 0, stats)
    except UnknownPrimitive as e:
        print(f"[static:window] REFUSED: {e}")
        return 4

    deepest = -1                     # deepest tainted OWNED local index
    n_scored = 0
    for o in outs:
        if not hasattr(o, "ndim"):
            continue
        axes = horizontal_axes(lay, o.shape, lay.nb)
        if axes is None:
            continue
        n_scored += 1
        b = np.asarray(o[w0], bool)
        e0 = o.shape[axes[0]] + (lay.m_a - lay.W)
        e1 = o.shape[axes[1]] + (lay.m_a - lay.W)
        own = _owner_mask(lay, w0, e0, e1)          # (wi, wj) owned cells
        ai, aj = axes[0] - 1, axes[1] - 1           # axes within o[w0]
        red = tuple(a for a in range(b.ndim) if a not in (ai, aj))
        b2 = b.any(axis=red) if red else b
        if ai > aj:
            b2 = b2.T
        hit = b2 & own
        if hit.any():
            pi, pj = np.nonzero(hit)
            wi, wj = hit.shape
            # depth of the deepest tainted owned cell measured from the
            # NEAREST window boundary on either axis
            d = int(np.max(np.minimum(np.minimum(pi, wi - 1 - pi),
                                      np.minimum(pj, wj - 1 - pj))))
            deepest = max(deepest, d)
            print(f"[static:window]   output {np.shape(o)}: {int(hit.sum())} "
                  f"tainted OWNED cells, deepest {d} from a window boundary "
                  f"(owned starts at {pad})")
    over = 0 if deepest < 0 else deepest - pad + 1
    for w in (_SCATTER_FALLBACKS, _CONTROL_TAINTS, _WHILE_LOOPS, _LONG_SCANS):
        if w:
            print(f"[static:window] WARNING: over-approximating construct "
                  f"count {len(w)} -- the number below is an over-estimate")
    print(f"[static:window] {n_scored} outputs scored; deepest tainted owned "
          f"local index {deepest}; overshoot into owned cells = {over}")
    verdict = ("CERTIFIED" if over <= 0
               else f"INSUFFICIENT, required >= {pad + over}")
    print(f"[static:window] VERDICT C{args.n} km={args.km} "
          f"{'NH' if args.nh else 'hydro'} kt={kt}: pad {pad} {verdict}")
    return 0 if over <= 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="analyse toy programs of KNOWN reach and exit")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
    ap.add_argument("--window", type=str, default=None, metavar="KT:PAD",
                    help="WINDOW mode: trace the substep on the tiled "
                         "port's windows (fv3_duo_windows, kt tiles per "
                         "face, pad cells), seed the whole west pad zone "
                         "of one interior window as garbage, and report "
                         "how deep the taint enters that tile's OWNED "
                         "cells: 0 = this pad is certified for this deck")
    ap.add_argument("--seed-point", action="store_true",
                    help="seed ONE cell (i=seam, j=centre) instead of the "
                         "mid-line: an isotropy check -- the tainted box "
                         "must be roughly square")
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--nh", action="store_true")
    ap.add_argument("--compare", type=int, default=None, metavar="R",
                    help="the perturbation ladder's reach; the probe may "
                         "read LESS than the static bound, never more")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.core.fv3_acoustic_3d import acoustic_substep_3d
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

    grid = create_fv3_duo_grid(args.n)
    cfg = FV3DuoConfig(km=args.km, hydrostatic=not args.nh)
    model = FV3DuoDynamicsModel(grid, cfg)
    ctx = model._ctx_jax
    ng = grid.ng
    bundle = model.dcmip16_initial_state(do_pert=True)
    state = dict(bundle["state"])
    nh = bundle.get("nh")
    dp0 = None
    if args.nh:
        ak, bk = np.asarray(model._ak), np.asarray(model._bk)
        dp0 = (ak[1:] - ak[:-1]) + (bk[1:] - bk[:-1]) * 1.0e5

    fields = [k for k in ("delp", "pt", "u", "v", "w") if k in state]
    order = list(state)

    if args.window:
        return _window_mode(args, model, ctx, state, nh, dp0, fields, order)

    def fn(*vals):
        st = dict(zip(order, vals))
        return acoustic_substep_3d(
            ctx, st, args.dt / 3.0, args.km, first_substep=True,
            ptop=float(model._ptop), akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
            exchange=False, hydrostatic=not args.nh, nh=nh, dp0=dp0,
            remap_step=False, remap_follows=True)

    print(f"[static] tracing ONE substep, C{args.n} km={args.km} "
          f"{'NH' if args.nh else 'hydro'}, exchanges OFF ...")
    closed = jax.make_jaxpr(fn)(*[state[k] for k in order])
    n_eqns = len(closed.jaxpr.eqns)
    print(f"[static] jaxpr: {n_eqns} equations at top level")

    seam = ng + args.n // 2
    # Seed ONE face, and only the middle of the seam line.  A full line
    # touches the face's j-edges, where the entry/barrier halo TABLES (which
    # ``exchange=False`` does not remove -- job 9630988 traced the taint
    # into the (1800,) A-table and the (72,) corner table) copy it into the
    # neighbouring faces' halos at EVERY i, from where it floods back and
    # saturates the compute domain at every face size (12/24/48 at
    # C24/48/96).  With a margin of ``m`` cells to each j-edge, taint can
    # only reach a table entry after travelling >= m cells, so any reach
    # below m is certified intra-face; a reach >= m is reported as
    # CONFOUNDED, never as a bound.
    margin = max(ng + 1, args.n // 4)
    taints = []
    for k in order:
        a = np.asarray(state[k])
        t = np.zeros(a.shape, bool)
        if k in fields and a.ndim >= 3:
            if args.seed_point:
                t[0, seam, ng + args.n // 2, ...] = True   # one cell
            else:
                t[0, seam, ng + margin:a.shape[2] - ng - margin, ...] = True
        taints.append(jnp.asarray(t))
    print(f"[static] seed: face 0, i={seam}, j in [{ng + margin}, "
          f"{args.n + ng - margin}) (margin {margin} to each j-edge)")

    # NUMPY, not jnp: the traced substep is ~80k equations (C48) and ~115k
    # (NH), and per-op JAX dispatch would dominate. The taint abstraction is
    # boolean arrays of the state's own shape, so numpy is a drop-in and
    # keeps the analysis off the device entirely.
    stats = {"axis": 1, "seam": seam, "per_prim": {}, "jumps": [], "ng": ng,
             "hist": [], "leak": (0, None, None),
             "extent": (args.n + 2 * ng, args.n + 2 * ng + 1)}
    try:
        outs = _interp(closed.jaxpr, closed.consts,
                       [np.asarray(t) for t in taints], np, 0, stats)
    except UnknownPrimitive as e:
        print(f"[static] REFUSED: {e}")
        return 4

    # Only arrays that still carry the face-stacked layout (6, i, j, ...)
    # have their seam on axis 1; a per-face (i, j, k) array would be
    # measured along j -- the seam LINE itself -- and read n/2+ng, i.e.
    # saturation by construction (job 9630893 traced the 15/27 numbers to
    # exactly this).  Refuse to fold any other layout into the answer.
    ext = (args.n + 2 * ng, args.n + 2 * ng + 1)
    worst = -1
    n_used = n_skipped = 0
    for o in outs:
        if not (hasattr(o, "ndim") and o.ndim >= 3 and o.shape[0] == 6
                and o.shape[1] in ext):
            n_skipped += 1
            print(f"[static] output of shape {np.shape(o)} is not "
                  f"face-stacked (6, i, j, ...); NOT folded into the bound")
            continue
        n_used += 1
        r = _reach_of(o[0], 0, seam)            # face 0 only, along i
        others = int(np.asarray(o[1:], bool).sum())
        b0 = np.asarray(o[0], bool)
        ii = np.nonzero(np.any(b0, axis=tuple(range(1, b0.ndim))))[0]
        jj = np.nonzero(np.any(b0, axis=tuple(a for a in range(b0.ndim)
                                              if a != 1)))[0]
        box = (f"i [{ii.min()},{ii.max()}] j [{jj.min()},{jj.max()}]"
               if ii.size else "empty")
        print(f"[static]   output {np.shape(o)}: reach {r} along i on "
              f"face 0; box {box}; {others} tainted cells on the other "
              f"five faces")
        worst = max(worst, r)
    flat_out = [o for o in outs if hasattr(o, "ndim") and o.ndim >= 3]
    if _SCATTER_FALLBACKS:
        print(f"[static] WARNING: {len(_SCATTER_FALLBACKS)} scatter(s) had "
              f"no concrete indices and were tainted WHOLE-OUTPUT -- the "
              f"number below is then an OVER-estimate, not a reach")
    if _CONTROL_TAINTS:
        print(f"[static] WARNING: {len(_CONTROL_TAINTS)} control predicate(s) "
              f"(cond branch / while trip count) depended on seam data, so "
              f"their outputs were tainted WHOLE -- over-estimate, not reach")
    if _WHILE_LOOPS:
        print(f"[static] WARNING: {len(_WHILE_LOOPS)} while_loop(s) -- their "
              f"trip count is not a trace-time constant, so their carry was "
              f"driven to a FIXPOINT and saturates; the number below is then "
              f"an over-estimate, not a reach")
    if _LONG_SCANS:
        print(f"[static] WARNING: scan(s) of length {_LONG_SCANS} exceeded "
              f"the {_MAX_UNROLL}-step unroll cap without reaching a "
              f"fixpoint, so their carry was saturated rather than "
              f"under-reported -- over-estimate, not reach")
    leak = stats["leak"]
    print(f"[static] other-face taint, maximum over EVERY equation: "
          f"{leak[0]} cells" + (f" (first peak at {leak[1]} {leak[2]})"
                               if leak[0] else ""))
    lf = stats.get("leak_first")
    if lf:
        print(f"[static] FIRST other-face taint: {lf[0]} cells at {lf[1]} "
              f"depth {lf[2]} out {lf[3]} in {lf[4][:4]}")
        for h in lf[5]:
            print(f"[static]     <- {h[0]} depth {h[1]} out {h[2]} "
                  f"tainted_cells {h[3]}")
    # reach along j on face 0 too: an operator that is nonlocal along the
    # seam line (cumsum, contraction, saturated gather axis) shows up here
    # long before it reaches a face edge
    closest_j = None                       # closest approach to a j-edge
    for o in outs:
        if (hasattr(o, "ndim") and o.ndim >= 3 and o.shape[0] == 6
                and o.shape[1] in ext):
            b = np.asarray(o[0], bool)
            js = np.nonzero(np.any(b, axis=tuple(a for a in range(b.ndim)
                                                 if a != 1)))[0]
            if js.size:
                d = min(int(js.min()), int(b.shape[1] - 1 - js.max()))
                closest_j = d if closest_j is None else min(closest_j, d)
    print(f"[static] along j on face 0: seeded j in [{ng + margin}, "
          f"{args.n + ng - margin}); tainted j comes within "
          f"{closest_j} of an array edge (seed was {ng + margin} away)")
    if worst >= margin or leak[0]:
        print(f"[static] CONFOUNDED: reach {worst} vs margin {margin}, "
              f"other-face taint max {leak[0]} -- the taint crossed (or may "
              f"have crossed) a face edge through the halo tables; this "
              f"number is NOT an intra-face bound")
    print(f"[static] STATIC REACH (exact, cannot be swallowed by a "
          f"limiter): {worst} cells from the seam, over {len(flat_out)} "
          f"output arrays")
    # WHERE the reach grows by more than a stencil width in one op: the
    # operation that saturates a bound, named with depth and shapes
    for j in sorted(stats["jumps"], key=lambda t: -t[0])[:3]:
        print(f"[static] REACH JUMP +{j[0]} ({j[1]} -> {j[2]}) at {j[3]} "
              f"depth {j[4]} out {j[5]} in {j[6][:4]}")
        for h in j[7]:
            print(f"[static]     <- {h[0]} depth {h[1]} out {h[2]} "
                  f"tainted_cells {h[3]}")
    top = sorted(stats["per_prim"].items(), key=lambda kv: -kv[1])[:6]
    print(f"[static] operations contributing the largest reach: "
          + ", ".join(f"{k}={v}" for k, v in top))
    if args.compare is not None:
        ok = args.compare <= worst
        rel = "<=" if ok else ">"
        note = ("CONSISTENT -- the probe may under-report (a limiter can "
                "swallow a perturbation); it may never exceed the static "
                "bound" if ok else
                "VIOLATED -- the probe found a dependence this analyser "
                "missed, so the analyser has a gap and its number is not "
                "usable")
        print(f"[static] CONTRACT CHECK vs the perturbation ladder "
              f"({args.compare} cells): probe {rel} static {worst} -- "
              f"{note}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
