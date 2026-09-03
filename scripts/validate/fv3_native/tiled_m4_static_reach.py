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


def _rule_gather(eqn, invals, jnp):
    """Conservative and SOUND: a gather with traced indices can read
    anywhere along the gathered axes, so taint the whole output if any
    input element is tainted."""
    x = jnp.asarray(invals[0], bool)
    out_shape = eqn.outvars[0].aval.shape
    return [jnp.broadcast_to(jnp.any(x), out_shape)]


def _rule_dynamic_slice(eqn, invals, jnp):
    x = jnp.asarray(invals[0], bool)
    out_shape = eqn.outvars[0].aval.shape
    return [jnp.broadcast_to(jnp.any(x), out_shape)]


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
    if upd_t is None or not upd_t.any():
        return [op_t]
    if idx is None:
        _SCATTER_FALLBACKS.append(eqn.primitive.name)
        return [np.ones(op_t.shape, bool) if op_t.any() or upd_t.any()
                else op_t]
    out = op_t.copy()
    dn = eqn.params["dimension_numbers"]
    try:
        ii = np.asarray(idx).reshape(-1, np.asarray(idx).shape[-1])
        odims = list(dn.scatter_dims_to_operand_dims)
        for row in ii:
            sl = [slice(None)] * out.ndim
            for d, v in zip(odims, row):
                sl[d] = int(v)
            out[tuple(sl)] |= bool(upd_t.any())
        return [out]
    except Exception:
        _SCATTER_FALLBACKS.append(eqn.primitive.name)
        return [np.ones(op_t.shape, bool)]


_SCATTER_FALLBACKS = []


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
            const_taints=None):
    """Taint-interpret one jaxpr.  ``stats`` accumulates, per primitive,
    the largest horizontal reach its output achieved -- so the reported
    number arrives with the operation that produced it.

    ``const_taints`` covers the case where a caller hoisted the
    sub-jaxpr's constants into leading operands (see ``_bind_call``):
    the constvars then carry the CALLER'S taint rather than being clean
    literals, and zeroing them would under-report."""
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
        for cv, t in zip(jaxpr.constvars, const_taints):
            # No VALUES available on this path, so the integer
            # side-channel goes dark inside this sub-jaxpr: a scatter in
            # here falls back to whole-output taint, which OVER-reports
            # (announced by _SCATTER_FALLBACKS) and never under-reports.
            write(cv, np.asarray(t, bool))
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
    for iv, a in zip(jaxpr.invars, args):
        write(iv, a)

    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        invals = [read(v) for v in eqn.invars]
        if name in _HIGHER_ORDER:
            outs = _interp_higher_order(eqn, invals, jnp, depth, stats)
        elif name in _SCATTER_PRIMS:
            idx = read_val(eqn.invars[1]) if len(eqn.invars) > 1 else None
            outs = _rule_scatter(eqn, invals, jnp, idx)
        else:
            rule = _RULES.get(name)
            if rule is None:
                raise UnknownPrimitive(
                    f"no taint rule for primitive {name!r} (depth {depth}) "
                    f"-- refusing to skip it: an unhandled operation drops a "
                    f"data dependence and UNDER-reports the reach, which is "
                    f"the exact failure this analyser exists to remove")
            outs = rule(eqn, invals, jnp)
        cval = _concrete(eqn, [read_val(v) for v in eqn.invars])
        for i, (v, o) in enumerate(zip(eqn.outvars, outs)):
            write(v, o)
            if cval is not None and i < len(cval):
                env_val[v] = np.asarray(cval[i])
        if stats is not None:
            for o in outs:
                r = _reach_of(o, stats["axis"], stats["seam"])
                if r > stats["per_prim"].get(name, -1):
                    stats["per_prim"][name] = r
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


def _call_sub(name, sub, invals, jnp, depth, stats):
    body, consts, ctaints, args = _bind_call(name, sub, invals)
    return _interp(body, consts, args, jnp, depth + 1, stats,
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


def _interp_higher_order(eqn, invals, jnp, depth, stats):
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
            pred = _call_sub("while.cond", p["cond_jaxpr"],
                             cconst + carry, jnp, depth, stats)[0]
            new = _call_sub("while.body", p["body_jaxpr"],
                            bconst + carry, jnp, depth, stats)
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
            o = _call_sub(f"{name}.branch", br, invals[1:], jnp, depth, stats)
            outs = o if outs is None else [
                np.asarray(x, bool) | np.asarray(y, bool)
                for x, y in zip(outs, o)]
        return _ctrl_or(outs, invals[0], jnp)

    if name != "scan":
        return _call_sub(name, sub, invals, jnp, depth, stats)

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
    for _ in range(min(length, _MAX_UNROLL)):
        outs = _call_sub(name, sub, cs + carry + xs_slices, jnp, depth, stats)
        new_carry = outs[:n_carry]
        ys = outs[n_carry:]
        if all(bool(jnp.array_equal(a, b))
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="analyse toy programs of KNOWN reach and exit")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
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
    taints = []
    for k in order:
        a = np.asarray(state[k])
        t = np.zeros(a.shape, bool)
        if k in fields and a.ndim >= 3:
            t[:, seam, ...] = True
        taints.append(jnp.asarray(t))

    # NUMPY, not jnp: the traced substep is ~80k equations (C48) and ~115k
    # (NH), and per-op JAX dispatch would dominate. The taint abstraction is
    # boolean arrays of the state's own shape, so numpy is a drop-in and
    # keeps the analysis off the device entirely.
    stats = {"axis": 1, "seam": seam, "per_prim": {}}
    try:
        outs = _interp(closed.jaxpr, closed.consts,
                       [np.asarray(t) for t in taints], np, 0, stats)
    except UnknownPrimitive as e:
        print(f"[static] REFUSED: {e}")
        return 4

    flat_out = [o for o in outs if hasattr(o, "ndim") and o.ndim >= 3]
    worst = -1
    for o in flat_out:
        worst = max(worst, _reach_of(o, 1, seam))
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
    print(f"[static] STATIC REACH (exact, cannot be swallowed by a "
          f"limiter): {worst} cells from the seam, over {len(flat_out)} "
          f"output arrays")
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
