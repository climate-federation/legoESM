#!/usr/bin/env python
"""Round 351: which construct makes the EOS consumers read ONE stored copy on XLA CPU?

In the compiled one-core ORCA2 step XLA re-evaluates the NEMO Roquet density polynomial
inside many consumer kernels (69 complete copies where 5 suffice; study
``orca2_r351_eos_research/report_v2.md``).  This microbench builds the step's two EOS
evaluations from (T, S, depth) exactly as the round-349 fast_literal card resolves them
(re-read from the card at run time):

  * the pressure density ``nemo_roquet_density_anomaly_ratio`` (nemo_eos80, geometric
    depth, rho0 = card rho_0, 3-D tmask), as ``ocean_pe_latlon_cgrid.py`` calls it;
  * NEMO rab ``nemo_roquet_alpha_beta`` (eos80, source-rounded), as N2's bn2 calls it.

and feeds them to consumers shaped like the step's: the real NEMO hpg_sco level-by-level
pressure recurrence, the real bn2 assembly, and slope/pad/vertical reads.  One construct
per arm sits between the producer and the consumers:

  a   none (baseline)                     d   lax.while_loop, runtime trip count 1, identity
  b   optimization_barrier (control)      d2  lax.while_loop, EOS computed in the body
  c   lax.cond, runtime predicate,        e   two compiled programs: EOS, then consumers
      identity branches                   f   jax.pure_callback identity (opaque host call)
  c2  lax.cond, EOS computed in           g   EOS fields also returned as program outputs
      both branches
c and d are what was asked; JAX prunes a branch/carry that forwards its input unchanged
(conditionals.py and loops.py "_jaxpr_forwarding"), so c2/d2 put the EOS INSIDE the construct.

Per arm: EOS copies (below), output bits vs arm a (int64 views; the guarded/exact consumers,
and separately one unguarded jnp.sum whose association XLA may change when a fusion
boundary moves), median wall per call (pinned by the caller), the same copies and bits when
the arm sits (i) inside an outer jit that does not return the EOS fields and (ii) inside a
lax.scan over 2 steps, and whether jax.grad of a scalar loss runs, its copies and its bits.

Copy counter (the study's method, ``hlo_eos_sites.parse`` + POLY rule + ``evaluations``,
uncommitted there, ported here): a polynomial multiply is an HLO ``multiply`` whose
innermost eos.py frame is the Roquet ``_fmul`` (51 per density copy) or the rab ``m``
(70 per alpha/beta pair).  A kernel is a top-level op of an executed computation; nested
fusions belong to their kernel.  Per kernel, copies = floor(multiplies / per-copy), times how
often the kernel runs (while/scan known trip counts; a runtime trip count is reported as
per-trip); ``copy_equivalents`` = multiplies x runs / per-copy counts a copy split across
kernels (rab is never whole in one kernel, even stored once: alpha 35 / beta 34 / 1).
In a reverse-mode program the transposed multiplies carry the same source frames, so its
counts mix forward and backward work.  Conditional branches are walked separately:
executed copies count only the taken branch (every arm's predicate is n > 0 with n = 1, JAX
branch index 1), dead ones are reported apart.  ``grid_evals`` = polynomial evaluations
actually emitted (index flow through slices/pads/reduces, equal offsets merged) in units of
one full-grid evaluation.  The run
refuses unless the density alone compiles to exactly one copy and arm a duplicates.

Usage: orca2_round351_eos_store_microbench.py --report R.json [--calls 20] [--arms a,b]
       [--no-grad | --skip-forward] [--no-placement] [--grad-levels 12] [--previous OLD.json]
Gradients run on (148, 180, --grad-levels): XLA CPU compile time of the reverse pass grows
steeply with the level count (MEASURED: 22 s at 6, 104 s at 20; arm a unfinished after
18 min at 30).
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in (REPO_ROOT / "packages/ocean", REPO_ROOT / "packages/core", REPO_ROOT):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.extend.core import jaxpr_as_fun  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import nemo_hpg_sco_literal_cgrid  # noqa: E402
from legoesm.ocean.eos import (  # noqa: E402
    compute_buoyancy_frequency_nemo_bn2,
    nemo_roquet_alpha_beta,
    nemo_roquet_density_anomaly_ratio,
)

from legoesm import constants  # noqa: E402

SHAPE = (148, 180, 30)
# What the round-349 fast_literal ORCA2 card resolves (re-read by resolve_card()):
CARD = {"eos": "nemo_eos80", "eos_depth": "geometric", "rho_0": 1026.0, "g": 9.80665,
        "pgf_scheme": "nemo_sco", "transcendentals": "native",
        "n2_eos_form": "eos80", "bn2_source_rounded": True}
POLY = {"rho": ("_nemo_roquet_eos_impl.<locals>._fmul", 51),   # one density copy
        "rab": ("nemo_roquet_alpha_beta.<locals>.m", 70)}      # one alpha/beta pair
POLY_FILE = "ocean/eos.py"
TAKEN_BRANCH = 1  # lax.cond binds branches (false, true); every arm's predicate is true
CONSTRUCT_OPS = ("conditional", "while", "custom-call", "opt-barrier")

# --------------------------------------------------------------------------- copy counter
_INSTR = re.compile(r"^\s*(ROOT )?%(\S+) = (.*)$")
_OPCODE = re.compile(r" ([a-z][a-z0-9-]*)\(")
_COMP = re.compile(r"^(ENTRY )?%(\S+) \(.*\{\s*$")
_META = re.compile(r", metadata=\{[^}]*\}")
_NAMEREF = re.compile(r"%([\w.\-]+)")
_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
_PASS = {"slice", "dynamic-slice", "broadcast", "transpose", "reshape", "bitcast", "reverse",
         "copy", "gather"}
_SHARE = {"pad", "concatenate", "dynamic-update-slice"}
_REDUCE = {"reduce", "reduce-window"}


def _ref(attrs, key):
    m = re.search(rf"\b{key}=%([\w.\-]+)", attrs)
    return m.group(1) if m else None


def parse_hlo(text: str):
    """-> (computations, instructions, frame -> innermost eos.py function).

    Port of the study's hlo_eos_sites.parse, plus while/conditional/call structure."""
    files, funcs, locs, frames = {}, {}, {}, {}
    comps: dict[str, dict] = {}
    instrs: dict[str, dict] = {}
    section = comp = None
    for line in text.splitlines():
        if line.startswith(_TABLES):
            section = line.strip()
            continue
        if section and line[:1].isdigit():
            i, rest = line.split(" ", 1)
            if section == "FileNames":
                files[int(i)] = rest.strip().strip('"')
            elif section == "FunctionNames":
                funcs[int(i)] = rest.strip().strip('"')
            else:
                kv = dict(re.findall(r"(\w+)=(\d+)", rest))
                if section == "FileLocations":
                    locs[int(i)] = (int(kv["file_name_id"]), int(kv["function_name_id"]))
                else:
                    frames[int(i)] = (int(kv["file_location_id"]), int(kv["parent_frame_id"]))
            continue
        m = _COMP.match(line)
        if m and not line.startswith(" "):
            section, comp = None, m.group(2)
            comps[comp] = dict(entry=bool(m.group(1)), instrs=[], root=None)
            continue
        m = _INSTR.match(line)
        if not m or comp is None:
            continue
        root, name, rest = m.groups()
        rest = rest if rest.startswith("(") else " " + rest
        om = _OPCODE.search(rest)
        if om is None:
            continue
        i, depth = om.end(), 1
        while depth:  # operand list ends at the matching parenthesis
            depth += (rest[i] == "(") - (rest[i] == ")")
            i += 1
        op, ops_text, attrs = om.group(1), rest[om.end():i - 1], rest[i:]
        fr = re.search(r"stack_frame_id=(\d+)", attrs)
        trip = re.search(r'"known_trip_count":\{"n":"(\d+)"', attrs)
        attrs = _META.sub("", attrs)
        branches = re.search(r"branch_computations=\{([^}]*)\}", attrs)
        branches = (_NAMEREF.findall(branches.group(1)) if branches else
                    [c for c in (_ref(attrs, "false_computation"), _ref(attrs, "true_computation"))
                     if c])
        instrs[name] = dict(
            comp=comp, op=op, type=rest[:om.start()].strip(), attrs=attrs,
            operands=[] if op in ("constant", "parameter") else _NAMEREF.findall(ops_text),
            frame=int(fr.group(1)) if fr else None, calls=_ref(attrs, "calls"),
            body=_ref(attrs, "body"), condition=_ref(attrs, "condition"),
            to_apply=_ref(attrs, "to_apply"), branches=branches,
            trip=int(trip.group(1)) if trip else None)
        comps[comp]["instrs"].append(name)
        if root:
            comps[comp]["root"] = name

    @functools.cache
    def innermost_eos_function(frame):
        # The dump prints parent_frame_id one above the parent's label (study, checked there).
        seen, f = set(), frame
        while f and f in frames and f not in seen:
            seen.add(f)
            loc, parent = frames[f]
            fid, fn = locs[loc]
            if files[fid].endswith(POLY_FILE):
                return funcs[fn]
            if parent - 1 == f:
                break
            f = parent - 1
        return None

    return comps, instrs, innermost_eos_function


def cells(type_text: str) -> int:
    dims = re.search(r"\[([\d,]*)\]", type_text)
    return int(np.prod([int(d) for d in dims.group(1).split(",") if d])) if dims else 1


def evaluations(comp, root_e, comps, instrs) -> dict[str, float]:
    """Scalar evaluations of every node of one fused computation whose root is evaluated
    root_e times (port of the study's hlo_eos_elements.evaluations, equal offsets merged)."""
    order = comps[comp]["instrs"]
    root = comps[comp]["root"] or order[-1]
    sig = collections.defaultdict(dict)  # node -> {index chain: evaluations}
    sig[root][()] = root_e
    for u in reversed(order):
        if u not in sig:
            continue
        x = instrs[u]
        for i, o in enumerate(x["operands"]):
            if o not in instrs or instrs[o]["comp"] != comp:
                continue
            for s, e in sig[u].items():
                key = u if x["op"] == "gather" else (x["op"], x["attrs"], x["type"], i)
                if x["op"] in _REDUCE:
                    if i >= len(x["operands"]) // 2:
                        continue  # init values
                    s2, e2 = s + (key,), e * cells(instrs[o]["type"]) / max(1, cells(x["type"]))
                elif x["op"] in _PASS:
                    s2, e2 = s + (key,), e
                elif x["op"] in _SHARE:
                    if x["op"] == "dynamic-update-slice" and i > 1:
                        continue
                    s2 = s + ((key, i),)
                    e2 = e * cells(instrs[o]["type"]) / max(1, cells(x["type"]))
                    if x["op"] == "dynamic-update-slice" and i == 0:
                        e2 = e  # ponytail: base read everywhere outside the update, upper estimate
                else:
                    s2, e2 = s, e  # elementwise: same index as the user
                sig[o][s2] = max(sig[o].get(s2, 0), e2)
    return {n: sum(d.values()) for n, d in sig.items()}


def kernel_rows(text: str) -> list[dict]:
    """One row per executed kernel holding polynomial multiplies, with its run count."""
    comps, instrs, eos_fn = parse_hlo(text)
    kind_of = {fn: k for k, (fn, _) in POLY.items()}

    def kind(x):
        return kind_of.get(eos_fn(x["frame"])) if x["op"] == "multiply" else None

    def fused(comp, root_e, tot):
        e = evaluations(comp, root_e, comps, instrs)
        for name in comps[comp]["instrs"]:
            x = instrs[name]
            if x["op"] == "fusion" and x["calls"]:
                fused(x["calls"], e.get(name, 0.0), tot)
            elif kind(x):
                tot[kind(x)][0] += 1
                tot[kind(x)][1] += e.get(name, 0.0)
        return tot

    rows = []

    def visit(comp, runs, known, branch):
        for name in comps[comp]["instrs"]:
            x = instrs[name]
            tot = {k: [0, 0.0] for k in POLY}
            if x["op"] == "fusion" and x["calls"]:
                fused(x["calls"], float(cells(x["type"])), tot)
            elif kind(x):
                tot[kind(x)] = [1, float(cells(x["type"]))]
            elif x["op"] == "while":
                visit(x["body"], runs * (x["trip"] or 1), known and x["trip"] is not None, branch)
            elif x["op"] == "conditional":
                for i, b in enumerate(x["branches"]):
                    visit(b, runs, known, branch + (i,))
            elif x["op"] == "call" and x["to_apply"]:
                visit(x["to_apply"], runs, known, branch)
            if any(n for n, _ in tot.values()):
                rows.append(dict(kernel=name, runs=runs, known_trips=known, branch=branch,
                                 **{k: tuple(v) for k, v in tot.items()}))

    visit(next(c for c, d in comps.items() if d["entry"]), 1, True, ())
    return rows


def copy_summary(texts: list[str], full_cells: int) -> dict:
    rows = [dict(r, program=p) for p, t in enumerate(texts) for r in kernel_rows(t)]
    out = {}
    for k, (_, per) in POLY.items():
        held = [r for r in rows if r[k][0]]
        live = [r for r in held if all(b == TAKEN_BRANCH for b in r["branch"])]
        dead = [r for r in held if not all(b == TAKEN_BRANCH for b in r["branch"])]
        out[k] = dict(
            copies=sum(r[k][0] // per * r["runs"] for r in live),
            partial_kernels=sum(bool(r[k][0] % per) for r in live),
            multiplies=sum(r[k][0] * r["runs"] for r in live),
            copy_equivalents=round(sum(r[k][0] * r["runs"] for r in live) / per, 3),
            grid_evals=round(sum(r[k][1] * r["runs"] for r in live) / (per * full_cells), 4),
            dead_branch_copies=sum(r[k][0] // per * r["runs"] for r in dead),
            runtime_trip_count=any(not r["known_trips"] for r in live))
    out["construct_ops"] = {op: sum(len(re.findall(rf" {op}\(", t)) for t in texts)
                            for op in CONSTRUCT_OPS}
    out["kernels"] = [[r["program"], r["kernel"], r["runs"], list(r["branch"]), r["rho"][0],
                       r["rab"][0]] for r in rows]
    return out


# --------------------------------------------------------------------------- model pieces
def make_inputs(shape=SHAPE, seed=351):
    """Synthetic ORCA2-like fp64 fields (j, i, k): t -1.8..~30 degC, s 33-36.5 psu,
    live gdept 5..4750 m = gdept_1d*(1+r3t), e3w from the same ladder, random-depth tmask."""
    rng = np.random.default_rng(seed)
    nj, ni, nk = shape
    k = np.arange(nk)
    t1d = 5.0 + 4745.0 * (k / (nk - 1)) ** 2
    stretch = 1.0 + 1e-4 * rng.standard_normal((nj, ni, 1))
    depth = t1d * stretch
    e3w = np.concatenate([2.0 * t1d[:1], np.diff(t1d)]) * stretch  # NEMO depth_to_e3 w-spacing
    t = (-1.8 + 30.0 * np.exp(-t1d / 700.0) * rng.uniform(0.3, 1.0, (nj, ni, 1))
         + 0.05 * rng.standard_normal(shape))
    s = np.clip(34.7 + 0.5 * rng.standard_normal(shape), 33.0, 36.5)
    tmask = (k < rng.integers(0, nk + 1, (nj, ni, 1))).astype(np.float64)
    return tuple(jnp.asarray(a, jnp.float64) for a in (t, s, depth, tmask, e3w))


def density(t, s, depth, tmask):
    """The pressure-density call of ocean_pe_latlon_cgrid.py (density_anomaly_ratio_fn lambda)
    for the card: p_eos = rho_0*constants.g*gdept as iterate_eos_and_pressure_anomaly builds it
    (constants.g, not the card g, is what that code uses; p is unread on the geometric path)."""
    p_eos = (CARD["rho_0"] * constants.g) * depth
    return nemo_roquet_density_anomaly_ratio(
        t, s, p_eos, rho0=CARD["rho_0"], geometric_depth_m=depth, tmask=tmask)


def eos_fields(t, s, depth, tmask):
    """Both EOS evaluations: pressure density, and rab as N2's bn2 calls it (default rho0
    1026 = card rho_0; the step's literal zrw depth is replaced by the same live gdept)."""
    alpha, beta = nemo_roquet_alpha_beta(t, s, depth, eos_form=CARD["n2_eos_form"],
                                         source_rounded=CARD["bn2_source_rounded"])
    return density(t, s, depth, tmask), alpha, beta


@functools.cache
def _grid(nj, ni):
    # ponytail: uniform beta-plane box, no ORCA north fold (the fold only swaps the last row's
    # neighbour; the step's rolls/level loop are the same)
    with jax.ensure_compile_time_eval():  # cached: must hold arrays, not a trace's tracers
        return create_beta_plane_cgrid_geometry(nj, ni, dx_m=2.0e5, dy_m=2.0e5, f0=0.0,
                                                beta=0.0, cartesian_pseudo_lat=True,
                                                dtype=jnp.float64)


def consumers(fields, t, s, depth, e3w):
    """The step's real pressure consumer (NEMO hpg_sco level-by-level recurrence), the real bn2
    assembly on rab, and slope/pad/vertical reads (shapes from partD_elements.txt)."""
    r, alpha, beta = fields
    hpg_u, hpg_v = nemo_hpg_sco_literal_cgrid(r, e3w, depth, _grid(*r.shape[:2]), CARD["g"])
    n2 = compute_buoyancy_frequency_nemo_bn2(
        t, s, depth, 0.5 * (depth[..., 1:] + depth[..., :-1]), g=CARD["g"],
        eos_form=CARD["n2_eos_form"], e3w_int=e3w[..., 1:], e3w_source="mesh_reference",
        source_rounded=CARD["bn2_source_rounded"], _alpha_beta_override=(alpha, beta))
    return (
        hpg_u, hpg_v,
        jnp.roll(r, -1, axis=1) - r,                          # slope stencil: i+1 and i
        jnp.roll(r, -1, axis=0) - r,                          # slope stencil: j+1 and j
        0.5 * (jnp.roll(r, -1, axis=1) + jnp.roll(r, 1, axis=1)),  # i+1 and i-1
        jnp.pad(r, ((6, 6), (0, 0), (0, 0))),                 # halo pad
        r[..., 1:] - r[..., :-1],                             # vertical difference
        _ordered_level_sum(nemo_source_round(r * e3w)),       # level sum, NEMO-literal order
        n2,                                                   # bn2 on alpha/beta
        jnp.sum(r * e3w, axis=-1),                            # level sum, XLA's order (unguarded)
    )


GROUPS = {"hpg (real)": (0, 2), "slope/pad/vertical/sum extras": (2, 8), "N2 (real bn2)": (8, 9),
          "unguarded jnp.sum": (9, 10)}
N_EXACT = 9   # outputs written in guarded/exact arithmetic
N_OUT = 10    # consumer outputs; arm g appends the EOS fields


def _ordered_level_sum(x):
    acc = nemo_source_round(x[..., 0])
    for k in range(1, x.shape[-1]):
        acc = nemo_source_round(acc + x[..., k])
    return acc


def _identity_callback(fields):
    shapes = tuple(jax.ShapeDtypeStruct(f.shape, f.dtype) for f in fields)
    return jax.pure_callback(lambda *x: tuple(np.asarray(v) for v in x), shapes, *fields)


def _while1(n, body, init):
    return lax.while_loop(lambda c: c[0] < n, body, (jnp.int32(0), init))[1]


ARMS = {
    "a": lambda t, s, d, m, w, n: consumers(eos_fields(t, s, d, m), t, s, d, w),
    "b": lambda t, s, d, m, w, n: consumers(
        lax.optimization_barrier(eos_fields(t, s, d, m)), t, s, d, w),
    "c": lambda t, s, d, m, w, n: consumers(
        lax.cond(n > 0, lambda x: x, lambda x: x, eos_fields(t, s, d, m)), t, s, d, w),
    "c2": lambda t, s, d, m, w, n: consumers(
        lax.cond(n > 0, eos_fields, eos_fields, t, s, d, m), t, s, d, w),
    "d": lambda t, s, d, m, w, n: consumers(
        _while1(n, lambda c: (c[0] + 1, c[1]), eos_fields(t, s, d, m)), t, s, d, w),
    "d2": lambda t, s, d, m, w, n: consumers(
        _while1(n, lambda c: (c[0] + 1, eos_fields(t, s, d, m)), (jnp.zeros_like(t),) * 3),
        t, s, d, w),
    "f": lambda t, s, d, m, w, n: consumers(_identity_callback(eos_fields(t, s, d, m)),
                                            t, s, d, w),
    "g": lambda t, s, d, m, w, n: (lambda f: consumers(f, t, s, d, w) + f)(eos_fields(t, s, d, m)),
}
SPLIT = "e"  # two programs
ARM_ORDER = ("a", "b", "c", "c2", "d", "d2", SPLIT, "f", "g")
_P1, _P2 = jax.jit(eos_fields), jax.jit(consumers)


def _split(t, s, d, m, w, n):
    return _P2(_P1(t, s, d, m), t, s, d, w)


def arm_fn(name):
    return _split if name == SPLIT else ARMS[name]


def build_arm(name, args):
    """-> (forward callable, [compiled HLO text of every program it runs])."""
    if name == SPLIT:
        t, s, d, m, w, _ = args
        ft = jax.eval_shape(eos_fields, t, s, d, m)
        return _split, [_P1.lower(t, s, d, m).compile().as_text(),
                        _P2.lower(ft, t, s, d, w).compile().as_text()]
    f = jax.jit(ARMS[name])
    return f, [f.lower(*args).compile().as_text()]


PLACEMENTS = {
    # (i) mid-step: an outer jit that does not return the EOS fields
    "outer_jit": lambda fn: lambda *a: fn(*a)[:N_OUT],
    # (ii) two steps of a lax.scan; each step's T feeds the next
    "scan2": lambda fn: lambda t, s, d, m, w, n: lax.scan(
        lambda tc, _: (lambda o: (tc + 1e-3 * o[2], o))(fn(tc, s, d, m, w, n)),
        t, None, length=2),
}


def _loss(outs):
    # Exact consumers only (the unguarded jnp.sum would feed its own fusion-dependent bits
    # into the cotangent); arm g's EOS outputs ride along as aux so XLA must keep them.
    return sum(jnp.sum(o * o) for o in outs[:N_EXACT]), tuple(outs[N_OUT:])


def build_grad(name, gargs):
    """-> (callable returning dL/d(t, s), [compiled HLO texts]).  Arm e's gradient is jax.grad
    of the two-program function run eagerly: each top-level jit of its jaxpr is one program."""
    g = jax.grad(lambda *a: _loss(arm_fn(name)(*a)), argnums=(0, 1), has_aux=True)
    if name != SPLIT:
        c = jax.jit(g).lower(*gargs).compile()
        return (lambda *a: c(*a)[0]), [c.as_text()]
    closed = jax.make_jaxpr(g)(*gargs)
    texts = [jax.jit(jaxpr_as_fun(q.params["jaxpr"])).lower(*[v.aval for v in q.invars])
             .compile().as_text() for q in closed.jaxpr.eqns if "jaxpr" in q.params
             and q.primitive.name in ("pjit", "jit")]
    return (lambda *a: jaxpr_as_fun(closed)(*a)[:2]), texts


def int64_bits(arrays):
    return [np.asarray(a).view(np.int64) for a in jax.tree_util.tree_leaves(arrays)]


def bits_equal(x, y) -> bool:
    bx, by = int64_bits(x), int64_bits(y)
    return len(bx) == len(by) and all(a.shape == b.shape and np.array_equal(a, b)
                                      for a, b in zip(bx, by))


def bit_diff(x, y) -> tuple[int, int, int]:
    """(differing elements, elements, max |int64 bit-pattern difference| = ulps when same sign)."""
    bx, by = int64_bits(x), int64_bits(y)
    return (sum(int((a != b).sum()) for a, b in zip(bx, by)), sum(a.size for a in bx),
            max(int(np.abs(a - b).max()) for a, b in zip(bx, by)))


def median_ms(fn, args, calls, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    walls = []
    for _ in range(calls):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        walls.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(walls)


def _copies(s):
    return {k: s[k] for k in POLY} | {"construct_ops": s["construct_ops"]}


def single_copy_check(args) -> dict:
    """Known case: the density alone, stored, must be exactly one copy."""
    t, s, d, m = args[:4]
    return copy_summary([jax.jit(density).lower(t, s, d, m).compile().as_text()], t.size)


def resolve_card() -> dict:
    """Re-read the round-349 fast_literal card the benchmark child builds; fail if CARD drifted."""
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round343_speed_benchmark as child,
    )
    card = child.build_fast_literal_mode_orca2_k0_card(
        child.bulk.build_nemo_orca2_k0_ocean_card(child.DEFAULT_DECK_ROOT))
    mc = card.ocean.recipe.model_config
    tke = mc.physics.vertical_mixing.tke
    got = {"eos": mc.eos, "eos_depth": mc.eos_depth, "rho_0": float(mc.rho_0), "g": float(mc.g),
           "pgf_scheme": mc.pgf_scheme, "transcendentals": card.ocean.transcendentals,
           "n2_eos_form": tke.n2_eos_form, "bn2_source_rounded": tke.nemo_bn2_source_rounded}
    if got != CARD:
        raise SystemExit(f"card resolves {got}, microbench assumes {CARD}")
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--arms", default=",".join(ARM_ORDER),
                    help="comma list; arm a always runs first")
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--skip-forward", action="store_true", help="gradient check only")
    ap.add_argument("--no-placement", action="store_true")
    ap.add_argument("--grad-levels", type=int, default=12,
                    help="grads on (148, 180, K): reverse-mode compile time grows steeply with K")
    ap.add_argument("--previous", type=Path, help="earlier report whose arm rows to keep beside")
    a = ap.parse_args(argv)
    if a.calls < 20:
        raise SystemExit("--calls must be >= 20")
    wanted = set(a.arms.split(",")) | {"a"}
    if wanted - set(ARM_ORDER):
        raise SystemExit(f"unknown arms {sorted(wanted - set(ARM_ORDER))}")
    names = [n for n in ARM_ORDER if n in wanted]
    rep = {"card": resolve_card(), "shape": SHAPE, "jax": jax.__version__, "argv": sys.argv,
           "inputs": "make_inputs(seed=351): synthetic T/S/live-gdept/tmask/e3w", "arms": {}}
    if a.previous:
        rep["previous"] = {"path": str(a.previous),
                           "arms": json.loads(a.previous.read_text()).get("arms")}
    rows = rep["arms"]

    def save():
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=2, default=str) + "\n")

    args = (*make_inputs(), jnp.int32(1))
    full = args[0].size
    rep["single_copy_check"] = single = _copies(single_copy_check(args))
    print(f"single-copy check: {single['rho']}", flush=True)
    if (single["rho"]["copies"], single["rho"]["partial_kernels"], single["rho"]["multiplies"],
            single["rab"]["multiplies"]) != (1, 0, POLY["rho"][1], 0):
        rep["refused"] = f"single-copy known case gave {single}, expected 1 density copy"
    ref = None
    for name in [] if (a.skip_forward or "refused" in rep) else names:
        fwd, texts = build_arm(name, args)
        row = rows[name] = _copies(copy_summary(texts, full))
        outs = jax.block_until_ready(fwd(*args))
        ref = outs if name == "a" else ref
        row["bits_equal_a"] = bits_equal(outs[:N_EXACT], ref[:N_EXACT])
        row["unordered_sum_diff_vs_a"] = bit_diff(outs[N_EXACT:N_OUT], ref[N_EXACT:N_OUT])
        row["median_ms"] = median_ms(fwd, args, a.calls)
        if name == "a":
            row["by_consumer_group"] = {
                g: _copies(copy_summary([jax.jit(lambda *x, lo=lo, hi=hi: ARMS["a"](*x)[lo:hi])
                                         .lower(*args).compile().as_text()], full))
                for g, (lo, hi) in GROUPS.items()}
        save()
        print(f"arm {name:2s} rho {row['rho']['copies']} rab {row['rab']['copy_equivalents']} "
              f"grid_evals {row['rho']['grid_evals']}/{row['rab']['grid_evals']} "
              f"dead {row['rho']['dead_branch_copies']} bits_equal_a {row['bits_equal_a']} "
              f"unordered {row['unordered_sum_diff_vs_a']} median {row['median_ms']:.1f} ms",
              flush=True)
        if name == "a" and row["rho"]["copies"] < 2:
            rep["refused"] = "arm a has < 2 density copies: microbench does not represent the step"
            break
    pref = {}
    for name in [] if (a.skip_forward or a.no_placement or "refused" in rep) else names:
        if name not in ("a", "c2", "d2", SPLIT, "f", "g"):
            continue
        for place, wrap in PLACEMENTS.items():
            f = jax.jit(wrap(arm_fn(name)))
            c = f.lower(*args).compile()
            outs = jax.block_until_ready(c(*args))
            exact = (outs[0], outs[1][:N_EXACT]) if place == "scan2" else outs[:N_EXACT]
            pref.setdefault(place, exact)
            rows[name][place] = _copies(copy_summary([c.as_text()], full)) | {
                "bits_equal_a_same_placement": bits_equal(exact, pref[place])}
            save()
            p = rows[name][place]
            print(f"arm {name:2s} {place:9s} rho {p['rho']['copies']} "
                  f"rab {p['rab']['copy_equivalents']} "
                  f"partial {p['rho']['partial_kernels']} grid_evals {p['rho']['grid_evals']} "
                  f"bits_equal_a {p['bits_equal_a_same_placement']}", flush=True)
    gshape = SHAPE[:2] + (a.grad_levels,)
    gargs, gref = (*make_inputs(gshape), jnp.int32(1)), None
    for name in [] if (a.no_grad or "refused" in rep) else names:
        row = rows.setdefault(name, {})
        row["grad_shape"] = gshape
        try:
            grad, texts = build_grad(name, gargs)
            row["grad_program"] = _copies(copy_summary(texts, int(np.prod(gshape))))
            row["grad_program"]["programs"] = len(texts)
            g = jax.block_until_ready(grad(*gargs))
            gref = g if name == "a" else gref
            row["grad"] = "runs"
            row["grad_bits_equal_a"] = bits_equal(g, gref)
            row["grad_diff_vs_a"] = bit_diff(g, gref)
        except Exception as exc:  # report, never hide, a mechanism that breaks reverse mode
            row["grad"] = f"FAILS: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        save()
        gp = row.get("grad_program", {})
        print(f"arm {name:2s} grad on {gshape}: {row['grad']} diff {row.get('grad_diff_vs_a')} "
              f"program rho {gp.get('rho', {}).get('copies')} "
              f"rab {gp.get('rab', {}).get('copy_equivalents')}", flush=True)
    save()
    if "refused" in rep:
        print("REFUSED:", rep["refused"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
