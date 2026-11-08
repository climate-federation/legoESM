#!/usr/bin/env python
"""Round 351: which construct makes the density's consumers read ONE stored copy on XLA CPU?

In the compiled one-core ORCA2 step XLA re-evaluates the NEMO Roquet density polynomial
inside many consumer kernels (69 complete copies where 5 suffice; study
``orca2_r351_eos_research/report_v2.md``).  This microbench feeds the REAL pressure-density
call (``nemo_roquet_density_anomaly_ratio`` exactly as ``ocean_pe_latlon_cgrid.py`` calls it
for the round-349 fast_literal card: nemo_eos80, geometric depth, rho0 = card rho_0, 3-D
tmask; guards untouched) into consumers shaped like the step's, and tries one construct per
arm between producer and consumers:

  a   none (baseline)                     d   lax.while_loop, runtime trip count 1, identity
  b   optimization_barrier (control)      d2  lax.while_loop, density computed in the body
  c   lax.cond, runtime predicate,        e   two compiled programs: density, then consumers
      identity branches                   f   jax.pure_callback identity (opaque host call)
  c2  lax.cond, density computed in       g   density also returned as a program output
      both branches
c and d are what was asked; JAX prunes a branch/carry that forwards its input unchanged
(conditionals.py and loops.py "_jaxpr_forwarding"), so c2/d2 put the density INSIDE the construct.

Per arm: kernels holding the polynomial (complete / partial copies), output bits vs arm a
(int64 views; the guarded/exact consumers, and separately one unguarded jnp.sum whose
association XLA may change when a fusion boundary moves), median wall per call (pinned by
the caller), and whether jax.grad of a scalar loss runs and matches arm a's gradient bits.

Copy counter = the study's method (``hlo_eos_sites.parse`` + ``hlo_eos_kernels`` POLY rule,
uncommitted, ported here): a polynomial multiply is an HLO ``multiply`` whose innermost
eos.py stack frame is ``_nemo_roquet_eos_impl.<locals>._fmul``; one source copy = 51 of
them; a kernel is a top-level op of a non-fusion computation, with nested fusion bodies
attributed to it.  The run refuses to report arms unless (1) the density alone compiles to
exactly one complete copy and (2) arm a reproduces the duplication (>= 2 complete copies).

Usage: orca2_round351_eos_store_microbench.py --report R.json [--calls 20] [--arms a,b]
       [--no-grad | --skip-forward] [--grad-levels 12]
Gradients run on (148, 180, --grad-levels): XLA CPU compile time of the reverse pass grows
steeply with the level count (MEASURED: 22 s at 6, 104 s at 20; arm a unfinished after
18 min at 30).
"""
from __future__ import annotations

import argparse
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
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import nemo_hpg_sco_literal_cgrid  # noqa: E402
from legoesm.ocean.eos import nemo_roquet_density_anomaly_ratio  # noqa: E402

from legoesm import constants  # noqa: E402

SHAPE = (148, 180, 30)
# What the round-349 fast_literal ORCA2 card resolves (re-read by resolve_card()):
CARD = {"eos": "nemo_eos80", "eos_depth": "geometric", "rho_0": 1026.0,
        "pgf_scheme": "nemo_sco", "transcendentals": "native"}
POLY_FILE = "ocean/eos.py"
POLY_FN = "_nemo_roquet_eos_impl.<locals>._fmul"
POLY_PER_COPY = 51  # _fmul calls in one Roquet evaluation (study: 51 multiplies = one copy)
CONSTRUCT_OPS = ("conditional", "while", "custom-call", "opt-barrier")

# --------------------------------------------------------------------------- copy counter
_INSTR = re.compile(r"^\s*(?:ROOT )?%(\S+) = (.*)$")
_OPCODE = re.compile(r" ([a-z][a-z0-9-]*)\(")
_COMP = re.compile(r"^(ENTRY )?%(\S+) \(.*\{\s*$")
_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


def parse_hlo(text: str):
    """-> ({computation: [(name, opcode, frame, calls)]}, frame -> innermost eos.py function).

    Port of the study's hlo_eos_sites.parse for the fields the counter needs."""
    files, funcs, locs, frames = {}, {}, {}, {}
    comps: dict[str, list] = {}
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
            comps[comp] = []
            continue
        m = _INSTR.match(line)
        if not m or comp is None:
            continue
        name, rest = m.groups()
        om = _OPCODE.search(rest if rest.startswith("(") else " " + rest)
        if om is None:
            continue
        fr = re.search(r"stack_frame_id=(\d+)", rest)
        calls = re.search(r"calls=%([\w.\-]+)", rest)
        comps[comp].append((name, om.group(1), int(fr.group(1)) if fr else None,
                            calls.group(1) if calls else None))

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

    return comps, innermost_eos_function


def polynomial_kernels(text: str) -> list[tuple[str, int]]:
    """[(kernel, polynomial multiplies)] for every kernel holding at least one."""
    comps, eos_fn = parse_hlo(text)
    bodies = {c for ins in comps.values() for _, op, _, c in ins if op == "fusion" and c}

    def count(comp):
        n = 0
        for _, op, fr, calls in comps[comp]:
            if op == "fusion" and calls:
                n += count(calls)
            elif op == "multiply" and eos_fn(fr) == POLY_FN:
                n += 1
        return n

    out = []
    for cname, ins in comps.items():
        if cname in bodies:
            continue
        for name, op, fr, calls in ins:
            n = count(calls) if op == "fusion" and calls else int(
                op == "multiply" and eos_fn(fr) == POLY_FN)
            if n:
                out.append((name, n))
    return out


def copy_summary(texts: list[str]) -> dict:
    ks = [k for t in texts for k in polynomial_kernels(t)]
    ops = {op: sum(len(re.findall(rf" {op}\(", t)) for t in texts) for op in CONSTRUCT_OPS}
    return dict(complete=sum(n >= POLY_PER_COPY for _, n in ks),
                partial=sum(0 < n < POLY_PER_COPY for _, n in ks),
                poly_multiplies=sum(n for _, n in ks), construct_ops=ops,
                kernels={k: n for k, n in ks})


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
    for the card: p_eos = rho_0*constants.g*gdept as iterate_eos_and_pressure_anomaly builds it."""
    p_eos = (CARD["rho_0"] * constants.g) * depth
    return nemo_roquet_density_anomaly_ratio(
        t, s, p_eos, rho0=CARD["rho_0"], geometric_depth_m=depth, tmask=tmask)


@functools.cache
def _grid(nj, ni):
    # ponytail: uniform beta-plane box, no ORCA north fold (the fold only swaps the last row's
    # neighbour; the step's rolls/level loop are the same)
    with jax.ensure_compile_time_eval():  # cached: must hold arrays, not a trace's tracers
        return create_beta_plane_cgrid_geometry(nj, ni, dx_m=2.0e5, dy_m=2.0e5, f0=0.0,
                                                beta=0.0, cartesian_pseudo_lat=True,
                                                dtype=jnp.float64)


def consumers(r, depth, e3w):
    """The step's real pressure consumer (NEMO hpg_sco level-by-level recurrence) plus
    slope/N2/MLE-shaped reads (shapes from partD_elements.txt)."""
    hpg_u, hpg_v = nemo_hpg_sco_literal_cgrid(r, e3w, depth, _grid(*r.shape[:2]), constants.g)
    return (
        hpg_u, hpg_v,
        jnp.roll(r, -1, axis=1) - r,                          # shift along i
        jnp.roll(r, -1, axis=0) - r,                          # shift along j
        0.5 * (jnp.roll(r, -1, axis=1) + jnp.roll(r, 1, axis=1)),  # two offsets per cell
        jnp.pad(r, ((6, 6), (0, 0), (0, 0))),                 # halo pad
        r[..., 1:] - r[..., :-1],                             # vertical difference
        _ordered_level_sum(nemo_source_round(r * e3w)),       # level sum, NEMO-literal order
        jnp.sum(r * e3w, axis=-1),                            # level sum, XLA's order (unguarded)
    )


def _ordered_level_sum(x):
    acc = nemo_source_round(x[..., 0])
    for k in range(1, x.shape[-1]):
        acc = nemo_source_round(acc + x[..., k])
    return acc


def _identity_callback(r):
    return jax.pure_callback(lambda x: np.asarray(x), jax.ShapeDtypeStruct(r.shape, r.dtype), r)


def _while1(n, body, init):
    return lax.while_loop(lambda c: c[0] < n, body, (jnp.int32(0), init))[1]


ARMS = {
    "a": lambda t, s, d, m, w, n: consumers(density(t, s, d, m), d, w),
    "b": lambda t, s, d, m, w, n: consumers(lax.optimization_barrier(density(t, s, d, m)), d, w),
    "c": lambda t, s, d, m, w, n: consumers(
        lax.cond(n > 0, lambda x: x, lambda x: x, density(t, s, d, m)), d, w),
    "c2": lambda t, s, d, m, w, n: consumers(lax.cond(n > 0, density, density, t, s, d, m), d, w),
    "d": lambda t, s, d, m, w, n: consumers(
        _while1(n, lambda c: (c[0] + 1, c[1]), density(t, s, d, m)), d, w),
    "d2": lambda t, s, d, m, w, n: consumers(
        _while1(n, lambda c: (c[0] + 1, density(t, s, d, m)), jnp.zeros_like(t)), d, w),
    "f": lambda t, s, d, m, w, n: consumers(_identity_callback(density(t, s, d, m)), d, w),
    "g": lambda t, s, d, m, w, n: (lambda r: consumers(r, d, w) + (r,))(density(t, s, d, m)),
}
SPLIT = "e"  # two programs, built in build_arm
ARM_ORDER = ("a", "b", "c", "c2", "d", "d2", SPLIT, "f", "g")
N_OUT = 9  # consumer outputs; arm g appends the density itself
N_EXACT = 8  # outputs written in guarded/exact arithmetic; output 9 is XLA-ordered jnp.sum


def _loss(outs):
    # Exact consumers only: the XLA-ordered jnp.sum output would feed its own fusion-dependent
    # bits into the cotangent and make every gradient comparison fail for that reason alone.
    return sum(jnp.sum(o * o) for o in outs[:N_EXACT])


_P1, _P2 = jax.jit(density), jax.jit(consumers)


def _split(t, s, d, m, w, n):
    return _P2(_P1(t, s, d, m), d, w)


def build_arm(name, args):
    """-> (forward callable, [compiled HLO text of every program it runs])."""
    if name == SPLIT:
        t, s, d, m, w, _ = args
        return _split, [_P1.lower(t, s, d, m).compile().as_text(),
                        _P2.lower(jax.ShapeDtypeStruct(t.shape, t.dtype), d, w).compile().as_text()]
    f = jax.jit(ARMS[name])
    return f, [f.lower(*args).compile().as_text()]


def build_grad(name):
    """jax.grad of the scalar loss w.r.t. (t, s); arm e's two programs stay two programs."""
    if name == SPLIT:
        return jax.grad(lambda *a: _loss(_split(*a)), argnums=(0, 1))
    return jax.jit(jax.grad(lambda *a: _loss(ARMS[name](*a)), argnums=(0, 1)))


def int64_bits(arrays):
    return [np.asarray(a).view(np.int64) for a in arrays]


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


def single_copy_check(args) -> dict:
    """Known case: the density alone, stored, must be exactly one complete copy."""
    t, s, d, m = args[:4]
    return copy_summary([jax.jit(density).lower(t, s, d, m).compile().as_text()])


def resolve_card() -> dict:
    """Re-read the round-349 fast_literal card the benchmark child builds; fail if CARD drifted."""
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round343_speed_benchmark as child,
    )
    card = child.build_fast_literal_mode_orca2_k0_card(
        child.bulk.build_nemo_orca2_k0_ocean_card(child.DEFAULT_DECK_ROOT))
    mc = card.ocean.recipe.model_config
    got = {"eos": mc.eos, "eos_depth": mc.eos_depth, "rho_0": float(mc.rho_0),
           "pgf_scheme": mc.pgf_scheme, "transcendentals": card.ocean.transcendentals}
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
    ap.add_argument("--grad-levels", type=int, default=12,
                    help="grads on (148, 180, K): reverse-mode compile time grows steeply with K")
    a = ap.parse_args(argv)
    if a.calls < 20:
        raise SystemExit("--calls must be >= 20")
    wanted = set(a.arms.split(",")) | {"a"}
    if wanted - set(ARM_ORDER):
        raise SystemExit(f"unknown arms {sorted(wanted - set(ARM_ORDER))}")
    names = [n for n in ARM_ORDER if n in wanted]
    rep = {"card": resolve_card(), "shape": SHAPE, "jax": jax.__version__, "argv": sys.argv,
           "inputs": "make_inputs(seed=351): synthetic t/s/live-gdept/tmask/e3w", "arms": {}}
    rows = rep["arms"]

    def save():
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=2, default=str) + "\n")

    args = (*make_inputs(), jnp.int32(1))
    rep["single_copy_check"] = single = single_copy_check(args)
    print(f"single-copy check: {single['complete']} complete, {single['partial']} partial, "
          f"{single['poly_multiplies']} polynomial multiplies", flush=True)
    if (single["complete"], single["partial"], single["poly_multiplies"]) != (1, 0, POLY_PER_COPY):
        rep["refused"] = f"single-copy known case gave {single}, expected 1 complete copy"
    ref = None
    for name in [] if (a.skip_forward or "refused" in rep) else names:
        fwd, texts = build_arm(name, args)
        row = rows[name] = copy_summary(texts)
        outs = jax.block_until_ready(fwd(*args))
        ref = outs if name == "a" else ref
        row["bits_equal_a"] = bits_equal(outs[:N_EXACT], ref[:N_EXACT])
        row["unordered_sum_bits_equal_a"] = bits_equal(outs[N_EXACT:N_OUT], ref[N_EXACT:])
        row["unordered_sum_diff_vs_a"] = bit_diff(outs[N_EXACT:N_OUT], ref[N_EXACT:])
        row["median_ms"] = median_ms(fwd, args, a.calls)
        save()
        print(f"arm {name:2s} complete {row['complete']:2d} partial {row['partial']:2d} "
              f"constructs {row['construct_ops']} bits_equal_a {row['bits_equal_a']} "
              f"(unordered sum {row['unordered_sum_bits_equal_a']}, "
              f"diff {row['unordered_sum_diff_vs_a']}) "
              f"median {row['median_ms']:.2f} ms", flush=True)
        if name == "a" and row["complete"] < 2:
            rep["refused"] = "arm a has < 2 complete copies: microbench does not represent the step"
            break
    gshape = SHAPE[:2] + (a.grad_levels,)
    gargs, gref = (*make_inputs(gshape), jnp.int32(1)), None
    for name in [] if (a.no_grad or "refused" in rep) else names:
        row = rows.setdefault(name, {})
        row["grad_shape"] = gshape
        try:
            grad = build_grad(name)
            if name != SPLIT:  # copies left in the reverse-mode program itself
                grad = grad.lower(*gargs).compile()
                gs = copy_summary([grad.as_text()])
                row["grad_program_copies"] = (gs["complete"], gs["partial"])
            g = jax.block_until_ready(grad(*gargs))
            gref = g if name == "a" else gref
            row["grad"] = "runs"
            row["grad_bits_equal_a"] = bits_equal(g, gref)
            row["grad_diff_vs_a"] = bit_diff(g, gref)
        except Exception as exc:  # report, never hide, a mechanism that breaks reverse mode
            row["grad"] = f"FAILS: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        save()
        print(f"arm {name:2s} grad on {gshape}: {row['grad']} "
              f"bits_equal_a {row.get('grad_bits_equal_a')} diff {row.get('grad_diff_vs_a')} "
              f"program copies (complete, partial) {row.get('grad_program_copies')}", flush=True)
    save()
    if "refused" in rep:
        print("REFUSED:", rep["refused"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
