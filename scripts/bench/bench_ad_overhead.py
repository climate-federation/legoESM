#!/usr/bin/env python
"""Cost of reverse-mode AD: gradient step vs forward step, matched config.

WHY THIS EXISTS
---------------
Every throughput number in this repo (``docs/performance/scaling/``, the
route-B campaign behind ``plot_routeb_campaign_paper.py``) is a FORWARD-ONLY
measurement: the benchmarks time ``model.step`` and explicitly build segments
with ``gradient_checkpoint=False`` so the AD path is never taken.  Separately,
reverse-mode gradients are verified CORRECT through MPI halo exchange and
global-sum reductions (``tests/distributed/test_mpi_differentiability.py``),
but only at <=6 ranks and without any timing.

So the repo can say "it is fast" and "gradients are correct", but has never
been able to say what a gradient COSTS.  This benchmark measures exactly that
one number and nothing else.

WHAT IT MEASURES
----------------
Wall time per segment for three modes, at an otherwise IDENTICAL config
(same grid, resolution, nlev, dt, precision, step count, same ``step_fn``
built by the very driver that produced the scaling curves):

  forward    : lax.scan of ``step_fn`` over ``--steps``            (baseline)
  grad       : d(loss)/d(initial state), full BPTT, no checkpointing
  grad_ckpt  : same, with per-step ``jax.checkpoint`` (remat)

Reported per mode: median/min wall time, and the ratios grad/forward and
grad_ckpt/forward in THREE currencies -- wall time, compiled FLOPs, and
compiled bytes accessed -- plus the reverse-mode tape size.

WHICH RATIO TO QUOTE.  Quote the **FLOP ratio**.  This dycore is strongly
bandwidth-bound (arithmetic intensity ~0.35 FLOP/byte), so the WALL ratio
mostly tracks bytes, i.e. whether the reverse-mode tape fit in the measuring
machine's cache.  That makes the wall ratio a statement about one memory
hierarchy: on the dev laptop it climbed 5.7x -> 8.4x -> 10.4x purely as the
tape grew 17MB -> 343MB -> 1GB, while the FLOP ratio stayed ~5.8-7.7x and
was NOT monotonic.  Reporting the wall ratio alone would dress a cache
artifact up as a property of reverse-mode AD.  (Even the FLOP ratio exceeds
the classic <=4x reverse-mode bound here, because transposed halo indexing
emits scatters: 852 -> 2018.)

SCOPE -- READ BEFORE QUOTING A NUMBER
-------------------------------------
* SINGLE DEVICE ONLY.  This says nothing about how AD scales.  Reverse mode
  adds tape memory, adds a transposed halo exchange per exchange, and adds
  recompute under checkpointing -- all three attack the very quantity the
  strong-scaling curves measure (work per device available to hide comms), so
  the forward scaling curve should be treated as an OPTIMISTIC bound for the
  gradient case, not a proxy for it.  Distributed reverse-mode AD has never
  been run above 6 ranks in this repo and has never been timed.
* The loss is a generic sum-of-mean-squares over the state's inexact leaves.
  AD cost is dominated by the transposed program, not the loss form, but this
  is a COST probe -- it is not a physically meaningful objective and no
  gradient VALUE from it should be interpreted.
* Full BPTT memory grows with ``--steps`` x state size.  Defaults are small
  deliberately.  If ``grad`` OOMs while ``grad_ckpt`` survives, that itself is
  the result -- it is recorded rather than raised.

Run:
  python scripts/bench/bench_ad_overhead.py --grid cubed-sphere --resolution 24
  python scripts/bench/bench_ad_overhead.py --grid latlon --resolution 64 \
      --steps 8 --repeats 5 --json results/ad_overhead.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

# Repo root on sys.path: the shared case builders in run_cpu_mpi_scaling
# import ``tests.test_cases.*``, and this script is invoked as a bare path
# (``python scripts/bench/bench_ad_overhead.py``) by PBS/CI alike.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODES = ("forward", "grad", "grad_ckpt")


def scalar_loss(state):
    """Generic differentiable scalar over a state pytree's inexact leaves.

    Cost probe only -- see the module docstring.  Sum of per-leaf mean squares
    keeps every float leaf in the backward graph (so the measured cost is the
    full transposed program) while staying finite and grid-agnostic.
    """
    import jax
    import jax.numpy as jnp

    total = jnp.array(0.0)
    for leaf in jax.tree_util.tree_leaves(state):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            # Squared MAGNITUDE, not square: the spectral state carries
            # complex128 coefficients, and `square` would keep the loss
            # complex -- `grad` then refuses ("requires real-valued outputs").
            # real(x * conj(x)) is the squared modulus for complex and reduces
            # to x**2 for real leaves, so one expression covers both.
            total = total + jnp.mean(jnp.real(leaf * jnp.conj(leaf)))
    return total


def build_runners(step_fn, dt: float, steps: int, state_template=None):
    """Return {mode: callable(state) -> pytree} for each timed mode.

    All three share ONE ``step_fn`` and ONE step count so the ratio is a
    controlled comparison: the only variable is the AD transform.

    ``state_template`` (a representative state) lets the jit wrapper be
    probed per grid; omit it to force plain ``jax.jit``.
    """
    import equinox as eqx
    import jax

    def _scan(state, *, remat: bool):
        def body(carry, _):
            return step_fn(carry, dt), None

        scan_body = jax.checkpoint(body) if remat else body
        out, _ = jax.lax.scan(scan_body, state, None, length=steps)
        return out

    def forward(state):
        return _scan(state, remat=False)

    def _loss(state, *, remat: bool):
        return scalar_loss(_scan(state, remat=remat))

    _grad = eqx.filter_grad(lambda s: _loss(s, remat=False))
    _grad_ckpt = eqx.filter_grad(lambda s: _loss(s, remat=True))

    # Plain-function shims: jax.jit treats a bare equinox Module callable as a
    # static (hashable) argument and dies on an unhashable state pytree.
    # Capturing it in a closure keeps jit's view a plain function.
    def grad_fn(state):
        return _grad(state)

    def grad_ckpt_fn(state):
        return _grad_ckpt(state)

    # jax.jit (NOT eqx.filter_jit) for every mode, for two reasons:
    #   1. symmetry -- all three modes go through the identical jit wrapper,
    #      so the ratio cannot pick up a wrapper artifact;
    #   2. equinox's Compiled wrapper exposes NO cost_analysis/memory_analysis,
    #      which would silently blank the FLOP and byte ratios -- the portable
    #      numbers -- for exactly the grad modes we care about.
    # eqx.filter_grad still does the differentiation, so integer/static leaves
    # in the state are filtered out of the gradient as intended.
    return {
        "forward": _jit_with_fallback(forward, state_template),
        "grad": _jit_with_fallback(grad_fn, state_template),
        "grad_ckpt": _jit_with_fallback(grad_ckpt_fn, state_template),
    }


def _jit_with_fallback(fn, state_template):
    """``jax.jit`` if the state allows it, else ``eqx.filter_jit``.

    ``jax.jit`` rejects a pytree carrying non-array leaves, which
    ``eqx.filter_jit`` would treat as static.  lat-lon and cubed-sphere
    states are all-array, but the icosahedral/spectral states are not
    verified, so probe rather than assume: a grid whose state has an
    exotic leaf must still produce a wall-clock ratio, even though the
    equinox wrapper exposes no cost analysis (FLOP/byte columns blank,
    and ``jit_wrapper`` in the JSON records which path was taken).
    """
    import equinox as eqx
    import jax

    if state_template is None:
        return jax.jit(fn)
    try:
        jitted = jax.jit(fn)
        jitted.lower(state_template)  # trace-only; does not execute
        jitted.jit_wrapper = "jax.jit"
        return jitted
    except Exception:
        fallback = eqx.filter_jit(fn)
        try:
            fallback.jit_wrapper = "eqx.filter_jit"
        except AttributeError:  # equinox Modules are frozen
            pass
        return fallback


def analyze_mode(fn, state):
    """Static metrics of the COMPILED program: FLOPs, bytes, temp memory.

    These matter more than the wall clock.  This dycore is strongly
    bandwidth-bound (arithmetic intensity ~0.35 FLOP/byte), so the wall-time
    ratio largely tracks BYTES and therefore whether the reverse-mode tape
    fit in cache on the measuring machine -- it is a property of that memory
    hierarchy, not of reverse-mode AD.  The FLOP ratio is the portable
    statement; ``temp_bytes`` tells a reader whether the case was cache- or
    DRAM-resident.  Returns {} if the backend exposes no analysis.
    """
    out: dict = {}
    try:
        compiled = fn.lower(state).compile()
    except Exception as exc:
        # Do NOT blank silently: this triggers a SECOND full compilation after
        # timing, which can OOM at large --steps/resolution.  Since the FLOP
        # ratio is the number meant to be quoted, a missing one must be
        # visible in the output, not an empty column.
        out["analysis_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, (list, tuple)):
            cost = cost[0]
        if isinstance(cost, dict):
            out["flops"] = cost.get("flops")
            out["bytes_accessed"] = cost.get("bytes accessed")
    except Exception:
        pass
    try:
        out["temp_bytes"] = compiled.memory_analysis().temp_size_in_bytes
    except Exception:
        pass
    return out


def time_modes_interleaved(runners, state, repeats: int, modes):
    """Time all modes with repeats INTERLEAVED, not blocked per mode.

    Blocking (all forward runs, then all grad runs) puts any monotonic
    machine drift -- thermal throttling, DVFS, allocator growth -- entirely
    on whichever mode ran last, biasing the ratio.  Measured on the dev
    laptop: blocked 9.64x vs interleaved 7.79x for the same case, a 24%
    swing from ordering alone.  Round-robin sampling shares the drift.

    Returns {mode: {"warmup_s", "runs_s"} | {"error"}}.  ``warmup_s`` is
    compile PLUS one execution -- it is NOT a compile time.
    """
    import jax

    def _call(fn):
        t0 = time.perf_counter()
        out = fn(state)
        jax.block_until_ready(jax.tree_util.tree_leaves(out))
        return time.perf_counter() - t0

    out: dict = {}
    live = []
    for mode in modes:
        try:
            out[mode] = {"warmup_s": _call(runners[mode]), "runs_s": []}
            live.append(mode)
        except Exception as exc:
            out[mode] = {"error": _classify(exc)}

    for _ in range(repeats):
        for mode in list(live):
            try:
                out[mode]["runs_s"].append(_call(runners[mode]))
            except Exception as exc:
                out[mode] = {"error": _classify(exc)}
                live = [m for m in live if m != mode]
    return out


def _classify(exc: Exception) -> str:
    """Record memory exhaustion as a result; re-raise everything else.

    A full-BPTT OOM while ``grad_ckpt`` survives is the finding this bench
    exists to surface.  A shape error or tracer leak is a BUG, and swallowing
    it as ``ok: False`` would report "AD failed" for a defect in the harness.
    """
    text = f"{type(exc).__name__}: {exc}"
    oom_markers = ("RESOURCE_EXHAUSTED", "Out of memory", "OUT_OF_MEMORY",
                   "MemoryError", "failed to allocate")
    if any(marker.lower() in text.lower() for marker in oom_markers):
        return text
    raise exc


def run(grid: str, resolution: int, nlev: int, precision: str, steps: int,
        repeats: int, dt: float | None, modes=MODES,
        physics_level: str = "none"):
    """Measure AD overhead. Returns a JSON-serializable result dict."""
    from scripts.bench.run_cpu_mpi_scaling import build_amip_step

    step_fn, state, dt_used, total_cells, _cells_per_rank, _pm = (
        build_amip_step(
            grid_type=grid, resolution=resolution, nlev=nlev, rank=0,
            n_ranks=1, precision=precision, physics_level=physics_level,
            dt=dt))

    runners = build_runners(step_fn, dt_used, steps, state_template=state)
    result = {
        "grid": grid, "resolution": resolution, "n_levels": nlev,
        "precision": precision, "steps": steps, "repeats": repeats,
        "dt_seconds": dt_used, "total_cells": total_cells,
        "physics_level": physics_level, "distributed": False, "n_devices": 1,
        "modes": {},
    }

    timings = time_modes_interleaved(runners, state, repeats, modes)
    for mode in modes:
        t = timings[mode]
        if "error" in t:
            result["modes"][mode] = {"ok": False, "error": t["error"]}
            continue
        runs = t["runs_s"]
        entry = {
            "ok": True,
            "warmup_s": t["warmup_s"],  # compile + one run, NOT compile alone
            "median_s": statistics.median(runs),
            "min_s": min(runs),
            "max_s": max(runs),
            "runs_s": runs,
            "jit_wrapper": getattr(runners[mode], "jit_wrapper", "unknown"),
        }
        entry.update(analyze_mode(runners[mode], state))
        result["modes"][mode] = entry

    base = result["modes"].get("forward", {})
    if base.get("ok"):
        for mode in modes:
            m = result["modes"][mode]
            if not m.get("ok"):
                continue
            m["ratio_vs_forward"] = m["median_s"] / base["median_s"]
            # min-based ratio: min is less noise-biased than median for
            # wall clock (noise is one-sided -- it only ever adds time).
            m["ratio_min_vs_forward"] = m["min_s"] / base["min_s"]
            for key, name in (("flops", "flop_ratio_vs_forward"),
                              ("bytes_accessed", "byte_ratio_vs_forward")):
                # `is not None` (not truthiness): a legitimate 0 must not be
                # silently reported as "unavailable".
                if m.get(key) is not None and base.get(key):
                    m[name] = m[key] / base[key]
    return result


def _mb(n):
    return f"{n / 1e6:.0f}MB" if n else "-"


def format_report(result) -> str:
    lines = [
        f"AD overhead — {result['grid']} r{result['resolution']} "
        f"L{result['n_levels']} {result['precision']} "
        f"{result['steps']} steps, physics={result.get('physics_level')}, "
        f"1 device (no MPI/SPMD)",
        f"{'mode':<11} {'median s':>9} {'min s':>9} {'spread':>7} "
        f"{'wall x':>7} {'FLOP x':>7} {'byte x':>7} {'tape':>8}",
    ]
    for mode, m in result["modes"].items():
        if not m.get("ok"):
            lines.append(f"{mode:<11} {'FAILED':>9}   {m['error'][:52]}")
            continue

        def _x(key):
            v = m.get(key)
            return f"{v:.2f}x" if v is not None else "-"

        # Spread = (max-min)/min: how much of the ratio is machine noise.
        lo, hi = m.get("min_s"), m.get("max_s")
        spread = ((hi - lo) / lo) if (lo and hi is not None) else 0.0
        lines.append(
            f"{mode:<11} {m['median_s']:>9.4f} {m['min_s']:>9.4f} "
            f"{spread * 100:>6.1f}% "
            f"{_x('ratio_vs_forward'):>7} {_x('flop_ratio_vs_forward'):>7} "
            f"{_x('byte_ratio_vs_forward'):>7} {_mb(m.get('temp_bytes')):>8}")
        if m.get("analysis_error"):
            lines.append(f"{'':<11}   analysis unavailable: "
                         f"{m['analysis_error'][:56]}")
        if m.get("jit_wrapper") == "eqx.filter_jit":
            lines.append(f"{'':<11}   note: eqx.filter_jit fallback "
                         f"(no FLOP/byte analysis on this grid)")
    lines.append(
        "NOTE: wall x is bandwidth-bound and machine-specific (it tracks "
        "whether the\n      reverse tape fit in cache — see 'tape'). Quote "
        "FLOP x for a portable claim.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--grid", default="cubed-sphere",
                   choices=["cubed-sphere", "latlon", "icosahedral",
                            "spectral"])
    p.add_argument("--resolution", type=int, default=24)
    p.add_argument("--nlev", type=int, default=26)
    p.add_argument("--precision", default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--steps", type=int, default=8,
                   help="steps per timed segment (full-BPTT memory scales "
                        "with this)")
    p.add_argument("--repeats", type=int, default=9,
                   help="timed samples per mode, interleaved across modes")
    p.add_argument("--physics-level", default="none",
                   help="physics tier passed to the shared case builder; "
                        "'none' matches the scaling campaign (dycore only)")
    p.add_argument("--dt", type=float, default=None,
                   help="override dt; default = the driver's auto dt, "
                        "matching the scaling campaign")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--json", default=None, help="write result JSON here")
    a = p.parse_args(argv)

    result = run(a.grid, a.resolution, a.nlev, a.precision, a.steps,
                 a.repeats, a.dt, modes=tuple(a.modes),
                 physics_level=a.physics_level)
    print(format_report(result))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {a.json}")
    # A missing or dead baseline means NO ratio was produced; exiting 0 would
    # let a CI or PBS job go green having measured nothing.  Note the guard
    # covers OMITTING forward too (`--modes grad grad_ckpt`), not just its
    # failure -- otherwise opting out of the baseline bypasses the check.
    if "forward" not in a.modes:
        print("FATAL: --modes must include 'forward' (the ratio baseline)",
              file=sys.stderr)
        raise SystemExit(1)
    if not result["modes"]["forward"].get("ok"):
        print("FATAL: forward baseline failed — no ratio produced",
              file=sys.stderr)
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
