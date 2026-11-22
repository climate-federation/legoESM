"""Per-step, per-stage value ranges of an MPAS AMIP run (numbers only).

Locates where a reduced-precision run first leaves the physical range, and
lets two arms (e.g. fp32 vs fp64, same IC) be compared step by step.  Wraps,
without changing what the model computes:

* the dycore step (``MPASPrimitiveEquationModel.step``): state ranges before
  and after every step (stage ``state_in`` / ``state_out``);
* every combined-physics component built by the factories in
  ``legoesm.atmosphere.physics.combined`` (radiation, convection, turbulence,
  surface bulk, microphysics, gravity-wave drag): ranges of every array leaf
  each call returns (tendencies and carries);
* the MPAS dry-mass fixer (``_fix_mass_mpas_hydro``): p_s in / out;
* the multilayer land step: every forcing leaf it receives and every leaf it
  returns.

Limits (codex review): ``call`` is a TRACE-time counter (distinguishes the
call sites inside one compiled step, e.g. macmic sub-steps), not an execution
order; rows inside one step are unordered.  Intermediate dynamics states and
the driver's post-step processing are not instrumented: a value created there
shows first in the next step's ``state_in``.  Must be installed before the
driver builds its physics (it is: install() runs before run_amip).

Each row: ``arm, step, stage, call, leaf, dtype, min, max, n_nonfinite,
argmin, argmax`` (flat indices into the leaf, to follow one column).
``step`` is the driver step during which the row was produced (rows from
traced code are flushed with ``jax.effects_barrier`` after every step).
``min``/``max`` deliberately use plain min/max (a NaN must show, not hide).
NO verdicts: thresholds and arm comparison are applied afterwards.

Usage (same argv as run_amip.py)::

    PROBE_CSV=out.csv PROBE_ARM=fp32 python scripts/validate/mpas_fp32_first_bad_stage.py \\
        scripts/run/run_amip.py --config config/amip/<deck>.yaml ...
"""
from __future__ import annotations

import csv
import functools
import os
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np

_CSV = os.environ.get("PROBE_CSV", "first_bad_stage.csv")
_ARM = os.environ.get("PROBE_ARM", "arm")
_STEP = [0]
_CALLS: dict[str, int] = {}
if "--report" not in sys.argv:
    _FH = open(_CSV, "w", newline="")
    _W = csv.writer(_FH)
    _W.writerow(["arm", "step", "stage", "call", "leaf", "dtype", "min", "max",
                 "n_nonfinite", "argmin", "argmax"])


def _leaves(tree):
    return [(jax.tree_util.keystr(p), x)
            for p, x in jax.tree_util.tree_leaves_with_path(tree)
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
            and getattr(x, "size", 0) > 0]


def _reduce(tree):
    """On-device reductions -> (names, dtypes, (n, 3) min/max/n_nonfinite)."""
    lv = _leaves(tree)
    acc = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    names = [n for n, _ in lv]
    dtypes = [str(x.dtype) for _, x in lv]
    stats = [jnp.stack([jnp.min(x).astype(acc), jnp.max(x).astype(acc),
                        jnp.sum(~jnp.isfinite(x)).astype(acc),
                        jnp.argmin(x).astype(acc), jnp.argmax(x).astype(acc)])
             for _, x in lv]
    return names, dtypes, (jnp.stack(stats) if stats else None)


def _write(stage, call, names, dtypes, arr):
    arr = np.asarray(arr)
    for n, d, (lo, hi, nn, ilo, ihi) in zip(names, dtypes, arr):
        _W.writerow([_ARM, _STEP[0], stage, call, n, d, repr(float(lo)),
                     repr(float(hi)), int(nn), int(ilo), int(ihi)])


def _report(stage, tree):
    """Works eagerly and under tracing (jax.debug.callback)."""
    call = _CALLS[stage] = _CALLS.get(stage, 0) + 1
    names, dtypes, arr = _reduce(tree)
    if arr is None:
        return
    if isinstance(arr, jax.core.Tracer):
        jax.debug.callback(functools.partial(_write, stage, call, names, dtypes), arr)
    else:
        _write(stage, call, names, dtypes, arr)


def _wrap_component(name, factory):
    @functools.wraps(factory)
    def make(*a, **k):
        fn = factory(*a, **k)

        @functools.wraps(fn)
        def wrapped(*fa, **fk):
            out = fn(*fa, **fk)
            _report(f"phys:{name}", out)
            return out
        wrapped.__dict__.update(getattr(fn, "__dict__", {}))
        return wrapped
    return make


def install():
    # jax.debug.callback needs a CPU device: fail now, not after compiling
    # (set JAX_PLATFORMS=cuda,cpu on a GPU node)
    jax.devices("cpu")
    import legoesm.atmosphere.physics.combined as CB
    for nm in ("make_radiation_physics", "make_convection_physics",
               "make_turbulence_physics", "make_surface_bulk_physics",
               "make_microphysics_physics", "make_gwd_physics"):
        if hasattr(CB, nm):
            setattr(CB, nm, _wrap_component(nm[len("make_"):-len("_physics")],
                                            getattr(CB, nm)))
    # surface bulk is imported locally inside make_physics: patch its source
    import legoesm.atmosphere.physics.turbulence.integration as TI
    TI.make_surface_bulk_physics = _wrap_component(
        "surface_bulk", TI.make_surface_bulk_physics)

    import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as PE
    _fix = PE._fix_mass_mpas_hydro

    def fix(state_new, *a, **k):
        _report("fixer:p_s_in", {"p_s": state_new.p_s.data})
        out = _fix(state_new, *a, **k)
        _report("fixer:p_s_out", {"p_s": out.p_s.data})
        return out
    PE._fix_mass_mpas_hydro = fix

    _step = PE.MPASPrimitiveEquationModel.step

    def step(self, state, *a, **k):
        # drain callbacks still pending from the previous step (incl. the
        # land step, which the driver runs AFTER model.step) before the label
        # moves on, so they keep the step they belong to
        jax.effects_barrier()
        _STEP[0] += 1
        _report("state_in", state)
        out = _step(self, state, *a, **k)
        _report("state_out", out)
        jax.effects_barrier()
        _FH.flush()
        return out
    PE.MPASPrimitiveEquationModel.step = step

    import legoesm.land.multilayer_land as ML
    _land = ML.step_multilayer_land_with_diagnostics

    def land(state, forcing, *a, **k):
        _report("land:forcing_in", forcing)
        _report("land:state_in", state)
        out = _land(state, forcing, *a, **k)
        _report("land:out", out)
        return out
    ML.step_multilayer_land_with_diagnostics = land


# gross sanity bounds for a state leaf (votes: author/codex/GLM, a/a/a)
_BOUNDS = {"T": (150.0, 350.0), "u": (-150.0, 150.0), "v": (-150.0, 150.0),
           "q_v": (-1e-6, 0.05), "p_s": (45000.0, 110000.0)}


def _bound_for(leaf):
    for k, b in _BOUNDS.items():
        if leaf.startswith(f".{k}") or f"'{k}'" in leaf:
            return b
    return None


def report(paths, stages=None, steps=None):
    """Numbers only: per arm the first step whose state leaves a bound or goes
    non-finite (with the stage rows of that step), then the per-step extrema
    of the requested stages side by side for every arm (no threshold)."""
    import pandas as pd
    df = pd.concat([pd.read_csv(p) for p in paths])
    for arm, d in df.groupby("arm"):
        st = d[d.stage.isin(["state_in", "state_out"])]
        bad = []
        for r in st.itertuples():
            b = _bound_for(r.leaf)
            if r.n_nonfinite > 0 or (b and (r.min < b[0] or r.max > b[1])):
                bad.append((r.step, r.stage, r.leaf, r.min, r.max, r.n_nonfinite))
        first = min(bad) if bad else None
        print(f"[{arm}] first out-of-bound state row: {first}")
        if first:
            k = first[0]
            sub = d[(d.step >= k - 1) & (d.step <= k)]
            print(sub.to_string(index=False, max_rows=400))
    if stages:
        piv = df[df.stage.isin(stages)]
        if steps:
            piv = piv[piv.step.between(*steps)]
        print(piv.pivot_table(index=["stage", "leaf", "step"], columns="arm",
                              values=["min", "max", "n_nonfinite"],
                              aggfunc="first").to_string(max_rows=2000))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--report":
    report(sys.argv[2:])
    sys.exit(0)


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        jax.effects_barrier()
        _FH.close()
