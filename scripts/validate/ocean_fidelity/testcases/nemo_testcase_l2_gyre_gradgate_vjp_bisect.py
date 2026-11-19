#!/usr/bin/env python3
"""Statement-boundary VJP bisect for the certified GYRE production step.

The taps are primal identities.  Each tap reports whether the cotangent at a
named boundary is finite; an operator owns the first poison when its output
cotangent is finite and the cotangent returned to one of its inputs is not.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_gradient_gate():
    path = Path(__file__).with_name("nemo_testcase_l2_gyre_gradient_gate.py")
    spec = importlib.util.spec_from_file_location("gyre_gradgate_bisect_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_vjp_tap(label: str, recorder):
    """Return a primal-identity array function that records its cotangent."""

    @jax.custom_vjp
    def tap(value):
        return value

    def forward(value):
        return value, None

    def backward(_residual, cotangent):
        finite = jnp.isfinite(cotangent)
        bad = ~finite
        flat_bad = bad.reshape(-1)
        first_bad = jnp.argmax(flat_bad)
        jax.debug.callback(
            lambda shape, bad_count, bad_index, bad_mask: recorder(
                label, shape, bad_count, bad_index, bad_mask),
            jnp.asarray(cotangent.shape, dtype=jnp.int32),
            jnp.sum(bad, dtype=jnp.int64),
            first_bad,
            bad,
            ordered=True,
        )
        return (cotangent,)

    tap.defvjp(forward, backward)
    return tap


def tap_pytree(tree, label: str, recorder):
    """Apply named identity taps to every inexact array leaf in ``tree``."""
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    tapped = []
    for path, leaf in path_leaves:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            suffix = jax.tree_util.keystr(path)
            tapped.append(make_vjp_tap(f"{label}{suffix}", recorder)(leaf))
        else:
            tapped.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, tapped)


def wrap_callable(original, label: str, recorder):
    """Tap floating inputs and outputs of one callable."""

    def wrapped(*args, **kwargs):
        args = tap_pytree(args, f"{label}.input", recorder)
        kwargs = tap_pytree(kwargs, f"{label}.kwinput", recorder)
        result = original(*args, **kwargs)
        return tap_pytree(result, f"{label}.output", recorder)

    return wrapped


def install_boundary_taps(context, recorder):
    """Instrument the major closures executed inside production ``_step_jitted``."""
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    originals = []

    def patch_module(name):
        original = getattr(model_module, name)
        setattr(model_module, name, wrap_callable(original, name, recorder))
        originals.append((model_module, name, original))

    for name in (
        "compute_frozen_geom_density",
        "_nemo_cen2_tracer_rhs",
        "_nemo_ws_rk3_tracer_pair_step",
        "barotropic_substeps_latlon_cgrid",
    ):
        patch_module(name)

    model = context.model

    def patch_method(name):
        original = getattr(model, name)
        wrapped = wrap_callable(original, name, recorder)
        setattr(model, name, types.MethodType(lambda _self, *a, **k: wrapped(*a, **k), model))
        originals.append((model, name, original))

    for name in (
        "_tke_step_entry_n2_bundle",
        "tendencies",
        "_tke_step_entry_p_sh2",
        "_apply_implicit_vertical_mixing",
    ):
        patch_method(name)
    return originals


def restore_boundary_taps(originals) -> None:
    for owner, name, original in reversed(originals):
        setattr(owner, name, original)


def measure() -> dict:
    """Run the registered one-step T-only reverse row with boundary taps."""
    gate = _load_gradient_gate()
    context = gate.build_context(gate.GYRE_CARD)
    events = []

    def recorder(label, shape, bad_count, first_bad, bad_mask):
        shape_values = [int(value) for value in np.asarray(shape)]
        count = int(np.asarray(bad_count))
        event = {
            "label": str(label),
            "shape": shape_values,
            "nonfinite_count": count,
            "first_nonfinite_flat_index": (
                int(np.asarray(first_bad)) if count else None
            ),
        }
        if count:
            bad_array = np.asarray(bad_mask)
            if bad_array.ndim == 3:
                horizontal = np.argwhere(np.any(bad_array, axis=-1))
                event["nonfinite_horizontal_cell_count"] = int(horizontal.shape[0])
                event["nonfinite_horizontal_cells"] = horizontal.tolist()
                event["nonfinite_levels"] = np.flatnonzero(
                    np.any(bad_array, axis=(0, 1))).tolist()
            elif bad_array.ndim == 2:
                event["nonfinite_cells"] = np.argwhere(bad_array).tolist()
        events.append(event)

    # Install before the first call to _step_jitted.  That call traces and
    # caches the production executable; installing afterward would leave the
    # taps outside the cached graph and produce a vacuous empty event list.
    originals = install_boundary_taps(context, recorder)
    zero = jnp.zeros_like(context.initial_state.T.data)
    truth = gate.production_rollout(context, zero, 1)
    jax.block_until_ready(truth)
    reference = gate.build_reference(context, truth)
    base, direction = gate.perturbations(context)
    try:
        def loss(d_t0):
            candidate = gate._score_state(gate.gradient_rollout(context, d_t0, 1))
            return gate.volume_weighted_mse(
                candidate.T,
                reference.state.T,
                reference.T_volume,
            ) / reference.T_variance

        def line(alpha):
            return loss(base + alpha * direction)

        reverse = jax.grad(line)(jnp.asarray(0.0, dtype=base.dtype))
        jax.block_until_ready(reverse)
        jax.effects_barrier()
    finally:
        restore_boundary_taps(originals)

    return {
        "case": gate.CASE,
        "steps": 1,
        "objective": "T-only registered normalized after-state loss",
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "reverse": float(np.asarray(reverse)) if bool(jnp.isfinite(reverse)) else None,
        "reverse_finite": bool(jnp.isfinite(reverse)),
        "events": events,
    }


def _git_output(*args: str) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", *args], cwd=_repo_root(), text=True).strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    actual = _git_output("rev-parse", "HEAD")
    if actual != args.expect_commit:
        raise RuntimeError(f"measurement commit {actual} != expected {args.expect_commit}")
    dirty = _git_output("status", "--porcelain")
    if dirty:
        raise RuntimeError(f"tracked/untracked worktree is not clean:\n{dirty}")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {args.output}")
    row = measure()
    row["provenance"] = {"commit": actual, "dirty": False}
    row["precision_policy"] = "PrecisionPolicy.fp64(transcendentals='libm')"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if not row["reverse_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
