"""Probe whether wrapping the spectral PE step in a jit-with-sharding
on emulated multi-CPU devices actually scales.  Bypasses
``model.step`` (which has ``dt`` as a static arg and conflicts with
outer JIT) by jitting ``model._do_step`` directly with primed caches.

Use:
    JAX_ENABLE_X64=1 XLA_FLAGS=--xla_force_host_platform_device_count=N \\
        PYTHONPATH=. .venv/bin/python scripts/probe_spectral_shard.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-max", type=int, default=21,
                        help="spectral truncation T<n>")
    parser.add_argument("--nlev", type=int, default=8)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument(
        "--shard-axis", choices=["none", "level"], default="level",
        help="Sharding axis: none = single-device baseline, level = "
             "shard along the vertical level axis using the JAX device "
             "mesh exposed via XLA_FLAGS.",
    )
    parser.add_argument(
        "--scan-steps", type=int, default=1,
        help="If >1, fuse this many ``_do_step`` calls inside a single "
             "``jax.lax.scan`` body (combines the iter-204 scan-steps "
             "win with the iter-210 _do_step bypass).",
    )
    args = parser.parse_args()

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral, spectral_pe_tendencies,
    )

    print(f"devices: {jax.devices()}  ({jax.device_count()} total)")

    grid = create_gaussian_grid(n_max=args.n_max)
    sigma = create_sigma_coordinate(args.nlev)
    state = isothermal_rest_state_spectral(grid, sigma, perturbation_amplitude=0.1)
    model = SpectralPrimitiveEquationModel(grid, sigma, SpectralPEConfig())
    dt = float(args.dt)

    # Prime caches (these are Python-level, dt-keyed, must run before any JIT).
    model._ensure_si_data(dt)
    model._ensure_sponge_factor(dt)
    model._ensure_hyperdiff_filter(dt)
    model._ensure_tracer_filter(dt)

    def tendency_fn(s):
        return spectral_pe_tendencies(
            s, model.grid, model.sigma_coord, model.config, None,
        )

    if args.shard_axis == "level":
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(jax.devices(), axis_names=("level",))
        # Shard the vertical level axis of every leaf that has one.
        def _leaf_shard(leaf):
            if not hasattr(leaf, "shape"):
                return None
            # Only shard if last axis matches nlev *and* is divisible.
            if leaf.shape and leaf.shape[-1] == args.nlev \
                    and args.nlev % jax.device_count() == 0:
                return NamedSharding(mesh, P(*([None] * (leaf.ndim - 1) + ["level"])))
            return NamedSharding(mesh, P())  # replicated
        out_shardings = jax.tree.map(_leaf_shard, state)

        @jax.jit  # output sharding via constraint annotations on leaves
        def step_jit(s, dt_arr):
            new = model._do_step(s, dt_arr, tendency_fn)
            return jax.tree.map(
                lambda leaf, sh: jax.lax.with_sharding_constraint(leaf, sh)
                                  if sh is not None else leaf,
                new, out_shardings,
            )
        # The level-shard path wraps a single ``_do_step`` (no scan fusion), so
        # the timing-loop counters must be set here too — without these the
        # ``for _ in range(n_outer)`` loop below raised UnboundLocalError and the
        # entire level branch was dead (only the ``else`` path set them).
        n_outer = args.n_steps
        n_eff = args.n_steps
    else:
        @jax.jit
        def _step_one(s, dt_arr):
            return model._do_step(s, dt_arr, tendency_fn)

        if args.scan_steps > 1:
            @jax.jit
            def step_jit(s, dt_arr):
                def body(carry, _):
                    return _step_one(carry, dt_arr), None
                new, _ = jax.lax.scan(body, s, None, length=args.scan_steps)
                return new
            n_outer = (args.n_steps + args.scan_steps - 1) // args.scan_steps
            n_eff = n_outer * args.scan_steps
        else:
            step_jit = _step_one
            n_outer = args.n_steps
            n_eff = args.n_steps

    dt_arr = jnp.asarray(dt)
    # warm-up JIT
    state = step_jit(state, dt_arr)
    jax.block_until_ready(jax.tree.leaves(state))

    t0 = time.time()
    for _ in range(n_outer):
        state = step_jit(state, dt_arr)
    jax.block_until_ready(jax.tree.leaves(state))
    elapsed = time.time() - t0
    sps = n_eff / elapsed
    print(f"shard={args.shard_axis}  scan={args.scan_steps:>2d}  "
          f"T{args.n_max}/{args.nlev}L  "
          f"{n_eff} steps ({n_outer} chunks) in {elapsed:.2f}s  → {sps:.4f} steps/s")


if __name__ == "__main__":
    main()
