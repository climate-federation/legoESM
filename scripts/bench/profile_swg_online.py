#!/usr/bin/env python3
"""Profile forward/reverse SWG closure segments without optimizing weights."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ml.training import TrainingConfig
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    build_swg_card,
)
from legoesm.training.swg_online import (
    BalancedNormalizers,
    SWGNetworkSpec,
    SWGScalarPotentialCNN,
    assert_network_gradients_live,
    build_closure_geometry,
    build_segment_fn,
    create_swg_optimizer,
    deterministic_trajectory_loss,
    network_parameter_count,
)

PROFILE_NETWORK_SPEC = SWGNetworkSpec(
    hidden_channels=16,
    hidden_layers=4,
    kernel_size=3,
    ssh_scale_m=1.0,
    velocity_scale_m_s=1.0,
    potential_scale_m3_s2=1.0,
)


def profile_metadata() -> dict:
    """Return the architecture and optimization choices frozen for profiling."""
    network = SWGScalarPotentialCNN(
        PROFILE_NETWORK_SPEC, key=jax.random.PRNGKey(20261013))
    return {
        "network": {
            **PROFILE_NETWORK_SPEC._asdict(),
            "input_channels": [
                "ssh_plus_noise", "u_at_t", "v_at_t", "wet_mask"],
            "output": "boundary-zero native F-point potential",
            "convolution_count": PROFILE_NETWORK_SPEC.hidden_layers + 1,
            "receptive_field_cells": (
                1 + (PROFILE_NETWORK_SPEC.hidden_layers + 1)
                * (PROFILE_NETWORK_SPEC.kernel_size - 1)),
            "parameter_count": network_parameter_count(network),
        },
        "checkpointing": "uniform production-step rematerialization",
        "optimizer_factory": "legoesm.ml.training.create_optimizer",
        "training_performed": False,
    }


def run_profile(window_days: int, mode: str, objective_operator: str) -> dict:
    """Compile and time one real-network sample; never update parameters."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if not jax.config.jax_enable_x64:
        raise ValueError("SWG online profile requires JAX x64")
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    network = SWGScalarPotentialCNN(
        PROFILE_NETWORK_SPEC, key=jax.random.PRNGKey(20261013))
    _, segment = build_segment_fn(
        card, geometry, window_days=window_days, checkpoint=True)
    n_steps = window_days * 24 * int(3_600 / card.dt_s)
    noise = jnp.zeros((n_steps, 3) + geometry.tmask.shape, dtype=jnp.float64)
    key = jax.random.PRNGKey(311)
    initial = card.recipe.initial_state
    # Construct through the shared factory and initialize state, but take no
    # update: this round proves plumbing and gradient reachability, not fit.
    optimizer = create_swg_optimizer(TrainingConfig(
        lr=1.0e-3, warmup_steps=0, total_steps=100,
        weight_decay=4.0e-3, optimizer="adamw", schedule="constant",
        clip_gradients=False, adam_epsilon=1.0e-7))
    optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    normalizers = BalancedNormalizers(1.0, 1.0, 1.0)

    def objective(net):
        _, hourly = segment(net, initial, noise, key, jnp.asarray(0.0))
        predicted = tuple(value[None] for value in hourly)
        target = tuple(jnp.zeros_like(value) for value in predicted)
        return deterministic_trajectory_loss(
            predicted, target, geometry, objective=objective_operator,
            normalizers=normalizers,
            beta_m1_s1=card.resolved_namelist.rn_beta_m_s,
            basin_length_m=card.resolved_namelist.rn_domsiz_m)

    if mode == "forward":
        measured = eqx.filter_jit(objective)

        def result_getter(result):
            return result
    elif mode == "reverse":
        measured = eqx.filter_jit(eqx.filter_value_and_grad(objective))

        def result_getter(result):
            value, grads = result
            assert_network_gradients_live(grads)
            return value
    else:
        raise ValueError("profile mode must be 'forward' or 'reverse'")

    started = time.perf_counter()
    first = measured(network)
    value = result_getter(first)
    jax.block_until_ready(value)
    compile_and_first_s = time.perf_counter() - started
    started = time.perf_counter()
    second = measured(network)
    value = result_getter(second)
    jax.block_until_ready(value)
    execution_s = time.perf_counter() - started
    return {
        **profile_metadata(),
        "window_days": window_days,
        "mode": mode,
        "objective_operator": objective_operator,
        "coarse_steps": n_steps,
        "hourly_observations": window_days * 24,
        "backend": jax.default_backend(),
        "compile_and_first_s": compile_and_first_s,
        "execution_s": execution_s,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "objective": float(value),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=int, choices=(1, 2, 4, 8),
                        required=True)
    parser.add_argument("--mode", choices=("forward", "reverse"), required=True)
    parser.add_argument("--objective", choices=("symmetric", "template"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_profile(args.window_days, args.mode, args.objective)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
