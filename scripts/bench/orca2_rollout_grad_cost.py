#!/usr/bin/env python3
"""Cost of one ORCA2 rollout gradient: device memory and wall time per schedule.

Feasibility probe for rollout training of a forced ORCA2 ocean (MEASUREMENT
ONLY; no model file is touched).  One ``jax.value_and_grad`` of

    L = volume-weighted mean of T^2 over wet cells after ``--steps`` steps

with respect to an additive perturbation of the step-1 temperature, through
``legoesm.training.checkpoint_schedule.checkpointed_loop`` with the chosen
adjoint-memory schedule.  The loop body is the production
``LatLonCGridOceanModel.step`` of the normal-mode standalone ORCA2 K0 card
(round 343's builder), fed per step by the freshwater, surface forcing and
damping target RECORDED from a forward run of the full composed K0 step
(bulk formulas + observed ice).  Inside the differentiated rollout those
fluxes are therefore fixed: the bulk-formula feedback is absent; the ocean
step, which dominates the cost, is the production one.

Reported per run: XLA's compiled memory estimate; the device's bytes in use,
peak bytes and allocator limit just before the gradient and after it (peak is
since process start, so the gradient's own share is peak-after minus
in-use-before); compile time and first-call run time; loss, gradient norm and
finiteness.  Only the stacked inputs stay on the device during the gradient (a
damping rate that is the same every step is held once).  ``--parity``, run
before the gradient, compares the loop's forward final state with an eager
chain of ``model.step`` calls (production path).  Step 1 runs eagerly, so
``--steps N`` differentiates N - 1 steps.  A RESOURCE_EXHAUSTED error is
recorded as ``"oom": true``, not raised.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean",
                      REPO_ROOT / "packages/ml")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from legoesm.training.checkpoint_schedule import (  # noqa: E402
    VALID_SCHEDULES,
    checkpointed_loop,
)


def stack_inputs(inputs):
    """Stack per-step ``step`` kwargs; each damping target must carry kt = i + 2.

    ``inputs`` are the kwargs of steps 2..N (step 1 runs eagerly).  The stamp is
    checked on the host, then cleared so the stacked pytree holds arrays only;
    the loop body re-stamps it with the traced counter (as ``integrate_scan``).
    A damping rate identical on every step is returned once, not stacked.
    Returns ``(stacked, gamma)``; ``gamma`` is None when the rate varies.
    """
    clean = []
    for i, kw in enumerate(inputs):
        kt = kw["sponge"].nemo_kt
        if kt != i + 2:
            raise ValueError(f"damping target for loop step {i} has kt={kt!r}, expected {i + 2}")
        clean.append(dict(kw, sponge=kw["sponge"]._replace(nemo_kt=None)))
    g0 = np.asarray(clean[0]["sponge"].gamma)
    gamma = None
    if all(np.array_equal(g0, np.asarray(kw["sponge"].gamma)) for kw in clean):
        gamma = clean[0]["sponge"].gamma
        clean = [dict(kw, sponge=kw["sponge"]._replace(gamma=None)) for kw in clean]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *clean), gamma


def make_rollout(model, dt, n_loop, schedule, checkpoints=None):
    """``(dT, stacked, gamma, state1) -> state`` after ``n_loop`` steps from
    ``state1`` with T += dT (every array an operand, none a closure constant)."""

    def rollout(dT, stacked, gamma, state1):
        st = state1._replace(T=state1.T.replace(data=state1.T.data + dT))

        def body(carry):
            i, s = carry
            kw = jax.tree.map(lambda a: a[i], stacked)
            kt = i + 2  # one traced object for both: step() checks identity
            sponge = kw["sponge"]._replace(nemo_kt=kt)
            kw["sponge"] = sponge if gamma is None else sponge._replace(gamma=gamma)
            return i + 1, model.step(s, dt, nemo_kt=kt, **kw)

        _, final = checkpointed_loop(body, (jnp.asarray(0, jnp.int32), st), n_loop,
                                     schedule=schedule, checkpoints=checkpoints)
        return final

    return rollout


def volume_mean_t2(model, z_coord):
    area = jnp.asarray(model.grid.area)
    wet = jnp.asarray(z_coord.is_active, dtype=jnp.float64)
    w = area[..., None] * jnp.asarray(z_coord.h_partial) * wet
    return lambda state: jnp.sum(w * state.T.data ** 2) / jnp.sum(w)


def _mem():
    stats = jax.devices()[0].memory_stats() or {}
    return {k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}


def _is_oom(error: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(error) or "Out of memory" in str(error)


def record_forced_inputs(steps: int, card_mode: str = "normal"):
    """K0 card (``normal`` mode, or the certified ``exact`` card), its ocean model
    and initial state, and the per-step ocean ``step`` kwargs of a forward
    composed-K0 run over ``steps`` steps."""
    from legoesm.ocean import bulk_flux_omip as bulk
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round343_speed_benchmark as r343,
    )

    base = bulk.build_nemo_orca2_k0_ocean_card(r343.DEFAULT_DECK_ROOT)
    if card_mode == "exact":
        card = base
    elif card_mode == "normal":
        card = r343.build_normal_mode_orca2_k0_card(base, variant="cat_a_plus_b")
    else:
        raise ValueError(f"unknown card mode {card_mode!r}; expected 'normal' or 'exact'")
    controls = r343.precompute_step_controls(
        card, r343.DEFAULT_DECK_ROOT, r343.DEFAULT_CORE_MANIFEST,
        r343.DEFAULT_ICE_COVER_PATH, steps)
    r = card.ocean.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config, iwm_forcing=r.iwm_forcing)
    state0 = state = r.initial_state
    carry_s0 = carry_s2 = prior = None
    inputs = []
    for kt in range(1, steps + 1):
        ctrl = controls[kt - 1]
        s0, s2 = ctrl["s0_base"], ctrl["s2_base"]
        if carry_s0 is not None:  # round 343's carry hand-off, verbatim
            s0 = s0._replace(prior=carry_s0.prior,
                             shortwave_fraction_top=carry_s0.shortwave_fraction_top)
            s2 = s2._replace(
                prior_runoff_mass=carry_s2.prior_runoff_mass,
                prior_runoff_temperature_content=carry_s2.prior_runoff_temperature_content,
                prior_runoff_salinity_content=carry_s2.prior_runoff_salinity_content,
                prior_restoring_volume_flux=carry_s2.prior_restoring_volume_flux,
                prior_restoring_heat_flux=carry_s2.prior_restoring_heat_flux,
                prior_restoring_coefficient=carry_s2.prior_restoring_coefficient)
        res = bulk.nemo_orca2_k0_ocean_step(
            card, state, model=model, nemo_kt=kt, s0f=ctrl["s0f"], ice_if=ctrl["ice_if"],
            s0=s0, s2=s2, chl=ctrl["chl"],
            geothermal_heat_flux_mwm2=ctrl["geothermal_heat_flux_mwm2"],
            prior_surface=prior, sponge=ctrl["sponge"])
        inputs.append({"freshwater": res.surface.freshwater_forcing,
                       "surface_forcing": res.surface.surface_forcing,
                       "sponge": ctrl["sponge"]})
        state = res.ocean.state_after if hasattr(res.ocean, "state_after") else res.ocean
        carry_s0, carry_s2, prior = res.next_s0, res.next_s2, res.surface
    return card, model, state0, inputs


def _state_of(result):
    return result.state_after if hasattr(result, "state_after") else result


def measure(steps, schedule, checkpoints, parity, save_grad, card_mode="normal"):
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    if not jax.config.jax_enable_x64:
        raise SystemExit("needs JAX_ENABLE_X64=1")
    if steps < 2:
        raise SystemExit("--steps must be >= 2 (step 1 runs outside the loop)")
    # the exact card's scalar-libm path is CPU-only (transcendentals.py refuses others)
    transcendentals = "libm" if card_mode == "exact" else "native"
    if card_mode == "exact" and jax.default_backend() != "cpu":
        raise SystemExit("--card exact runs on CPU only")
    set_policy(PrecisionPolicy.fp64(transcendentals=transcendentals))
    if get_policy().transcendentals != transcendentals:
        raise SystemExit(f"{transcendentals} transcendental policy is not active")

    out = {"steps": steps, "differentiated_steps": steps - 1, "schedule": schedule,
           "card": card_mode, "transcendentals": transcendentals,
           "checkpoints": checkpoints, "backend": jax.default_backend(),
           "device": str(jax.devices()[0]),
           "env": {k: os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE",
                                                  "XLA_PYTHON_CLIENT_MEM_FRACTION",
                                                  "XLA_PYTHON_CLIENT_ALLOCATOR")},
           "legoesm_ocean_file": __import__("legoesm.ocean").ocean.__file__}
    t0 = time.perf_counter()
    card, model, state0, inputs = record_forced_inputs(steps, card_mode)
    dt = card.ocean.dt_s
    state1 = _state_of(model.step(state0, dt, nemo_kt=1, **inputs[0]))
    state1 = model.seed_scan_carry(state1, dt, nemo_kt=2, **inputs[1])
    del state0
    stacked, gamma = stack_inputs(inputs[1:])
    out["setup_s"] = time.perf_counter() - t0
    out["dt_s"] = dt
    out["stacked_input_bytes"] = int(sum(a.nbytes for a in jax.tree.leaves((stacked, gamma))))
    out["damping_rate_shared"] = gamma is not None
    out["state_bytes"] = int(sum(np.asarray(a).nbytes for a in jax.tree.leaves(state1)))

    rollout = make_rollout(model, dt, steps - 1, schedule, checkpoints)
    dT0 = jnp.zeros_like(state1.T.data)
    if parity:  # before the gradient, so an OOM there cannot lose it
        loop_final = jax.jit(rollout)(dT0, stacked, gamma, state1)
        st = state1
        for kt in range(2, steps + 1):
            st = _state_of(model.step(st, dt, nemo_kt=kt, **inputs[kt - 1]))
        out["parity_max_abs"] = {
            name: float(np.max(np.abs(np.asarray(getattr(loop_final, name).data)
                                      - np.asarray(getattr(st, name).data))))
            for name in ("T", "S", "eta", "u", "v")}
        del loop_final, st
    del inputs  # the gradient holds only the stacked copy
    gc.collect()
    out["mem_before_grad"] = _mem()

    loss_fn = volume_mean_t2(model, card.ocean.recipe.z_coord)
    vg = jax.jit(jax.value_and_grad(lambda dT, *ops: loss_fn(rollout(dT, *ops))))
    try:
        t0 = time.perf_counter()
        compiled = vg.lower(dT0, stacked, gamma, state1).compile()
        out["compile_s"] = time.perf_counter() - t0
        mem = compiled.memory_analysis()
        if mem is not None:
            out["xla_temp_bytes"] = int(mem.temp_size_in_bytes)
            out["xla_argument_bytes"] = int(mem.argument_size_in_bytes)
            out["xla_output_bytes"] = int(mem.output_size_in_bytes)
        t0 = time.perf_counter()
        loss, grad = jax.block_until_ready(compiled(dT0, stacked, gamma, state1))
        out["grad_run_s_first_call"] = time.perf_counter() - t0
    except Exception as error:  # noqa: BLE001 - recorded, see docstring
        if not _is_oom(error):
            raise
        out.update(oom=True, error=str(error)[:2000], mem_at_oom=_mem())
        return out
    g = np.asarray(grad)
    out.update(oom=False, loss=float(loss), grad_l2=float(np.sqrt(np.sum(g * g))),
               grad_max_abs=float(np.max(np.abs(g))), grad_nonfinite=int((~np.isfinite(g)).sum()),
               mem_after_grad=_mem())
    if save_grad is not None:
        np.save(save_grad, g)
    return out


def locate_nan(card_mode):
    """Every backward operation of ONE in-loop step (kt = 2) whose cotangent turns
    non-finite, via round 315's ``walk_pullback`` (fused: non-T cotangents enter
    as symbolic zeros, as in ``jax.grad`` of the T loss); raw and sanitized."""
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round315_nan_locate import (  # noqa: E501
        walk_pullback,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm" if card_mode == "exact" else "native"))
    card, model, state0, inputs = record_forced_inputs(3, card_mode)
    dt = card.ocean.dt_s
    state1 = _state_of(model.step(state0, dt, nemo_kt=1, **inputs[0]))
    loss_fn = volume_mean_t2(model, card.ocean.recipe.z_coord)
    st2, vjp_fn = jax.vjp(lambda st: _state_of(model.step(st, dt, nemo_kt=2, **inputs[1])),
                          state1)
    ct = jax.vjp(loss_fn, st2)[1](jnp.float64(1.0))[0]
    paths = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_leaves_with_path(st2)]
    zeros = tuple(q for q in paths if not (q == ".T" or q.startswith(".T[")))
    jit_dT = np.asarray(vjp_fn(ct)[0].T.data)
    out = {"card": card_mode, "step_kt": 2,
           "jit_dT_nonfinite": int((~np.isfinite(jit_dT)).sum()), "runs": {}}
    for sanitize in (False, True):
        g, events = walk_pullback(vjp_fn, ct, zeros, sanitize)
        dT = np.asarray(g[0].T.data)
        out["runs"]["sanitized" if sanitize else "raw"] = {
            "dT_nonfinite": int((~np.isfinite(dT)).sum()), "events": events}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--locate-nan", action="store_true",
                        help="instead of costing: list the one-step backward NaN sources")
    parser.add_argument("--steps", type=int, help="total steps N (dt = 3 h)")
    parser.add_argument("--schedule", choices=VALID_SCHEDULES)
    parser.add_argument("--checkpoints", type=int, default=None,
                        help="binomial budget K (default ceil(sqrt(N-1)))")
    parser.add_argument("--parity", action="store_true",
                        help="also compare the loop's forward state with eager model.step")
    parser.add_argument("--save-grad", type=Path, default=None)
    parser.add_argument("--card", choices=("normal", "exact"), default="normal",
                        help="normal-mode K0 card (native maths) or the certified exact "
                             "card (scalar libm, CPU only)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp(repo=REPO_ROOT)
    if args.locate_nan:
        result = locate_nan(args.card)
        result.update(provenance=stamp, argv=sys.argv)
        args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")
        print(f"STATUS LOCATED card={args.card} "
              f"sources={len(result['runs']['sanitized']['events'])}")
        return 0
    if args.steps is None or args.schedule is None:
        parser.error("--steps and --schedule are required unless --locate-nan")
    result = measure(args.steps, args.schedule, args.checkpoints, args.parity, args.save_grad,
                     args.card)
    result["provenance"] = stamp
    result["argv"] = sys.argv
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print(f"STATUS {'OOM' if result['oom'] else 'MEASURED'} steps={args.steps} "
          f"schedule={args.schedule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
