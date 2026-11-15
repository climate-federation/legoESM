#!/usr/bin/env python3
"""How much do near-neutral interfaces drive the ORCA2 gradient? (MEASUREMENT ONLY)

The generic internal-wave-mixing path takes sqrt(max(N2, 0)); its derivative
1/(2 sqrt(N2)) is finite but unbounded as N2 -> 0+.  This probe, on the
normal-mode K0 card with the forcing recorded by
``scripts/bench/orca2_rollout_grad_cost.py``:

1. counts, over the forward rollout's IWM calls, the wet interfaces with
   N2 <= 0 and with 0 < N2 below 1e-12, 1e-10 and 1e-8 s^-2;
2. with ``--floor F``, recomputes d(volume-mean T^2)/d(T at step 1) with the
   sqrt's DERIVATIVE evaluated at max(N2, F) (the forward is untouched), and
   saves it, so runs at several F can be compared with the unfloored gradient.

The model is not modified: the IWM module's ``safe_sqrt_nan`` is swapped in
this process only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean",
           REPO_ROOT / "packages/ml", REPO_ROOT / "scripts/bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BINS = (1e-12, 1e-10, 1e-8)
_COUNTS: list[np.ndarray] = []
_RECORD = {"on": False}


def floored_sqrt(floor: float):
    """sqrt(max(x, 0)) as safe_sqrt_nan, with the derivative taken at max(x, floor)."""
    from legoesm.core.profile_metrics import safe_sqrt_nan

    @jax.custom_jvp
    def f(x):
        return safe_sqrt_nan(x)

    @f.defjvp
    def f_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        slope = jnp.where(x > 0, 0.5 / jnp.sqrt(jnp.maximum(x, floor)), 0.0)
        return f(x), slope * t

    return f


def counting(original):
    def wrapped(forcing, depth_cell, dz_w, H, N2, **kw):
        if _RECORD["on"]:
            wet = dz_w > 0

            def host(n2, w):
                n2, w = np.asarray(n2), np.asarray(w)
                row = [w.sum(), (w & (n2 <= 0)).sum()]
                row += [(w & (n2 > 0) & (n2 < b)).sum() for b in BINS]
                _COUNTS.append(np.array(row, dtype=np.int64))

            jax.debug.callback(host, N2, wet)
        return original(forcing, depth_cell, dz_w, H, N2, **kw)

    return wrapped


def main(argv=None) -> int:
    import orca2_rollout_grad_cost as cost
    import legoesm.ocean.physics.vertical_mixing.internal_wave_mixing as iwm
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--floor", type=float, default=None, help="derivative floor on N2 [s^-2]")
    p.add_argument("--save-grad", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if not jax.config.jax_enable_x64:
        raise SystemExit("needs JAX_ENABLE_X64=1")
    stamp = worktree_stamp(repo=REPO_ROOT)
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    if args.floor is not None:
        iwm.safe_sqrt_nan = floored_sqrt(args.floor)
    iwm.compute_iwm_diffusivity = counting(iwm.compute_iwm_diffusivity)

    card, model, state0, inputs = cost.record_forced_inputs(args.steps)
    dt = card.ocean.dt_s
    state1 = cost._state_of(model.step(state0, dt, nemo_kt=1, **inputs[0]))
    state1 = model.seed_scan_carry(state1, dt, nemo_kt=2, **inputs[1])
    stacked, gamma = cost.stack_inputs(inputs[1:])
    del inputs, state0
    rollout = cost.make_rollout(model, dt, args.steps - 1, "uniform")
    loss_fn = cost.volume_mean_t2(model, card.ocean.recipe.z_coord)
    dT0 = jnp.zeros_like(state1.T.data)

    _RECORD["on"] = True                       # traced into this forward only
    jax.block_until_ready(jax.jit(rollout)(dT0, stacked, gamma, state1))
    _RECORD["on"] = False
    counts = np.sum(_COUNTS, axis=0)

    g = np.asarray(jax.jit(jax.grad(lambda d, *o: loss_fn(rollout(d, *o))))(
        dT0, stacked, gamma, state1))
    np.save(args.save_grad, g)
    out = {"steps": args.steps, "floor": args.floor, "iwm_calls_counted": len(_COUNTS),
           "wet_interface_samples": int(counts[0]), "n2_nonpositive": int(counts[1]),
           **{f"n2_pos_below_{b:g}": int(c) for b, c in zip(BINS, counts[2:])},
           "grad_nonfinite": int((~np.isfinite(g)).sum()),
           "grad_l2": float(np.sqrt(np.sum(g * g))), "provenance": stamp}
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
