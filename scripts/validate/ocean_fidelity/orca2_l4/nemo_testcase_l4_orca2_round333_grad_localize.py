#!/usr/bin/env python3
"""Localize the stored-LDF-divisor reverse NaN on rung-10's real grid."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
)


class GateError(RuntimeError):
    """The real-grid primitive localization is incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _finite(value) -> bool:
    return bool(np.all(np.isfinite(np.asarray(value))))


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_safe_stored_ldf_divisor,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "gradient localization requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64) and jax.default_backend() == "cpu",
            "gradient localization requires CPU fp64")

    _, shipped = rung10.card_fields(deck_root)
    card = r329.selected_card(shipped, arm="recordice", rung="rung10")
    z_coord = card.recipe.z_coord
    active = np.asarray(z_coord.is_active, dtype=bool)
    stored_reference = np.asarray(z_coord.h_partial, dtype=np.float64)
    rebuilt_reference = np.asarray(z_coord.nemo_e3t_0, dtype=np.float64)
    require(active.shape == stored_reference.shape == rebuilt_reference.shape,
            "rung-10 thickness shapes moved")

    inactive = ~active
    surface_wet = np.broadcast_to(active[..., :1], active.shape)
    coastal_subbottom = inactive & surface_wet
    fully_dry = inactive & ~surface_wet
    require(bool(np.any(coastal_subbottom)) and bool(np.any(fully_dry)),
            "rung-10 dry-cell classes are empty")
    require(not bool(np.any((stored_reference == 0.0) & active)),
            "stored reference has a zero active thickness")
    require(not bool(np.any((rebuilt_reference == 0.0) & inactive)),
            "rebuilt reference no longer supplies an inert dry divisor")

    coastal_index = tuple(int(i) for i in np.argwhere(coastal_subbottom)[0])
    land_index = tuple(int(i) for i in np.argwhere(fully_dry)[0])

    # This is the exact primitive tail of gm_redi's rectangular evaluation:
    # divide first, then multiply by tmask.  The LDF numerator is zero where
    # the operator does not execute.  A later selection can discard the NaN
    # primal, but a zero cotangent entering div's transpose still computes 0/0.
    numerator = jnp.zeros(active.shape, dtype=jnp.float64)
    active_jax = jnp.asarray(active, dtype=jnp.float64)
    stored_jax = jnp.asarray(stored_reference)
    rebuilt_jax = jnp.asarray(rebuilt_reference)

    def rectangular_loss(divisor):
        quotient = numerator / divisor
        masked = quotient * active_jax
        return jnp.sum(jnp.where(active_jax != 0.0, masked, 0.0))

    checked_grad = checkify.checkify(
        jax.grad(rectangular_loss), errors=checkify.float_checks)
    unsafe_error, unsafe_gradient = checked_grad(stored_jax)
    rebuilt_error, rebuilt_gradient = checked_grad(rebuilt_jax)

    def safe_loss(divisor):
        safe = _nemo_safe_stored_ldf_divisor(divisor, active_jax)
        return rectangular_loss(safe)

    safe_error, safe_gradient = checkify.checkify(
        jax.grad(safe_loss), errors=checkify.float_checks)(stored_jax)

    # Per-primitive reverse probe at one actual coastal sub-bottom cell.  The
    # incoming cotangent is exactly zero, as supplied by the inactive mask.
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    _quotient, divide_pullback = jax.vjp(lambda num, div: num / div, zero, zero)
    divide_num_bar, divide_div_bar = divide_pullback(zero)
    _masked, multiply_pullback = jax.vjp(lambda value, mask: value * mask,
                                         jnp.asarray(np.nan), zero)
    multiply_value_bar, multiply_mask_bar = multiply_pullback(zero)

    debug_nans_message = None
    try:
        with jax.debug_nans(True):
            jax.grad(lambda div: jnp.where(
                zero != 0.0, (zero / div) * zero, zero))(zero)
    except (FloatingPointError, RuntimeError) as error:
        debug_nans_message = str(error).splitlines()[0]

    quotient = np.asarray(numerator / stored_jax)
    result = {
        "format": "nemo-testcase-l4-orca2-round333-grad-localize-v1",
        "status": "PASS_R333_GRAD_LOCALIZE",
        "execution": "rung10-N1-real-grid-cpu-fp64",
        "worktree": stamp,
        "shape": list(active.shape),
        "cell_census": {
            "total": int(active.size),
            "active": int(np.count_nonzero(active)),
            "inactive": int(np.count_nonzero(inactive)),
            "coastal_subbottom": int(np.count_nonzero(coastal_subbottom)),
            "fully_dry": int(np.count_nonzero(fully_dry)),
            "stored_zero_active": int(np.count_nonzero(
                (stored_reference == 0.0) & active)),
            "stored_zero_inactive": int(np.count_nonzero(
                (stored_reference == 0.0) & inactive)),
            "rebuilt_zero_inactive": int(np.count_nonzero(
                (rebuilt_reference == 0.0) & inactive)),
            "unsafe_quotient_nonfinite": int(np.count_nonzero(
                ~np.isfinite(quotient))),
        },
        "examples": {
            "coastal_subbottom": {
                "index": list(coastal_index),
                "stored": float(stored_reference[coastal_index]),
                "rebuilt": float(rebuilt_reference[coastal_index]),
            },
            "fully_dry": {
                "index": list(land_index),
                "stored": float(stored_reference[land_index]),
                "rebuilt": float(rebuilt_reference[land_index]),
            },
        },
        "checkify": {
            "unsafe_error": unsafe_error.get(),
            "unsafe_gradient_finite": _finite(unsafe_gradient),
            "rebuilt_error": rebuilt_error.get(),
            "rebuilt_gradient_finite": _finite(rebuilt_gradient),
            "safe_error": safe_error.get(),
            "safe_gradient_finite": _finite(safe_gradient),
        },
        "per_primitive_zero_cotangent": {
            "divide_numerator_bar": float(divide_num_bar),
            "divide_divisor_bar": float(divide_div_bar),
            "divide_bars_finite": _finite((divide_num_bar, divide_div_bar)),
            "multiply_value_bar": float(multiply_value_bar),
            "multiply_mask_bar": float(multiply_mask_bar),
            "multiply_bars_finite": _finite(
                (multiply_value_bar, multiply_mask_bar)),
        },
        "debug_nans": debug_nans_message,
    }
    require(result["cell_census"]["unsafe_quotient_nonfinite"]
            == result["cell_census"]["inactive"],
            "unsafe nonfinite cells are not exactly the inactive cells")
    require(unsafe_error.get() is not None and not _finite(unsafe_gradient),
            "unsafe stored path did not reproduce the reverse failure")
    require(rebuilt_error.get() is None and _finite(rebuilt_gradient),
            "rebuilt path is not finite")
    require(safe_error.get() is None and _finite(safe_gradient),
            "safe stored path is not finite")
    require(not result["per_primitive_zero_cotangent"]["divide_bars_finite"],
            "divide transpose did not localize the first bad cotangent")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.expect_commit)
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
