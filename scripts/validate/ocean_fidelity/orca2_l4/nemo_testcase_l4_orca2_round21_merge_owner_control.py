#!/usr/bin/env python3
"""Run one-variable ORCA2 ladder controls for the two merge owners.

``fold`` reuses the committed pre-merge fold-descriptor control.  ``e3f``
restores the bridge-carried NEMO operands consumed by the pre-``ddb70da1a4``
live vorticity-thickness producer.  ``both`` installs both substitutions.
Every remaining argument is forwarded unchanged to the round-1 ladder gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
from jax import lax

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
for package in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

import legoesm.ocean.vertical as vertical  # noqa: E402


def bridge_carried_vorticity_e3f(
    eta,
    z_coord,
    dtype,
    nn_e3f_typ=0,
    *,
    grid=None,
    e3t_0=None,
    tmask=None,
    plant_ulp=False,
):
    """Reproduce the pre-rebuild producer from NEMO's carried operands.

    ``e3t_0`` and ``tmask`` are accepted because the current production call
    supplies them, but deliberately ignored: this control changes exactly the
    operand source back to the admitted NEMO bridge record.  The arithmetic is
    the implementation immediately before commit ``ddb70da1a4``.
    """
    del e3t_0, tmask
    if nn_e3f_typ not in (0, 1):
        raise ValueError("nn_e3f_typ must be 0 or 1")
    raw = getattr(z_coord, "nemo_een_barotropic", None)
    e3t0 = getattr(z_coord, "nemo_e3t_0", None)
    active = getattr(z_coord, "is_active", None)
    if raw is None or e3t0 is None or active is None:
        raise ValueError(
            "bridge control requires carried e3t_0, masks, and EEN operands")

    b = lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    e3t0 = jnp.asarray(e3t0, dtype=dtype)
    active = jnp.asarray(active, dtype=dtype)

    def east(value):
        return jnp.roll(value, -1, axis=1)

    def north(value):
        return jnp.concatenate([value[1:], jnp.zeros_like(value[:1])], axis=0)

    masked = b(e3t0 * active)
    masked_n = north(masked)
    ref_sum = b(b(masked + east(masked)) + b(masked_n + east(masked_n)))
    active_n = north(active)
    wet_sum = b(b(active + east(active)) + b(active_n + east(active_n)))
    divisor = (jnp.asarray(4.0, dtype=dtype) if nn_e3f_typ == 0
               else jnp.maximum(wet_sum, one))
    e3f0vor = b(ref_sum / divisor)
    e3f0vor = jnp.where(
        e3f0vor == 0.0, jnp.asarray(raw.e3f_0, dtype=dtype), e3f0vor)
    e3f0vor = vertical.nemo_t_fold_f_owned(e3f0vor, grid)

    area_eta = b(
        b(jnp.asarray(raw.e1t, dtype=dtype)
          * jnp.asarray(raw.e2t, dtype=dtype)) * eta)
    area_eta_n = north(area_eta)
    quad = b(b(area_eta + east(area_eta))
             + b(area_eta_n + east(area_eta_n)))
    hf0 = jnp.asarray(raw.hf_0, dtype=dtype)
    wet_f = (hf0 > 0.0).astype(dtype)
    r1_hf0 = b(wet_f / b(hf0 + one - wet_f))
    area_f = b(jnp.asarray(raw.e1f, dtype=dtype)
               * jnp.asarray(raw.e2f, dtype=dtype))
    r3f = b(b(quarter * quad) * r1_hf0 / area_f)
    r3f = vertical.nemo_t_fold_f_owned(r3f, grid)
    e3f_native = b(e3f0vor * b(
        one + r3f[..., None] * jnp.asarray(raw.fe3mask, dtype=dtype)))
    with_south = jnp.concatenate([e3f_native[:1], e3f_native], axis=0)
    result = jnp.concatenate([with_south[:, -1:], with_south], axis=1)
    if plant_ulp:
        result = jnp.where(
            result != 0.0,
            jnp.nextafter(result, jnp.asarray(jnp.inf, dtype=dtype)),
            result,
        )
    return result


def install_bridge_control(*, plant_ulp: bool = False) -> None:
    """Install the bridge-carried producer before the ladder imports model code."""
    original = vertical.nemo_qco_live_vorticity_e3f_cgrid

    def controlled(*args, **kwargs):
        return bridge_carried_vorticity_e3f(
            *args, **kwargs, plant_ulp=plant_ulp)

    controlled.__wrapped__ = original
    vertical.nemo_qco_live_vorticity_e3f_cgrid = controlled


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--owner", choices=("fold", "e3f", "both"), required=True)
    parser.add_argument("--plant-vorticity-ulp", action="store_true")
    args, forwarded = parser.parse_known_args()
    if args.plant_vorticity_ulp and args.owner == "fold":
        parser.error("--plant-vorticity-ulp requires owner e3f or both")

    if args.owner in ("fold", "both"):
        from nemo_testcase_l4_orca2_merge_gyre_fold_layout_control import (
            install_parent_layout,
        )
        install_parent_layout()
    if args.owner in ("e3f", "both"):
        install_bridge_control(plant_ulp=args.plant_vorticity_ulp)

    sys.argv = [sys.argv[0], *forwarded]
    import nemo_testcase_l4_orca2_round1_ladder_gate as gate
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
