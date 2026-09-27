#!/usr/bin/env python
"""Trace every production ``bn2`` raw-e3w operand in the DINO day-180 twin.

This is a diagnostic wrapper around the committed DINO twin, not a second
harness.  It labels each Python call site while JAX traces the production step
and prints that call's runtime raw-mesh divisor minimum and invalid-cell count.
The wrapped kernel, configuration, state bridge, and step are unchanged.

Run with the normal ``kamm_twin_90d.py`` positional/optional arguments after
``--``.  ``--plant-call N`` sets the first divisor cell of traced call N to
zero and must make the existing fail-closed raw-mesh guard fire.
"""
from __future__ import annotations

import argparse
import inspect
import runpy
import sys

import jax
import jax.numpy as jnp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-call", type=int, default=-1)
    parser.add_argument("--sanitize-invalid", action="store_true",
                        help="diagnostic only: report then replace invalid "
                             "divisors, allowing later call sites to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    args = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
    if not args:
        parser.error("pass the kamm_twin_90d.py arguments after --")

    from legoesm.ocean import eos

    original = eos.compute_buoyancy_frequency_nemo_bn2
    call_count = 0

    def traced(*call_args, **call_kwargs):
        nonlocal call_count
        call_count += 1
        call_id = call_count
        site = "unknown"
        chain = []
        for frame in inspect.stack()[1:]:
            if ("/legoesm/ocean/" in frame.filename
                    and not frame.filename.endswith("/eos.py")):
                chain.append(
                    f"{frame.filename.rsplit('/', 1)[-1]}:"
                    f"{frame.function}:{frame.lineno}")
                if site == "unknown":
                    site = chain[-1]
                if len(chain) == 5:
                    break
        e3w = call_kwargs.get("e3w_int")
        print(f"TRACE_SITE call={call_id} site={site} "
              f"chain={' > '.join(chain)} has_e3w={e3w is not None}")
        if e3w is not None:
            e3w = jnp.asarray(e3w)
            if call_id == ns.plant_call:
                e3w = e3w.at[(0,) * e3w.ndim].set(0.0)
                call_kwargs["e3w_int"] = e3w
                print(f"TRACE_PLANT call={call_id} cell=all-zero-index value=0")
            nonfinite = jnp.sum(~jnp.isfinite(e3w))
            zero = jnp.sum(e3w == 0.0)
            negative = jnp.sum(e3w < 0.0)
            invalid = nonfinite + zero + negative
            jax.debug.print(
                "TRACE_VALUE call={call} min={minimum:.17e} "
                "nonfinite={nonfinite} zero={zero} negative={negative} "
                "invalid={invalid}",
                call=call_id, minimum=jnp.nanmin(e3w),
                nonfinite=nonfinite, zero=zero, negative=negative,
                invalid=invalid,
                ordered=True,
            )
            if ns.sanitize_invalid:
                call_kwargs["e3w_int"] = jnp.where(
                    jnp.isfinite(e3w) & (e3w > 0.0), e3w,
                    jnp.ones_like(e3w))
        return original(*call_args, **call_kwargs)

    eos.compute_buoyancy_frequency_nemo_bn2 = traced
    sys.argv = ["kamm_twin_90d.py", *args]
    runpy.run_path(
        "scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
