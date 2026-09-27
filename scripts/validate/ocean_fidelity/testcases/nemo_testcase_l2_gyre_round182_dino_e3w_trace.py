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
    parser.add_argument(
        "--trace-step-boundaries", action="store_true",
        help="print non-finite counts at the bridged-entry, post-surface-"
             "forcing, and returned-step boundaries")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    args = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
    if not args:
        parser.error("pass the kamm_twin_90d.py arguments after --")

    from legoesm.ocean import eos

    def trace_state(label, state, model=None):
        fields = {
            "eta": state.eta.data,
            "T": state.T.data,
            "S": state.S.data,
            "u": state.u.data,
            "v": state.v.data,
        }
        jax.debug.print(
            label + " " + " ".join(
                f"{name}_nonfinite={{{name}}}" for name in fields),
            ordered=True,
            **{
                name: jnp.sum(~jnp.isfinite(value))
                for name, value in fields.items()
            },
        )
        if model is not None:
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                compute_face_masks_3d,
            )
            wet_t = jnp.asarray(model.z_coord.is_active, dtype=bool)
            wet_u, wet_v = compute_face_masks_3d(wet_t, model.grid)
            wet_eta = jnp.asarray(state.land_mask.data, dtype=bool)
            jax.debug.print(
                label + "_ACTIVE "
                "eta_nonfinite={eta} T_nonfinite={T} S_nonfinite={S} "
                "u_nonfinite={u} v_nonfinite={v} "
                "eta_cells={eta_n} T_cells={T_n} u_cells={u_n} v_cells={v_n}",
                eta=jnp.sum(wet_eta & ~jnp.isfinite(state.eta.data)),
                T=jnp.sum(wet_t & ~jnp.isfinite(state.T.data)),
                S=jnp.sum(wet_t & ~jnp.isfinite(state.S.data)),
                u=jnp.sum(wet_u & ~jnp.isfinite(state.u.data)),
                v=jnp.sum(wet_v & ~jnp.isfinite(state.v.data)),
                eta_n=jnp.sum(wet_eta), T_n=jnp.sum(wet_t),
                u_n=jnp.sum(wet_u), v_n=jnp.sum(wet_v),
                ordered=True,
            )

    if ns.trace_step_boundaries:
        from legoesm.ocean.experiments import dino

        original_surface = dino.apply_dino_lat_lon_surface_forcing

        def traced_surface(state, *surface_args, **surface_kwargs):
            trace_state("TRACE_STEP_ENTRY", state)
            result = original_surface(state, *surface_args, **surface_kwargs)
            state_out = result[0] if isinstance(result, tuple) else result
            trace_state("TRACE_AFTER_SURFACE", state_out)
            return result

        dino.apply_dino_lat_lon_surface_forcing = traced_surface

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

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    original_step = LatLonCGridOceanModel.step

    def traced_step(model, state, *step_args, **step_kwargs):
        if ns.trace_step_boundaries:
            trace_state("TRACE_MODEL_ENTRY", state, model)
        result = original_step(model, state, *step_args, **step_kwargs)
        fields = {
            "eta": result.eta.data,
            "T": result.T.data,
            "S": result.S.data,
            "u": result.u.data,
            "v": result.v.data,
        }
        jax.debug.print(
            "TRACE_STEP_OUT " + " ".join(
                f"{name}_nonfinite={{{name}}}" for name in fields),
            ordered=True,
            **{
                name: jnp.sum(~jnp.isfinite(value))
                for name, value in fields.items()
            },
        )
        if ns.trace_step_boundaries:
            trace_state("TRACE_STEP_OUT_ACTIVE_CHECK", result, model)
        return result

    LatLonCGridOceanModel.step = traced_step
    sys.argv = ["kamm_twin_90d.py", *args]
    try:
        runpy.run_path(
            "scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py",
            run_name="__main__",
        )
    except Exception as exc:
        if (ns.plant_call > 0
                and "raw-mesh e3w_int must contain only finite values > 0"
                in str(exc)):
            print("STATUS PLANT-FIRED")
        elif ns.plant_call > 0:
            print("STATUS PLANT-MISSED")
        raise
    if ns.plant_call > 0:
        print("STATUS PLANT-MISSED")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
