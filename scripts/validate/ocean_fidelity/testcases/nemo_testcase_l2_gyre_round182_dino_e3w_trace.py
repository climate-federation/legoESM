#!/usr/bin/env python
"""Trace every production ``bn2`` raw-e3w operand in the DINO day-180 twin.

This is a diagnostic wrapper around the committed DINO twin, not a second
harness.  It labels each Python call site while JAX traces the production step
and prints that call's runtime raw-mesh divisor minimum and invalid-cell count.
The wrapped kernel, configuration, state bridge, and step are unchanged.

Run with the wrapped harness's own positional/optional arguments after
``--``; ``--script`` selects the harness (default: the developed-state 90-day
twin; the from-rest year screen is the other caller of the same step).  ``--plant-call N`` sets the first divisor cell of traced call N to
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
    parser.add_argument(
        "--trace-baro-entry", action="store_true",
        help="trace the first NEMO ssh-average entry-inverse operands")
    parser.add_argument(
        "--plant-closed-v-boundary", action="store_true",
        help="after the repaired ssh-average entry inverse, restore one "
             "non-finite closed V-face boundary value; the production run "
             "must fail and print STATUS PLANT-FIRED")
    parser.add_argument(
        "--script",
        default="scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py",
        help="the committed DINO harness to wrap; the default is the "
             "developed-state 90-day twin, and the from-rest year screen "
             "(dino_year_screen_fullframe.py) is the other one that runs the "
             "same production step")
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
            wet_u = jnp.asarray(wet_u, dtype=bool)
            wet_v = jnp.asarray(wet_v, dtype=bool)
            wet_eta = jnp.asarray(state.land_mask.data, dtype=bool)
            jax.debug.print(
                label + "_ACTIVE "
                "eta_nonfinite={eta} T_nonfinite={T} S_nonfinite={S} "
                "u_nonfinite={u} v_nonfinite={v} "
                "eta_cells={eta_n} T_cells={T_n} u_cells={u_n} v_cells={v_n} "
                "eta_maxabs={eta_max:.17e} T_maxabs={T_max:.17e} "
                "S_maxabs={S_max:.17e} u_maxabs={u_max:.17e} "
                "v_maxabs={v_max:.17e}",
                eta=jnp.sum(wet_eta & ~jnp.isfinite(state.eta.data)),
                T=jnp.sum(wet_t & ~jnp.isfinite(state.T.data)),
                S=jnp.sum(wet_t & ~jnp.isfinite(state.S.data)),
                u=jnp.sum(wet_u & ~jnp.isfinite(state.u.data)),
                v=jnp.sum(wet_v & ~jnp.isfinite(state.v.data)),
                eta_n=jnp.sum(wet_eta), T_n=jnp.sum(wet_t),
                u_n=jnp.sum(wet_u), v_n=jnp.sum(wet_v),
                eta_max=jnp.nanmax(jnp.where(wet_eta, jnp.abs(state.eta.data), 0.0)),
                T_max=jnp.nanmax(jnp.where(wet_t, jnp.abs(state.T.data), 0.0)),
                S_max=jnp.nanmax(jnp.where(wet_t, jnp.abs(state.S.data), 0.0)),
                u_max=jnp.nanmax(jnp.where(wet_u, jnp.abs(state.u.data), 0.0)),
                v_max=jnp.nanmax(jnp.where(wet_v, jnp.abs(state.v.data), 0.0)),
                ordered=True,
            )

    if ns.trace_step_boundaries:
        from legoesm.ocean.experiments import dino

        original_surface = dino.apply_dino_lat_lon_surface_forcing

        def traced_surface(state, *surface_args, **surface_kwargs):
            trace_state("TRACE_STEP_ENTRY", state)
            result = original_surface(state, *surface_args, **surface_kwargs)
            state_out = result if hasattr(result, "eta") else result[0]
            trace_state("TRACE_AFTER_SURFACE", state_out)
            return result

        dino.apply_dino_lat_lon_surface_forcing = traced_surface

    if ns.trace_baro_entry:
        from legoesm.ocean.dynamics import barotropic_latlon_cgrid as baro

        original_ssh_avg = baro._nemo_ssh_avg_apply
        ssh_avg_calls = 0

        def traced_ssh_avg(eta_dyn, u_mask, v_mask, grid, area, prep, **kwargs):
            nonlocal ssh_avg_calls
            ssh_avg_calls += 1
            result = original_ssh_avg(
                eta_dyn, u_mask, v_mask, grid, area, prep, **kwargs)
            if kwargs.get("return_entry_inverse", False):
                if ns.plant_closed_v_boundary:
                    planted_v = result[3].at[0, 0].set(jnp.nan)
                    result = (*result[:3], planted_v, *result[4:])
                _, _, _, r1_e1e2v, _ = prep
                area_pad = baro.pad_ns_zero(area)
                eta_pad = baro.pad_ns_zero(eta_dyn)
                ssh_v_s = baro.nemo_source_round(area_pad[:-1] * eta_pad[:-1])
                ssh_v_n = baro.nemo_source_round(area_pad[1:] * eta_pad[1:])
                ssh_v_sum = baro.nemo_source_round(ssh_v_s + ssh_v_n)
                r3_v_half_sum = baro.nemo_source_round(0.5 * ssh_v_sum)
                jax.debug.print(
                    "TRACE_BARO_ENTRY call={call} "
                    "eta_nonfinite={eta_bad} inv_area_nonfinite={metric_bad} "
                    "zero_half_sum={zero_sum} zero_times_nonfinite={overlap} "
                    "r1_u_nonfinite={u_bad} r1_v_nonfinite={v_bad}",
                    call=ssh_avg_calls,
                    eta_bad=jnp.sum(~jnp.isfinite(eta_dyn)),
                    metric_bad=jnp.sum(~jnp.isfinite(r1_e1e2v)),
                    zero_sum=jnp.sum(r3_v_half_sum == 0.0),
                    overlap=jnp.sum(
                        (r3_v_half_sum == 0.0) & ~jnp.isfinite(r1_e1e2v)),
                    u_bad=jnp.sum(~jnp.isfinite(result[2])),
                    v_bad=jnp.sum(~jnp.isfinite(result[3])),
                    ordered=True,
                )
            return result

        baro._nemo_ssh_avg_apply = traced_ssh_avg

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
    sys.argv = [ns.script.rsplit("/", 1)[-1], *args]
    try:
        runpy.run_path(ns.script, run_name="__main__")
    except Exception as exc:
        if ns.plant_closed_v_boundary:
            print("STATUS PLANT-FIRED")
            raise
        if (ns.plant_call > 0
                and "raw-mesh e3w_int must contain only finite values > 0"
                in str(exc)):
            print("STATUS PLANT-FIRED")
        elif ns.plant_call > 0:
            print("STATUS PLANT-MISSED")
        raise
    if ns.plant_closed_v_boundary:
        print("STATUS PLANT-MISSED")
        raise SystemExit(2)
    if ns.plant_call > 0:
        print("STATUS PLANT-MISSED")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
