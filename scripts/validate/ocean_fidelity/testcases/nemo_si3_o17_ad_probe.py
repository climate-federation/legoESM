#!/usr/bin/env python3
"""CPU-fp64 value-and-grad probe for one active mixed-ice O17 deck step."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.dynamics import SI3CGridAEVPForcing, SI3CGridAEVPState, SI3CGridMetrics
from legoesm.ice.si3_finalize import SI3StressInputs
from legoesm.ice.si3_o17 import (
    SI3O17LiveForcing,
    SI3O17StepState,
    SI3O17SurfaceForcing,
    build_orca2_sas_ice_o17_card,
    si3_o17_step,
)
from legoesm.ice.si3_orca2_state import (
    SI3ORCA2AdvectionGrid,
    SI3State,
    si3_lbc_lnk_orca2,
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
)
from legoesm.ice.si3_surface import SI3SurfaceConfig

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    DEFAULT_PPSRC,
    parse_resolved_ice_namelist,
)

DEFAULT_MASTER = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master"
)


def _case(master: Path, shape: tuple[int, int] = (7, 7)):
    """Mixed-regime O17 step entry; ``shape`` (184, 152) is the real ORCA2 rank grid."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    run = master / "record_100step"
    nml = parse_resolved_ice_namelist(
        run / "output.namelist.ice", run / "output.namelist.dyn", DEFAULT_PPSRC
    )
    surface_cfg = SI3SurfaceConfig.from_resolved_namelists(
        run / "output.namelist.dyn", run / "output.namelist.ice"
    )
    card = build_orca2_sas_ice_o17_card(surface=surface_cfg, thermo_namelist=nml)
    # The owned tile exercises three regimes at separated columns: open water,
    # active ice, and a nonzero category column below epsi10 (icethd.F90:121-138).
    (ni, nj), h = shape, 2
    ii = (ni - 2 * h, nj - 2 * h)
    full = (ni, nj)
    zf, of = jnp.zeros(full, jnp.float64), jnp.ones(full, jnp.float64)
    oi = jnp.ones(ii, jnp.float64)
    ix, iy = jnp.meshgrid(
        jnp.arange(ni, dtype=jnp.float64), jnp.arange(nj, dtype=jnp.float64), indexing="ij"
    )
    ocean_u = 0.10 + 0.001 * ix + 0.0002 * iy
    ocean_v = 0.05 + 0.0003 * ix - 0.0007 * iy

    def lbc(value, nature="T", sign=1.0):
        return si3_lbc_lnk_orca2(
            jnp.asarray(value, dtype=jnp.float64),
            nature,
            sign,
            nn_hls=card.state.nn_hls,
            ldfull=True,
        )

    pattern = np.asarray(  # icethd.F90:123,131: ice, open water, nonzero <= epsi10
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0e-12]], dtype=np.float64
    )
    reps = (-(-ii[0] // 3), -(-ii[1] // 3))
    ice_owned = jnp.asarray(np.tile(pattern, reps)[: ii[0], : ii[1]])
    ice_mask = lbc(zf.at[h:-h, h:-h].set(ice_owned))
    category_area = jnp.asarray([0.12, 0.16, 0.18, 0.14, 0.10])
    category_thickness = jnp.asarray([0.25, 0.80, 1.60, 2.80, 4.50])
    a_i = lbc(ice_mask[..., None] * category_area)
    v_i = lbc(a_i * category_thickness)
    v_s = lbc(a_i * 0.12)
    sv_i = lbc(v_i * 5.0)
    a_ip = lbc(a_i * 0.08)
    v_ip = lbc(a_ip * 0.04)
    v_il = lbc(a_ip * 0.008)
    oa_i = lbc(a_i * 30.0)
    t_su = lbc(jnp.where(a_i > 0.0, card.constants.T0 - 5.0, card.constants.T0))
    e_i = lbc(
        jnp.broadcast_to(
            v_i[..., None, :] * (3.0e8 / card.state.nlay_i),
            (*full, card.state.nlay_i, card.state.jpl),
        )
    )
    e_s = lbc(
        jnp.broadcast_to(
            v_s[..., None, :] * (1.1e8 / card.state.nlay_s),
            (*full, card.state.nlay_s, card.state.jpl),
        )
    )
    szv_i = lbc(
        jnp.broadcast_to(
            v_i[..., None, :] * (5.0 / card.state.nlay_i),
            (*full, card.state.nlay_i, card.state.jpl),
        )
    )
    u_ice = lbc((0.04 + 0.0005 * ix) * ice_mask, "U", -1.0)
    v_ice = lbc((-0.02 + 0.0003 * iy) * ice_mask, "V", -1.0)
    eqv = si3_var_glo2eqv(
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        a_i=a_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        kn=2,
        hi_max_top=99.0,
        sss_m=jnp.full(full, 34.0),
        config=card.state,
    )
    t_si = jnp.broadcast_to(
        jnp.where(ice_mask[..., None] > 0.0, card.constants.T0 - 1.0, 0.0),
        (*full, card.state.jpl),
    )
    agg = si3_var_agg(
        v_i=v_i,
        v_s=eqv.v_s,
        sv_i=sv_i,
        a_i=eqv.a_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        t_su=t_su,
        t_si=t_si,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        kn=2,
        config=card.state,
    )
    snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
        vt_i=agg.vt_i,
        vt_s=agg.vt_s,
        vt_ip=agg.vt_ip,
        vt_il=agg.vt_il,
        snwice_mass_prev=zf,
        tmask=of,
        config=card.state,
        is_init=True,
    )
    base = SI3State.zeros(ni, nj, card.state)
    state = base._replace(
        v_i=v_i,
        v_s=eqv.v_s,
        sv_i=sv_i,
        a_i=eqv.a_i,
        t_su=t_su,
        u_ice=u_ice,
        v_ice=v_ice,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        t1_ice=t_su,
        t_si=t_si,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        snwice_fmass=snwice_fmass,
        equivalent=eqv,
        aggregated=agg,
    )
    length = jnp.full(full, 2.0e5)
    area = jnp.full(full, 4.0e10)
    metrics = SI3CGridMetrics(
        length,
        length,
        length,
        length,
        length,
        length,
        length,
        length,
        area,
        area,
        area,
        area,
    )
    aevp_forcing = SI3CGridAEVPForcing(
        concentration_t=agg.at_i,
        ice_volume_t=agg.vt_i,
        snow_volume_t=agg.vt_s,
        pond_volume_t=agg.vt_ip,
        lid_volume_t=agg.vt_il,
        air_stress_u_t=jnp.full(full, 0.08),
        air_stress_v_t=jnp.full(full, 0.03),
        drag_io_t=jnp.full(full, 0.005),
        ocean_u_u=ocean_u,
        ocean_v_v=ocean_v,
        ssh_t=None,
        coriolis_t=jnp.full(full, 1.0e-4),
        tmask_t=of,
        umask_u=of,
        vmask_v=of,
        fast_tmask=zf,
        depth_t=jnp.full(full, 1000.0),
        depth_u=jnp.full(full, 1000.0),
        depth_v=jnp.full(full, 1000.0),
        iceberg_tmask=zf,
        iceberg_umask=zf,
        iceberg_vmask=zf,
    )
    aevp = SI3CGridAEVPState(
        u_ice,
        v_ice,
        jnp.full(full, 100.0),
        jnp.full(full, -60.0),
        jnp.full(full, 20.0),
    )
    grid = SI3ORCA2AdvectionGrid(
        e1u=length,
        e2v=length,
        r1_e1u=1.0 / length,
        r1_e2v=1.0 / length,
        e2u=length,
        e1v=length,
        e1e2t=area,
        r1_e1e2t=1.0 / area,
        tmask=of,
        umask=of,
        vmask=of,
    )
    surface = SI3O17SurfaceForcing(
        wndi=jnp.full(ii, 5.0),
        wndj=jnp.full(ii, 2.0),
        theta_air_zt=jnp.full(ii, 270.0),
        q_air_zt=jnp.full(ii, 0.002),
        slp=jnp.full(ii, 101000.0),
        rhoa=jnp.full(ii, 1.3),
        cloud_fra=jnp.full(ii, 0.5),
        smask0=oi,
        qlw=jnp.full(ii, 250.0),
        precip=jnp.full(ii, 1.0e-7),
        snow=jnp.full(ii, 4.0e-8),
        qsr=jnp.full(ii, 80.0),
        emp=jnp.full(ii, 2.0e-8),
        qns_oce=jnp.full(ii, 20.0),
        qsr_oce=jnp.full(ii, 80.0),
        ssu_m=ocean_u,
        ssv_m=ocean_v,
        utau=jnp.full(ii, 0.08),
        vtau=jnp.full(ii, 0.03),
        frq_m=jnp.full(ii, 0.02),
        e3t_m=jnp.full(ii, 10.0),
    )
    stress = SI3StressInputs(
        uu_kbb=ocean_u,
        vv_kbb=ocean_v,
        umask_k1=of,
        vmask_k1=of,
        u_ice=u_ice,
        v_ice=v_ice,
        u_oce=ocean_u,
        v_oce=ocean_v,
        drag_io=jnp.full(full, 0.005),
        at_i=agg.at_i,
        utau=jnp.full(full, 0.08),
        vtau=jnp.full(full, 0.03),
        tmod_io=jnp.full(full, 0.04),
        taum=jnp.full(ii, 0.09),
        utau_oce=jnp.full(ii, 0.04),
        vtau_oce=jnp.full(ii, 0.02),
    )
    forcing = SI3O17LiveForcing(
        kt=1,
        sss_m=jnp.full(full, 34.0),
        sst_m=jnp.full(full, -1.8),
        rn_amax_2d=jnp.full(full, 0.95),
        advection_grid=grid,
        aevp_forcing=aevp_forcing,
        aevp_metrics=metrics,
        ssh_m=zf,
        surface=surface,
        stress=stress,
    )
    return SI3O17StepState(state, aevp), forcing, card


def probe(master: Path, *, execution_mode: str = "compiled") -> dict:
    if execution_mode not in {"compiled", "eager"}:
        raise ValueError(f"unknown AD execution mode {execution_mode!r}")
    entry, forcing, card = _case(master)
    # Scheduling/routing values are literal static inputs.  Only floating
    # state/forcing leaves enter reverse-mode AD; this prevents ``kt`` and the
    # deck's category/iteration counts from becoming JAX tracers.
    static_inputs = {
        "forcing.kt": int(forcing.kt),
        "card.state.jpl": int(card.state.jpl),
        "card.state.nlay_i": int(card.state.nlay_i),
        "card.state.nlay_s": int(card.state.nlay_s),
        "card.state.nn_hls": int(card.state.nn_hls),
        "card.aevp.n_subcycles": int(card.aevp.n_subcycles),
        "card.aevp.category_count": int(card.aevp.category_count),
        "card.state.ln_icedyn": bool(card.state.ln_icedyn),
        "card.state.ln_icethd": bool(card.state.ln_icethd),
    }
    inputs = (entry, forcing[1:])

    def objective(x):
        step_entry, live_forcing_dynamic = x
        live_forcing = SI3O17LiveForcing(static_inputs["forcing.kt"], *live_forcing_dynamic)
        out, trace = si3_o17_step(step_entry, live_forcing, card)
        exchange = trace.finalize.final
        primal = (out, exchange.fluxes, exchange.outputs, trace.finalize.stress)
        leaves = [
            v
            for v in jax.tree_util.tree_leaves(primal)
            if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating)
        ]
        return sum(jnp.sum(v) for v in leaves), primal

    jax.eval_shape(objective, inputs)
    if execution_mode == "compiled":
        transformed = jax.jit(jax.value_and_grad(objective, has_aux=True))
        compile_started = time.monotonic()
        compiled = transformed.lower(inputs).compile()
        compile_seconds = time.monotonic() - compile_started
        execute_started = time.monotonic()
        (value, primal1), grad = compiled(inputs)
        execution_seconds = time.monotonic() - execute_started
    else:
        compile_seconds = None
        execute_started = time.monotonic()
        with jax.disable_jit():
            (value, primal1), grad = jax.value_and_grad(objective, has_aux=True)(inputs)
        execution_seconds = time.monotonic() - execute_started

    def primal_outputs(x):
        result = si3_o17_step(x[0], SI3O17LiveForcing(static_inputs["forcing.kt"], *x[1]), card)
        return (
            result[0],
            result[1].finalize.final.fluxes,
            result[1].finalize.final.outputs,
            result[1].finalize.stress,
        )

    if execution_mode == "compiled":
        primal0 = jax.jit(primal_outputs)(inputs)
    else:
        with jax.disable_jit():
            primal0 = primal_outputs(inputs)
    unchanged = all(
        np.array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))
        for a, b in zip(jax.tree_util.tree_leaves(primal0), jax.tree_util.tree_leaves(primal1))
        if hasattr(a, "dtype") and a.dtype == jnp.float64
    )
    rows = []
    input_paths, _ = jax.tree_util.tree_flatten_with_path(inputs)
    grad_leaves = jax.tree_util.tree_leaves(grad)
    if len(input_paths) != len(grad_leaves):
        raise AssertionError("input/gradient pytree leaf count changed")
    for (path, input_leaf), leaf in zip(input_paths, grad_leaves):
        a = np.asarray(leaf)
        input_dtype = getattr(input_leaf, "dtype", None)
        differentiable = input_dtype is not None and jnp.issubdtype(input_dtype, jnp.floating)
        leaf_name = jax.tree_util.keystr(path)
        if leaf_name.startswith("[0]"):
            leaf_name = "entry" + leaf_name[3:]
        else:
            for index, name in enumerate(SI3O17LiveForcing._fields[1:]):
                prefix = f"[1][{index}]"
                if leaf_name.startswith(prefix):
                    leaf_name = f"forcing.{name}" + leaf_name[len(prefix) :]
                    break
        row = {
            "leaf": leaf_name,
            "shape": list(np.shape(input_leaf)),
            "input_dtype": str(
                input_dtype if input_dtype is not None else type(input_leaf).__name__
            ),
            "gradient_dtype": str(a.dtype),
            "differentiable": bool(differentiable),
        }
        if differentiable:
            row.update(
                finite=int(np.isfinite(a).sum()),
                nan=int(np.isnan(a).sum()),
                inf=int(np.isinf(a).sum()),
                nonzero=int(np.count_nonzero(a)),
            )
        else:
            raise AssertionError(f"non-floating leaf entered differentiated pytree: {path}")
        rows.append(row)
    finite_rows = [row for row in rows if row["differentiable"]]

    def zero_reason(leaf_name: str) -> str:
        if any(
            leaf_name.startswith(f"forcing.aevp_forcing.{name}")
            for name in (
                "concentration_t",
                "ice_volume_t",
                "snow_volume_t",
                "pond_volume_t",
                "lid_volume_t",
                "air_stress_u_t",
                "air_stress_v_t",
            )
        ):
            return "overwritten from the live ice state or O7 stress before the aEVP call"
        if any(
            leaf_name.startswith(f"forcing.stress.{name}")
            for name in (
                "u_ice",
                "v_ice",
                "at_i",
            )
        ):
            return "overwritten from the completed O16 state before the stress exchange"
        if any(
            token in leaf_name
            for token in (
                ".tmask",
                ".umask",
                ".vmask",
                ".smask",
                "iceberg_",
                "fast_tmask",
            )
        ):
            return "fixed routing/mask leaf; branch selection is intentionally non-differentiable"
        if any(token in leaf_name for token in ("depth_t", "depth_u", "depth_v")):
            return "landfast basal drag is disabled by the frozen deck"
        if ".prather.sal." in leaf_name:
            return "bulk-salinity Prather moments are inactive when nn_icesal=4 uses layer salt"
        return "active program has an exact zero local derivative at this mixed-regime state"

    zero_gradient_leaves = [
        {"leaf": row["leaf"], "reason": zero_reason(row["leaf"])}
        for row in finite_rows
        if row["nonzero"] == 0
    ]
    return {
        "status": "PASS"
        if unchanged and all(r["nan"] == r["inf"] == 0 for r in finite_rows)
        else "FAIL",
        "objective": float(value),
        "forward_bits_unchanged": unchanged,
        "compile_seconds": compile_seconds,
        "execution_seconds": execution_seconds,
        "execution_mode": execution_mode,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "memory_rule_reason": (
            "exact full O17 reverse-mode compile; run alone on a small conforming mixed-regime tile"
        ),
        "synthetic_regime": {
            "full_shape": [7, 7],
            "owned_shape": [3, 3],
            "ice_owned_columns": 1,
            "open_water_owned_columns": 7,
            "near_empty_owned_columns": 1,
            "ice_categories": 5,
            "snow": True,
            "ponds": True,
            "nonzero_velocity_and_stress": True,
        },
        "static_literal_inputs": static_inputs,
        "zero_gradient_leaves": zero_gradient_leaves,
        "leaves": rows,
    }


def eager_jit_probe(master: Path, shape: tuple[int, int] = (184, 152)) -> dict:
    """One production-card O17 step (100 aEVP subcycles) eager and compiled, bit-compared.

    The entry is the mixed-regime fixture laid out on the real ORCA2 rank grid
    (jpi=184, jpj=152 from Ni0glo=180, Nj0glo=148, nn_hls=2), with its fold LBC
    applied, so ice-covered, open-water and nonzero-below-epsi10 columns all occur.
    """
    entry, forcing, card = _case(master, shape)
    h = card.state.nn_hls
    conc = np.asarray(entry.ice.a_i)[h:-h, h:-h].sum(axis=-1)
    dynamic = tuple(forcing[1:])

    def step(step_entry, live_dynamic):
        return si3_o17_step(step_entry, SI3O17LiveForcing(1, *live_dynamic), card)

    started = time.monotonic()
    with jax.disable_jit():
        eager = step(entry, dynamic)
    jax.block_until_ready(eager)
    eager_seconds = time.monotonic() - started
    eager_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.monotonic()
    executable = jax.jit(step).lower(entry, dynamic).compile()
    compile_seconds = time.monotonic() - started
    started = time.monotonic()
    compiled = executable(entry, dynamic)
    jax.block_until_ready(compiled)
    execute_seconds = time.monotonic() - started
    mismatched, leaves, nonfinite = [], 0, []
    for (path, a), (_, b) in zip(
        jax.tree_util.tree_flatten_with_path(eager)[0],
        jax.tree_util.tree_flatten_with_path(compiled)[0],
        strict=True,
    ):
        a, b = np.asarray(a), np.asarray(b)
        leaves += 1
        bits = (a.view(np.uint64) != b.view(np.uint64)) if a.dtype == np.float64 else (a != b)
        if a.shape != b.shape or np.count_nonzero(bits):
            signed_zero_only = bool(a.dtype == np.float64 and np.all(a[bits] == 0.0) and np.all(b[bits] == 0.0))
            mismatched.append({
                "leaf": jax.tree_util.keystr(path), "count": int(np.count_nonzero(bits)),
                "signed_zero_only": signed_zero_only,
                "eager_negative_zero": int(np.count_nonzero(np.signbit(a[bits]))) if signed_zero_only else None,
                "jit_negative_zero": int(np.count_nonzero(np.signbit(b[bits]))) if signed_zero_only else None,
            })
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            nonfinite.append(jax.tree_util.keystr(path))
    return {
        "status": "PASS" if not mismatched and not nonfinite else "FAIL",
        "full_shape": list(shape),
        "owned_shape": [shape[0] - 2 * h, shape[1] - 2 * h],
        "aevp_subcycles": card.aevp.n_subcycles,
        "owned_columns": {
            "ice_covered": int(np.count_nonzero(conc > 1.0e-10)),
            "ice_free": int(np.count_nonzero(conc == 0.0)),
            "near_empty_le_epsi10": int(np.count_nonzero((conc > 0.0) & (conc <= 1.0e-10))),
        },
        "leaves_compared": leaves,
        "bitwise_mismatched_leaves": mismatched,
        "all_mismatches_signed_zero_only": all(row["signed_zero_only"] for row in mismatched),
        "nonfinite_leaves": nonfinite,
        "eager_seconds": eager_seconds,
        "eager_peak_rss_kib": eager_rss,
        "jit_compile_seconds": compile_seconds,
        "jit_execute_seconds": execute_seconds,
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-mode", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--eager-jit-probe", action="store_true")
    args = parser.parse_args()
    report = (
        eager_jit_probe(args.master)
        if args.eager_jit_probe
        else probe(args.master, execution_mode=args.execution_mode)
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
