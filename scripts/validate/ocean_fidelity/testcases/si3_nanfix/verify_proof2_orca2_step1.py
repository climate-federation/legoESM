#!/usr/bin/env python3
"""Proof (2): REAL ORCA2 step 1 (kt=1, rank=0) eager O17 step has 0 NaN in owned outputs and fluxes, and the 9 fields at the 1,813 open-water snowfall points equal NEMO's record bit for bit."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.dynamics import SI3CGridAEVPForcing, SI3CGridAEVPState, SI3CGridMetrics
from legoesm.ice.si3_finalize import SI3StressInputs, _sum_flux_terms
from legoesm.ice.si3_o17 import (
    SI3O17LiveForcing,
    SI3O17StepState,
    SI3O17SurfaceForcing,
    build_orca2_sas_ice_o17_card,
    si3_o17_step,
)
from legoesm.ice.si3_orca2_state import (
    SI3AllPratherMoments,
    SI3CategoryBounds,
    SI3ORCA2AdvectionGrid,
    SI3PratherMoments,
    SI3State,
    si3_var_agg,
    si3_var_glo2eqv,
)
from legoesm.ice.si3_surface import SI3SurfaceConfig
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    DEFAULT_PPSRC,
    parse_resolved_ice_namelist,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_composition_scorer import (
    FLUX_FIELDS,
    PRATHER_PREFIX,
    STATE_FIELDS,
    read_record,
)

_PRIMARY_EVIDENCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step"
)
_REFUSED_EVIDENCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/oracle_o17_full_step_20261104T140742Z_rc1_452003"
)
EVIDENCE_ROOT = _PRIMARY_EVIDENCE if _PRIMARY_EVIDENCE.exists() else _REFUSED_EVIDENCE
METRICS = (
    "e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
    "e1e2t", "e1e2u", "e1e2v", "e1e2f",
)
NINE_FIELDS = (
    "hfx_spr",
    "hfx_sub",
    "wfx_err_sub",
    "wfx_snw_sub",
    "wfx_spr",
    "wfx_sub",
    "emp",
    "fwfice",
    "qns",
)


def _as_full(value: np.ndarray, full: tuple[int, int], h: int) -> jnp.ndarray:
    value = np.asarray(value)
    if value.shape[:2] == full:
        return jnp.asarray(value)
    if value.shape[:2] != (full[0] - 2 * h, full[1] - 2 * h):
        raise ValueError(f"cannot place {value.shape} on the {full} grid")
    out = np.zeros(full + value.shape[2:], dtype=np.float64)
    out[h:-h, h:-h, ...] = value
    return jnp.asarray(out)


def _as_owned(value: np.ndarray, full: tuple[int, int], h: int) -> jnp.ndarray:
    value = np.asarray(value)
    if value.shape[:2] == full:
        return jnp.asarray(value[h:-h, h:-h, ...])
    if value.shape[:2] != (full[0] - 2 * h, full[1] - 2 * h):
        raise ValueError(f"cannot read {value.shape} as owned on the {full} grid")
    return jnp.asarray(value)


def _moments(fields: dict[str, np.ndarray], full: tuple[int, int], h: int) -> SI3AllPratherMoments:
    def family(prefix: str) -> SI3PratherMoments:
        names = (f"sx{prefix}", f"sy{prefix}", f"sxx{prefix}", f"syy{prefix}", f"sxy{prefix}")
        return SI3PratherMoments(*(_as_full(fields[name], full, h) for name in names))
    return SI3AllPratherMoments(**{name: family(prefix) for name, prefix in PRATHER_PREFIX.items()})


def _entry_state_real(records: dict[int, dict[str, Any]], card) -> SI3O17StepState:
    f = records[22]["fields"]
    full = (records[22]["meta"]["jpi"], records[22]["meta"]["jpj"])
    h = int(card.state.nn_hls)

    def F(name: str):
        return _as_full(f[name], full, h)

    state_cfg = card.state
    eqv = si3_var_glo2eqv(
        v_i=F("v_i"), v_s=F("v_s"), sv_i=F("sv_i"), a_i=F("a_i"), oa_i=F("oa_i"),
        a_ip=F("a_ip"), v_ip=F("v_ip"), v_il=F("v_il"), e_s=F("e_s"), e_i=F("e_i"),
        szv_i=F("szv_i"), kn=2, hi_max_top=float(np.asarray(f["hi_max"])[-1]),
        sss_m=F("sss_m"), config=state_cfg,
    )
    agg = si3_var_agg(
        v_i=F("v_i"), v_s=eqv.v_s, sv_i=F("sv_i"), a_i=eqv.a_i, oa_i=F("oa_i"),
        a_ip=F("a_ip"), v_ip=F("v_ip"), v_il=F("v_il"), t_su=F("t_su"), t_si=F("t_si"),
        e_s=F("e_s"), e_i=F("e_i"), szv_i=F("szv_i"), a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i, t_s=eqv.t_s, kn=2, config=state_cfg,
    )
    state = SI3State.zeros(*full, state_cfg)._replace(
        **{name: F(name) for name in STATE_FIELDS},
        prather=_moments(records[29]["fields"], full, h), equivalent=eqv, aggregated=agg,
        bounds=SI3CategoryBounds(jnp.asarray(f["hi_max"]), jnp.asarray(f["hi_mean"])),
        cflice_delayed=(
            jnp.asarray(f["cflice_delayed"]) if "cflice_delayed" in f else None
        ),
    )
    aevp = SI3CGridAEVPState(
        state.u_ice, state.v_ice, F("stress1_i"), F("stress2_i"), F("stress12_i")
    )
    return SI3O17StepState(state, aevp)


def _live_forcing_real(records: dict[int, dict[str, Any]], card) -> SI3O17LiveForcing:
    f, masks = records[22]["fields"], records[23]["fields"]
    r1, r2 = records[32]["fields"], records[33]["fields"]
    full = (records[22]["meta"]["jpi"], records[22]["meta"]["jpj"])
    h = int(card.state.nn_hls)

    def F(src, name: str):
        return _as_full(src[name], full, h)

    def O(src, name: str):
        return _as_owned(src[name], full, h)

    zero = jnp.zeros(full, jnp.float64)
    metrics = SI3CGridMetrics(*(F(f, name) for name in METRICS))
    grid = SI3ORCA2AdvectionGrid(
        e1u=F(f, "e1u"), e2v=F(f, "e2v"), r1_e1u=F(f, "r1_e1u"), r1_e2v=F(f, "r1_e2v"),
        e2u=F(f, "e2u"), e1v=F(f, "e1v"), e1e2t=F(f, "e1e2t"), r1_e1e2t=F(f, "r1_e1e2t"),
        tmask=F(f, "tmask"), umask=F(f, "umask"), vmask=F(f, "vmask"),
    )
    aevp_forcing = SI3CGridAEVPForcing(
        *(zero for _ in range(7)), F(f, "drag_io"), F(f, "u_oce"), F(f, "v_oce"),
        None,
        F(f, "ff_t"), F(f, "tmask"), F(f, "umask"), F(f, "vmask"),
        F(masks, "fast_tmask"), zero, zero, zero,
        F(masks, "icb_tmask"), F(masks, "icb_umask"), F(masks, "icb_vmask"),
    )
    surface = SI3O17SurfaceForcing(
        wndi=O(r1, "wndi"), wndj=O(r1, "wndj"), theta_air_zt=O(r1, "tair"),
        q_air_zt=O(r1, "qair"), slp=O(r1, "slp"), rhoa=O(r1, "rhoa"),
        cloud_fra=O(f, "cloud_fra"), smask0=O(r1, "smask"), qlw=O(r2, "qlw"),
        precip=O(r2, "precip"), snow=O(r2, "snow"), qsr=O(r2, "qsr"), emp=O(r2, "emp"),
        qns_oce=O(f, "qns_oce"), qsr_oce=O(f, "qsr_oce"), ssu_m=F(f, "ssu_m"),
        ssv_m=F(f, "ssv_m"), utau=O(f, "utau"), vtau=O(f, "vtau"), frq_m=O(f, "frq_m"),
        e3t_m=O(f, "e3t_m"),
    )
    stress = SI3StressInputs(
        uu_kbb=F(f, "uu_kbb"), vv_kbb=F(f, "vv_kbb"), umask_k1=F(f, "umask"),
        vmask_k1=F(f, "vmask"), u_ice=F(f, "u_ice"), v_ice=F(f, "v_ice"),
        u_oce=F(f, "u_oce"), v_oce=F(f, "v_oce"), drag_io=F(f, "drag_io"), at_i=zero,
        utau=F(f, "utau"), vtau=F(f, "vtau"), tmod_io=zero, taum=O(f, "taum"),
        utau_oce=O(f, "utau"), vtau_oce=O(f, "vtau"),
    )
    return SI3O17LiveForcing(
        kt=int(records[22]["meta"]["kt"]), sss_m=F(f, "sss_m"), sst_m=F(f, "sst_m"),
        rn_amax_2d=F(f, "rn_amax_2d"), advection_grid=grid, aevp_forcing=aevp_forcing,
        aevp_metrics=metrics, ssh_m=F(f, "ssh_m"), surface=surface, stress=stress,
    )


def _compare_bits(got: np.ndarray, want: np.ndarray, mask: np.ndarray | None = None) -> dict:
    g = np.asarray(got, dtype=np.float64)
    w = np.asarray(want, dtype=np.float64)
    assert g.shape == w.shape, f"shape mismatch: {g.shape} vs {w.shape}"
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        g = g[m]
        w = w[m]
    else:
        g = g.ravel()
        w = w.ravel()
    gb = np.ascontiguousarray(g).view(np.uint64)
    wb = np.ascontiguousarray(w).view(np.uint64)
    eq = gb == wb
    diff_count = int(np.sum(~eq))
    max_ulp = 0
    if diff_count > 0:
        for gi, wi in zip(gb[~eq], wb[~eq]):
            max_ulp = max(max_ulp, int(abs(int(gi) - int(wi))))
    return {
        "compared": int(g.size),
        "equal": int(np.sum(eq)),
        "differing": diff_count,
        "max_ulp": max_ulp,
    }


def main() -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    run = EVIDENCE_ROOT / "record_100step"
    rec_dir = run / "o17_master_records"
    kt, rank = 1, 0

    def rec(kind: int) -> dict:
        matches = sorted(rec_dir.glob(f"o17_kind*{kind:02d}*_step*{kt:04d}_rank*{rank:04d}.bin"))
        assert len(matches) == 1, f"expected 1 record for kind={kind}, got {matches}"
        return read_record(matches[0])

    records = {k: rec(k) for k in (22, 23, 26, 27, 28, 29, 32, 33)}

    nml = parse_resolved_ice_namelist(
        run / "output.namelist.ice", run / "output.namelist.dyn", DEFAULT_PPSRC
    )
    surface_cfg = SI3SurfaceConfig.from_resolved_namelists(
        run / "output.namelist.dyn", run / "output.namelist.ice"
    )
    card = build_orca2_sas_ice_o17_card(surface=surface_cfg, thermo_namelist=nml)
    h = int(card.state.nn_hls)
    owned = (slice(h, -h), slice(h, -h))

    entry = _entry_state_real(records, card)
    forcing = _live_forcing_real(records, card)

    t0 = time.perf_counter()
    with jax.disable_jit():
        out, trace = si3_o17_step(entry, forcing, card)
    elapsed_s = time.perf_counter() - t0

    # 1. Check for NaNs in all owned outputs and fluxes
    nan_counts = {}
    for name in STATE_FIELDS:
        arr = np.asarray(getattr(out.ice, name))
        arr_owned = arr[owned] if arr.shape[:2] == (184, 152) else arr
        nan_counts[f"state.{name}"] = int(np.isnan(arr_owned).sum())
    for name in FLUX_FIELDS:
        arr = np.asarray(getattr(trace.finalize.final.fluxes, name))
        nan_counts[f"fluxes.{name}"] = int(np.isnan(arr).sum())
    for name in trace.finalize.final.outputs._fields:
        arr = np.asarray(getattr(trace.finalize.final.outputs, name))
        nan_counts[f"outputs.{name}"] = int(np.isnan(arr).sum())
    for name in trace.finalize.stress._fields:
        arr = np.asarray(getattr(trace.finalize.stress, name))
        arr_owned = arr[owned] if arr.shape[:2] == (184, 152) else arr
        nan_counts[f"stress.{name}"] = int(np.isnan(arr_owned).sum())

    total_owned_nans = sum(nan_counts.values())

    # 2. Identify the 1,813 open-water snowfall points
    a_i_pre_thermo = np.asarray(trace.postdyn_correction.a_i)[owned]
    at_i_pre_thermo = np.sum(a_i_pre_thermo, axis=-1)
    sprecip_owned = np.asarray(trace.surface_stress.blk_ice_2.sprecip)
    open_water_snow_mask = (at_i_pre_thermo == 0.0) & (sprecip_owned > 0.0)
    n_open_water_snow = int(np.sum(open_water_snow_mask))

    # Reconstruct NEMO kind 28 fwfice per iceupdate.F90:177 / si3_finalize.py:978
    f28 = records[28]["fields"]
    with jax.disable_jit():
        nemo_fwfice = np.asarray(
            _sum_flux_terms(
                f28["wfx_ice"], f28["wfx_snw"], f28["wfx_pnd"], f28["wfx_err_sub"]
            )
        )

    got_9 = {
        "hfx_spr": np.asarray(trace.finalize.final.fluxes.hfx_spr),
        "hfx_sub": np.asarray(trace.finalize.final.fluxes.hfx_sub),
        "wfx_err_sub": np.asarray(trace.finalize.final.fluxes.wfx_err_sub),
        "wfx_snw_sub": np.asarray(trace.finalize.final.fluxes.wfx_snw_sub),
        "wfx_spr": np.asarray(trace.finalize.final.fluxes.wfx_spr),
        "wfx_sub": np.asarray(trace.finalize.final.fluxes.wfx_sub),
        "emp": np.asarray(trace.finalize.final.outputs.emp),
        "fwfice": np.asarray(trace.finalize.final.outputs.fwfice),
        "qns": np.asarray(trace.finalize.final.outputs.qns),
    }
    want_9 = {
        "hfx_spr": np.asarray(_as_owned(f28["hfx_spr"], (184, 152), h)),
        "hfx_sub": np.asarray(_as_owned(f28["hfx_sub"], (184, 152), h)),
        "wfx_err_sub": np.asarray(_as_owned(f28["wfx_err_sub"], (184, 152), h)),
        "wfx_snw_sub": np.asarray(_as_owned(f28["wfx_snw_sub"], (184, 152), h)),
        "wfx_spr": np.asarray(_as_owned(f28["wfx_spr"], (184, 152), h)),
        "wfx_sub": np.asarray(_as_owned(f28["wfx_sub"], (184, 152), h)),
        "emp": np.asarray(_as_owned(f28["emp"], (184, 152), h)),
        "fwfice": np.asarray(_as_owned(nemo_fwfice, (184, 152), h)),
        "qns": np.asarray(_as_owned(f28["qns"], (184, 152), h)),
    }

    nine_fields_1813 = {}
    nine_fields_all_owned = {}
    for field in NINE_FIELDS:
        nine_fields_1813[field] = _compare_bits(
            got_9[field], want_9[field], open_water_snow_mask
        )
        nine_fields_all_owned[field] = _compare_bits(got_9[field], want_9[field])

    # Also check t_si on inactive categories at kind 27 and kind 28
    inactive_cat_mask = a_i_pre_thermo <= float(card.constants.epsi10)
    r27_tsi = np.asarray(records[27]["fields"]["t_si"])
    if r27_tsi.shape[:2] == (184, 152):
        r27_tsi = r27_tsi[owned]
    f28_tsi = np.asarray(f28["t_si"])
    if f28_tsi.shape[:2] == (184, 152):
        f28_tsi = f28_tsi[owned]
    t_si_inactive_k27 = _compare_bits(
        np.asarray(trace.finalize.correction.state.t_si)[owned],
        r27_tsi,
        inactive_cat_mask,
    )
    t_si_inactive_k28 = _compare_bits(
        np.asarray(out.ice.t_si)[owned],
        f28_tsi,
        inactive_cat_mask,
    )

    report = {
        "status": (
            "PASS"
            if total_owned_nans == 0
            and n_open_water_snow == 1813
            and all(
                v["differing"] == 0 and v["max_ulp"] == 0
                for v in nine_fields_1813.values()
            )
            else "FAIL"
        ),
        "elapsed_seconds": elapsed_s,
        "total_owned_nans": total_owned_nans,
        "open_water_snowfall_points": n_open_water_snow,
        "nine_fields_at_1813_points": nine_fields_1813,
        "nine_fields_all_26640_owned": nine_fields_all_owned,
        "t_si_inactive_categories_kind27": t_si_inactive_k27,
        "t_si_inactive_categories_kind28": t_si_inactive_k28,
    }
    out_path = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_nanfix_scratch/proof2_orca2_step1.json"
    )
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
