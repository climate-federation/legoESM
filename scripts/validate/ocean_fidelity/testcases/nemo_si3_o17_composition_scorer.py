#!/usr/bin/env python3
"""Score unified O17 records by executing the production O17 entry.

Extents are never assumed: every record field carries its own shape.  NEMO
allocates many SI3/SBC arrays on the owned interior only (``A2D(0)``, e.g.
``t_si`` at ``ice.F90:451``, the flux accumulators at ``ice.F90:400-412``,
``utau_oce`` at ``iceupdate.F90:55``) while the lane's state carries full-halo
arrays.  Owned entries are padded with the lane's halo convention (zero, as
``si3_orca2_state.py`` builds ``t_si``); owned records are compared on the owned
interior only.  Records are streamed one step at a time.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import re
import resource
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

MAGIC = b"O17S0001"
VERSION = 1
VALID_LABELS = ("INDEPENDENT", "GIVEN-NEMO-ENTRY")
TRAJECTORY_BAR = 1.0e-15
KINDS = {
    22: "step_entry", 23: "after_rheology", 24: "after_advection",
    25: "after_rdgrft", 26: "after_dyn_cor", 27: "after_thermo",
    28: "step_exit", 29: "prather_entry", 30: "prather_exit",
    31: "after_flx_other", 32: "raw_blk1", 33: "raw_blk2", 34: "tau_private",
}
_NAME = re.compile(r"o17_kind(\d{2})_(\w+?)_step(\d{4})_rank(\d{4})\.bin$")
STATE_FIELDS = (
    "v_i", "v_s", "sv_i", "a_i", "t_su", "u_ice", "v_ice", "oa_i",
    "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i", "t_si",
    "snwice_mass", "snwice_mass_b", "snwice_fmass",
)
FLUX_FIELDS = (
    "wfx_snw", "wfx_snw_sni", "wfx_snw_sum", "wfx_pnd", "wfx_spr", "wfx_sub",
    "wfx_snw_sub", "wfx_ice_sub", "wfx_ice", "wfx_sni", "wfx_opw", "wfx_bog",
    "wfx_dyn", "wfx_bom", "wfx_sum", "wfx_lam", "wfx_res", "wfx_err_sub",
    "sfx_bog", "sfx_bom", "sfx_lam", "sfx_sum", "sfx_sni", "sfx_opw", "sfx_bri",
    "sfx_dyn", "sfx_res", "sfx_sub", "hfx_bog", "hfx_bom", "hfx_sum", "hfx_opw",
    "hfx_dif", "hfx_snw", "hfx_err_dif", "hfx_sub", "hfx_spr", "hfx_thd",
    "hfx_dyn", "hfx_res", "qlead", "qsb_ice_bot", "fhld",
)
# Kind 25 is written after ice_dyn_rdgrft (icedyn.F90:134), kind 26 after
# ice_cor(kt,1) (icedyn.F90:135); these are the recorded members of each carry.
RDGRFT_FIELDS = (
    "a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i",
    "wfx_dyn", "sfx_dyn", "hfx_dyn", "wfx_pnd",
)
DYN_COR_FIELDS = (
    "a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i",
    "t_su", "wfx_res", "sfx_res", "hfx_res",
)
PRATHER_PREFIX = {
    "ice": "ice", "sn": "sn", "a": "a", "age": "age", "sal": "sal",
    "ap": "ap", "vp": "vp", "vl": "vl", "c0": "c0", "e": "e", "si": "si",
}
METRICS = ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
           "e1e2t", "e1e2u", "e1e2v", "e1e2f")


def read_record(path: Path) -> dict[str, Any]:
    """Read a self-describing stream record; no halo extent is inferred."""

    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError(f"bad O17 magic: {path}")
        if struct.unpack("=i", stream.read(4))[0] != VERSION:
            raise ValueError(f"bad O17 version: {path}")
        meta = struct.unpack("=8i", stream.read(32))
        fields: dict[str, np.ndarray] = {}
        while True:
            raw_name = stream.read(32)
            if len(raw_name) != 32:
                raise ValueError(f"truncated field header: {path}")
            name = raw_name.decode("ascii").rstrip(" \x00")
            rank = struct.unpack("=i", stream.read(4))[0]
            dims = struct.unpack("=4i", stream.read(16))
            if name == "__END__" and rank == -1:
                break
            if name in fields or rank not in range(5):
                raise ValueError(f"bad/duplicate O17 field {name!r}: {path}")
            shape = tuple(dims[:rank])
            count = int(np.prod(shape, dtype=np.int64)) if rank else 1
            raw = stream.read(count * 8)
            if len(raw) != count * 8:
                raise ValueError(f"truncated O17 field {name}: {path}")
            value = np.frombuffer(raw, dtype="=f8").copy()
            fields[name] = value.reshape(shape, order="F") if rank else value.reshape(())
        if stream.read(1):
            raise ValueError(f"trailing bytes after O17 terminator: {path}")
    kind, kt, rank_id, jpi, jpj, jpl, nlay_i, nlay_s = meta
    if kind not in KINDS or min(kt, jpi, jpj, jpl, nlay_i, nlay_s) < 1 or rank_id < 0:
        raise ValueError(f"invalid O17 metadata: {path}")
    return {
        "meta": {"kind": kind, "kt": kt, "rank": rank_id, "jpi": jpi, "jpj": jpj,
                 "jpl": jpl, "nlay_i": nlay_i, "nlay_s": nlay_s},
        "fields": fields,
    }


def record_paths(root: Path) -> dict[tuple[int, int], dict[int, Path]]:
    """Index file NAMES only (no payload is loaded); require a complete grid."""
    index: dict[tuple[int, int], dict[int, Path]] = {}
    for path in sorted(root.glob("o17_kind*.bin")):
        match = _NAME.search(path.name)
        if not match:
            raise ValueError(f"unparsable O17 record name: {path.name}")
        kind, label, kt, rank = int(match[1]), match[2], int(match[3]), int(match[4])
        if KINDS.get(kind) != label:
            raise ValueError(f"O17 record label {label!r} is not kind {kind}")
        slot = index.setdefault((kt, rank), {})
        if kind in slot:
            raise ValueError(f"duplicate O17 record {(kind, kt, rank)}")
        slot[kind] = path
    if not index:
        raise ValueError(f"no O17 records in {root}")
    steps = sorted({kt for kt, _ in index})
    ranks = sorted({rank for _, rank in index})
    missing = [(kind, kt, rank) for kt in steps for rank in ranks for kind in KINDS
               if kind not in index.get((kt, rank), {})]
    if missing:
        raise ValueError(f"O17 index incomplete; missing={missing[:8]}")
    return index


def load_step(paths: dict[int, Path], kt: int, rank: int) -> dict[int, dict[str, Any]]:
    records = {kind: read_record(path) for kind, path in paths.items()}
    for kind, record in records.items():
        if (record["meta"]["kind"], record["meta"]["kt"], record["meta"]["rank"]) != (kind, kt, rank):
            raise ValueError(f"O17 header {record['meta']} disagrees with its name")
    return records


def _as_full(value: np.ndarray, full: tuple[int, int], h: int) -> jnp.ndarray:
    """A2D(0) record -> lane full-halo array (halo zero, the lane's t_si convention)."""
    value = np.asarray(value)
    if value.shape[:2] == full:
        return jnp.asarray(value)
    if value.shape[:2] != (full[0] - 2 * h, full[1] - 2 * h):
        raise ValueError(f"cannot place {value.shape} on the {full} grid")
    out = np.zeros(full + value.shape[2:], dtype=np.float64)
    out[h:-h, h:-h, ...] = value
    return jnp.asarray(out)


def _as_owned(value: np.ndarray, full: tuple[int, int], h: int) -> jnp.ndarray:
    """Full-halo record -> owned interior consumed by an A2D(0) kernel argument."""
    value = np.asarray(value)
    if value.shape[:2] == full:
        return jnp.asarray(value[h:-h, h:-h, ...])
    if value.shape[:2] != (full[0] - 2 * h, full[1] - 2 * h):
        raise ValueError(f"cannot read {value.shape} as owned on the {full} grid")
    return jnp.asarray(value)


def _moments(fields: dict[str, np.ndarray], full, h) -> SI3AllPratherMoments:
    def family(prefix: str) -> SI3PratherMoments:
        names = (f"sx{prefix}", f"sy{prefix}", f"sxx{prefix}", f"syy{prefix}", f"sxy{prefix}")
        return SI3PratherMoments(*(_as_full(fields[name], full, h) for name in names))
    return SI3AllPratherMoments(**{name: family(prefix) for name, prefix in PRATHER_PREFIX.items()})


def _grid(record: dict[str, Any], card) -> tuple[tuple[int, int], int]:
    return (record["meta"]["jpi"], record["meta"]["jpj"]), int(card.state.nn_hls)


def entry_state(records: dict[int, dict[str, Any]], card) -> SI3O17StepState:
    """Kinds 22 (state) and 29 (Prather entry) -> the lane's full-halo step entry."""
    f = records[22]["fields"]
    full, h = _grid(records[22], card)
    def F(name):  # noqa: N802
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
        cflice_delayed=(  # mpp_allreduce_generic.h90:147-152; unallocated at kt=1
            jnp.asarray(f["cflice_delayed"]) if "cflice_delayed" in f else None
        ),
    )
    aevp = SI3CGridAEVPState(state.u_ice, state.v_ice, F("stress1_i"), F("stress2_i"), F("stress12_i"))
    return SI3O17StepState(state, aevp)


def live_forcing(records: dict[int, dict[str, Any]], card) -> SI3O17LiveForcing:
    """Kinds 22, 23 (icedyn masks), 32/33 (raw bulk arguments) -> live forcing."""
    f, masks = records[22]["fields"], records[23]["fields"]
    r1, r2 = records[32]["fields"], records[33]["fields"]
    full, h = _grid(records[22], card)
    def F(src, name):  # noqa: N802
        return _as_full(src[name], full, h)

    def O(src, name):  # noqa: N802, E743
        return _as_owned(src[name], full, h)

    zero = jnp.zeros(full, jnp.float64)
    metrics = SI3CGridMetrics(*(F(f, name) for name in METRICS))
    grid = SI3ORCA2AdvectionGrid(
        e1u=F(f, "e1u"), e2v=F(f, "e2v"), r1_e1u=F(f, "r1_e1u"), r1_e2v=F(f, "r1_e2v"),
        e2u=F(f, "e2u"), e1v=F(f, "e1v"), e1e2t=F(f, "e1e2t"), r1_e1e2t=F(f, "r1_e1e2t"),
        tmask=F(f, "tmask"), umask=F(f, "umask"), vmask=F(f, "vmask"),
    )
    aevp_forcing = SI3CGridAEVPForcing(  # first seven are overwritten in-step (si3_o17.py)
        *(zero for _ in range(7)), F(f, "drag_io"), F(f, "u_oce"), F(f, "v_oce"),
        None,  # ssh_t: the ORCA2 solver derives it from ssh_m (icedyn_rhg_evp.F90:268)
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


def record_inputs(entry: SI3O17StepState, forcing: SI3O17LiveForcing) -> dict[int, dict[str, Any]]:
    """Inverse of ``entry_state``/``live_forcing``: model inputs as writer fields."""
    s, g, a, sf, st = entry.ice, forcing.advection_grid, forcing.aevp_forcing, forcing.surface, forcing.stress
    k22 = {name: getattr(s, name) for name in STATE_FIELDS}
    k22.update(stress1_i=entry.aevp.stress1_t, stress2_i=entry.aevp.stress2_t,
               stress12_i=entry.aevp.stress12_f, hi_max=s.bounds.hi_max, hi_mean=s.bounds.hi_mean,
               sss_m=forcing.sss_m, sst_m=forcing.sst_m, rn_amax_2d=forcing.rn_amax_2d,
               ssh_m=forcing.ssh_m, drag_io=a.drag_io_t, u_oce=a.ocean_u_u, v_oce=a.ocean_v_v,
               ff_t=a.coriolis_t, tmask=g.tmask, umask=g.umask, vmask=g.vmask,
               r1_e1u=g.r1_e1u, r1_e2v=g.r1_e2v, r1_e1e2t=g.r1_e1e2t,
               cloud_fra=sf.cloud_fra, qns_oce=sf.qns_oce, qsr_oce=sf.qsr_oce, ssu_m=sf.ssu_m,
               ssv_m=sf.ssv_m, frq_m=sf.frq_m, e3t_m=sf.e3t_m, utau=st.utau, vtau=st.vtau,
               taum=st.taum, uu_kbb=st.uu_kbb, vv_kbb=st.vv_kbb)
    k22.update(zip(METRICS, forcing.aevp_metrics))
    if s.cflice_delayed is not None:
        k22["cflice_delayed"] = s.cflice_delayed
    k29 = {f"{stem}{prefix}": getattr(getattr(s.prather, family), member)
           for family, prefix in PRATHER_PREFIX.items()
           for member, stem in zip(SI3PratherMoments._fields, ("sx", "sy", "sxx", "syy", "sxy"), strict=True)}
    return {
        22: k22,
        23: {"fast_tmask": a.fast_tmask, "icb_tmask": a.iceberg_tmask,
             "icb_umask": a.iceberg_umask, "icb_vmask": a.iceberg_vmask},
        29: k29,
        32: {"wndi": sf.wndi, "wndj": sf.wndj, "tair": sf.theta_air_zt, "qair": sf.q_air_zt,
             "slp": sf.slp, "rhoa": sf.rhoa, "smask": sf.smask0},
        33: {"tair": sf.theta_air_zt, "qair": sf.q_air_zt, "slp": sf.slp, "qlw": sf.qlw,
             "precip": sf.precip, "snow": sf.snow, "qsr": sf.qsr, "emp": sf.emp},
    }


def scored_fields(out: SI3O17StepState, trace) -> list[tuple[int, str, Any]]:
    """Every (kind, record field, model value) compared; also the synthetic writer's source."""
    rows: list[tuple[int, str, Any]] = []
    aevp = trace.aevp
    rows += [(23, "u_ice", aevp.u_ice_u), (23, "v_ice", aevp.v_ice_v),
             (23, "stress1_i", aevp.stress1_t), (23, "stress2_i", aevp.stress2_t),
             (23, "stress12_i", aevp.stress12_f)]
    rows += [(24, n, getattr(trace.advection_state, n)) for n in STATE_FIELDS]
    rows += [(25, n, getattr(trace.redistribution.state, n)) for n in RDGRFT_FIELDS]
    rows += [(26, n, getattr(trace.postdyn_correction, n)) for n in DYN_COR_FIELDS]
    # Kind 27 follows `CALL ice_thd(kt)` (icestp.F90:206); ice_thd itself ends with
    # ice_cor(kt,2) and the ageing/LBC/velocity tail (icethd.F90:183-201).
    rows += [(27, n, getattr(trace.finalize.correction.state, n)) for n in STATE_FIELDS]
    rows += [(28, n, getattr(out.ice, n)) for n in STATE_FIELDS]
    rows += [(28, n, getattr(trace.finalize.final.fluxes, n)) for n in FLUX_FIELDS]
    rows += [(28, n, getattr(trace.finalize.final.outputs, n))
             for n in ("qsr", "qns", "emp", "sfx", "snwice_mass", "snwice_mass_b", "snwice_fmass")]
    rows += [(28, n, getattr(trace.finalize.stress, n)) for n in ("utau", "vtau", "taum")]
    for family, prefix in PRATHER_PREFIX.items():
        moments = getattr(out.ice.prather, family)
        for member, stem in zip(moments._fields, ("sx", "sy", "sxx", "syy", "sxy"), strict=True):
            rows.append((30, f"{stem}{prefix}", getattr(moments, member)))
    rows += [(31, n, getattr(trace.surface_stress.flx_other, n)) for n in ("qlead", "qsb_ice_bot", "fhld")]
    rows += [(34, n, getattr(trace.finalize.stress, n)) for n in ("tmod_io", "utau_oce", "vtau_oce")]
    return rows


def compare(actual: Any, expected: Any, full: tuple[int, int], h: int) -> dict[str, Any]:
    """Exact-bit and relative comparison; an owned operand restricts both to the owned interior."""
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    owned = (full[0] - 2 * h, full[1] - 2 * h)
    if a.shape != b.shape and a.shape[2:] == b.shape[2:]:
        if a.shape[:2] == full and b.shape[:2] == owned:
            a = a[h:-h, h:-h, ...]
        elif b.shape[:2] == full and a.shape[:2] == owned:
            b = b[h:-h, h:-h, ...]
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)], "different": max(a.size, b.size),
                "nonfinite": 0, "max_relative": float("inf")}
    nonfinite = int(np.count_nonzero(~np.isfinite(a)) + np.count_nonzero(~np.isfinite(b)))
    different = int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
    with np.errstate(invalid="ignore", over="ignore"):
        rel = np.abs(a - b) / np.maximum(np.abs(b), np.finfo(np.float64).tiny)
    max_relative = float("inf") if nonfinite else float(np.max(rel, initial=0.0))
    return {"different": different, "nonfinite": nonfinite, "max_relative": max_relative}


def _passes(row: dict[str, Any], label: str) -> bool:
    if "shape_mismatch" in row or row["nonfinite"]:
        return False  # any non-finite compared value fails the step (both labels)
    if label == "GIVEN-NEMO-ENTRY":
        return row["different"] == 0
    return row["max_relative"] <= TRAJECTORY_BAR


def score_step(out, trace, records: dict[int, dict[str, Any]], card, label: str) -> dict[str, Any]:
    full, h = _grid(records[22], card)
    fields: dict[str, Any] = {}
    for kind, name, actual in scored_fields(out, trace):
        row = compare(actual, records[kind]["fields"][name], full, h)
        row["pass"] = _passes(row, label)
        fields[f"{KINDS[kind]}.{name}"] = row
    return {"pass": all(row["pass"] for row in fields.values()),
            "different": sum(row["different"] for row in fields.values()),
            "nonfinite": sum(row["nonfinite"] for row in fields.values()),
            "failed_fields": [key for key, row in fields.items() if not row["pass"]],
            "fields": fields}


def kt_program(kt: int, nn_fsbc: int) -> int:
    """Representative kt of the same compiled program.

    The O17 graph reads ``kt`` only through ``si3_dyn_orca2`` -> Prather's sweep
    parity ``MOD((kt-1)/nn_fsbc, 2)`` (``icedyn_adv_pra.F90:267-367``,
    ``si3_orca2_state.py`` ``kt_parity``) and its ``kt >= 1`` guard.
    """
    return 1 + ((kt - 1) // nn_fsbc) % 2 * nn_fsbc


def make_runner(card, execution: str) -> Callable[[SI3O17StepState, SI3O17LiveForcing], Any]:
    """One production step.  ``jit`` compiles once per (sweep parity, pytree structure).

    Eager is the default: compiled XLA rewrites ``x * convert(mask)`` into a select
    and stores +0 where NEMO's arithmetic (and eager) store -0 (Resume 5 receipt).
    """
    if execution == "eager":
        def eager(entry, forcing):
            with jax.disable_jit():
                return si3_o17_step(entry, forcing, card)
        return eager
    if execution != "jit":
        raise ValueError(f"unknown execution mode {execution!r}")
    compiled: dict[int, Any] = {}

    def run(entry, forcing):
        kt = kt_program(forcing.kt, card.state.nn_fsbc)
        if kt not in compiled:
            compiled[kt] = jax.jit(
                lambda e, d, kt=kt: si3_o17_step(e, SI3O17LiveForcing(kt, *d), card)
            )
        return compiled[kt](entry, tuple(forcing[1:]))
    return run


def _same_bits(left: Any, right: Any) -> bool:
    a, at = jax.tree_util.tree_flatten(left)
    b, bt = jax.tree_util.tree_flatten(right)
    return at == bt and all(
        np.asarray(x).shape == np.asarray(y).shape
        and np.asarray(x).tobytes() == np.asarray(y).tobytes()
        for x, y in zip(a, b, strict=True)
    )


def _release_free_heap() -> None:
    """Return freed heap pages to the OS so VmRSS samples live memory (glibc only)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _rss_kib() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1


def card_for(run: Path):
    nml = parse_resolved_ice_namelist(run / "output.namelist.ice", run / "output.namelist.dyn", DEFAULT_PPSRC)
    surface_cfg = SI3SurfaceConfig.from_resolved_namelists(run / "output.namelist.dyn", run / "output.namelist.ice")
    return build_orca2_sas_ice_o17_card(surface=surface_cfg, thermo_namelist=nml)


CARRIED_KIND22_FIELDS = frozenset(
    (*STATE_FIELDS, "stress1_i", "stress2_i", "stress12_i", "cflice_delayed")
)
TIMING_KEYS = ("scorer_wall_seconds", "scorer_rss_kib_after_each_step", "scorer_peak_rss_kib")


def parse_step_range(spec: str) -> tuple[int, int]:
    """Parse a 1-based inclusive step range ``START..END`` (e.g. ``1..20``)."""
    match = re.fullmatch(r"\s*(\d+)\.\.(\d+)\s*", spec)
    if not match:
        raise ValueError(f"step range must have form START..END (e.g. 1..20), got {spec!r}")
    start, stop = int(match[1]), int(match[2])
    if start < 1 or stop < start:
        raise ValueError(f"invalid step range {start}..{stop}: require 1 <= START <= END")
    return start, stop


def strip_timing(report: dict[str, Any]) -> dict[str, Any]:
    """Return ``report`` without runtime wall-clock/RSS telemetry fields."""
    return {k: v for k, v in report.items() if k not in TIMING_KEYS}


def exit_state(records: dict[int, dict[str, Any]], card) -> SI3O17StepState:
    """Kinds 22, 23, 27, 28, and 30 -> the step-exit ``SI3O17StepState`` of a recorded step."""
    f22, f23 = records[22]["fields"], records[23]["fields"]
    f27, f28 = records[27]["fields"], records[28]["fields"]
    full, h = _grid(records[28], card)

    def F(src, name):  # noqa: N802
        return _as_full(src[name], full, h)

    state_cfg = card.state
    eqv = si3_var_glo2eqv(
        v_i=F(f27, "v_i"), v_s=F(f27, "v_s"), sv_i=F(f27, "sv_i"), a_i=F(f27, "a_i"),
        oa_i=F(f27, "oa_i"), a_ip=F(f27, "a_ip"), v_ip=F(f27, "v_ip"), v_il=F(f27, "v_il"),
        e_s=F(f27, "e_s"), e_i=F(f27, "e_i"), szv_i=F(f27, "szv_i"), kn=2,
        hi_max_top=float(np.asarray(f22["hi_max"])[-1]), sss_m=F(f22, "sss_m"),
        config=state_cfg,
    )
    agg = si3_var_agg(
        v_i=F(f27, "v_i"), v_s=eqv.v_s, sv_i=F(f27, "sv_i"), a_i=eqv.a_i,
        oa_i=F(f27, "oa_i"), a_ip=F(f27, "a_ip"), v_ip=F(f27, "v_ip"), v_il=F(f27, "v_il"),
        t_su=F(f27, "t_su"), t_si=F(f27, "t_si"), e_s=F(f27, "e_s"), e_i=F(f27, "e_i"),
        szv_i=F(f27, "szv_i"), a_ip_eff=eqv.a_ip_eff, t_i=eqv.t_i, t_s=eqv.t_s,
        kn=2, config=state_cfg,
    )
    state = SI3State.zeros(*full, state_cfg)._replace(
        **{name: F(f28, name) for name in STATE_FIELDS},
        prather=_moments(records[30]["fields"], full, h), equivalent=eqv, aggregated=agg,
        bounds=SI3CategoryBounds(jnp.asarray(f22["hi_max"]), jnp.asarray(f22["hi_mean"])),
        cflice_delayed=(
            jnp.asarray(f28["cflice_delayed"]) if "cflice_delayed" in f28 else None
        ),
    )
    aevp = SI3CGridAEVPState(
        F(f23, "u_ice"), F(f23, "v_ice"),
        F(f23, "stress1_i"), F(f23, "stress2_i"), F(f23, "stress12_i"),
    )
    return SI3O17StepState(state, aevp)


def external_forcing(records: dict[int, dict[str, Any]], card) -> SI3O17LiveForcing:
    """Build ``SI3O17LiveForcing`` reading ONLY external (non-carried) fields from ``records``."""
    external_k22 = {
        "meta": records[22]["meta"],
        "fields": {k: v for k, v in records[22]["fields"].items() if k not in CARRIED_KIND22_FIELDS},
    }
    full, h = _grid(records[22], card)
    zero = jnp.zeros(full, jnp.float64)
    external_k22["fields"]["u_ice"] = zero
    external_k22["fields"]["v_ice"] = zero
    filtered = {22: external_k22, 23: records[23], 32: records[32], 33: records[33]}
    return live_forcing(filtered, card)


def carry_step_state(
    prev_out: SI3O17StepState,
    records: dict[int, dict[str, Any]],
    card,
) -> SI3O17StepState:
    """Return the carried ice state for the next free-run step from ``prev_out``."""
    del records, card
    return prev_out


def _carried_mismatches(
    actual: SI3O17StepState,
    expected: SI3O17StepState,
    full: tuple[int, int],
    h: int,
) -> dict[str, dict[str, Any]]:
    """Bit-for-bit comparison of every carried state leaf entering a free-run step."""
    mismatches: dict[str, dict[str, Any]] = {}
    pairs: list[tuple[str, Any, Any]] = [
        (f"ice.{name}", getattr(actual.ice, name), getattr(expected.ice, name))
        for name in (*STATE_FIELDS, "cnd_ice", "t1_ice")
    ]
    for family, prefix in PRATHER_PREFIX.items():
        act_m = getattr(actual.ice.prather, family)
        exp_m = getattr(expected.ice.prather, family)
        for member, stem in zip(act_m._fields, ("sx", "sy", "sxx", "syy", "sxy"), strict=True):
            pairs.append((f"prather.{stem}{prefix}", getattr(act_m, member), getattr(exp_m, member)))
    for member in actual.ice.bounds._fields:
        pairs.append((f"ice.bounds.{member}", getattr(actual.ice.bounds, member), getattr(expected.ice.bounds, member)))
    for member in actual.ice.equivalent._fields:
        pairs.append((f"ice.equivalent.{member}", getattr(actual.ice.equivalent, member), getattr(expected.ice.equivalent, member)))
    for member in actual.ice.aggregated._fields:
        pairs.append((f"ice.aggregated.{member}", getattr(actual.ice.aggregated, member), getattr(expected.ice.aggregated, member)))
    if actual.ice.cflice_delayed is not None or expected.ice.cflice_delayed is not None:
        pairs.append(("ice.cflice_delayed", actual.ice.cflice_delayed, expected.ice.cflice_delayed))
    for member, rec_name in (("stress1_t", "stress1_i"), ("stress2_t", "stress2_i"), ("stress12_f", "stress12_i")):
        pairs.append((f"aevp.{rec_name}", getattr(actual.aevp, member), getattr(expected.aevp, member)))
    for name, a_val, e_val in pairs:
        if a_val is None or e_val is None:
            row = {"different": 1, "nonfinite": 0, "max_relative": float("inf"), "pass": False}
            mismatches[f"carried_entry.{name}"] = row
            continue
        row = compare(a_val, e_val, full, h)
        row["pass"] = _passes(row, "GIVEN-NEMO-ENTRY")
        if not row["pass"]:
            mismatches[f"carried_entry.{name}"] = row
    return mismatches


def _summarize_label(steps_dict: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    rows = list(steps_dict.values())
    return {
        "steps": steps_dict,
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "different": sum(row["different"] for row in rows),
        "nonfinite": sum(row["nonfinite"] for row in rows),
        "max_relative": max(
            (r["max_relative"] for row in rows for r in row["fields"].values()),
            default=0.0,
        ),
        "ulp_bar": 0,
        "trajectory_relative_bar": TRAJECTORY_BAR,
    }


def merge_reports(
    reports: list[dict[str, Any]],
    *,
    expect_steps: int = 100,
) -> dict[str, Any]:
    """Combine parallel step-range reports into the exact serial scorer report."""
    if not reports:
        raise ValueError("no range reports provided for merge")
    labels = list(VALID_LABELS)
    expected_steps = list(range(1, expect_steps + 1))
    first = reports[0]
    execution = first.get("execution")
    if execution not in ("eager", "jit"):
        raise ValueError(f"invalid execution mode {execution!r} in range report")
    ranks = list(first.get("ranks", []))
    if not ranks or ranks != sorted(set(ranks)):
        raise ValueError(f"invalid ranks {ranks!r} in range report")

    step_owner: dict[int, dict[str, Any]] = {}
    rss_by_step: dict[int, int] = {}
    duplicates: list[int] = []

    for rep in reports:
        if rep.get("status") not in ("PASS", "FAIL"):
            raise ValueError(f"cannot merge non-scored report with status={rep.get('status')!r}")
        if rep.get("production_entry") != "legoesm.ice.si3_o17.si3_o17_step":
            raise ValueError(f"mismatched production_entry {rep.get('production_entry')!r}")
        if rep.get("execution") != execution:
            raise ValueError(f"mismatched execution mode {rep.get('execution')!r} != {execution!r}")
        if list(rep.get("ranks", [])) != ranks:
            raise ValueError(f"mismatched ranks {rep.get('ranks')!r} != {ranks!r}")
        rep_labels = rep.get("labels", {})
        if list(rep_labels.keys()) != labels:
            raise ValueError(f"range report labels must be {labels}, got {list(rep_labels.keys())}")
        rep_steps = rep.get("steps", [])
        if not isinstance(rep_steps, list) or not rep_steps:
            raise ValueError("range report has empty or invalid steps list")
        rep_rss = rep.get("scorer_rss_kib_after_each_step", [])
        if len(rep_rss) != len(rep_steps):
            raise ValueError("range report rss list length disagrees with its steps list")
        expected_keys = {f"{kt}:{rank}" for kt in rep_steps for rank in ranks}
        for label in labels:
            step_map = rep_labels[label].get("steps", {})
            if set(step_map.keys()) != expected_keys:
                raise ValueError(
                    f"range report step keys {sorted(step_map.keys())[:4]} disagree with steps {rep_steps}"
                )
        for kt, rss_val in zip(rep_steps, rep_rss, strict=True):
            if not isinstance(kt, int):
                raise ValueError(f"non-integer step {kt!r} in range report")
            if kt in step_owner:
                duplicates.append(kt)
            else:
                step_owner[kt] = rep
                rss_by_step[kt] = int(rss_val)

    if duplicates:
        raise ValueError(f"duplicate O17 step(s) in merge: {sorted(set(duplicates))}")

    present_steps = sorted(step_owner.keys())
    if present_steps != expected_steps:
        missing = sorted(set(expected_steps) - set(present_steps))
        extra = sorted(set(present_steps) - set(expected_steps))
        raise ValueError(
            f"O17 merge requires steps 1..{expect_steps}; missing={missing}, extra={extra}"
        )

    per_label: dict[str, Any] = {}
    for label in labels:
        steps_dict: dict[str, dict[str, Any]] = {}
        for kt in expected_steps:
            rep = step_owner[kt]
            for rank in ranks:
                key = f"{kt}:{rank}"
                raw_row = rep["labels"][label]["steps"][key]
                fields: dict[str, Any] = {}
                for fname, frow in raw_row["fields"].items():
                    checked = dict(frow)
                    checked["pass"] = _passes(checked, label)
                    fields[fname] = checked
                row = {
                    "pass": all(r["pass"] for r in fields.values()),
                    "different": sum(r["different"] for r in fields.values()),
                    "nonfinite": sum(r["nonfinite"] for r in fields.values()),
                    "failed_fields": [k for k, r in fields.items() if not r["pass"]],
                    "fields": fields,
                    "reused_bitwise_identical_entry": bool(raw_row["reused_bitwise_identical_entry"]),
                }
                steps_dict[key] = row
        per_label[label] = _summarize_label(steps_dict, label)

    status = "PASS" if all(per_label[label]["status"] == "PASS" for label in labels) else "FAIL"
    return {
        "status": status,
        "labels": per_label,
        "steps": expected_steps,
        "ranks": ranks,
        "execution": execution,
        "production_entry": "legoesm.ice.si3_o17.si3_o17_step",
        "scorer_wall_seconds": sum(float(rep["scorer_wall_seconds"]) for rep in reports),
        "scorer_rss_kib_after_each_step": [rss_by_step[kt] for kt in expected_steps],
        "scorer_peak_rss_kib": max(int(rep["scorer_peak_rss_kib"]) for rep in reports),
    }


def score(
    evidence: Path,
    labels: list[str],
    *,
    execution: str = "eager",
    expect_steps: int = 100,
    step_range: tuple[int, int] | None = None,
    card=None,
    runner=None,
) -> dict[str, Any]:
    started = time.monotonic()
    if list(labels) != list(VALID_LABELS):
        raise ValueError(f"labels must be exactly {VALID_LABELS}")
    run = evidence / "record_100step"
    index = record_paths(run / "o17_master_records")
    ranks = sorted({rank for _, rank in index})
    all_steps = sorted({kt for kt, _ in index})
    if all_steps != list(range(1, expect_steps + 1)):
        raise ValueError(f"O17 requires steps 1..{expect_steps}, got {all_steps[:3]}..{all_steps[-3:]}")
    if step_range is None:
        steps = all_steps
    else:
        start, stop = step_range
        if start < 1 or stop < start or stop > expect_steps:
            raise ValueError(f"O17 step range {start}..{stop} outside 1..{expect_steps}")
        steps = list(range(start, stop + 1))
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = card if card is not None else card_for(run)
    runner = runner if runner is not None else make_runner(card, execution)
    per_label: dict[str, Any] = {label: {"steps": {}} for label in labels}
    independent: dict[int, SI3O17StepState] = {}
    if steps[0] > 1:
        for rank in ranks:
            prev_records = load_step(index[(steps[0] - 1, rank)], steps[0] - 1, rank)
            independent[rank] = exit_state(prev_records, card)
            del prev_records
    rss_after_step: list[int] = []
    for kt in steps:
        for rank in ranks:
            records = load_step(index[(kt, rank)], kt, rank)
            given_entry = entry_state(records, card)
            forcing = live_forcing(records, card)
            given = runner(given_entry, forcing)
            entries = {"GIVEN-NEMO-ENTRY": given_entry,
                       "INDEPENDENT": given_entry if kt == 1 else independent[rank]}
            for label in labels:
                entry = entries[label]
                reused = label == "INDEPENDENT" and _same_bits(entry, given_entry)
                out, trace = given if reused else runner(entry, forcing)
                row = score_step(out, trace, records, card, label)
                row["reused_bitwise_identical_entry"] = reused
                per_label[label]["steps"][f"{kt}:{rank}"] = row
                if label == "INDEPENDENT":
                    independent[rank] = out
                del out, trace, entry
            # Only the INDEPENDENT state crosses a step; records and traces are dropped.
            del records, given, given_entry, forcing, entries
        if execution == "eager":
            # JAX's eager per-primitive executable cache gained ~640 entries per
            # step even when the same step was scored again (receipt Resume 5,
            # item 5); it is bounded only by a 4096-entry LRU.  Drop it each step.
            jax.clear_caches()
        gc.collect()
        _release_free_heap()
        rss_after_step.append(_rss_kib())
    for label in labels:
        rows = list(per_label[label]["steps"].values())
        per_label[label].update(
            status="PASS" if all(row["pass"] for row in rows) else "FAIL",
            different=sum(row["different"] for row in rows),
            nonfinite=sum(row["nonfinite"] for row in rows),
            max_relative=max((r["max_relative"] for row in rows for r in row["fields"].values()),
                             default=0.0),
            ulp_bar=0, trajectory_relative_bar=TRAJECTORY_BAR,
        )
    status = "PASS" if all(per_label[label]["status"] == "PASS" for label in labels) else "FAIL"
    return {"status": status, "labels": per_label, "steps": steps, "ranks": ranks,
            "execution": execution, "production_entry": "legoesm.ice.si3_o17.si3_o17_step",
            "scorer_wall_seconds": time.monotonic() - started,
            "scorer_rss_kib_after_each_step": rss_after_step,
            "scorer_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def free_run(
    evidence: Path,
    *,
    execution: str = "eager",
    expect_steps: int = 100,
    n_steps: int | None = None,
    card=None,
    runner=None,
    post_step_hook: Callable[[int, int, SI3O17StepState], SI3O17StepState] | None = None,
) -> dict[str, Any]:
    """Run ``si3_o17_step`` freely for ``n_steps`` from step 1's recorded entry state.

    Step 1 starts from NEMO's recorded state before step 1 (``entry_state``).
    Every subsequent step takes its carried ice state from our previous step's
    output (``carry_step_state``); only external inputs (atmospheric forcing,
    prescribed SAS ocean fields, grid/masks, time ``kt``) come from the record
    via ``external_forcing``.  Every recorded output field is compared after
    every step with NEMO bit for bit (signed zeros and NaN bits included),
    without stopping at the first difference.
    """
    started = time.monotonic()
    run = evidence / "record_100step"
    index = record_paths(run / "o17_master_records")
    ranks = sorted({rank for _, rank in index})
    all_steps = sorted({kt for kt, _ in index})
    if all_steps != list(range(1, expect_steps + 1)):
        raise ValueError(f"O17 requires steps 1..{expect_steps}, got {all_steps[:3]}..{all_steps[-3:]}")
    target_steps = expect_steps if n_steps is None else n_steps
    if target_steps < 1 or target_steps > expect_steps:
        raise ValueError(f"O17 free_run requires 1 <= n_steps <= {expect_steps}, got {target_steps}")
    steps = list(range(1, target_steps + 1))

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = card if card is not None else card_for(run)
    runner = runner if runner is not None else make_runner(card, execution)
    carried: dict[int, SI3O17StepState] = {}
    per_step: dict[str, Any] = {}
    first_difference: dict[str, Any] | None = None
    rss_after_step: list[int] = []

    for kt in steps:
        for rank in ranks:
            records = load_step(index[(kt, rank)], kt, rank)
            full, h = _grid(records[22], card)
            if kt == 1:
                entry = entry_state(records, card)
                carry_mismatches: dict[str, dict[str, Any]] = {}
            else:
                prev_out = carried[rank]
                entry = carry_step_state(prev_out, records, card)
                carry_mismatches = _carried_mismatches(entry, prev_out, full, h)
            forcing = external_forcing(records, card)
            out, trace = runner(entry, forcing)
            scored = score_step(out, trace, records, card, "GIVEN-NEMO-ENTRY")
            fields: dict[str, Any] = {}
            if carry_mismatches:
                fields.update(carry_mismatches)
            fields.update(scored["fields"])
            failed_fields = [key for key, frow in fields.items() if not frow["pass"]]
            differing_per_field = {key: int(frow["different"]) for key, frow in fields.items()}
            step_pass = len(failed_fields) == 0 and not carry_mismatches
            step_key = f"{kt}:{rank}"
            if first_difference is None and not step_pass:
                first_field = failed_fields[0]
                frow = fields[first_field]
                first_difference = {
                    "step": kt,
                    "rank": rank,
                    "step_key": step_key,
                    "field": first_field,
                    "different": int(frow["different"]),
                    "nonfinite": int(frow["nonfinite"]),
                    "max_relative": float(frow["max_relative"]),
                }
            per_step[step_key] = {
                "pass": step_pass,
                "carry_chain_intact": len(carry_mismatches) == 0,
                "different": sum(frow["different"] for frow in fields.values()),
                "nonfinite": sum(frow["nonfinite"] for frow in fields.values()),
                "failed_fields": failed_fields,
                "differing_values_per_field": differing_per_field,
                "fields": fields,
            }
            carried[rank] = post_step_hook(kt, rank, out) if post_step_hook is not None else out
            del out, trace, entry, records, forcing
        if execution == "eager":
            jax.clear_caches()
        gc.collect()
        _release_free_heap()
        rss_after_step.append(_rss_kib())

    rows = list(per_step.values())
    status = "PASS" if all(row["pass"] for row in rows) else "FAIL"
    return {
        "status": status,
        "mode": "FREE_RUN",
        "steps": steps,
        "ranks": ranks,
        "execution": execution,
        "production_entry": "legoesm.ice.si3_o17.si3_o17_step",
        "ulp_bar": 0,
        "different": sum(row["different"] for row in rows),
        "nonfinite": sum(row["nonfinite"] for row in rows),
        "max_relative": max(
            (r["max_relative"] for row in rows for r in row["fields"].values()),
            default=0.0,
        ),
        "first_difference": first_difference,
        "per_step": per_step,
        "scorer_wall_seconds": time.monotonic() - started,
        "scorer_rss_kib_after_each_step": rss_after_step,
        "scorer_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def schema() -> dict[str, Any]:
    return {"status": "PASS", "magic": MAGIC.decode(), "version": VERSION, "kinds": KINDS,
            "state_fields": STATE_FIELDS, "flux_fields": FLUX_FIELDS, "labels": VALID_LABELS,
            "bars": {"local_ulp": 0, "trajectory_relative": TRAJECTORY_BAR}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--labels", nargs=2, default=list(VALID_LABELS))
    parser.add_argument("--execution", choices=("jit", "eager"), default="eager")
    parser.add_argument("--expect-steps", type=int, default=100)
    parser.add_argument("--steps", type=parse_step_range, default=None,
                        help="1-based inclusive step range START..END (e.g. 1..20)")
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="Merge step-range report JSON files into a full report")
    parser.add_argument("--free-run", action="store_true",
                        help="Run carried-state free run from step 1 entry")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Number of steps N for --free-run (defaults to --expect-steps)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write JSON report")
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--read-record", type=Path)
    args = parser.parse_args()
    try:
        if args.schema:
            result = schema()
        elif args.read_record:
            record = read_record(args.read_record)
            result = {"status": "PASS", "meta": record["meta"], "fields": sorted(record["fields"])}
        elif args.merge:
            loaded = [json.loads(path.read_text()) for path in args.merge]
            result = merge_reports(loaded, expect_steps=args.expect_steps)
        elif args.evidence and args.free_run:
            result = free_run(
                args.evidence.resolve(),
                execution=args.execution,
                expect_steps=args.expect_steps,
                n_steps=args.n_steps,
            )
        elif args.evidence:
            result = score(
                args.evidence.resolve(),
                args.labels,
                execution=args.execution,
                expect_steps=args.expect_steps,
                step_range=args.steps,
            )
        else:
            raise ValueError("one of --schema, --read-record, --merge, or --evidence is required")
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
    except (OSError, ValueError, KeyError, struct.error, json.JSONDecodeError) as exc:
        err_text = json.dumps({"status": "REFUSE", "reason": str(exc)}, indent=2)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(err_text + "\n")
        print(err_text)
        return 2
    print(text)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
