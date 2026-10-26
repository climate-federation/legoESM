#!/usr/bin/env python3
"""Fail-closed reader and per-rung projection gate for the SI3 O8--O16 master record.

Binary/log/provenance helpers are imported from the admitted O5 and O7 tools.
This file owns only the new stage registry and its variable-extent record format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o5_orca2_sas_gate import (
    SameProgramError,
    _producer_is_ancestor_or_rebased_equivalent,
    _verify_run_completion_and_logs,
    max_ulp_diff,
    same_program_observers,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o7_orca2_sas_gate import (
    derive_sas_time_level,
    verify_o7_log,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NEMO_ROOT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
MAGIC = b"O816M001"
VERSION = 1
N_META = 20
SCALAR_NAMES = (
    "rDt_ice", "r1_Dt_ice", "rn_beta", "rn_dmin", "rn_hinew", "picard_iterations",
    "rcnd_s", "rn_kappa_i", "rn_kappa_s", "rn_snwblow", "rn_sinew", "nn_icesal",
    "r1_nlay_i", "r1_nlay_s", "rn_simin", "rn_himin",
    "ln_drainage", "ln_flushing", "nn_drainage", "nn_flushing",
    "rn_sal_himin", "rn_flushrate", "rn_vbrc", "nn_sal_scheme",
    "rn_alpha_RJW", "rn_Rc_RJW", "nice_pnd", "rn_apnd_min", "rn_apnd_max",
    "rn_pnd_flush", "ln_pnd_lids", "ln_pnd_rain",
    "rn_icesal", "rn_sal_gd", "rn_time_gd", "rn_sal_fl", "rn_time_fl",
    "nn_liquidus", "rn_alpha_GN", "rn_Rc_GN", "rn_alpha_CW", "ln_sal_chk",
    "ln_pnd", "ln_pnd_LEV", "ln_pnd_CST", "ln_pnd_TOPO", "rn_apnd",
    "rn_hpnd", "ln_pnd_alb", "rn_pnd_hl_min", "rn_pnd_hl_max",
    "rn_alb_sdry", "rn_alb_smlt", "rn_alb_idry", "rn_alb_imlt",
    "rn_alb_dpnd", "rn_alb_lpnd", "rn_alb_hpiv",
)
N_SCALARS = len(SCALAR_NAMES)
HEADER_BYTES = 8 + 4 + 4 * N_META + 8 * N_SCALARS
STEPS = tuple(range(1, 101))
PICARD_STEPS = (1, 2, 100)


@dataclass(frozen=True)
class Field:
    name: str
    extent: str
    dtype: str = "f8"


def _fields(names: str, extent: str, dtype: str = "f8") -> tuple[Field, ...]:
    return tuple(Field(name, extent, dtype) for name in names.split())


CAT_BASE = _fields(
    "a_i at_i h_i h_s t_su s_i t_bo sss sst qns_ice qsr_ice dqns_ice "
    "qtr_ice_top qsb_ice_bot fhld frq_m sprecip qprec_ice evap_ice "
    "hfx_err_dif hfx_dif qml_ice qcn_ice sfx_bri sfx_lam wfx_lam o_i",
    "G1",
) + _fields("t_i e_i sz_i", "GI") + _fields("t_s e_s", "GS")

DH_DIAGS = _fields(
    "dh_s_tot dh_i_sum dh_i_itm dh_i_bom dh_i_bog dh_i_sub "
    "dh_s_sum dh_s_itm dh_snowice",
    "G1",
)
DH_FLUXES = _fields(
    "hfx_res wfx_snw_sum hfx_spr wfx_spr hfx_sub wfx_snw_sub hfx_snw "
    "wfx_res sfx_res hfx_thd hfx_sum wfx_sum sfx_sum wfx_ice_sub sfx_sub "
    "wfx_err_sub hfx_bog wfx_bog sfx_bog hfx_bom wfx_bom sfx_bom sfx_sni "
    "wfx_sni wfx_snw_sni",
    "G1",
)

STATE_I3 = _fields(
    "a_i v_i v_s h_i h_s s_i sv_i o_i oa_i t_su "
    "a_ip v_ip h_ip v_il h_il a_ip_frac a_ip_eff",
    "I3",
)
STATE_II4 = _fields("t_i e_i sz_i szv_i", "II4")
STATE_IS4 = _fields("t_s e_s", "IS4")
STATE = STATE_I3 + STATE_II4 + STATE_IS4
TOTALS = _fields(
    "at_i ato_i vt_i vt_s st_i et_i et_s tm_i tm_s sm_i tm_su hm_i hm_s "
    "om_i at_ip at_ip_eff hm_ip vt_ip hm_il vt_il tm_si",
    "I2",
)
POND = _fields("dh_i_sum_2d dh_s_sum_2d", "I3")
GLOBAL_AUX = (
    _fields("h_i_b a_i_b", "I3")
    + _fields("t_bo ht_i_new fraz_frac sss_m sst_m rn_amax_2d", "I2")
    + _fields("tmask_k1", "F2")
    + _fields("smask0", "I2")
    + _fields("hi_max", "C0")
    + _fields("hi_mean", "C1")
)
POND_FORCING = _fields("sprecip tprecip", "I2") + _fields("qsr_ice qns_ice", "I3")
FLUX = _fields(
    "wfx_snw wfx_snw_sni wfx_snw_sum wfx_pnd wfx_spr wfx_sub wfx_snw_sub "
    "wfx_ice_sub wfx_ice wfx_sni wfx_opw wfx_bog wfx_dyn wfx_bom wfx_sum "
    "wfx_lam wfx_res wfx_err_sub sfx_bog sfx_bom sfx_lam sfx_sum sfx_sni "
    "sfx_opw sfx_bri sfx_dyn sfx_res sfx_sub hfx_bog hfx_bom hfx_sum hfx_opw "
    "hfx_dif hfx_snw hfx_err_dif hfx_sub hfx_spr hfx_thd hfx_dyn hfx_res "
    "qlead qsb_ice_bot fhld qt_atm_oi qt_oce_ai",
    "I2",
)


@dataclass(frozen=True)
class Kind:
    label: str
    fields: tuple[Field, ...]
    category: bool = False


KINDS: dict[int, Kind] = {
    1: Kind(
        "frazil_entry",
        _fields("qlead smask0", "I2")
        + _fields("at_i u_ice v_ice umask_k1 vmask_k1 utau_ice vtau_ice", "F2")
        + _fields("hi_max", "C0"),
    ),
    2: Kind("frazil_exit", _fields("ht_i_new fraz_frac", "I2")),
    3: Kind("category_entry", _fields("nptidx", "G1", "i4") + CAT_BASE + DH_DIAGS + DH_FLUXES, True),
    4: Kind(
        "zdf_exit",
        _fields("nptidx", "G1", "i4")
        + _fields("t_su qns_ice qcn_ice_top qcn_ice_bot qtr_ice_bot hfx_err_dif hfx_dif cnd_ice t1_ice t_si", "G1")
        + _fields("t_i e_i", "GI")
        + _fields("t_s e_s", "GS")
        + _fields("o_i", "G1"),
        True,
    ),
    5: Kind(
        "dh_exit",
        _fields("nptidx", "G1", "i4") + CAT_BASE + DH_DIAGS + DH_FLUXES,
        True,
    ),
    6: Kind("temp1_exit_sal_entry", _fields("nptidx", "G1", "i4") + CAT_BASE + DH_DIAGS + DH_FLUXES, True),
    7: Kind("sal_exit", _fields("nptidx", "G1", "i4") + CAT_BASE + DH_DIAGS + DH_FLUXES, True),
    8: Kind("temp2_exit_da_entry", _fields("nptidx", "G1", "i4") + CAT_BASE + DH_DIAGS + DH_FLUXES, True),
    9: Kind("da_exit", _fields("nptidx", "G1", "i4") + CAT_BASE + DH_DIAGS + DH_FLUXES, True),
    10: Kind("pond_entry_category_done", STATE + TOTALS + POND + FLUX + GLOBAL_AUX + POND_FORCING),
    11: Kind("pond_exit_itd_entry", STATE + TOTALS + POND + FLUX + GLOBAL_AUX + POND_FORCING),
    12: Kind("itd_exit_do_entry", STATE + TOTALS + POND + FLUX + GLOBAL_AUX + POND_FORCING),
    13: Kind("do_exit_cor_entry", STATE + TOTALS + POND + FLUX + GLOBAL_AUX + POND_FORCING),
    14: Kind("cor_exit", STATE + TOTALS + POND + FLUX + GLOBAL_AUX + POND_FORCING),
    15: Kind(
        "update_flx_entry",
        STATE + TOTALS + FLUX
        + _fields("qsr_oce qns_oce qemp_oce qemp_ice emp_oce emp_ice qns_tot qsr_tot frq_m", "I2")
        + _fields("a_i_b qsr_ice qns_ice qtr_ice_bot qevap_ice", "I3")
        + _fields("qcn_ice qml_ice qtr_ice_top", "I3")
        + _fields("at_i_b wfx_snw_dyn cloud_fra", "I2")
        + _fields("tmask_k1", "F2")
        + _fields("smask0", "I2")
        + _fields("snwice_mass snwice_mass_b", "F2"),
    ),
    16: Kind(
        "update_flx_exit_ocean_boundary",
        _fields(
            "qsr qns emp sfx fwfice qt_atm_oi qt_oce_ai wfx_ice wfx_snw wfx_pnd",
            "I2",
        )
        + _fields("fr_i", "F2")
        + _fields("tn_ice alb_ice", "I3")
        + _fields("snwice_mass snwice_mass_b snwice_fmass", "F2"),
    ),
    17: Kind(
        "picard_final",
        _fields("nptidx", "G1", "i4")
        + _fields("l_t_converged", "G1", "i4")
        + _fields("t_su qns_ice", "G1")
        + _fields("t_i", "GI")
        + _fields("t_s", "GS"),
        True,
    ),
    18: Kind(
        "pre_aggregation",
        STATE,
    ),
}

# The final Picard record also carries the final scratch state of NEMO's
# tridiagonal solve.  These arrays are passed directly by icethd_zdf_bl99.
KINDS[17] = Kind(
    "picard_final",
    KINDS[17].fields
        + _fields("zkappa_i", "GI0")
        + _fields("zradab_i", "GI")
        + _fields("zradtr_i", "GI0")
        + _fields("zkappa_s", "GS0")
        + _fields("zradab_s", "GS")
        + _fields("zradtr_s", "GS0"),
        True,
)

KINDS[19] = Kind(
    "tau_entry",
    _fields("uu_kbb vv_kbb umask_k1 vmask_k1 u_ice v_ice u_oce v_oce drag_io at_i utau vtau", "F2")
    + _fields("taum", "I2"),
)
KINDS[20] = Kind("tau_exit", _fields("utau vtau", "F2") + _fields("taum", "I2") + _fields("rCdU_ice", "F2"))
KINDS[21] = Kind("pre_category_state", STATE)

PROJECTIONS: dict[str, tuple[int, ...]] = {
    "O8": (1, 2),
    "O9": (21, 3, 4, 5, 6, 17),
    "O10": (21, 6, 7, 8),
    "O11": (21, 8, 9),
    "O12": (10, 11),
    "O13": (11, 12),
    "O14": (12, 13),
    "O15": (13, 14),
    "O16": (18, 15, 16, 19, 20),
}


def _refuse(message: str) -> ValueError:
    return ValueError(f"STATUS PLANT-FIRED: {message}")


def _shape(field: Field, meta: dict[str, int]) -> tuple[int, ...]:
    ni = meta["ntei"] - meta["ntsi"] + 1
    nj = meta["ntej"] - meta["ntsj"] + 1
    dims = {
        "I2": (ni, nj),
        "F2": (meta["jpi"], meta["jpj"]),
        "I3": (ni, nj, meta["jpl"]),
        "II4": (ni, nj, meta["nlay_i"], meta["jpl"]),
        "IS4": (ni, nj, meta["nlay_s"], meta["jpl"]),
        "G1": (meta["npti"],),
        "GI": (meta["npti"], meta["nlay_i"]),
        "GS": (meta["npti"], meta["nlay_s"]),
        "GI0": (meta["npti"], meta["nlay_i"] + 1),
        "GS0": (meta["npti"], meta["nlay_s"] + 1),
        "C0": (meta["jpl"] + 1,),
        "C1": (meta["jpl"],),
    }
    try:
        return dims[field.extent]
    except KeyError as exc:
        raise _refuse(f"unknown extent code {field.extent}") from exc


def record_name(kind: int, kt: int, jl: int = 0, rank: int = 0) -> str:
    suffix = f"_cat{jl:02d}" if KINDS[kind].category else ""
    return f"o8o16_kind{kind:02d}_{KINDS[kind].label}{suffix}_step{kt:04d}_rank{rank:04d}.bin"


def read_master_record(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Read one master record using only declared extents and the registered schema."""
    p = Path(path)
    raw = bytearray(p.read_bytes())
    if plant == "truncation" and raw:
        del raw[-1]
    if plant == "header" and raw:
        raw[0] ^= 1
    if len(raw) < HEADER_BYTES or bytes(raw[:8]) != MAGIC:
        raise _refuse(f"bad/truncated master header in {p}")
    version = struct.unpack_from("<i", raw, 8)[0]
    if version != VERSION:
        raise _refuse(f"unsupported master version {version}")
    vals = tuple(map(int, np.frombuffer(raw, dtype="<i4", count=N_META, offset=12)))
    keys = (
        "kind", "kt", "jl", "rank", "jpi", "jpj", "jpl", "nlay_i", "nlay_s",
        "ntsi", "ntei", "ntsj", "ntej", "npti", "nfields", "endian", "nit000",
        "kmm", "forcing_mode", "dynamics_mode",
    )
    meta = dict(zip(keys, vals, strict=True))
    kind = meta["kind"]
    if kind not in KINDS or meta["nfields"] != len(KINDS[kind].fields):
        raise _refuse(f"unknown kind/field count {kind}/{meta['nfields']}")
    if (
        meta["jpl"] != 5
        or meta["nlay_i"] != 10
        or meta["nlay_s"] != 5
        or meta["endian"] != 1
        or meta["jpi"] < 1
        or meta["jpj"] < 1
        or not (1 <= meta["ntsi"] <= meta["ntei"] <= meta["jpi"])
        or not (1 <= meta["ntsj"] <= meta["ntej"] <= meta["jpj"])
        or not (0 <= meta["npti"] <= meta["jpi"] * meta["jpj"])
        or meta["forcing_mode"] != 1
        or meta["dynamics_mode"] != 1
    ):
        raise _refuse(f"invalid master metadata {meta}")
    if KINDS[kind].category != (meta["jl"] > 0):
        raise _refuse(f"category/header mismatch for kind {kind}")
    if meta["jl"] not in (0, 1, 2, 3, 4, 5):
        raise _refuse(f"invalid category {meta['jl']}")
    expected = record_name(kind, meta["kt"], meta["jl"], meta["rank"])
    if p.name != expected:
        raise _refuse(f"filename/header disagreement: {p.name} != {expected}")
    scalars = np.frombuffer(raw, dtype="<f8", count=N_SCALARS, offset=8 + 4 + 4 * N_META).copy()
    if not np.all(np.isfinite(scalars)):
        raise _refuse(f"nonfinite scalar header in {p}")
    offset = HEADER_BYTES
    fields: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    specs = list(KINDS[kind].fields)
    if plant == "swapped-field" and len(specs) > 1:
        specs[0], specs[1] = specs[1], specs[0]
    for spec in specs:
        shape = _shape(spec, meta)
        dtype = np.dtype("<i4" if spec.dtype == "i4" else "<f8")
        count = int(np.prod(shape))
        stop = offset + count * dtype.itemsize
        if stop > len(raw):
            raise _refuse(f"truncated field {spec.name} in {p}")
        value = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(shape, order="F").copy()
        if spec.dtype == "f8" and not np.all(np.isfinite(value)):
            raise _refuse(f"nonfinite field {spec.name} in {p}")
        fields[spec.name] = value
        digests[spec.name] = hashlib.sha256(raw[offset:stop]).hexdigest()
        offset = stop
    if offset != len(raw):
        raise _refuse(f"trailing bytes in {p}: {len(raw) - offset}")
    if kind == 3:
        nonzero = [field.name for field in DH_DIAGS if np.any(fields[field.name] != 0.0)]
        if nonzero:
            raise _refuse(f"category-entry diagnostics were not zeroed: {nonzero}")
    return {
        "meta": meta,
        "scalars": dict(zip(SCALAR_NAMES, map(float, scalars), strict=True)),
        "fields": fields,
        "sha256": digests,
        "record_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _expected_projection(rung: str) -> set[tuple[int, int, int]]:
    expected: set[tuple[int, int, int]] = set()
    for kind in PROJECTIONS[rung]:
        steps = PICARD_STEPS if kind == 17 else STEPS
        cats = range(1, 6) if KINDS[kind].category else (0,)
        expected.update((kind, kt, jl) for kt in steps for jl in cats)
    return expected


def validate_projection_index(
    entries: Iterable[tuple[int, int, int]], rung: str, plant: str | None = None
) -> dict[str, Any]:
    """Prove that one rung sees exactly its registered kinds, steps and categories."""
    if rung not in PROJECTIONS:
        raise _refuse(f"unknown projection {rung}")
    got = set(entries)
    expected = _expected_projection(rung)
    if plant == "missing-step":
        got.discard(next(iter(sorted(expected))))
    elif plant == "foreign-kind":
        got.add((99, 1, 0))
    elif plant == "wrong-category":
        one = next(row for row in sorted(expected) if row[2] > 0)
        got.remove(one)
        got.add((one[0], one[1], 0))
    if got != expected:
        missing = sorted(expected - got)[:4]
        extra = sorted(got - expected)[:4]
        raise _refuse(f"{rung} projection mismatch missing={missing} extra={extra}")
    return {
        "rung": rung,
        "kinds": list(PROJECTIONS[rung]),
        "records": len(expected),
        "steps": 100,
        "forcing_disposition": "O7 exact surface products feed O8-O16 stage views",
        "production_dynamics_disposition": "unmodified production deck; dynamics outputs are outside every projection",
    }


def projection_from_directory(root: Path, rung: str) -> dict[str, Any]:
    entries = []
    files = []
    for path in sorted(root.glob("*.bin")):
        match = re.match(r"o8o16_kind(\d{2})_", path.name)
        if match is None or int(match.group(1)) not in PROJECTIONS[rung]:
            continue
        rec = read_master_record(path)
        meta = rec["meta"]
        entries.append((meta["kind"], meta["kt"], meta["jl"]))
        files.append({"path": path.name, "sha256": rec["record_sha256"]})
    result = validate_projection_index(entries, rung)
    result["files"] = files
    return result


def field_bytes(field: Field, *, jpi: int, jpj: int, ni: int, nj: int, npti: int) -> int:
    meta = dict(jpi=jpi, jpj=jpj, jpl=5, nlay_i=10, nlay_s=5,
                ntsi=1, ntei=ni, ntsj=1, ntej=nj, npti=npti)
    return int(np.prod(_shape(field, meta))) * (4 if field.dtype == "i4" else 8)


def size_report(jpi: int = 182, jpj: int = 149, ni: int = 180, nj: int = 147) -> dict[str, Any]:
    """Worst-case sizing: every category gathers every declared interior cell."""
    npti = ni * nj
    kinds: dict[str, Any] = {}
    total = 0
    for kind, spec in KINDS.items():
        per_record = HEADER_BYTES + sum(
            field_bytes(field, jpi=jpi, jpj=jpj, ni=ni, nj=nj, npti=npti)
            for field in spec.fields
        )
        nsteps = len(PICARD_STEPS) if kind == 17 else len(STEPS)
        ncats = 5 if spec.category else 1
        value = per_record * nsteps * ncats
        total += value
        kinds[str(kind)] = {
            "label": spec.label,
            "steps": nsteps,
            "categories": ncats,
            "fields": len(spec.fields),
            "bytes_per_record_worst_case": per_record,
            "bytes_total_worst_case": value,
        }
    projections = {}
    for rung, projection in PROJECTIONS.items():
        value = sum(kinds[str(kind)]["bytes_total_worst_case"] for kind in projection)
        projections[rung] = {"kinds": list(projection), "bytes_worst_case": value}
    return {
        "grid": {"jpi": jpi, "jpj": jpj, "interior_i": ni, "interior_j": nj,
                 "jpl": 5, "nlay_i": 10, "nlay_s": 5},
        "assumption": "worst case npti=interior_i*interior_j for every category",
        "header_bytes_per_record": HEADER_BYTES,
        "kinds": kinds,
        "projections": projections,
        "master_bytes_worst_case": total,
        "master_gib_worst_case": total / 2**30,
        "retention": (
            "retain immutable raw master plus manifests until O8-O16 all admit; "
            "publish per-rung relative-path manifests without copying shared boundary files"
        ),
    }


def _synthetic_record(
    path: Path, kind: int, kt: int = 1, jl: int | None = None, *, nonzero_entry_dh: bool = False
) -> None:
    spec = KINDS[kind]
    jl = 1 if jl is None and spec.category else (0 if jl is None else jl)
    meta = dict(kind=kind, kt=kt, jl=jl, rank=0, jpi=6, jpj=6, jpl=5,
                nlay_i=10, nlay_s=5, ntsi=2, ntei=5, ntsj=2, ntej=5,
                npti=3 if spec.category else 0, nfields=len(spec.fields), endian=1,
                nit000=1, kmm=2, forcing_mode=1, dynamics_mode=1)
    ordered = (
        "kind", "kt", "jl", "rank", "jpi", "jpj", "jpl", "nlay_i", "nlay_s",
        "ntsi", "ntei", "ntsj", "ntej", "npti", "nfields", "endian", "nit000",
        "kmm", "forcing_mode", "dynamics_mode",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", VERSION))
        f.write(np.asarray([meta[key] for key in ordered], dtype="<i4").tobytes())
        scalar_values = np.zeros(N_SCALARS, dtype="<f8")
        scalar_values[:6] = [10800.0, 1 / 10800.0, 1.0, 8.0, 0.1, 0.0]
        f.write(scalar_values.tobytes())
        seed = 1
        for field in spec.fields:
            shape = _shape(field, meta)
            dtype = np.dtype("<i4" if field.dtype == "i4" else "<f8")
            data = np.arange(seed, seed + int(np.prod(shape)), dtype=dtype).reshape(shape, order="F")
            if kind == 3 and field in DH_DIAGS and not nonzero_entry_dh:
                data.fill(0)
            if field.dtype == "f8" and seed == 1:
                data.flat[0] = -0.0
            f.write(data.tobytes(order="F"))
            seed += data.size


def self_test(plant: str | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = []
        for kind in KINDS:
            path = root / record_name(kind, 1, 1 if KINDS[kind].category else 0)
            _synthetic_record(path, kind)
            paths.append(path)
        if plant == "category-dh":
            target = root / record_name(3, 1, 1)
            _synthetic_record(target, 3, nonzero_entry_dh=True)
        else:
            target = paths[2] if plant == "swapped-field" else paths[0]
        rec = read_master_record(target, plant=plant if plant in {"header", "truncation", "swapped-field"} else None)
        if plant == "swapped-field":
            raise _refuse("swapped-field plant survived parsing")
        if not np.signbit(rec["fields"][KINDS[1].fields[0].name].flat[0]):
            raise _refuse("signed zero was not retained")
        if max_ulp_diff(np.array([-0.0]), np.array([0.0])) != 0:
            raise _refuse("imported O5 ULP helper changed zero semantics")
        if plant == "one-bit":
            raw = bytearray(target.read_bytes())
            raw[-1] ^= 1
            target.write_bytes(raw)
            reread = read_master_record(target)
            if reread["sha256"] == rec["sha256"]:
                raise _refuse("one-bit payload plant was not visible")
            raise _refuse("one-bit payload plant fired")
    return {"records": len(KINDS), "signed_zero": True, "schema_kinds": len(KINDS)}


def verify_manifest(root: Path, producer: str | None = None) -> dict[str, Any]:
    manifest_path = root / "manifest/run_manifest.json"
    if not manifest_path.is_file():
        raise _refuse("manifest was not written before the run")
    manifest = json.loads(manifest_path.read_text())
    required = {
        "producer_commit", "binary_sha256", "source_sha256", "nemo_source_sha256", "deck_sha256",
        "forcing_disposition", "dynamics_disposition", "record_schema_sha256",
        "estimated_master_bytes", "max_evidence_disk_kib", "wall_clock_budget_seconds",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise _refuse(f"manifest missing keys {missing}")
    if producer is not None and manifest["producer_commit"] != producer:
        raise _refuse("requested producer differs from run manifest")
    if not _producer_is_ancestor_or_rebased_equivalent(manifest["producer_commit"], REPO_ROOT):
        raise _refuse("producer commit is not an ancestor or patch-equivalent")
    if manifest["forcing_disposition"] != "O7_EXACT_PRODUCTS_TO_O8_O16":
        raise _refuse("forcing disposition mismatch")
    if manifest["dynamics_disposition"] != "PRODUCTION_DECK_ON_EXCLUDED_FROM_PROJECTIONS":
        raise _refuse("dynamics disposition mismatch")
    if manifest.get("steps") != 100 or manifest.get("ranks") != 1:
        raise _refuse("master window/rank mismatch")
    if manifest.get("ice_namelist") != "UNMODIFIED_PRODUCTION" or manifest.get("ln_zdf_chkcvg_overlay") is not False:
        raise _refuse("production ice namelist/overlay disposition mismatch")
    if manifest.get("estimated_master_bytes") != size_report()["master_bytes_worst_case"]:
        raise _refuse("manifest storage estimate differs from schema")
    if manifest.get("max_evidence_disk_kib") != 150_000_000:
        raise _refuse("manifest disk-use gate differs from acquisition contract")
    if manifest.get("wall_clock_budget_seconds") != 14_400:
        raise _refuse("manifest wall-clock gate differs from acquisition contract")
    for rel, want in manifest["source_sha256"].items():
        path = REPO_ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != want:
            raise _refuse(f"source digest mismatch: {rel}")
    nemo_ppsrc = DEFAULT_NEMO_ROOT / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo"
    for unit, want in manifest["nemo_source_sha256"].items():
        path = nemo_ppsrc / f"{unit}.f90"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != want:
            raise _refuse(f"NEMO source digest mismatch: {unit}")
    for name, want in manifest["deck_sha256"].items():
        path = root / "manifest/production_deck" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != want:
            raise _refuse(f"deck digest mismatch: {name}")
    for label, want in manifest["binary_sha256"].items():
        path = root / "manifest/binaries" / ("nemo.ref" if label == "reference" else "nemo.rec")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != want:
            raise _refuse(f"binary digest mismatch: {label}")
    return manifest


def admit(root: Path, producer: str | None = None) -> dict[str, Any]:
    manifest = verify_manifest(root, producer)
    if not (root / "RUN_COMPLETE").is_file():
        raise _refuse("missing acquisition RUN_COMPLETE")
    _verify_run_completion_and_logs(
        root / "reference_100step", "reference_100step", expect_icedyn=True, expect_icethd=True
    )
    _verify_run_completion_and_logs(
        root / "record_100step", "record_100step", expect_icedyn=True, expect_icethd=True
    )
    verify_o7_log(root / "record_100step/ocean.output", expected_nn_itend=100)
    for rung in PROJECTIONS:
        result = projection_from_directory(root / "record_100step/o8_o16_master_records", rung)
        retained = root / f"manifest/per_rung/{rung}.json"
        if not retained.is_file():
            raise _refuse(f"missing/stale relative projection manifest for {rung}")
        retained_result = json.loads(retained.read_text())
        if retained_result.get("records") != result["records"] or retained_result.get("files") != result["files"]:
            raise _refuse(f"missing/stale relative projection manifest for {rung}")
    ref_restart = sorted((root / "reference_100step").glob("*restart*.nc"))
    rec_restart = sorted((root / "record_100step").glob("*restart*.nc"))
    if not ref_restart or [p.name for p in ref_restart] != [p.name for p in rec_restart]:
        raise _refuse("reference/record restart sets differ")
    for ref, rec in zip(ref_restart, rec_restart, strict=True):
        if hashlib.sha256(ref.read_bytes()).digest() != hashlib.sha256(rec.read_bytes()).digest():
            raise _refuse(f"observer changed restart bytes: {ref.name}")
    usage_path = root / "manifest/resource_usage.json"
    if not usage_path.is_file():
        raise _refuse("missing measured resource usage")
    usage = json.loads(usage_path.read_text())
    record_bytes = sum(p.stat().st_size for p in (root / "record_100step/o8_o16_master_records").glob("*.bin"))
    if usage.get("record_bytes") != record_bytes or not (0 < record_bytes <= manifest["estimated_master_bytes"]):
        raise _refuse("measured record size missing or over schema gate")
    if not (0 < usage.get("evidence_disk_kib", 0) <= manifest["max_evidence_disk_kib"]):
        raise _refuse("measured evidence disk use exceeds gate")
    if not (0 <= usage.get("wall_clock_seconds", -1) <= manifest["wall_clock_budget_seconds"]):
        raise _refuse("measured wall clock exceeds gate")
    return {"status": "ADMITTED", "manifest": manifest, "projections": list(PROJECTIONS)}


def schema_document() -> dict[str, Any]:
    return {
        "magic": MAGIC.decode(), "version": VERSION, "header_bytes": HEADER_BYTES,
        "scalar_names": list(SCALAR_NAMES),
        "kinds": {str(k): {"label": v.label, "category": v.category,
                           "fields": [field.__dict__ for field in v.fields]}
                  for k, v in KINDS.items()},
        "projections": {key: list(value) for key, value in PROJECTIONS.items()},
    }


def write_stub_records(root: Path) -> dict[str, Any]:
    """Write a complete tiny record index for Checklist 15's stub-NEMO driver."""
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    for kind, spec in KINDS.items():
        steps = PICARD_STEPS if kind == 17 else STEPS
        cats = range(1, 6) if spec.category else (0,)
        for kt in steps:
            for jl in cats:
                _synthetic_record(root / record_name(kind, kt, jl), kind, kt, jl)
                count += 1
    return {"status": "STUB_RECORDS_WRITTEN", "records": count}


def verify_same_program(
    new_bld: Path,
    ref_bld: Path,
    patch_dir: Path,
    expected_ppsrc_dir: Path,
    new_cfg: str,
    ref_cfg: str,
    plant: str | None = None,
) -> dict[str, Any]:
    units = (
        "icestp", "icethd", "icethd_do", "icethd_da", "icethd_sal",
        "icethd_pnd", "icethd_zdf_bl99", "icealb",
    )
    markers = {unit: ("! O8O16_OBSERVER_BEGIN", "! O8O16_OBSERVER_END") for unit in units}
    expected = {unit: expected_ppsrc_dir / f"{unit}.f90" for unit in units}
    if plant == "same-program":
        with tempfile.TemporaryDirectory() as raw:
            planted = Path(raw)
            for unit, source in expected.items():
                target = planted / source.name
                target.write_bytes(source.read_bytes())
            target = planted / "icethd.f90"
            text = target.read_text()
            text = text.replace("CALL o8o16_category(3,jl)", "CALL o8o16_category(4,jl)", 1)
            target.write_text(text)
            expected = {unit: planted / f"{unit}.f90" for unit in units}
            return same_program_observers(
                new_bld, ref_bld,
                {unit: patch_dir / f"{unit}_o8_o16.patch" for unit in units},
                ("o8_o16_master_writer",), new_cfg=new_cfg, ref_cfg=ref_cfg,
                preprocessed_markers=markers, preprocessed_patch_units=expected,
            )
    return same_program_observers(
        new_bld, ref_bld,
        {unit: patch_dir / f"{unit}_o8_o16.patch" for unit in units},
        ("o8_o16_master_writer",), new_cfg=new_cfg, ref_cfg=ref_cfg,
        preprocessed_markers=markers, preprocessed_patch_units=expected,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--schema", action="store_true")
    action.add_argument("--size", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--projection-self-test", choices=sorted(PROJECTIONS))
    action.add_argument("--read-record", type=Path)
    action.add_argument("--read-projection", choices=sorted(PROJECTIONS))
    action.add_argument("--admit-existing", type=Path)
    action.add_argument("--same-program", type=Path, metavar="NEW_BLD")
    action.add_argument("--write-stub-records", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--plant")
    parser.add_argument("--producer")
    parser.add_argument("--nemo-root", type=Path, default=DEFAULT_NEMO_ROOT)
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--patch-dir", type=Path)
    parser.add_argument("--expected-ppsrc-dir", type=Path)
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    args = parser.parse_args(argv)
    try:
        if args.schema:
            result = schema_document()
        elif args.size:
            result = size_report()
        elif args.self_test:
            result = self_test(args.plant)
        elif args.projection_self_test:
            result = validate_projection_index(
                _expected_projection(args.projection_self_test), args.projection_self_test, args.plant
            )
        elif args.read_record:
            rec = read_master_record(args.read_record, args.plant)
            result = {"meta": rec["meta"], "fields": sorted(rec["fields"])}
        elif args.read_projection:
            if args.records is None:
                raise _refuse("--read-projection requires --records")
            result = projection_from_directory(args.records, args.read_projection)
        elif args.same_program:
            if None in (
                args.reference_bld, args.patch_dir, args.expected_ppsrc_dir,
                args.new_cfg, args.reference_cfg,
            ):
                raise _refuse("--same-program needs reference BLD, patch/expected ppsrc dirs and config names")
            result = verify_same_program(
                args.same_program, args.reference_bld, args.patch_dir, args.expected_ppsrc_dir,
                args.new_cfg, args.reference_cfg, args.plant,
            )
        elif args.write_stub_records:
            result = write_stub_records(args.write_stub_records)
        else:
            result = admit(args.admit_existing, args.producer)
        result["sas_time_level"] = derive_sas_time_level(args.nemo_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError, SameProgramError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
