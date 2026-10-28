#!/usr/bin/env python3
"""Fail-closed 0-ULP scoring driver for admitted SI3 production rungs.

Scores the admitted 100-step O8--O16 master record at
``/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master``
in order O8 through O16 against legoESM's production implementations under
``jax.jit`` (CPU ``fp64``, ``PrecisionPolicy.fp64(transcendentals="libm")``,
inputs passed as runtime JIT arguments, and the rung's resolved namelist from
``record_100step/output.namelist.ice`` and ``record_100step/output.namelist.dyn``).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants as canonical_constants
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.bitz_lipscomb import (
    SI3ColumnArrays,
    SI3DHFluxes,
    SI3DHIncrements,
    SI3SurfaceForcing,
    si3_column_step_arrays,
    si3_ice_thd_category_stages,
    si3_ice_thd_dh,
    si3_ice_thd_frazil,
    si3_ice_thd_do,
    si3_ice_thd_o10_category_stages,
    si3_ice_thd_sal,
    si3_ice_thd_temp,
    si3_ice_thd_zdf_bl99,
)
from legoesm.ice.config import SI3LateralMeltConfig, SI3PondConfig, SI3ThermoConfig
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.itd import si3_ice_itd_rem
from legoesm.ice.ponds import step_ponds
from legoesm.ice.si3_thd_da import si3_ice_thd_da
from legoesm.ice.si3_thd_pnd import si3_ice_thd_pnd
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o5_orca2_sas_gate import (
    max_ulp_diff,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_master_gate import (
    CAT_BASE,
    DEFAULT_NEMO_ROOT,
    DH_DIAGS,
    DH_FLUXES,
    HEADER_BYTES,
    KINDS,
    MAGIC,
    N_META,
    N_SCALARS,
    PICARD_STEPS,
    PROJECTIONS,
    SCALAR_NAMES,
    STEPS,
    VERSION,
    Field,
    record_name,
    verify_sas_drag_resolution,
)

DEFAULT_MASTER_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master"
)
DEFAULT_PPSRC = DEFAULT_NEMO_ROOT / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo"

# Pinned admission digests from docs/ice/fidelity/si3_r18_o8_o16_receipt.md:160-176.
PINNED_RUN_MANIFEST_SHA256 = (
    "bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0"
)
PINNED_RESOURCE_USAGE_SHA256 = (
    "b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c"
)
PINNED_RECORDS_INVENTORY_SHA256 = (
    "fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85"
)
PINNED_PROJECTION_SHA256: dict[str, str] = {
    "O8": "303a29d76034c3856e4cf6fa60ed551b7146e5decd3618038b058501c818ed82",
    "O9": "3d97b7bb71c91530a89f60498c8a84316829200e89d8c45470f5a8d1814ee0a3",
    "O10": "ded24ddbc825a9a331d763665e51559022c97288cb7c7f516b3a587693deb554",
    "O11": "0c5197c52331975ff77d813f0154e0df6ea0aa1e0593e9ca062990d664465666",
    "O12": "fe395cab681d59a8c28a722cd7fc8222ec3ac0ee8c4e569d4043643fd4a47e50",
    "O13": "a94d126ddaee3194a053ef437d762707dff93045287392d4e0586aaee995a4aa",
    "O14": "8130e5ac516cf8c1549badbab2caf90cf5cce03f3167c317a10532b836572cd6",
}

SCORE_RUNGS = ("O8", "O9", "O10", "O11", "O12", "O13")
N_PAD_1D = 4099

ZDF_UPDATED_FIELDS = (
    "t_su",
    "qns_ice",
    "qcn_ice_top",
    "qcn_ice_bot",
    "qtr_ice_bot",
    "hfx_err_dif",
    "hfx_dif",
    "cnd_ice",
    "t1_ice",
    "t_si",
    "t_i",
    "e_i",
    "t_s",
    "e_s",
)
DH_UPDATED_CAT_FIELDS = (
    "h_i",
    "h_s",
    "t_su",
    "a_i",
    "e_i",
    "e_s",
    "t_s",
    "sz_i",
    "s_i",
    "qml_ice",
)
DH_DIAG_FIELDS = tuple(f.name for f in DH_DIAGS)
DH_FLUX_FIELDS = tuple(f.name for f in DH_FLUXES)
SAL_UPDATED_FIELDS = ("sz_i", "sfx_bri", "sfx_res")
DA_UPDATED_FIELDS = ("a_i", "h_i", "h_s", "sfx_lam", "hfx_thd", "wfx_lam")
PND_UPDATED_FIELDS = (
    "at_i",
    "a_ip",
    "v_ip",
    "h_ip",
    "v_il",
    "h_il",
    "a_ip_frac",
    "wfx_pnd",
)
O13_REMAP_KEYS = (
    "h_i_b", "a_i_b", "h_i", "a_i", "v_i", "v_s", "sv_i", "oa_i",
    "a_ip", "v_ip", "v_il", "t_su", "e_s", "e_i", "szv_i", "hi_max",
    "rn_amax_2d",
)
DO_UPDATED_FIELDS = (
    "a_i",
    "v_i",
    "sv_i",
    "e_i",
    "szv_i",
    "at_i",
    "sfx_opw",
    "wfx_opw",
    "hfx_thd",
    "hfx_opw",
)
CAT_ENTRY_KEYS = (
    "h_i",
    "h_s",
    "t_su",
    "t_i",
    "t_s",
    "e_i",
    "e_s",
    "sz_i",
    "a_i",
    "t_bo",
    "qns_ice",
    "qsr_ice",
    "dqns_ice",
    "qtr_ice_top",
    "hfx_err_dif",
    "hfx_dif",
    "at_i",
    "s_i",
    "sss",
    "sst",
    "qsb_ice_bot",
    "fhld",
    "frq_m",
    "sprecip",
    "qprec_ice",
    "evap_ice",
)

ICE_NAMELIST_SPECS: dict[str, type] = {
    "ln_icedH": bool,
    "ln_icedA": bool,
    "ln_icedO": bool,
    "ln_leadhfx": bool,
    "rn_hinew": float,
    "ln_frazil": bool,
    "rn_maxfraz": float,
    "rn_vfraz": float,
    "rn_Cfraz": float,
    "rn_himin": float,
    "rn_kappa_i": float,
    "rn_kappa_s": float,
    "ln_zdf_BL99": bool,
    "ln_cndi_U64": bool,
    "ln_cndi_P07": bool,
    "rn_cnd_s": float,
    "nn_qtrice": int,
    "ln_zdf_chkcvg": bool,
    "ln_cndflx": bool,
    "ln_cndemulate": bool,
    "nn_snwfra": int,
    "rn_beta": float,
    "rn_dmin": float,
    "rn_snwblow": float,
    "nn_icesal": int,
    "rn_icesal": float,
    "rn_sinew": float,
    "rn_simin": float,
    "ln_drainage": bool,
    "ln_flushing": bool,
    "nn_sal_scheme": int,
    "nn_drainage": int,
    "nn_flushing": int,
    "rn_flushrate": float,
    "rn_vbrc": float,
    "nn_liquidus": int,
    "rn_sal_himin": float,
    "rn_alpha_RJW": float,
    "rn_Rc_RJW": float,
    "rn_alpha_GN": float,
    "rn_Rc_GN": float,
    "rn_alpha_CW": float,
    "ln_sal_chk": bool,
    "ln_pnd": bool,
    "ln_pnd_LEV": bool,
    "rn_apnd_min": float,
    "rn_apnd_max": float,
    "rn_pnd_flush": float,
    "ln_pnd_lids": bool,
    "ln_pnd_rain": bool,
    "ln_pnd_CST": bool,
    "rn_apnd": float,
    "rn_hpnd": float,
    "ln_pnd_TOPO": bool,
    "ln_pnd_alb": bool,
    "rn_pnd_hl_min": float,
    "rn_pnd_hl_max": float,
    "ln_virtual_itd": bool,
    "ln_icediachk": bool,
}


def _refuse(message: str) -> ValueError:
    return ValueError(f"STATUS PLANT-FIRED: {message}")


def _shape(field: Field, meta: dict[str, int]) -> tuple[int, ...]:
    ni = meta["ntei"] - meta["ntsi"] + 1
    nj = meta["ntej"] - meta["ntsj"] + 1
    dims = {
        "I2": (ni, nj),
        "H1": (ni + 2, nj + 2),
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


def parse_resolved_ice_namelist(
    output_namelist_ice: Path,
    output_namelist_dyn: Path,
    ppsrc: Path = DEFAULT_PPSRC,
    plant: str | None = None,
) -> dict[str, Any]:
    """Parse the resolved sea-ice and dynamics namelists without fallback defaults."""
    if not output_namelist_ice.is_file():
        raise _refuse(f"missing resolved ice namelist: {output_namelist_ice}")
    drag_info = verify_sas_drag_resolution(output_namelist_dyn, ppsrc)
    text = output_namelist_ice.read_text(errors="replace")
    parsed: dict[str, Any] = {"ln_drgice_imp": drag_info["ln_drgice_imp"]}
    for raw_line in text.splitlines():
        code = raw_line.split("!", 1)[0].strip()
        if not code:
            continue
        for key, typ in ICE_NAMELIST_SPECS.items():
            m = re.match(rf"^{re.escape(key)}\s*=\s*(\S+)", code, flags=re.I)
            if not m:
                continue
            tok = m.group(1).rstrip(",")
            if typ is bool:
                low = tok.lower()
                if low in ("t", ".true."):
                    parsed[key] = True
                elif low in ("f", ".false."):
                    parsed[key] = False
                else:
                    raise _refuse(f"invalid boolean token {tok!r} for {key}")
            elif typ is int:
                parsed[key] = int(tok)
            else:
                parsed[key] = float(tok.replace("d", "e").replace("D", "E"))
    missing = sorted(set(ICE_NAMELIST_SPECS) - set(parsed))
    if missing:
        raise _refuse(f"missing resolved namelist parameters in {output_namelist_ice}: {missing}")
    if plant == "namelist-mismatch":
        parsed["rn_sinew"] = parsed["rn_sinew"] + 0.05
    return parsed


def verify_master_admission_manifests(
    master_root: Path = DEFAULT_MASTER_ROOT,
    plant: str | None = None,
) -> dict[str, Any]:
    """Verify the admitted master record's top-level and per-rung projection digests."""
    manifest_dir = master_root / "manifest"
    run_manifest_path = manifest_dir / "run_manifest.json"
    resource_path = manifest_dir / "resource_usage.json"
    inventory_path = manifest_dir / "records.sha256"
    for p in (run_manifest_path, resource_path, inventory_path):
        if not p.is_file():
            raise _refuse(f"missing master manifest file: {p}")

    run_sha = hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
    res_sha = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    inv_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    if plant == "manifest-digest":
        run_sha = "0" * 64
    if run_sha != PINNED_RUN_MANIFEST_SHA256:
        raise _refuse(f"run_manifest.json SHA-256 mismatch: {run_sha} != {PINNED_RUN_MANIFEST_SHA256}")
    if res_sha != PINNED_RESOURCE_USAGE_SHA256:
        raise _refuse(f"resource_usage.json SHA-256 mismatch: {res_sha} != {PINNED_RESOURCE_USAGE_SHA256}")
    if inv_sha != PINNED_RECORDS_INVENTORY_SHA256:
        raise _refuse(f"records.sha256 SHA-256 mismatch: {inv_sha} != {PINNED_RECORDS_INVENTORY_SHA256}")

    inventory_map: dict[str, str] = {}
    for line in inventory_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            inventory_map[Path(parts[1]).name] = parts[0]

    rung_digests: dict[str, dict[str, str]] = {}
    for rung, expected_sha in PINNED_PROJECTION_SHA256.items():
        proj_path = manifest_dir / "per_rung" / f"{rung}.json"
        if not proj_path.is_file():
            raise _refuse(f"missing projection manifest: {proj_path}")
        got_sha = hashlib.sha256(proj_path.read_bytes()).hexdigest()
        if got_sha != expected_sha:
            raise _refuse(f"{rung} projection manifest SHA-256 mismatch: {got_sha} != {expected_sha}")
        proj_data = json.loads(proj_path.read_text())
        file_map: dict[str, str] = {}
        for item in proj_data["files"]:
            fname = Path(item["path"]).name
            fsha = item["sha256"]
            if inventory_map.get(fname) != fsha:
                raise _refuse(f"{rung} record {fname} digest disagrees with records.sha256")
            file_map[fname] = fsha
        rung_digests[rung] = file_map

    return {
        "run_manifest_sha256": run_sha,
        "resource_usage_sha256": res_sha,
        "records_inventory_sha256": inv_sha,
        "projection_manifest_sha256": dict(PINNED_PROJECTION_SHA256),
        "rung_record_digests": rung_digests,
    }


def read_record_verified(
    path: Path,
    expected_record_sha256: str | None = None,
) -> dict[str, Any]:
    """Read one master binary record with full header/extent/finiteness checks."""
    raw = bytearray(path.read_bytes())
    if len(raw) < HEADER_BYTES or bytes(raw[:8]) != MAGIC:
        raise _refuse(f"bad/truncated master header in {path}")
    record_sha = hashlib.sha256(raw).hexdigest()
    if expected_record_sha256 is not None and record_sha != expected_record_sha256:
        raise _refuse(
            f"record SHA-256 mismatch for {path.name}: {record_sha} != {expected_record_sha256}"
        )
    version = struct.unpack_from("<i", raw, 8)[0]
    if version != VERSION:
        raise _refuse(f"unsupported master version {version}")
    vals = tuple(map(int, np.frombuffer(raw, dtype="<i4", count=N_META, offset=12)))
    keys = (
        "kind",
        "kt",
        "jl",
        "rank",
        "jpi",
        "jpj",
        "jpl",
        "nlay_i",
        "nlay_s",
        "ntsi",
        "ntei",
        "ntsj",
        "ntej",
        "npti",
        "nfields",
        "endian",
        "nit000",
        "kmm",
        "forcing_mode",
        "dynamics_mode",
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
    expected_name = record_name(kind, meta["kt"], meta["jl"], meta["rank"])
    if path.name != expected_name:
        raise _refuse(f"filename/header disagreement: {path.name} != {expected_name}")
    scalars = np.frombuffer(
        raw, dtype="<f8", count=N_SCALARS, offset=8 + 4 + 4 * N_META
    ).copy()
    if not np.all(np.isfinite(scalars)):
        raise _refuse(f"nonfinite scalar header in {path}")
    offset = HEADER_BYTES
    fields: dict[str, np.ndarray] = {}
    for spec in KINDS[kind].fields:
        shape = _shape(spec, meta)
        dtype = np.dtype("<i4" if spec.dtype == "i4" else "<f8")
        count = int(np.prod(shape))
        stop = offset + count * dtype.itemsize
        if stop > len(raw):
            raise _refuse(f"truncated field {spec.name} in {path}")
        value = (
            np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
            .reshape(shape, order="F")
            .copy()
        )
        if spec.dtype == "f8" and not np.all(np.isfinite(value)):
            raise _refuse(f"nonfinite field {spec.name} in {path}")
        fields[spec.name] = value
        offset = stop
    if offset != len(raw):
        raise _refuse(f"trailing bytes in {path}: {len(raw) - offset}")
    if kind == 3:
        nonzero = [field.name for field in DH_DIAGS if np.any(fields[field.name] != 0.0)]
        if nonzero:
            raise _refuse(f"category-entry diagnostics were not zeroed: {nonzero}")
    return {
        "meta": meta,
        "scalars": dict(zip(SCALAR_NAMES, map(float, scalars), strict=True)),
        "fields": fields,
        "record_sha256": record_sha,
    }


def _verify_record_scalars_against_namelist(
    scalars: dict[str, float],
    namelist: dict[str, Any],
    kind: int,
) -> None:
    """Confirm that the binary record's scalar header matches the resolved namelist."""
    checks = [
        ("rcnd_s", float(namelist["rn_cnd_s"])),
        ("rn_kappa_i", float(namelist["rn_kappa_i"])),
        ("rn_kappa_s", float(namelist["rn_kappa_s"])),
        ("rn_snwblow", float(namelist["rn_snwblow"])),
        ("rn_sinew", float(namelist["rn_sinew"])),
        ("nn_icesal", float(namelist["nn_icesal"])),
        ("rn_simin", float(namelist["rn_simin"])),
        ("rn_himin", float(namelist["rn_himin"])),
        ("ln_drainage", 1.0 if namelist["ln_drainage"] else 0.0),
        ("ln_flushing", 1.0 if namelist["ln_flushing"] else 0.0),
        ("nn_drainage", float(namelist["nn_drainage"])),
        ("nn_flushing", float(namelist["nn_flushing"])),
        ("rn_sal_himin", float(namelist["rn_sal_himin"])),
        ("rn_flushrate", float(namelist["rn_flushrate"])),
        ("rn_vbrc", float(namelist["rn_vbrc"])),
        ("nn_sal_scheme", float(namelist["nn_sal_scheme"])),
        ("rn_alpha_RJW", float(namelist["rn_alpha_RJW"])),
        ("rn_Rc_RJW", float(namelist["rn_Rc_RJW"])),
        ("rn_apnd_min", float(namelist["rn_apnd_min"])),
        ("rn_apnd_max", float(namelist["rn_apnd_max"])),
        ("rn_pnd_flush", float(namelist["rn_pnd_flush"])),
        ("ln_pnd_lids", 1.0 if namelist["ln_pnd_lids"] else 0.0),
        ("ln_pnd_rain", 1.0 if namelist["ln_pnd_rain"] else 0.0),
        ("nn_liquidus", float(namelist["nn_liquidus"])),
        ("ln_sal_chk", 1.0 if namelist["ln_sal_chk"] else 0.0),
        ("ln_pnd", 1.0 if namelist["ln_pnd"] else 0.0),
        ("ln_pnd_LEV", 1.0 if namelist["ln_pnd_LEV"] else 0.0),
        ("ln_pnd_CST", 1.0 if namelist["ln_pnd_CST"] else 0.0),
        ("ln_pnd_TOPO", 1.0 if namelist["ln_pnd_TOPO"] else 0.0),
        ("rn_apnd", float(namelist["rn_apnd"])),
        ("rn_hpnd", float(namelist["rn_hpnd"])),
        ("ln_pnd_alb", 1.0 if namelist["ln_pnd_alb"] else 0.0),
        ("rn_pnd_hl_min", float(namelist["rn_pnd_hl_min"])),
        ("rn_pnd_hl_max", float(namelist["rn_pnd_hl_max"])),
        ("nn_snwfra", float(namelist["nn_snwfra"])),
    ]
    if kind in (1, 2, 12, 13):
        checks += [("rn_hinew", float(namelist["rn_hinew"]))] + ([("ln_icedO", 1.0), ("ln_leadhfx", 1.0)] if kind in (12, 13) else [])
    if kind in (8, 9):
        checks.extend(
            [
                ("rn_beta", float(namelist["rn_beta"])),
                ("rn_dmin", float(namelist["rn_dmin"])),
            ]
        )
    for sname, expected in checks:
        got = float(namelist[sname]) if sname in ("ln_icedO", "ln_leadhfx") else scalars[sname]
        if np.float64(got).view(np.uint64) != np.float64(expected).view(np.uint64):
            raise _refuse(
                f"record identity {sname}={got!r} disagrees with resolved/preregistered value {expected!r} in kind {kind}"
            )


def flip_1ulp(arr: np.ndarray, index: tuple[int, ...]) -> np.ndarray:
    out = arr.copy()
    u = out.view(np.uint64)
    u[index] ^= np.uint64(1)
    return out


class FieldScoreAccumulator:
    """Accumulates 0-ULP bitwise comparisons across steps and categories."""

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, Any]] = {}

    def update(
        self,
        field: str,
        got: np.ndarray | jax.Array,
        want: np.ndarray,
        *,
        kt: int,
        category: int,
    ) -> None:
        g = np.asarray(got)
        w = np.asarray(want)
        if g.shape != w.shape or g.dtype != w.dtype:
            raise _refuse(
                f"shape/dtype mismatch on {field}: got {g.shape}/{g.dtype}, want {w.shape}/{w.dtype}"
            )
        if w.dtype == np.float64:
            neq_mask = g.view(np.uint64) != w.view(np.uint64)
            abs_diff = float(np.max(np.abs(g - w))) if w.size else 0.0
            ulp = int(max_ulp_diff(g, w)) if w.size else 0
        else:
            neq_mask = g != w
            abs_diff = float(np.max(np.abs(g.astype(np.int64) - w.astype(np.int64)))) if w.size else 0.0
            ulp = int(abs_diff)

        neq_count = int(np.sum(neq_mask))
        tot_count = int(w.size)
        entry = self.stats.setdefault(
            field,
            {
                "unequal": 0,
                "total": 0,
                "max_abs_diff": 0.0,
                "max_ulp": 0,
                "first_non_bit": None,
            },
        )
        if neq_count > 0 and entry["first_non_bit"] is None:
            first_idx = [int(i) for i in np.argwhere(neq_mask)[0]]
            entry["first_non_bit"] = {
                "kt": int(kt),
                "category": int(category),
                "cell": first_idx,
                "field": field,
            }
        entry["unequal"] += neq_count
        entry["total"] += tot_count
        if abs_diff > entry["max_abs_diff"]:
            entry["max_abs_diff"] = abs_diff
        if ulp > entry["max_ulp"]:
            entry["max_ulp"] = ulp

    def summary(self) -> dict[str, Any]:
        total_unequal = sum(v["unequal"] for v in self.stats.values())
        total_elements = sum(v["total"] for v in self.stats.values())
        max_abs = max((v["max_abs_diff"] for v in self.stats.values()), default=0.0)
        max_ulp = max((v["max_ulp"] for v in self.stats.values()), default=0)
        first_non_bit = None
        for v in self.stats.values():
            if v["first_non_bit"] is not None:
                cand = (
                    v["first_non_bit"]["kt"],
                    v["first_non_bit"]["category"],
                    v["first_non_bit"]["cell"],
                    v["first_non_bit"]["field"],
                )
                if first_non_bit is None or cand[:2] < first_non_bit[:2]:
                    first_non_bit = cand
        return {
            "status": "PASS_0_ULP" if total_unequal == 0 else "NON_BIT_EXACT",
            "total_unequal": total_unequal,
            "total_elements": total_elements,
            "max_abs_diff": max_abs,
            "max_ulp": max_ulp,
            "first_non_bit": (
                {
                    "kt": first_non_bit[0],
                    "category": first_non_bit[1],
                    "cell": first_non_bit[2],
                    "field": first_non_bit[3],
                }
                if first_non_bit is not None
                else None
            ),
            "fields": self.stats,
        }


def _pad_1d(arr: np.ndarray, n_pad: int = N_PAD_1D) -> jnp.ndarray:
    n = arr.shape[0]
    if n > n_pad:
        raise _refuse(f"npti={n} exceeds N_PAD_1D={n_pad}")
    if n == n_pad:
        return jnp.asarray(arr, dtype=jnp.float64)
    pad_width = [(0, n_pad - n)] + [(0, 0)] * (arr.ndim - 1)
    return jnp.asarray(np.pad(arr, pad_width, mode="edge"), dtype=jnp.float64)


def _pad_dict(fields: dict[str, np.ndarray], keys: Sequence[str], n_pad: int = N_PAD_1D) -> dict[str, jnp.ndarray]:
    return {k: _pad_1d(fields[k], n_pad) for k in keys}


@functools.lru_cache(maxsize=8)
def _build_compiled_kernels(
    rDt_ice: float,
    r1_Dt_ice: float,
    r1_nlay_i: float,
    namelist_items: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    """Compile and cache the production stage kernels for one resolved namelist."""
    namelist = dict(namelist_items)
    constants = NEMO_SI3_CONSTANTS_CONFIG._replace(k_snow=float(namelist["rn_cnd_s"]))
    gravity = canonical_constants.g_nemo
    rho_w = canonical_constants.rho_water

    @jax.jit
    def jit_o8_frazil(qlead: jax.Array):
        return si3_ice_thd_frazil(
            qlead,
            rn_hinew=float(namelist["rn_hinew"]),
            ln_frazil=bool(namelist["ln_frazil"]),
        )

    @jax.jit
    def jit_o9_zdf(inputs: dict[str, jax.Array]):
        return si3_ice_thd_zdf_bl99(
            **inputs,
            rDt_ice=rDt_ice,
            rn_cnd_s=float(namelist["rn_cnd_s"]),
            rn_kappa_i=float(namelist["rn_kappa_i"]),
            rn_kappa_s=float(namelist["rn_kappa_s"]),
            nn_qtrice=int(namelist["nn_qtrice"]),
            nn_snwfra=int(namelist["nn_snwfra"]),
            ln_cndi_P07=bool(namelist["ln_cndi_P07"]),
            ln_cndflx=bool(namelist["ln_cndflx"]),
            ln_virtual_itd=bool(namelist["ln_virtual_itd"]),
            ln_zdf_chkcvg=bool(namelist["ln_zdf_chkcvg"]),
            constants=constants,
        )

    @jax.jit
    def jit_o9_dh(inputs: dict[str, jax.Array], fluxes: SI3DHFluxes):
        return si3_ice_thd_dh(
            **inputs,
            fluxes=fluxes,
            rDt_ice=rDt_ice,
            rn_snwblow=float(namelist["rn_snwblow"]),
            rn_sinew=float(namelist["rn_sinew"]),
            nn_icesal=int(namelist["nn_icesal"]),
            ln_cndflx=bool(namelist["ln_cndflx"]),
            ln_cndemulate=bool(namelist["ln_cndemulate"]),
            constants=constants,
        )

    @jax.jit
    def jit_temp(h_i: jax.Array, e_i: jax.Array, sz_i: jax.Array):
        return si3_ice_thd_temp(h_i=h_i, e_i=e_i, sz_i=sz_i, constants=constants)

    @jax.jit
    def jit_o9_category_stages(entry: dict[str, jax.Array], fluxes: SI3DHFluxes):
        return si3_ice_thd_category_stages(
            entry=entry,
            fluxes=fluxes,
            namelist=namelist,
            rDt_ice=rDt_ice,
            constants=constants,
        )

    @jax.jit
    def jit_o9_picard(
        h_i: jax.Array,
        h_s: jax.Array,
        t_su: jax.Array,
        t_i: jax.Array,
        t_s: jax.Array,
        sz_i: jax.Array,
        t_bo: jax.Array,
        qns_ice: jax.Array,
        qsr_ice: jax.Array,
        dqns_ice: jax.Array,
        qtr_ice_top: jax.Array,
    ):
        core = bl._si3_zdf_bl99_core(
            h_i=h_i,
            h_s=h_s,
            t_su=t_su,
            t_i_old=t_i,
            t_s_old=t_s,
            sz_i=sz_i,
            t_bo=t_bo,
            qns_ice=qns_ice,
            qsr_ice=qsr_ice,
            dqns_ice=dqns_ice,
            qtr_ice_top=qtr_ice_top,
            dt=rDt_ice,
            constants=constants,
            rn_kappa_i=float(namelist["rn_kappa_i"]),
            rn_kappa_s=float(namelist["rn_kappa_s"]),
        )
        ab_s, ab_i, qtr_bot = bl._radiation(
            h_i,
            h_s,
            qtr_ice_top,
            t_su,
            rn_kappa_i=float(namelist["rn_kappa_i"]),
            rn_kappa_s=float(namelist["rn_kappa_s"]),
            _nlay=(10, 5),
        )
        return core, ab_s, ab_i, qtr_bot

    @jax.jit
    def jit_o10_sal(inputs: dict[str, jax.Array]):
        return si3_ice_thd_sal(
            **inputs,
            namelist=namelist,
            rDt_ice=rDt_ice,
            r1_Dt_ice=r1_Dt_ice,
            r1_nlay_i=r1_nlay_i,
            rho_i=constants.rho_ice,
            rho_s=constants.rho_snow,
            rho_w=rho_w,
            gravity=gravity,
            constants=constants,
        )

    @jax.jit
    def jit_o10_category_stages(
        entry: dict[str, jax.Array],
        fluxes: SI3DHFluxes,
        sfx_bri: jax.Array,
    ):
        return si3_ice_thd_o10_category_stages(
            entry=entry,
            fluxes=fluxes,
            sfx_bri=sfx_bri,
            namelist=namelist,
            rDt_ice=rDt_ice,
            r1_Dt_ice=r1_Dt_ice,
            r1_nlay_i=r1_nlay_i,
            rho_i=constants.rho_ice,
            rho_s=constants.rho_snow,
            rho_w=rho_w,
            gravity=gravity,
            constants=constants,
        )

    @jax.jit
    def jit_o11_da(inputs: dict[str, jax.Array]):
        return si3_ice_thd_da(
            **inputs,
            rn_beta=float(namelist["rn_beta"]),
            rn_dmin=float(namelist["rn_dmin"]),
            rDt_ice=rDt_ice,
            nn_icesal=int(namelist["nn_icesal"]),
            constants=constants,
        )

    @jax.jit
    def jit_o12_pnd(inputs: dict[str, jax.Array]):
        return si3_ice_thd_pnd(
            **inputs,
            namelist=namelist,
            rDt_ice=rDt_ice,
            r1_Dt_ice=r1_Dt_ice,
            rho_w=rho_w,
            gravity=gravity,
            nn_hls=0,
            constants=constants,
        )

    @jax.jit
    def jit_o14_do(inputs: dict[str, jax.Array]):
        return si3_ice_thd_do(
            **inputs,
            nn_icesal=int(namelist["nn_icesal"]),
            rn_sinew=float(namelist["rn_sinew"]),
            r1_Dt_ice=r1_Dt_ice,
            r1_nlay_i=r1_nlay_i,
            constants=constants,
        )

    si3_orca2_cfg = SI3ThermoConfig(
        n_ice_layers=10,
        n_snow_layers=5,
        salinity_scheme=int(namelist["nn_icesal"]),
        lateral_melt=bool(namelist["ln_icedA"]),
        namthd_da=SI3LateralMeltConfig(
            rn_beta=float(namelist["rn_beta"]),
            rn_dmin=float(namelist["rn_dmin"]),
        ),
    )
    pnd_orca2_cfg = SI3PondConfig(
        rn_himin=float(namelist["rn_himin"]),
        rn_apnd_min=float(namelist["rn_apnd_min"]),
        rn_apnd_max=float(namelist["rn_apnd_max"]),
        rn_pnd_flush=float(namelist["rn_pnd_flush"]),
        ln_pnd_lids=bool(namelist["ln_pnd_lids"]),
    )

    @jax.jit
    def jit_prod_column_step(
        entry: dict[str, jax.Array],
        fluxes: SI3DHFluxes,
        sfx_bri: jax.Array,
        sfx_lam: jax.Array,
        wfx_lam: jax.Array,
    ):
        zero = jnp.zeros_like(entry["h_i"])
        state = SI3ColumnArrays(
            concentration=entry["a_i"],
            h_ice=entry["h_i"],
            h_snow=entry["h_s"],
            T_surface=entry["t_su"],
            e_ice=entry["e_i"],
            e_snow=entry["e_s"],
            S_bulk=entry["s_i"],
            S_layers=entry["sz_i"],
            age_volume=zero,
        )
        forcing = SI3SurfaceForcing(
            qns_ice=entry["qns_ice"],
            qsr_ice=entry["qsr_ice"],
            dqns_ice=entry["dqns_ice"],
            qtr_ice_top=entry["qtr_ice_top"],
            t_bottom=entry["t_bo"],
            sss=entry["sss"],
            evaporation=entry["evap_ice"],
            snow_precipitation=entry["sprecip"],
            qprec_ice=entry["qprec_ice"],
            qcn_ice_bottom=zero,
            qsb_ice_bottom=entry["qsb_ice_bot"],
            fhld=entry["fhld"],
            qlead=zero,
            frq_m=entry["frq_m"],
            sst_m=entry["sst"],
        )
        return si3_column_step_arrays(
            state,
            forcing,
            rDt_ice,
            constants,
            si3=si3_orca2_cfg,
            namelist=namelist,
            at_i=entry["at_i"],
            t_i_entry=entry["t_i"],
            t_s_entry=entry["t_s"],
            hfx_err_dif=entry["hfx_err_dif"],
            hfx_dif=entry["hfx_dif"],
            fluxes=fluxes,
            sfx_bri=sfx_bri,
            sfx_lam=sfx_lam,
            wfx_lam=wfx_lam,
            _return_stage_outputs=True,
        )

    @jax.jit
    def jit_prod_step_ponds(inputs: dict[str, jax.Array]):
        zero_3d = jnp.zeros_like(inputs["a_i"])
        return step_ponds(
            inputs["a_ip_frac"],
            inputs["h_ip"],
            melt_water_m=zero_3d,
            rain_water_m=zero_3d,
            ice_mask=inputs["a_i"] > 0.0,
            h_snow=zero_3d,
            T_air=inputs["t_su"],
            dt=rDt_ice,
            drainage_timescale=86400.0,
            refreeze_threshold=constants.T0,
            pond_to_ice_max_area=0.6,
            depth_to_area_ratio=0.8,
            scheme="si3_lev",
            at_i=inputs["a_i"],
            a_i=inputs["a_i"],
            h_i=inputs["h_i"],
            t_su=inputs["t_su"],
            h_il=inputs["h_il"],
            dh_i_sum=inputs["dh_i_sum_2d"],
            dh_s_sum=inputs["dh_s_sum_2d"],
            sz_i=inputs["sz_i"],
            t_i=inputs["t_i"],
            namthd_pnd=pnd_orca2_cfg,
            rho_w=rho_w,
            gravity=gravity,
            ice_constants=constants,
            v_i=inputs["v_i"],
            a_ip=inputs["a_ip"],
            v_ip=inputs["v_ip"],
            v_il=inputs["v_il"],
            wfx_pnd=inputs["wfx_pnd"],
            nn_hls=0,
            _return_si3_outputs=True,
        )

    @jax.jit
    def jit_o13_remap(inputs: dict[str, jax.Array]):
        return si3_ice_itd_rem(
            **inputs,
            nn_icesal=int(namelist["nn_icesal"]),
            ln_pnd=bool(namelist["ln_pnd_LEV"] or namelist["ln_pnd_TOPO"]),
            ln_pnd_lids=bool(namelist["ln_pnd_lids"]),
            epsi10=constants.epsi10,
            epsi20=constants.epsi20,
            rt0=constants.T0,
        )

    return {
        "jit_o8_frazil": jit_o8_frazil,
        "jit_o9_zdf": jit_o9_zdf,
        "jit_o9_dh": jit_o9_dh,
        "jit_temp": jit_temp,
        "jit_o9_category_stages": jit_o9_category_stages,
        "jit_o9_picard": jit_o9_picard,
        "jit_o10_sal": jit_o10_sal,
        "jit_o10_category_stages": jit_o10_category_stages,
        "jit_o11_da": jit_o11_da,
        "jit_o12_pnd": jit_o12_pnd,
        "jit_o13_remap": jit_o13_remap,
        "jit_o14_do": jit_o14_do,
        "jit_prod_column_step": jit_prod_column_step,
        "jit_prod_step_ponds": jit_prod_step_ponds,
    }


def _score_o13_step(
    *, records_dir: Path, kt: int, first_kt: int,
    digest_map: dict[str, str], namelist: dict[str, Any],
    jit_o13_remap, accumulator: FieldScoreAccumulator, plant: str | None,
) -> dict[str, Any]:
    """Score one admitted Kind 11 -> Kind 12 pair through the production kernel."""
    n11 = record_name(11, kt, 0)
    n12 = record_name(12, kt, 0)
    r11 = read_record_verified(records_dir / n11, digest_map.get(n11))
    r12 = read_record_verified(records_dir / n12, digest_map.get(n12))
    _verify_record_scalars_against_namelist(r11["scalars"], namelist, 11)
    _verify_record_scalars_against_namelist(r12["scalars"], namelist, 12)
    f11 = r11["fields"]
    f12 = dict(r12["fields"])
    if plant == "o13-1ulp" and kt == first_kt:
        f12["a_i"] = flip_1ulp(f12["a_i"], (0, 0, 0))
    remap_in = {key: jnp.asarray(f11[key], dtype=jnp.float64) for key in O13_REMAP_KEYS}
    if plant == "o13-model-mutation" and kt == first_kt:
        remap_in["hi_max"] = remap_in["hi_max"].at[1].set(
            jnp.nextafter(remap_in["hi_max"][1], jnp.asarray(jnp.inf, jnp.float64))
        )
    remap_out = jit_o13_remap(remap_in)
    if plant == "o13-dropped-transfer" and kt == first_kt:
        planted = np.asarray(remap_out["v_i"]).copy()
        planted[np.unravel_index(np.argmax(np.abs(planted)), planted.shape)] = 0.0
        remap_out = {**remap_out, "v_i": jnp.asarray(planted, dtype=jnp.float64)}
    conservation = check_o13_conservation(f11, remap_out, frame=kt)
    for spec in KINDS[12].fields:
        got = remap_out[spec.name] if spec.name in remap_out else f11[spec.name]
        accumulator.update(spec.name, got, f12[spec.name], kt=kt, category=0)
    return conservation


def score_o13(
    master_root: Path = DEFAULT_MASTER_ROOT,
    ppsrc: Path = DEFAULT_PPSRC,
    steps: Sequence[int] = STEPS,
    plant: str | None = None,
    verify_manifests: bool = True,
) -> dict[str, Any]:
    """O13-only mode of the extended r21 scorer; uses the identical scoring block."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    manifest_info = (
        verify_master_admission_manifests(master_root, plant=plant)
        if verify_manifests else {"rung_record_digests": {"O13": {}}}
    )
    run_dir = master_root / "record_100step"
    records_dir = run_dir / "o8_o16_master_records"
    namelist = parse_resolved_ice_namelist(
        run_dir / "output.namelist.ice", run_dir / "output.namelist.dyn",
        ppsrc=ppsrc, plant=plant,
    )
    first = read_record_verified(records_dir / record_name(11, steps[0], 0))
    kernels = _build_compiled_kernels(
        float(first["scalars"]["rDt_ice"]),
        float(first["scalars"]["r1_Dt_ice"]),
        float(first["scalars"]["r1_nlay_i"]),
        tuple(sorted(namelist.items())),
    )
    acc = FieldScoreAccumulator()
    digest_map = manifest_info["rung_record_digests"]["O13"]
    conservation_reports = []
    for kt in steps:
        conservation_reports.append(_score_o13_step(
            records_dir=records_dir, kt=kt, first_kt=steps[0],
            digest_map=digest_map, namelist=namelist,
            jit_o13_remap=kernels["jit_o13_remap"], accumulator=acc, plant=plant,
        ))
    score = acc.summary()
    if score["total_unequal"]:
        raise _refuse(
            f"O13 non-bit-exact: unequal={score['total_unequal']}/{score['total_elements']}, "
            f"max_abs_diff={score['max_abs_diff']}, max_ulp={score['max_ulp']}, "
            f"first_non_bit={score['first_non_bit']}"
        )
    return {
        "status": "PASS_O13_0_ULP", "steps_scored": len(steps),
        "first_step": int(steps[0]), "last_step": int(steps[-1]),
        "manifest_verification": {
            key: manifest_info.get(key) for key in (
                "run_manifest_sha256", "resource_usage_sha256",
                "records_inventory_sha256", "projection_manifest_sha256",
            )
        },
        "resolved_namelist": namelist,
        "rungs": {"O13": {
            "production_function": "legoesm.ice.itd.si3_ice_itd_rem",
            "entry_kind": 11, "exit_kind": 12, "score": score,
            "conservation": _summarize_o13_conservation(conservation_reports),
        }},
    }


def score_o14(
    master_root: Path = DEFAULT_MASTER_ROOT,
    ppsrc: Path = DEFAULT_PPSRC,
    steps: Sequence[int] = STEPS,
    plant: str | None = None,
    verify_manifests: bool = True,
) -> dict[str, Any]:
    """Score O14 kind 12 -> kind 13 through production ``si3_ice_thd_do``."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    manifest_info = (
        verify_master_admission_manifests(master_root, plant=plant)
        if verify_manifests
        else {"rung_record_digests": {"O14": {}}}
    )
    o14_digests = manifest_info["rung_record_digests"]["O14"]
    run_dir = master_root / "record_100step"
    records_dir = run_dir / "o8_o16_master_records"
    namelist = parse_resolved_ice_namelist(
        run_dir / "output.namelist.ice",
        run_dir / "output.namelist.dyn",
        ppsrc=ppsrc,
        plant=plant,
    )
    first = read_record_verified(
        records_dir / record_name(12, steps[0], 0),
        o14_digests.get(record_name(12, steps[0], 0)),
    )
    kernels = _build_compiled_kernels(
        float(first["scalars"]["rDt_ice"]),
        float(first["scalars"]["r1_Dt_ice"]),
        float(first["scalars"]["r1_nlay_i"]),
        tuple(sorted(namelist.items())),
    )
    jit_o14_do = kernels["jit_o14_do"]
    input_keys = (
        "a_i",
        "v_i",
        "sv_i",
        "e_i",
        "szv_i",
        "qlead",
        "t_bo",
        "sfx_opw",
        "wfx_opw",
        "ht_i_new",
        "fraz_frac",
        "hfx_thd",
        "hfx_opw",
        "rn_amax_2d",
        "sss_m",
        "hi_max",
    )
    arg_names = {"ht_i_new": "ht_i_new"}
    acc = FieldScoreAccumulator()
    for kt in steps:
        n12 = record_name(12, kt, 0)
        n13 = record_name(13, kt, 0)
        r12 = read_record_verified(records_dir / n12, o14_digests.get(n12))
        r13 = read_record_verified(records_dir / n13, o14_digests.get(n13))
        _verify_record_scalars_against_namelist(r12["scalars"], namelist, 12)
        _verify_record_scalars_against_namelist(r13["scalars"], namelist, 13)
        f12 = r12["fields"]
        f13 = dict(r13["fields"])
        if plant == "o14-1ulp" and kt == steps[0]:
            f13["v_i"] = flip_1ulp(f13["v_i"], (0, 0, 0))
        inputs = {
            arg_names.get(name, name): jnp.asarray(f12[name], dtype=jnp.float64)
            for name in input_keys
        }
        out = jit_o14_do(inputs)
        for spec in KINDS[13].fields:
            if spec.name in DO_UPDATED_FIELDS:
                got = (
                    f12[spec.name]
                    if plant == "o14-model-bypass"
                    else getattr(out, spec.name)
                )
            else:
                got = f12[spec.name]
            acc.update(spec.name, got, f13[spec.name], kt=kt, category=0)

    score = acc.summary()
    if score["total_unequal"]:
        raise _refuse(
            f"O14 non-bit-exact: unequal={score['total_unequal']}/{score['total_elements']}, "
            f"max_abs_diff={score['max_abs_diff']}, max_ulp={score['max_ulp']}, "
            f"first_non_bit={score['first_non_bit']}"
        )
    return {
        "status": "PASS_O14_0_ULP",
        "steps_scored": len(steps),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "production_function": "legoesm.ice.bitz_lipscomb.si3_ice_thd_do",
        "entry_kind": 12,
        "exit_kind": 13,
        "score": score,
        "manifest_verification": {
            "run_manifest_sha256": manifest_info.get("run_manifest_sha256"),
            "resource_usage_sha256": manifest_info.get("resource_usage_sha256"),
            "records_inventory_sha256": manifest_info.get("records_inventory_sha256"),
            "projection_manifest_sha256": manifest_info.get("projection_manifest_sha256"),
        },
        "resolved_namelist": namelist,
    }


def score_o8_to_o12(
    master_root: Path = DEFAULT_MASTER_ROOT,
    ppsrc: Path = DEFAULT_PPSRC,
    steps: Sequence[int] = STEPS,
    plant: str | None = None,
    verify_manifests: bool = True,
) -> dict[str, Any]:
    """Score rungs O8 through O13 in order against the master record."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    manifest_info = (
        verify_master_admission_manifests(master_root, plant=plant)
        if verify_manifests
        else {"rung_record_digests": {r: {} for r in SCORE_RUNGS}}
    )
    digests_by_rung = manifest_info["rung_record_digests"]

    run_dir = master_root / "record_100step"
    records_dir = run_dir / "o8_o16_master_records"
    namelist = parse_resolved_ice_namelist(
        run_dir / "output.namelist.ice",
        run_dir / "output.namelist.dyn",
        ppsrc=ppsrc,
        plant=plant,
    )

    first_rec = read_record_verified(records_dir / record_name(3, steps[0], 1))
    rDt_ice = float(first_rec["scalars"]["rDt_ice"])
    r1_Dt_ice = float(first_rec["scalars"]["r1_Dt_ice"])
    r1_nlay_i = float(first_rec["scalars"]["r1_nlay_i"])

    kernels = _build_compiled_kernels(
        rDt_ice,
        r1_Dt_ice,
        r1_nlay_i,
        tuple(sorted(namelist.items())),
    )
    jit_o8_frazil = kernels["jit_o8_frazil"]
    jit_o9_zdf = kernels["jit_o9_zdf"]
    jit_o9_dh = kernels["jit_o9_dh"]
    jit_temp = kernels["jit_temp"]
    jit_o9_category_stages = kernels["jit_o9_category_stages"]
    jit_o9_picard = kernels["jit_o9_picard"]
    jit_o10_sal = kernels["jit_o10_sal"]
    jit_o10_category_stages = kernels["jit_o10_category_stages"]
    jit_o11_da = kernels["jit_o11_da"]
    jit_o12_pnd = kernels["jit_o12_pnd"]
    jit_o13_remap = kernels["jit_o13_remap"]
    jit_prod_column_step = kernels["jit_prod_column_step"]
    jit_prod_step_ponds = kernels["jit_prod_step_ponds"]

    # 1. Rung O8: Kind 1 -> Kind 2 (si3_ice_thd_frazil)
    o8_acc = FieldScoreAccumulator()
    for kt in steps:
        n1 = record_name(1, kt, 0)
        n2 = record_name(2, kt, 0)
        r1 = read_record_verified(records_dir / n1, digests_by_rung["O8"].get(n1))
        r2 = read_record_verified(records_dir / n2, digests_by_rung["O8"].get(n2))
        _verify_record_scalars_against_namelist(r1["scalars"], namelist, 1)
        _verify_record_scalars_against_namelist(r2["scalars"], namelist, 2)
        f1 = r1["fields"]
        f2 = dict(r2["fields"])
        if plant == "o8-1ulp" and kt == steps[0]:
            f2["ht_i_new"] = flip_1ulp(f2["ht_i_new"], (0, 0))
        if plant == "swapped-exit" and kt == steps[0]:
            f2["ht_i_new"], f2["fraz_frac"] = f2["fraz_frac"], f2["ht_i_new"]
        out_o8 = jit_o8_frazil(jnp.asarray(f1["qlead"], dtype=jnp.float64))
        o8_acc.update("ht_i_new", out_o8.ht_i_new, f2["ht_i_new"], kt=kt, category=0)
        o8_acc.update("fraz_frac", out_o8.fraz_frac, f2["fraz_frac"], kt=kt, category=0)

    # 2. Rungs O9, O10, O11 (category stages across kt and jl=1..5) and O12 (2D/3D pond stage)
    o9_zdf_acc = FieldScoreAccumulator()
    o9_dh_acc = FieldScoreAccumulator()
    o9_temp1_acc = FieldScoreAccumulator()
    o9_composed_acc = FieldScoreAccumulator()
    o9_picard_acc = FieldScoreAccumulator()
    o9_gather_acc = FieldScoreAccumulator()

    o10_sal_acc = FieldScoreAccumulator()
    o10_temp2_acc = FieldScoreAccumulator()
    o10_composed_acc = FieldScoreAccumulator()

    o11_da_acc = FieldScoreAccumulator()
    o12_pnd_acc = FieldScoreAccumulator()
    prod_column_acc = FieldScoreAccumulator()
    prod_step_ponds_acc = FieldScoreAccumulator()
    o13_remap_acc = FieldScoreAccumulator()
    o13_conservation_reports = []

    zdf_input_keys = (
        "h_i",
        "h_s",
        "t_su",
        "t_i",
        "t_s",
        "e_i",
        "e_s",
        "sz_i",
        "a_i",
        "t_bo",
        "qns_ice",
        "qsr_ice",
        "dqns_ice",
        "qtr_ice_top",
        "hfx_err_dif",
        "hfx_dif",
    )
    dh_input_keys_from_r3 = (
        "h_i",
        "h_s",
        "a_i",
        "at_i",
        "sz_i",
        "s_i",
        "t_bo",
        "sss",
        "sst",
        "qsr_ice",
        "qtr_ice_top",
        "qsb_ice_bot",
        "fhld",
        "frq_m",
        "sprecip",
        "qprec_ice",
        "evap_ice",
    )
    dh_input_keys_from_r4 = (
        "t_su",
        "t_i",
        "t_s",
        "e_i",
        "e_s",
        "qns_ice",
        "qcn_ice_top",
        "qcn_ice_bot",
        "qtr_ice_bot",
    )
    sal_input_keys = (
        "h_i",
        "t_su",
        "t_i",
        "sz_i",
        "t_bo",
        "sss",
        "a_i",
        "dh_i_sum",
        "dh_s_sum",
        "sfx_bri",
        "sfx_res",
    )
    da_input_keys = (
        "at_i",
        "a_i",
        "h_i",
        "h_s",
        "e_i",
        "e_s",
        "s_i",
        "sz_i",
        "sst",
        "t_bo",
        "sfx_lam",
        "hfx_thd",
        "wfx_lam",
    )
    pnd_input_keys = (
        "a_i",
        "v_i",
        "h_i",
        "t_su",
        "a_ip",
        "v_ip",
        "h_ip",
        "v_il",
        "h_il",
        "a_ip_frac",
        "dh_i_sum_2d",
        "dh_s_sum_2d",
        "t_i",
        "sz_i",
        "wfx_pnd",
    )

    picard_iconv_matches = 0
    unpadded_spot_checks = 0

    for kt in steps:
        n21 = record_name(21, kt, 0)
        r21 = read_record_verified(records_dir / n21, digests_by_rung["O9"].get(n21))
        _verify_record_scalars_against_namelist(r21["scalars"], namelist, 21)

        for jl in range(1, 6):
            n3 = record_name(3, kt, jl)
            n4 = record_name(4, kt, jl)
            n5 = record_name(5, kt, jl)
            n6 = record_name(6, kt, jl)
            n7 = record_name(7, kt, jl)
            n8 = record_name(8, kt, jl)
            n9 = record_name(9, kt, jl)

            r3 = read_record_verified(records_dir / n3, digests_by_rung["O9"].get(n3))
            r4 = read_record_verified(records_dir / n4, digests_by_rung["O9"].get(n4))
            r5 = read_record_verified(records_dir / n5, digests_by_rung["O9"].get(n5))
            r6 = read_record_verified(records_dir / n6, digests_by_rung["O9"].get(n6))
            r7 = read_record_verified(records_dir / n7, digests_by_rung["O10"].get(n7))
            r8 = read_record_verified(records_dir / n8, digests_by_rung["O10"].get(n8))
            r9 = read_record_verified(records_dir / n9, digests_by_rung["O11"].get(n9))

            for k_id, rec in ((3, r3), (4, r4), (5, r5), (6, r6), (7, r7), (8, r8), (9, r9)):
                _verify_record_scalars_against_namelist(rec["scalars"], namelist, k_id)

            npti = r3["meta"]["npti"]
            f3 = r3["fields"]
            f4 = dict(r4["fields"])
            f5 = dict(r5["fields"])
            f6 = dict(r6["fields"])
            f7 = dict(r7["fields"])
            f8 = dict(r8["fields"])
            f9 = dict(r9["fields"])

            if plant == "o9-1ulp" and kt == steps[0] and jl == 1:
                f4["t_su"] = flip_1ulp(f4["t_su"], (0,))
            if plant == "o10-1ulp" and kt == steps[0] and jl == 1:
                f7["sz_i"] = flip_1ulp(f7["sz_i"], (0, 0))
            if plant == "o11-1ulp" and kt == steps[0] and jl == 1:
                f9["a_i"] = flip_1ulp(f9["a_i"], (0,))

            # Check Kind 21 -> Kind 3 gather for intensive/state fields
            jpi = r3["meta"]["jpi"]
            ntsi, ntsj = r3["meta"]["ntsi"], r3["meta"]["ntsj"]
            nptidx = f3["nptidx"]
            jj_full = (nptidx - 1) // jpi + 1
            ji_full = (nptidx - 1) % jpi + 1
            ii = ji_full - ntsi
            ij = jj_full - ntsj
            for gname in ("a_i", "h_i", "h_s", "t_su", "s_i", "o_i"):
                o9_gather_acc.update(
                    gname, r21["fields"][gname][ii, ij, jl - 1], f3[gname], kt=kt, category=jl
                )
            for gname in ("t_i", "sz_i", "t_s"):
                o9_gather_acc.update(
                    gname,
                    r21["fields"][gname][ii, ij, :, jl - 1],
                    f3[gname],
                    kt=kt,
                    category=jl,
                )

            # O9 substage 1: Kind 3 -> Kind 4 (si3_ice_thd_zdf_bl99)
            zdf_in = _pad_dict(f3, zdf_input_keys)
            zdf_out = jit_o9_zdf(zdf_in)
            for fname in ZDF_UPDATED_FIELDS:
                o9_zdf_acc.update(
                    fname, getattr(zdf_out, fname)[:npti], f4[fname], kt=kt, category=jl
                )
            for fname in ("nptidx", "o_i"):
                o9_zdf_acc.update(fname, f3[fname], f4[fname], kt=kt, category=jl)

            # O9 substage 2: Kind 3 + Kind 4 -> Kind 5 (si3_ice_thd_dh)
            dh_in = {
                **_pad_dict(f3, dh_input_keys_from_r3),
                **_pad_dict(r4["fields"], dh_input_keys_from_r4),
            }
            flux_in = SI3DHFluxes(**_pad_dict(f3, DH_FLUX_FIELDS))
            dh_out = jit_o9_dh(dh_in, flux_in)
            for fname in DH_UPDATED_CAT_FIELDS:
                o9_dh_acc.update(
                    fname, getattr(dh_out, fname)[:npti], f5[fname], kt=kt, category=jl
                )
            for fname in DH_DIAG_FIELDS:
                got_diag = (
                    np.zeros_like(f5[fname])
                    if fname == "dh_s_tot"
                    else getattr(dh_out.dh, fname)[:npti]
                )
                o9_dh_acc.update(fname, got_diag, f5[fname], kt=kt, category=jl)
            for fname in DH_FLUX_FIELDS:
                o9_dh_acc.update(
                    fname, getattr(dh_out.fluxes, fname)[:npti], f5[fname], kt=kt, category=jl
                )
            for spec in KINDS[5].fields:
                if (
                    spec.name not in DH_UPDATED_CAT_FIELDS
                    and spec.name not in DH_DIAG_FIELDS
                    and spec.name not in DH_FLUX_FIELDS
                ):
                    src = r4["fields"][spec.name] if spec.name in r4["fields"] else f3[spec.name]
                    o9_dh_acc.update(spec.name, src, f5[spec.name], kt=kt, category=jl)

            # O9 substage 3: Kind 5 -> Kind 6 (si3_ice_thd_temp)
            t1_in = _pad_dict(r5["fields"], ("h_i", "e_i", "sz_i"))
            t1_out = jit_temp(t1_in["h_i"], t1_in["e_i"], t1_in["sz_i"])
            o9_temp1_acc.update("t_i", t1_out[:npti], f6["t_i"], kt=kt, category=jl)
            for spec in KINDS[6].fields:
                if spec.name != "t_i":
                    o9_temp1_acc.update(
                        spec.name, r5["fields"][spec.name], f6[spec.name], kt=kt, category=jl
                    )

            # O9 composed category path & production si3_column_step_arrays (Kind 3 -> Kind 4..9)
            entry_pad = _pad_dict(f3, CAT_ENTRY_KEYS)
            sfx_bri_pad = _pad_1d(f3["sfx_bri"])
            sfx_lam_pad = _pad_1d(r8["fields"]["sfx_lam"])
            wfx_lam_pad = _pad_1d(r8["fields"]["wfx_lam"])
            prod_trace, prod_o10, prod_da, _ = jit_prod_column_step(
                entry_pad,
                flux_in,
                sfx_bri_pad,
                sfx_lam_pad,
                wfx_lam_pad,
            )
            cat_out = prod_o10.o9
            for fname in ZDF_UPDATED_FIELDS:
                o9_composed_acc.update(
                    f"zdf.{fname}",
                    getattr(cat_out.zdf, fname)[:npti],
                    f4[fname],
                    kt=kt,
                    category=jl,
                )
            for fname in DH_UPDATED_CAT_FIELDS:
                o9_composed_acc.update(
                    f"dh.{fname}",
                    getattr(cat_out.dh, fname)[:npti],
                    f5[fname],
                    kt=kt,
                    category=jl,
                )
            for fname in DH_DIAG_FIELDS:
                got_diag = (
                    np.zeros_like(f5[fname])
                    if fname == "dh_s_tot"
                    else getattr(cat_out.dh.dh, fname)[:npti]
                )
                o9_composed_acc.update(
                    f"dh.{fname}", got_diag, f5[fname], kt=kt, category=jl
                )
            for fname in DH_FLUX_FIELDS:
                o9_composed_acc.update(
                    f"dh.{fname}",
                    getattr(cat_out.dh.fluxes, fname)[:npti],
                    f5[fname],
                    kt=kt,
                    category=jl,
                )
            o9_composed_acc.update(
                "temp1.t_i", cat_out.t_i[:npti], f6["t_i"], kt=kt, category=jl
            )

            # Spot-check unpadded (npti,) JIT execution on kt == steps[0]
            if kt == steps[0]:
                entry_unpad = {
                    k: jnp.asarray(f3[k], dtype=jnp.float64) for k in CAT_ENTRY_KEYS
                }
                flux_unpad = SI3DHFluxes(
                    **{k: jnp.asarray(f3[k], dtype=jnp.float64) for k in DH_FLUX_FIELDS}
                )
                unpad_o10 = jit_o10_category_stages(
                    entry_unpad,
                    flux_unpad,
                    jnp.asarray(f3["sfx_bri"], dtype=jnp.float64),
                )
                o10_composed_acc.update(
                    "sz_i", unpad_o10.sal.sz_i, f7["sz_i"], kt=kt, category=jl
                )
                o10_composed_acc.update(
                    "sfx_bri", unpad_o10.sal.sfx_bri, f7["sfx_bri"], kt=kt, category=jl
                )
                o10_composed_acc.update(
                    "sfx_res", unpad_o10.sal.sfx_res, f7["sfx_res"], kt=kt, category=jl
                )
                o10_composed_acc.update(
                    "t_i", unpad_o10.t_i, f8["t_i"], kt=kt, category=jl
                )
                unpadded_spot_checks += 1

            # O9 Kind 17 picard_final check at PICARD_STEPS
            if kt in PICARD_STEPS:
                n17 = record_name(17, kt, jl)
                r17 = read_record_verified(
                    records_dir / n17, digests_by_rung["O9"].get(n17)
                )
                _verify_record_scalars_against_namelist(r17["scalars"], namelist, 17)
                f17 = r17["fields"]
                core, ab_s, ab_i, _ = jit_o9_picard(
                    *(
                        zdf_in[k]
                        for k in (
                            "h_i",
                            "h_s",
                            "t_su",
                            "t_i",
                            "t_s",
                            "sz_i",
                            "t_bo",
                            "qns_ice",
                            "qsr_ice",
                            "dqns_ice",
                            "qtr_ice_top",
                        )
                    )
                )
                o9_picard_acc.update("nptidx", f3["nptidx"], f17["nptidx"], kt=kt, category=jl)
                conv_got = np.asarray(core.iterations[:npti] < 200, dtype=np.int32)
                o9_picard_acc.update(
                    "l_t_converged", conv_got, f17["l_t_converged"], kt=kt, category=jl
                )
                o9_picard_acc.update("t_su", core.t_su[:npti], f17["t_su"], kt=kt, category=jl)
                o9_picard_acc.update(
                    "qns_ice", core.qns_ice[:npti], f17["qns_ice"], kt=kt, category=jl
                )
                o9_picard_acc.update("t_i", core.t_i[:npti], f17["t_i"], kt=kt, category=jl)
                o9_picard_acc.update("t_s", core.t_s[:npti], f17["t_s"], kt=kt, category=jl)
                o9_picard_acc.update(
                    "zkappa_i", core.k_i[:npti], f17["zkappa_i"], kt=kt, category=jl
                )
                o9_picard_acc.update(
                    "zradab_i", ab_i[:npti], f17["zradab_i"], kt=kt, category=jl
                )
                o9_picard_acc.update(
                    "zkappa_s", core.k_s[:npti], f17["zkappa_s"], kt=kt, category=jl
                )
                o9_picard_acc.update(
                    "zradab_s", ab_s[:npti], f17["zradab_s"], kt=kt, category=jl
                )
                # Telescope radiation transmission check
                o9_picard_acc.update(
                    "zradtr_s_top",
                    f3["qtr_ice_top"],
                    f17["zradtr_s"][:, 0],
                    kt=kt,
                    category=jl,
                )
                o9_picard_acc.update(
                    "zradtr_s_diff",
                    np.asarray(ab_s[:npti], dtype=np.float64),
                    f17["zradtr_s"][:, :-1] - f17["zradtr_s"][:, 1:],
                    kt=kt,
                    category=jl,
                )
                o9_picard_acc.update(
                    "zradtr_i_bot",
                    np.asarray(core.qtr_ice_bot[:npti], dtype=np.float64),
                    f17["zradtr_i"][:, -1],
                    kt=kt,
                    category=jl,
                )
                o9_picard_acc.update(
                    "zradtr_i_diff",
                    np.asarray(ab_i[:npti], dtype=np.float64),
                    f17["zradtr_i"][:, :-1] - f17["zradtr_i"][:, 1:],
                    kt=kt,
                    category=jl,
                )
                max_it = int(np.max(np.asarray(core.iterations[:npti])))
                iconv_want = int(r17["scalars"]["picard_iterations"])
                if max_it != iconv_want:
                    raise _refuse(
                        f"Kind 17 picard_iterations mismatch at kt={kt} jl={jl}: {max_it} != {iconv_want}"
                    )
                picard_iconv_matches += 1

            # 3. Rung O10: Kind 6 -> Kind 7 (si3_ice_thd_sal) and Kind 7 -> Kind 8 (si3_ice_thd_temp)
            sal_in = _pad_dict(r6["fields"], sal_input_keys)
            sal_out = jit_o10_sal(sal_in)
            for fname in SAL_UPDATED_FIELDS:
                o10_sal_acc.update(
                    fname, getattr(sal_out, fname)[:npti], f7[fname], kt=kt, category=jl
                )
            for spec in KINDS[7].fields:
                if spec.name not in SAL_UPDATED_FIELDS:
                    o10_sal_acc.update(
                        spec.name, r6["fields"][spec.name], f7[spec.name], kt=kt, category=jl
                    )

            t2_in = _pad_dict(r7["fields"], ("h_i", "e_i", "sz_i"))
            t2_out = jit_temp(t2_in["h_i"], t2_in["e_i"], t2_in["sz_i"])
            o10_temp2_acc.update("t_i", t2_out[:npti], f8["t_i"], kt=kt, category=jl)
            for spec in KINDS[8].fields:
                if spec.name != "t_i":
                    o10_temp2_acc.update(
                        spec.name, r7["fields"][spec.name], f8[spec.name], kt=kt, category=jl
                    )

            # 4. Rung O11: Kind 8 -> Kind 9 (si3_ice_thd_da)
            da_in = _pad_dict(r8["fields"], da_input_keys)
            da_out = jit_o11_da(da_in)
            for fname in DA_UPDATED_FIELDS:
                o11_da_acc.update(
                    fname, getattr(da_out, fname)[:npti], f9[fname], kt=kt, category=jl
                )
            for spec in KINDS[9].fields:
                if spec.name not in DA_UPDATED_FIELDS:
                    o11_da_acc.update(
                        spec.name, r8["fields"][spec.name], f9[spec.name], kt=kt, category=jl
                    )

            # Production entry point: si3_column_step_arrays (Kind 3 -> Kind 4, 5, 6, 7, 8, 9)
            for fname in ZDF_UPDATED_FIELDS:
                prod_column_acc.update(
                    f"zdf.{fname}",
                    getattr(prod_o10.o9.zdf, fname)[:npti],
                    f4[fname],
                    kt=kt,
                    category=jl,
                )
            for fname in DH_UPDATED_CAT_FIELDS:
                prod_column_acc.update(
                    f"dh.{fname}",
                    getattr(prod_o10.o9.dh, fname)[:npti],
                    f5[fname],
                    kt=kt,
                    category=jl,
                )
            for fname in DH_DIAG_FIELDS:
                got_diag = (
                    np.zeros_like(f5[fname])
                    if fname == "dh_s_tot"
                    else getattr(prod_o10.o9.dh.dh, fname)[:npti]
                )
                prod_column_acc.update(
                    f"dh.{fname}", got_diag, f5[fname], kt=kt, category=jl
                )
            for fname in DH_FLUX_FIELDS:
                prod_column_acc.update(
                    f"dh.{fname}",
                    getattr(prod_o10.o9.dh.fluxes, fname)[:npti],
                    f5[fname],
                    kt=kt,
                    category=jl,
                )
            prod_column_acc.update(
                "temp1.t_i", prod_o10.o9.t_i[:npti], f6["t_i"], kt=kt, category=jl
            )
            for fname in SAL_UPDATED_FIELDS:
                prod_column_acc.update(
                    f"sal.{fname}",
                    getattr(prod_o10.sal, fname)[:npti],
                    f7[fname],
                    kt=kt,
                    category=jl,
                )
            prod_column_acc.update(
                "temp2.t_i", prod_o10.t_i[:npti], f8["t_i"], kt=kt, category=jl
            )
            for fname in DA_UPDATED_FIELDS:
                prod_column_acc.update(
                    f"da.{fname}",
                    getattr(prod_da, fname)[:npti],
                    f9[fname],
                    kt=kt,
                    category=jl,
                )
            prod_column_acc.update(
                "trace.post_zdf.t_su",
                prod_trace.post_zdf.T_surface[:npti],
                f4["t_su"],
                kt=kt,
                category=jl,
            )
            prod_column_acc.update(
                "trace.post_dh.h_i",
                prod_trace.post_dh.h_ice[:npti],
                f5["h_i"],
                kt=kt,
                category=jl,
            )
            prod_column_acc.update(
                "trace.post_sal.sz_i",
                prod_trace.post_sal.S_layers[:npti],
                f7["sz_i"],
                kt=kt,
                category=jl,
            )

        # 5. Rung O12: Kind 10 -> Kind 11 (si3_ice_thd_pnd and step_ponds(scheme='si3_lev'))
        n10 = record_name(10, kt, 0)
        n11 = record_name(11, kt, 0)
        r10 = read_record_verified(records_dir / n10, digests_by_rung["O12"].get(n10))
        r11 = read_record_verified(records_dir / n11, digests_by_rung["O12"].get(n11))
        _verify_record_scalars_against_namelist(r10["scalars"], namelist, 10)
        _verify_record_scalars_against_namelist(r11["scalars"], namelist, 11)
        f10 = r10["fields"]
        f11 = dict(r11["fields"])
        if plant == "o12-1ulp" and kt == steps[0]:
            f11["a_ip"] = flip_1ulp(f11["a_ip"], (0, 0, 0))

        pnd_in = {k: jnp.asarray(f10[k], dtype=jnp.float64) for k in pnd_input_keys}
        pnd_out = jit_o12_pnd(pnd_in)
        for fname in PND_UPDATED_FIELDS:
            o12_pnd_acc.update(
                fname, getattr(pnd_out, fname), f11[fname], kt=kt, category=0
            )
        for spec in KINDS[11].fields:
            if spec.name not in PND_UPDATED_FIELDS:
                o12_pnd_acc.update(
                    spec.name, f10[spec.name], f11[spec.name], kt=kt, category=0
                )
        prod_pnd_out = jit_prod_step_ponds(pnd_in)
        for fname in PND_UPDATED_FIELDS:
            prod_step_ponds_acc.update(
                fname, getattr(prod_pnd_out, fname), f11[fname], kt=kt, category=0
            )
        for spec in KINDS[11].fields:
            if spec.name not in PND_UPDATED_FIELDS:
                prod_step_ponds_acc.update(
                    spec.name, f10[spec.name], f11[spec.name], kt=kt, category=0
                )

        # 6. Rung O13: Kind 11 -> Kind 12 (production si3_ice_itd_rem).
        o13_conservation_reports.append(_score_o13_step(
            records_dir=records_dir, kt=kt, first_kt=steps[0],
            digest_map=digests_by_rung["O13"], namelist=namelist,
            jit_o13_remap=jit_o13_remap, accumulator=o13_remap_acc, plant=plant,
        ))

    rungs_report = {
        "O8": {
            "production_function": "legoesm.ice.bitz_lipscomb.si3_ice_thd_frazil",
            "entry_kind": 1,
            "exit_kind": 2,
            "score": o8_acc.summary(),
        },
        "O9": {
            "production_functions": [
                "legoesm.ice.bitz_lipscomb.si3_column_step_arrays",
                "legoesm.ice.bitz_lipscomb.si3_ice_thd_zdf_bl99",
                "legoesm.ice.bitz_lipscomb.si3_ice_thd_dh",
                "legoesm.ice.bitz_lipscomb.si3_ice_thd_temp",
                "legoesm.ice.bitz_lipscomb.si3_ice_thd_category_stages",
            ],
            "entry_kind": 3,
            "exit_kinds": [4, 5, 6, 17],
            "zdf_exit": o9_zdf_acc.summary(),
            "dh_exit": o9_dh_acc.summary(),
            "temp1_exit": o9_temp1_acc.summary(),
            "composed_category_stages": o9_composed_acc.summary(),
            "picard_final_kind17": {
                **o9_picard_acc.summary(),
                "iconv_matches": picard_iconv_matches,
            },
            "pre_category_gather_kind21": o9_gather_acc.summary(),
        },
        "O10": {
            "production_functions": [
                "legoesm.ice.bitz_lipscomb.si3_column_step_arrays",
                "legoesm.ice.bitz_lipscomb.si3_ice_thd_sal",
                "legoesm.ice.bitz_lipscomb.si3_ice_thd_temp",
                "legoesm.ice.bitz_lipscomb.si3_ice_thd_o10_category_stages",
            ],
            "entry_kind": 6,
            "exit_kinds": [7, 8],
            "sal_exit": o10_sal_acc.summary(),
            "temp2_exit": o10_temp2_acc.summary(),
            "unpadded_o10_composed_spot_check": {
                **o10_composed_acc.summary(),
                "categories_checked": unpadded_spot_checks,
            },
        },
        "O11": {
            "production_function": "legoesm.ice.si3_thd_da.si3_ice_thd_da",
            "production_entry_point": "legoesm.ice.bitz_lipscomb.si3_column_step_arrays",
            "entry_kind": 8,
            "exit_kind": 9,
            "score": o11_da_acc.summary(),
            "production_column_step_arrays_o9_o11": prod_column_acc.summary(),
        },
        "O12": {
            "production_function": "legoesm.ice.si3_thd_pnd.si3_ice_thd_pnd",
            "production_entry_point": "legoesm.ice.ponds.step_ponds(scheme='si3_lev')",
            "entry_kind": 10,
            "exit_kind": 11,
            "score": o12_pnd_acc.summary(),
            "production_step_ponds_si3_lev": prod_step_ponds_acc.summary(),
        },
        "O13": {
            "production_function": "legoesm.ice.itd.si3_ice_itd_rem",
            "entry_kind": 11,
            "exit_kind": 12,
            "score": o13_remap_acc.summary(),
            "conservation": _summarize_o13_conservation(o13_conservation_reports),
        },
    }

    # Check if any rung had a non-0-ULP difference (e.g. under a plant)
    failures: list[str] = []
    for rung_name, rdata in rungs_report.items():
        if rung_name == "O9":
            sub_scores = [
                rdata["zdf_exit"],
                rdata["dh_exit"],
                rdata["temp1_exit"],
                rdata["composed_category_stages"],
                rdata["picard_final_kind17"],
                rdata["pre_category_gather_kind21"],
            ]
        elif rung_name == "O10":
            sub_scores = [
                rdata["sal_exit"],
                rdata["temp2_exit"],
                rdata["unpadded_o10_composed_spot_check"],
            ]
        elif rung_name == "O11":
            sub_scores = [
                rdata["score"],
                rdata["production_column_step_arrays_o9_o11"],
            ]
        elif rung_name == "O12":
            sub_scores = [
                rdata["score"],
                rdata["production_step_ponds_si3_lev"],
            ]
        else:
            sub_scores = [rdata["score"]]
        for sub in sub_scores:
            if sub["total_unequal"] > 0:
                fnb = sub["first_non_bit"]
                failures.append(
                    f"{rung_name} non-bit-exact: unequal={sub['total_unequal']}/{sub['total_elements']}, "
                    f"max_abs_diff={sub['max_abs_diff']}, max_ulp={sub['max_ulp']}, "
                    f"first_non_bit={fnb}"
                )
    if failures:
        raise _refuse("; ".join(failures))

    return {
        "status": "PASS_ALL_O8_O13_0_ULP",
        "steps_scored": len(steps),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "manifest_verification": {
            "run_manifest_sha256": manifest_info.get("run_manifest_sha256"),
            "resource_usage_sha256": manifest_info.get("resource_usage_sha256"),
            "records_inventory_sha256": manifest_info.get("records_inventory_sha256"),
            "projection_manifest_sha256": manifest_info.get("projection_manifest_sha256"),
        },
        "resolved_namelist": namelist,
        "rungs": rungs_report,
    }


O13_CONSERVED_FIELDS = (
    "a_i", "v_i", "v_s", "sv_i", "oa_i", "a_ip", "v_ip", "v_il",
    "e_s", "e_i", "szv_i",
)


def check_o13_conservation(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    frame: int | str,
) -> dict[str, Any]:
    """Check every preregistered O13 extensive inventory across categories."""
    eps = np.finfo(np.float64).eps
    fields: dict[str, dict[str, float]] = {}
    for name in O13_CONSERVED_FIELDS:
        before_arr = np.asarray(before[name], dtype=np.float64)
        after_arr = np.asarray(after[name], dtype=np.float64)
        got = np.sum(after_arr, axis=-1, dtype=np.float64)
        if name == "a_i":
            if "a_i_sum_target" not in after:
                raise ValueError("O13 conservation requires a_i_sum_target")
            expected = np.asarray(after["a_i_sum_target"], dtype=np.float64)
        else:
            expected = np.sum(before_arr, axis=-1, dtype=np.float64)
        scale = np.sum(np.abs(before_arr), axis=-1, dtype=np.float64)
        allowed = 64.0 * eps * np.maximum(scale, np.finfo(np.float64).tiny)
        residual = np.abs(got - expected)
        if not np.all(np.isfinite(residual)) or np.any(residual > allowed):
            worst = np.unravel_index(np.argmax(residual - allowed), residual.shape)
            raise _refuse(
                f"O13 conservation failed for {name} at frame={frame}, "
                f"cell={list(worst)}, residual={residual[worst]!r}, "
                f"allowed={allowed[worst]!r}"
            )
        fields[name] = {
            "max_abs_residual": float(np.max(residual, initial=0.0)),
            "max_allowed": float(np.max(allowed, initial=0.0)),
        }
    return {"status": "PASS_CONSERVATION", "frame": frame, "fields": fields}


def _summarize_o13_conservation(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "status": "PASS_CONSERVATION",
        "frames_checked": len(reports),
        "fields": {
            name: {
                key: max(float(report["fields"][name][key]) for report in reports)
                for key in ("max_abs_residual", "max_allowed")
            }
            for name in O13_CONSERVED_FIELDS
        },
    }


from legoesm.ice.si3_finalize import (  # noqa: E402 - late-rung extension
    SI3Fluxes,
    SI3StressInputs,
    SI3UpdateFluxForcing,
    si3_ice_update_tau,
    si3_o15_step,
    si3_o16_step,
)
from legoesm.ice.si3_orca2_state import (  # noqa: E402 - late-rung extension
    SI3AggregatedVars,
    SI3CategoryBounds,
    SI3EquivalentVars,
    SI3ORCA2StateConfig,
    SI3State,
    si3_lbc_lnk_orca2,
)
from legoesm.ice.si3_surface import SI3SurfaceConfig  # noqa: E402
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_master_gate import (  # noqa: E402
    PRE_AGG_TOTALS,
    STATE,
    TOTALS,
)

PINNED_PROJECTION_SHA256.update({
    "O15": "e1cd64c354294a071aca8496b014b59eb0b8f7be93d0304219e5c94b192ea292",
    "O16": "6a3f9ed6f9330af1c65b603726c2ba407a28683f2de1b2a9dcaf9fd7dd3dc5e0",
})


_DIRECT_STATE_FIELDS = (
    "v_i", "v_s", "sv_i", "a_i", "t_su", "oa_i", "a_ip", "v_ip",
    "v_il", "e_s", "e_i", "szv_i",
)
_EQUIVALENT_FIELDS = SI3EquivalentVars._fields
_AGGREGATED_FIELDS = SI3AggregatedVars._fields


def _full_orca2_field(value: np.ndarray, nn_hls: int = 2) -> jax.Array:
    """Harness-only reconstruction of an interior record's production halos."""
    arr = jnp.asarray(value, dtype=jnp.float64)
    pad = ((nn_hls, nn_hls), (nn_hls, nn_hls)) + ((0, 0),) * (arr.ndim - 2)
    return si3_lbc_lnk_orca2(jnp.pad(arr, pad), "T", 1.0, nn_hls)


def _state_from_record(
    fields: dict[str, np.ndarray],
    config: SI3ORCA2StateConfig,
    *,
    full_orca2: bool,
) -> SI3State:
    sample = fields["a_i"]
    ni, nj = sample.shape[:2]
    h = config.nn_hls if full_orca2 else 0
    conv = (lambda x: _full_orca2_field(x, h)) if full_orca2 else (
        lambda x: jnp.asarray(x, dtype=jnp.float64)
    )
    state = SI3State.zeros(ni + 2 * h, nj + 2 * h, config=config)
    direct = {name: conv(fields[name]) for name in _DIRECT_STATE_FIELDS}
    eqv = state.equivalent._replace(
        **{name: conv(fields[name]) for name in _EQUIVALENT_FIELDS if name in fields}
    )
    agg = state.aggregated._replace(
        **{name: conv(fields[name]) for name in _AGGREGATED_FIELDS if name in fields}
    )
    t_si = conv(fields["t_si"]) if "t_si" in fields else state.t_si
    if "hi_max" not in fields:
        raise _refuse("state record lacks explicit hi_max")
    hi_mean = fields.get("hi_mean", np.asarray(state.bounds.hi_mean))
    bounds = SI3CategoryBounds(
        hi_max=jnp.asarray(fields["hi_max"], dtype=jnp.float64),
        hi_mean=jnp.asarray(hi_mean, dtype=jnp.float64),
    )
    return state._replace(
        **direct,
        t_si=t_si,
        equivalent=eqv,
        aggregated=agg,
        bounds=bounds,
    )


def _fluxes_from_record(fields: dict[str, np.ndarray]) -> SI3Fluxes:
    return SI3Fluxes(
        **{name: jnp.asarray(fields[name], dtype=jnp.float64) for name in SI3Fluxes._fields}
    )


def _state_score_fields(state: SI3State, *, interior_halo: int = 0) -> dict[str, jax.Array]:
    sl = (
        (slice(interior_halo, -interior_halo), slice(interior_halo, -interior_halo))
        if interior_halo
        else (slice(None), slice(None))
    )
    out = {name: getattr(state, name)[sl] for name in _DIRECT_STATE_FIELDS}
    out.update({name: getattr(state.equivalent, name)[sl] for name in _EQUIVALENT_FIELDS})
    out.update({name: getattr(state.aggregated, name)[sl] for name in _AGGREGATED_FIELDS})
    return out


def _late_config(namelist: dict[str, Any], scalars: dict[str, float], *, nn_hls: int) -> SI3ORCA2StateConfig:
    return SI3ORCA2StateConfig.for_orca2_sas_ice(
        nn_hls=nn_hls,
        ln_icethd=True,
        rn_simin=float(namelist["rn_simin"]),
        rn_sinew=float(namelist["rn_sinew"]),
        nn_icesal=int(namelist["nn_icesal"]),
        ln_pnd=bool(namelist["ln_pnd"]),
        ln_pnd_LEV=bool(namelist["ln_pnd_LEV"]),
        ln_pnd_TOPO=bool(namelist["ln_pnd_TOPO"]),
        ln_pnd_lids=bool(namelist["ln_pnd_lids"]),
        ln_pnd_alb=bool(namelist["ln_pnd_alb"]),
        nn_snwfra=int(namelist["nn_snwfra"]),
        nn_fsbc=1,
        rDt_ice=float(scalars["rDt_ice"]),
    )


def _update_forcing(fields: dict[str, np.ndarray]) -> SI3UpdateFluxForcing:
    values: dict[str, jax.Array] = {}
    for name in SI3UpdateFluxForcing._fields:
        source = "tmask_k1" if name == "tmask" else name
        values[name] = jnp.asarray(fields[source], dtype=jnp.float64)
    return SI3UpdateFluxForcing(**values)


def verify_kind15_translation_evidence(
    master_root: Path = DEFAULT_MASTER_ROOT,
    steps: Sequence[int] = (1, 2, 100),
    *,
    plant: bool = False,
) -> dict[str, Any]:
    """Verify kind-15 writer order without translating the admitted reader."""
    manifest = verify_master_admission_manifests(master_root)
    digests = manifest["rung_record_digests"]["O16"]
    records_dir = master_root / "record_100step/o8_o16_master_records"
    writer = (
        Path(__file__).with_name("nemo_si3_o8_o16_master_acquisition")
        / "o8_o16_master_writer.F90"
    ).read_text()
    writer = writer.split("ELSEIF(kind==15) THEN", 1)[1].split(
        "ELSEIF(kind==16) THEN", 1
    )[0]
    writer_tail = (
        "at_i_b", "wfx_snw_dyn", "qsr", "sfx", "cloud_fra",
        "tmask(:,:,1)", "smask0", "snwice_mass", "snwice_mass_b",
    )
    if plant:
        writer_tail = writer_tail[:2] + ("sfx", "qsr") + writer_tail[4:]
    positions = []
    for name in writer_tail:
        extent = "f2" if name in ("tmask(:,:,1)", "snwice_mass", "snwice_mass_b") else "i2"
        token = f"CALL put_{extent}(u,{name})"
        positions.append(writer.index(token))
    if positions != sorted(positions):
        raise _refuse("kind-15 writer tail order evidence failed")

    sbc_phy = (DEFAULT_NEMO_ROOT / "src/OCE/SBC/sbc_phy.F90").read_text()
    match = re.search(r"\bpp_cldf\s*=\s*([0-9.]+)", sbc_phy, re.I)
    if match is None:
        raise _refuse("cannot resolve independently known pp_cldf")
    pp_cldf = np.float64(match.group(1))
    cross_kind_fields = _DIRECT_STATE_FIELDS
    checked = 0
    for kt in steps:
        f15 = read_record_verified(
            records_dir / record_name(15, kt), digests[record_name(15, kt)]
        )["fields"]
        f18 = read_record_verified(
            records_dir / record_name(18, kt), digests[record_name(18, kt)]
        )["fields"]
        for name in cross_kind_fields:
            if not np.array_equal(f15[name].view(np.uint64), f18[name].view(np.uint64)):
                raise _refuse(f"kind-15 pre-tail alignment failed for {name} at kt={kt}")
            checked += 1
        # Under the fixed writer order, the reader-labelled cloud_fra block is
        # qsr and matches the independently recorded qsr_oce input bit-for-bit.
        if not np.array_equal(f15["cloud_fra"].view(np.uint64), f15["qsr_oce"].view(np.uint64)):
            raise _refuse(f"kind-15 qsr permutation evidence failed at kt={kt}")
        # The reader-labelled sfx block is the writer's cloud_fra block and
        # matches NEMO's independent pp_cldf source constant.
        if not np.all(f15["sfx"] == pp_cldf):
            raise _refuse(f"kind-15 cloud permutation evidence failed at kt={kt}")

    return {
        "status": "KIND15_TRANSLATION_EVIDENCE_ONLY",
        "steps": list(map(int, steps)),
        "writer_tail_order": list(writer_tail),
        "admitted_schema_tail_order": [
            "at_i_b", "wfx_snw_dyn", "cloud_fra", "qsr", "sfx",
            "tmask_k1", "smask0", "snwice_mass", "snwice_mass_b",
        ],
        "cross_kind_fields_checked": checked,
        "qsr_matches_qsr_oce": True,
        "cloud_fra_matches_pp_cldf": float(pp_cldf),
        "translation_applied": False,
    }


def score_o15_o16(
    master_root: Path = DEFAULT_MASTER_ROOT,
    ppsrc: Path = DEFAULT_PPSRC,
    steps: Sequence[int] = STEPS,
    plant: str | None = None,
) -> dict[str, Any]:
    """Score O15 and O16 kind 20 through their production entry points.

    Kind 15 remains deliberately unscored pending the user's reader-translation
    versus re-acquisition decision.
    """
    if not steps:
        raise _refuse("no steps requested")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    manifest = verify_master_admission_manifests(master_root, plant=plant)
    digests = manifest["rung_record_digests"]
    run_dir = master_root / "record_100step"
    records_dir = run_dir / "o8_o16_master_records"
    namelist = parse_resolved_ice_namelist(
        run_dir / "output.namelist.ice", run_dir / "output.namelist.dyn", ppsrc, plant
    )
    o15 = FieldScoreAccumulator()
    o16_tau = FieldScoreAccumulator()

    for kt in steps:
        records: dict[int, dict[str, Any]] = {}
        for kind in PROJECTIONS["O15"] + (19, 20):
            name = record_name(kind, kt, 0)
            rung = "O15" if kind in PROJECTIONS["O15"] else "O16"
            records[kind] = read_record_verified(records_dir / name, digests[rung].get(name))
            _verify_record_scalars_against_namelist(records[kind]["scalars"], namelist, kind)

        f13 = records[13]["fields"]
        f14 = dict(records[14]["fields"])
        cfg15 = _late_config(namelist, records[13]["scalars"], nn_hls=0)
        state13 = _state_from_record(f13, cfg15, full_orca2=False)
        flux13 = _fluxes_from_record(f13)
        skip_rebin = plant == "o15-model-mutation"
        strip_transfer_barriers = plant == "o15-strip-transfer-barriers"
        out15 = jax.jit(
            lambda state, flux, sss, sst, amax: si3_o15_step(
                state, flux, sss_m=sss, sst_m=sst, rn_amax_2d=amax,
                rn_himin=float(namelist["rn_himin"]), config=cfg15,
                _plant_skip_rebin=skip_rebin,
                _plant_strip_transfer_barriers=strip_transfer_barriers,
            )
        )(
            state13, flux13, jnp.asarray(f13["sss_m"]), jnp.asarray(f13["sst_m"]),
            jnp.asarray(f13["rn_amax_2d"]),
        )
        got15 = _state_score_fields(out15.state)
        got15.update({name: getattr(out15.fluxes, name) for name in SI3Fluxes._fields})
        if plant == "o15-1ulp" and kt == steps[0]:
            f14["a_i"] = flip_1ulp(f14["a_i"], (0, 0, 0))
        owned15 = set(_DIRECT_STATE_FIELDS) | set(_EQUIVALENT_FIELDS) | set(PRE_AGG_TOTALS)
        owned15 |= set(SI3Fluxes._fields)
        for spec in KINDS[14].fields:
            if spec.name in owned15:
                o15.update(spec.name, got15[spec.name], f14[spec.name], kt=kt, category=0)

        f19 = records[19]["fields"]
        f20 = dict(records[20]["fields"])
        cfg16 = _late_config(namelist, records[19]["scalars"], nn_hls=2)
        stress_values = {
            name: jnp.asarray(f19[name], dtype=jnp.float64) for name in SI3StressInputs._fields
        }
        # Version-1 kind 19 writer order is tmod_io, utau, vtau while its
        # admitted schema labels those byte blocks utau, vtau, tmod_io.
        stress_values.update(
            tmod_io=jnp.asarray(f19["utau"], dtype=jnp.float64),
            utau=jnp.asarray(f19["vtau"], dtype=jnp.float64),
            vtau=jnp.asarray(f19["tmod_io"], dtype=jnp.float64),
        )
        stress = SI3StressInputs(**stress_values)
        tau = jax.jit(lambda x: si3_ice_update_tau(
            x, config=cfg16, ln_drgice_imp=bool(namelist["ln_drgice_imp"])
        ))(stress)
        tau_meta = records[20]["meta"]
        owned = (
            slice(tau_meta["ntsi"] - 1, tau_meta["ntei"]),
            slice(tau_meta["ntsj"] - 1, tau_meta["ntej"]),
        )
        if plant == "o16-tau-1ulp" and kt == steps[0]:
            f20["utau"] = flip_1ulp(
                f20["utau"], (tau_meta["ntsi"] - 1, tau_meta["ntsj"] - 1)
            )
        for spec in KINDS[20].fields:
            got = getattr(tau, spec.name)
            expected = f20[spec.name]
            if spec.extent == "F2":
                got = got[owned]
                expected = expected[owned]
            o16_tau.update(spec.name, got, expected, kt=kt, category=0)

    summaries = {
        "O15": o15.summary(),
        "O16_kind20_owned_interior": o16_tau.summary(),
    }
    failures = [f"{name}: {score['first_non_bit']} max_ulp={score['max_ulp']}"
                for name, score in summaries.items() if score["total_unequal"]]
    if failures:
        raise _refuse("; ".join(failures))
    return {
        "status": "PASS_O15_AND_O16_KIND20_0_ULP",
        "steps_scored": len(steps),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "manifest_verification": manifest,
        "resolved_namelist": namelist,
        "scores": summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rung",
        choices=("O8-O13", "O13", "O14", "O15-O16"),
        default="O8-O13",
        help="Certified rung group to score",
    )
    parser.add_argument(
        "--master-root",
        type=Path,
        default=DEFAULT_MASTER_ROOT,
        help="Path to the admitted O8-O16 master record root",
    )
    parser.add_argument(
        "--ppsrc",
        type=Path,
        default=DEFAULT_PPSRC,
        help="Path to the NEMO ORCA2_SAS_ICE_O7_REF preprocessed sources",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1..100",
        help="Step range '1..100' or comma-separated list '1,2,100'",
    )
    parser.add_argument(
        "--plant",
        type=str,
        default=None,
        choices=(
            None,
            "o8-1ulp",
            "o9-1ulp",
            "o10-1ulp",
            "o11-1ulp",
            "o12-1ulp",
            "o13-1ulp",
            "o13-model-mutation",
            "o14-1ulp",
            "o14-model-bypass",
            "o13-dropped-transfer",
            "o15-1ulp",
            "o15-model-mutation",
            "o15-strip-transfer-barriers",
            "o16-tau-1ulp",
            "manifest-digest",
            "namelist-mismatch",
            "swapped-exit",
        ),
        help="Optional planted violation to verify fail-closed detection",
    )
    parser.add_argument(
        "--only-o13", action="store_true",
        help="run the shared O13 scoring block only (same manifest and EXIT-field gates)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the JSON scoring report",
    )
    args = parser.parse_args(argv)

    if ".." in args.steps:
        lo, hi = map(int, args.steps.split("..", 1))
        steps = tuple(range(lo, hi + 1))
    else:
        steps = tuple(int(x) for x in args.steps.split(",") if x.strip())

    scorer = (
        score_o14
        if args.rung == "O14"
        else score_o15_o16
        if args.rung == "O15-O16"
        else score_o13
        if args.rung == "O13" or args.only_o13
        else score_o8_to_o12
    )
    report = scorer(master_root=args.master_root, ppsrc=args.ppsrc, steps=steps, plant=args.plant)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {"status": report["status"], "steps_scored": report["steps_scored"]}
    if "rungs" in report:
        summary["rungs"] = {
            k: (
                v["score"]["status"]
                if "score" in v
                else v["composed_category_stages"]["status"]
                if "composed_category_stages" in v
                else v["sal_exit"]["status"]
            )
            for k, v in report["rungs"].items()
        }
    else:
        summary["score"] = (
            report["score"]["status"]
            if "score" in report
            else {key: value["status"] for key, value in report["scores"].items()}
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
