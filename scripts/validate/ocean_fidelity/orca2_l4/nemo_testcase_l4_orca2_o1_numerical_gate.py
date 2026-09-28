#!/usr/bin/env python3
"""Cellwise ORCA2 O1 ``fld_read`` and certified-NCAR boundary gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm import constants  # noqa: E402
from legoesm.ocean.bulk_flux_omip import (  # noqa: E402
    air_sea_fluxes,
    potential_air_temperature_10m,
    seawater_q_sat,
)
from legoesm.ocean.forcing.nemo_fld_read import (  # noqa: E402
    decode_source_indices,
    nemo_fld_interp,
    nemo_t_rotation_from_domain,
    rotate_en_to_ij,
)

RECORD = "oracle_sbcblk_o1_kt00000001.bin"
NX, NY = 90, 148
INPUT_FIELDS = (
    "wndi", "wndj", "tair", "humi", "qsr_down", "qlw_down",
    "precip_raw", "snow_raw", "slp",
)
OUTPUT_FIELDS = (
    "theta_air", "q_air", "precip", "sst", "ssu", "ssv", "tsk",
    "ssq", "cd_du", "sensible", "latent", "evap", "qlwn", "qsr",
    "qns", "emp", "utau", "vtau", "taum", "wndm",
)
FIELD_SPECS = (
    ("wndi", "u_10.15JUNE2009_fill.nc", "U_10_MOD", True),
    ("wndj", "v_10.15JUNE2009_fill.nc", "V_10_MOD", True),
    ("tair", "t_10.15JUNE2009_fill.nc", "T_10_MOD", False),
    ("humi", "q_10.15JUNE2009_fill.nc", "Q_10_MOD", False),
    ("qsr_down", "ncar_rad.15JUNE2009_fill.nc", "SWDN_MOD", False),
    ("qlw_down", "ncar_rad.15JUNE2009_fill.nc", "LWDN_MOD", False),
    ("precip_raw", "ncar_precip.15JUNE2009_fill.nc", "PRC_MOD1", False),
    ("snow_raw", "ncar_precip.15JUNE2009_fill.nc", "SNOW", False),
    ("slp", "slp.15JUNE2009_fill.nc", "SLP", False),
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_o1(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frames = []
    with path.open("rb", buffering=0) as handle:
        for kind, fields in enumerate((INPUT_FIELDS, OUTPUT_FIELDS)):
            require(handle.read(16).decode("ascii").rstrip() == "NEMO_L4_BLKIO_1",
                    f"frame {kind}: bad magic")
            header = struct.unpack("=8i", handle.read(32))
            require(header == (1, 1, kind, NX, NY, len(fields), 0, 64),
                    f"frame {kind}: bad header {header}")
            raw = np.fromfile(handle, np.float64, len(fields) * NX * NY)
            require(raw.size == len(fields) * NX * NY, f"frame {kind}: truncated")
            frames.append({
                field: raw[index * NX * NY:(index + 1) * NX * NY].reshape(NY, NX)
                for index, field in enumerate(fields)
            })
        require(handle.read(1) == b"", "trailing O1 payload")
    return frames[0], frames[1]


def cell_score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(candidate.shape == oracle.shape == mask.shape, "cell-score shape mismatch")
    require(np.isfinite(candidate[mask]).all() and np.isfinite(oracle[mask]).all(),
            "non-finite scored value")
    unequal_mask = candidate[mask].view(np.uint64) != oracle[mask].view(np.uint64)
    delta = np.abs(candidate[mask] - oracle[mask])
    indices = np.argwhere(mask)
    first = indices[np.flatnonzero(unequal_mask)[0]].tolist() if unequal_mask.any() else None
    return {
        "status": "AT_BAR" if not unequal_mask.any() else "DEBT",
        "unequal": int(unequal_mask.sum()),
        "count": int(unequal_mask.size),
        "max_abs": float(delta.max(initial=0.0)),
        "first_unequal_ji": first,
    }


def binding_plants(oracles: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, str]:
    results = {}
    first = tuple(np.argwhere(mask)[0])
    for field, expected in oracles.items():
        altered = expected.copy()
        altered[first] = np.nextafter(altered[first], np.inf)
        score = cell_score(altered, expected, mask)
        if score["unequal"] == 0:
            raise GateError(f"plant did not fail through cell_score: {field}")
        results[field] = "PASS_NONZERO"
    return results


def load_weights(root: Path, bicubic: bool) -> tuple[np.ndarray, np.ndarray]:
    filename = (
        "weights_core2_orca2_bicub.nc" if bicubic
        else "weights_core2_orca2_bilin.nc"
    )
    count = 16 if bicubic else 4
    with Dataset(root / filename) as dataset:
        sources = decode_source_indices(np.stack([
            np.asarray(dataset[f"src{index:02d}"][:].data)
            for index in range(1, 5)
        ]))
        weights = np.stack([
            np.asarray(dataset[f"wgt{index:02d}"][:].data, dtype=np.float64)
            for index in range(1, count + 1)
        ])
    return sources, weights


def validate(root: Path) -> dict:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "O1 certification is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    oracle_input, oracle_output = read_o1(root / RECORD)
    maps: dict[str, np.ndarray] = {}
    cache = {kind: load_weights(root, kind) for kind in (False, True)}
    for field, filename, variable, bicubic in FIELD_SPECS:
        with Dataset(root / filename) as dataset:
            # Resolved ln_tint=.false.; ocean.output pins record 1 at kt=1.
            source = np.asarray(dataset[variable][0].data, dtype=np.float64)
        indices, weights = cache[bicubic]
        operator = jax.jit(
            lambda value, src, wgt: nemo_fld_interp(
                value, src, wgt, bicubic=bicubic
            )
        )
        maps[field] = np.asarray(operator(source, indices, weights))

    with Dataset(root / "ORCA_R2_zps_domcfg.nc") as dataset:
        cosine, sine = nemo_t_rotation_from_domain(*(
            np.asarray(dataset[name][:].data, dtype=np.float64)
            for name in ("glamt", "gphit", "glamv", "gphiv")
        ))
    mapped_wind = jax.jit(rotate_en_to_ij)(
        maps["wndi"], maps["wndj"], cosine, sine
    )
    maps["wndi"], maps["wndj"] = map(np.asarray, mapped_wind)
    maps = {field: values[:, :NX] for field, values in maps.items()}
    all_cells = np.ones((NY, NX), dtype=bool)
    mapping_scores = {
        field: cell_score(maps[field], oracle_input[field], all_cells)
        for field in INPUT_FIELDS
    }

    with Dataset(root / "mesh_mask_0000.nc") as dataset:
        wet = np.asarray(dataset["tmask"][0, 0].data, dtype=bool)
    u, v = oracle_input["wndi"], oracle_input["wndj"]
    tair, qair, slp = (oracle_input[name] for name in ("tair", "humi", "slp"))
    sst = oracle_output["sst"]
    bulk = jax.jit(
        lambda ui, vj, ta, qa, ts, ps: air_sea_fluxes(
            ui, vj, ta, qa, ts + constants.T_freeze, slp_Pa=ps, algo="ncar"
        )
    )(u, v, tair, qair, sst, slp)
    tau_x, tau_y, sensible, latent, evap = map(np.asarray, bulk)
    theta_air = np.asarray(jax.jit(
        lambda ta, qa, ps: potential_air_temperature_10m(ta, qa, ps)[0]
    )(tair, qair, slp))
    ssq = np.asarray(jax.jit(
        lambda ts, ps: seawater_q_sat(ts + constants.T_freeze, ps)
    )(sst, slp))
    bulk_candidates = {
        "theta_air": theta_air,
        "ssq": ssq,
        "utau": -tau_x,
        "vtau": -tau_y,
        "sensible": sensible,
        "latent": latent,
        "evap": evap,
    }
    bulk_scores = {
        field: cell_score(values, oracle_output[field], wet)
        for field, values in bulk_candidates.items()
    }
    mapping_status = (
        "AT_BAR" if all(row["status"] == "AT_BAR" for row in mapping_scores.values())
        else "DEBT"
    )
    bulk_status = (
        "AT_BAR" if all(row["status"] == "AT_BAR" for row in bulk_scores.values())
        else "DEBT"
    )
    first = (
        {"boundary": "O1-M/fld_read", "owner": "ORCA2_OWNER"}
        if mapping_status != "AT_BAR"
        else ({"boundary": "O1-B/NCAR", "owner": "LANE3B_OWNER"}
              if bulk_status != "AT_BAR" else None)
    )
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "oracle_label": "VARIANT_V2_PROVISIONAL_O1_DEFINED_FIELDS",
        "record_sha256": sha256(root / RECORD),
        "execution": {
            "backend": jax.default_backend(),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 A2D(0), j=148 i=90; wet T cells for bulk",
        },
        "O1_M_fld_read": {"status": mapping_status, "owner": "ORCA2_OWNER",
                           "fields": mapping_scores},
        "O1_B_ncar": {"status": bulk_status, "owner": "LANE3B_OWNER",
                       "operand_mode": "ORACLE_SUPPLIED_O1_M", "fields": bulk_scores},
        "first_over_bar": first,
        "plants": {
            "mapping_cell_score": binding_plants(oracle_input, all_cells),
            "bulk_cell_score": binding_plants(
                {name: oracle_output[name] for name in bulk_candidates}, wet
            ),
        },
        "si3": "UNMEASURED_PENDING_ICE_MERGE_ORACLE_SUPPLIED_EXCHANGE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.run_dir)
    except (GateError, OSError, ValueError, struct.error) as exc:
        result = {"status": "FAIL", "error": str(exc)}
        code = 1
    else:
        code = 0
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
