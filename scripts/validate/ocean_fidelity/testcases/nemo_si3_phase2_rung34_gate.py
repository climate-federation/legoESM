#!/usr/bin/env python3
"""fp64 geometry and kt=1 dynALL gate for SI3 lane-3 rung 3.4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import cast

import netCDF4
import numpy as np
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import (
    ICE_RHEO_TRACERS,
    build_ice_rheo_card,
    step_ice_rheo_card,
)

from legoesm import constants

POINTWISE_BAR = 1.0e-15
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
HERE = Path(__file__).resolve().parent
ORACLE_GATE = HERE / "nemo_si3_oracle_gate.py"
MESH_FIELDS = (
    "e1t",
    "e2t",
    "e1u",
    "e2u",
    "e1v",
    "e2v",
    "e1f",
    "e2f",
    "tmask",
    "umask",
    "vmask",
)
DYNAMICS_FIELDS = (
    "u_ice",
    "v_ice",
    "stress1_i",
    "stress2_i",
    "stress12_i",
)
UNINFORMATIVE_ZERO_FIELDS = frozenset(("oa_i", "a_ip", "v_ip", "v_il"))
_PLANT_MAGNITUDE = 1.0e-10
_HALO_WIDTH = 2
_HASH_BLOCK_BYTES = 1024 * 1024
_FP64_STORAGE_BITS = 64


class Rung34GateError(RuntimeError):
    """Fail-closed rung-3.4 card or artifact error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rung34GateError(message)


def _load_oracle_gate():
    spec = importlib.util.spec_from_file_location("nemo_si3_oracle_gate_rung34", ORACLE_GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle_gate = _load_oracle_gate()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(
    name: str,
    oracle,
    candidate,
    *,
    exact: bool = False,
    uninformative_zero: bool = False,
) -> dict[str, object]:
    oracle = np.asarray(oracle)
    candidate = np.asarray(candidate)
    require(oracle.shape == candidate.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64 or exact, f"{name}: candidate is not fp64")
    require(
        bool(np.all(np.isfinite(oracle)) and np.all(np.isfinite(candidate))),
        f"{name}: non-finite input",
    )
    difference = np.abs(candidate.astype(np.float64) - oracle.astype(np.float64))
    nonzero = int(np.count_nonzero(difference))
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    maximum = float(difference[index])
    oracle_maximum = float(np.max(np.abs(oracle)))
    scale = max(oracle_maximum, 1.0)
    normalized = maximum / scale
    relative = maximum / oracle_maximum if oracle_maximum > 0.0 else None
    if uninformative_zero:
        require(oracle_maximum == 0.0, f"{name}: UNINFORMATIVE field is not oracle-zero")
        status = "UNINFORMATIVE"
    else:
        passed = bool(np.array_equal(candidate, oracle)) if exact else normalized <= POINTWISE_BAR
        status = "AT-BAR" if passed else "DEBT"
    return {
        "name": name,
        "status": status,
        "max_abs": maximum,
        "normalized_max_abs": normalized,
        "relative_max_abs": relative,
        "classification_metric": "normalized_max_abs",
        "relative_is_diagnostic": True,
        "oracle_max_abs": oracle_maximum,
        "max_abs_index_xy": [int(value) for value in index],
        "candidate_dtype": str(candidate.dtype),
        "oracle_dtype": str(oracle.dtype),
        "n": int(oracle.size),
        "bitwise_nonzero_over_n": f"{nonzero} / {oracle.size}",
        "bar": 0.0 if exact else POINTWISE_BAR,
    }


def _mesh_xy(value: np.ndarray) -> np.ndarray:
    while value.ndim > 2:
        value = value[0]
    return cast(np.ndarray, value.T)


def _oracle_field(frame: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name.startswith("e_s_l"):
        return cast(np.ndarray, frame["e_s"][..., int(name[-2:]) - 1, 0])
    if name.startswith("e_i_l"):
        return cast(np.ndarray, frame["e_i"][..., int(name[-2:]) - 1, 0])
    if name.startswith("szv_i_l"):
        return cast(np.ndarray, frame["szv_i"][..., int(name[-2:]) - 1, 0])
    value = frame[name]
    return cast(np.ndarray, value if value.ndim == 2 else value[..., 0])


def _candidate_fields(card, state) -> dict[str, np.ndarray]:
    index = ICE_RHEO_TRACERS.index
    fields = {
        name: np.asarray(state.contents[..., index(name)])
        for name in ICE_RHEO_TRACERS
    }
    fields.update(
        {
            "t_su": np.asarray(state.t_surface),
            "sv_i": np.asarray(state.bulk_salt_diagnostic),
            "u_ice": np.asarray(state.dynamics.u_ice_u),
            "v_ice": np.asarray(state.dynamics.v_ice_v),
            "stress1_i": np.asarray(state.dynamics.stress1_t),
            "stress2_i": np.asarray(state.dynamics.stress2_t),
            "stress12_i": np.asarray(state.dynamics.stress12_f),
        }
    )
    return fields


def run_gate(root: Path, *, plant_field: bool = False) -> dict[str, object]:
    frame1_path = root / "oracle_ice_step_entry_kt00000001.bin"
    frame2_path = root / "oracle_ice_step_entry_kt00000002.bin"
    mesh_path = root / "mesh_mask.nc"
    init_path = root / "output.init_ice.nc"
    for path in (frame1_path, frame2_path, mesh_path, init_path):
        require(path.is_file(), f"missing rung-3.4 artifact: {path}")
    header1, frame1 = oracle_gate.read_frame(frame1_path)
    header2, frame2 = oracle_gate.read_frame(frame2_path)
    require(header1["kt"] == 1 and header2["kt"] == 2, "entry frame clock mismatch")
    require(
        header1["storage_bits"] == _FP64_STORAGE_BITS,
        "oracle entry frame is not fp64",
    )
    with netCDF4.Dataset(mesh_path) as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in MESH_FIELDS}
    with netCDF4.Dataset(init_path) as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze

    card = build_ice_rheo_card(frame1, mesh, ocean_temperature_k)
    candidate = step_ice_rheo_card(card, completed_steps=0)
    fields = _candidate_fields(card, candidate)
    if plant_field:
        fields["v_s"] = fields["v_s"].copy()
        fields["v_s"][_HALO_WIDTH + 10, _HALO_WIDTH + 10] += _PLANT_MAGNITUDE

    rows: list[dict[str, object]] = []
    for name, value in zip(
        ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"),
        card.metrics[:8],
        strict=True,
    ):
        rows.append(
            _score(
                f"geometry.{name}",
                _mesh_xy(mesh[name]),
                np.asarray(value)[_HALO_WIDTH:-_HALO_WIDTH, _HALO_WIDTH:-_HALO_WIDTH],
            )
        )
    for name, value in (
        ("tmask", card.forcing_template.tmask_t),
        ("umask", card.forcing_template.umask_u),
        ("vmask", card.forcing_template.vmask_v),
    ):
        rows.append(
            _score(
                f"geometry.{name}",
                _mesh_xy(mesh[name]).astype(bool),
                np.asarray(value)[_HALO_WIDTH:-_HALO_WIDTH, _HALO_WIDTH:-_HALO_WIDTH].astype(bool),
                exact=True,
            )
        )
    for name, value in fields.items():
        rows.append(
            _score(
                f"kt1.{name}",
                _oracle_field(frame2, name)[
                    _HALO_WIDTH:-_HALO_WIDTH, _HALO_WIDTH:-_HALO_WIDTH
                ],
                value[_HALO_WIDTH:-_HALO_WIDTH, _HALO_WIDTH:-_HALO_WIDTH],
                uninformative_zero=name in UNINFORMATIVE_ZERO_FIELDS,
            )
        )

    failed = [row["name"] for row in rows if row["status"] == "DEBT"]
    return {
        "gate": "nemo-si3-phase2-rung34-kt1-v1",
        "case": "ICE_RHEO_OMIP_L3",
        "status": "KT1-VERIFIED" if not failed else "DEBT",
        "exit_code": 0 if not failed else 1,
        "bar": POINTWISE_BAR,
        "bar_definition": "max_abs / max(oracle_max_abs, 1.0)",
        "relative_column": "max_abs / oracle_max_abs; diagnostic only",
        "cpu_only": True,
        "precision_policy": "fp64",
        "candidate_dtypes": sorted(
            {row["candidate_dtype"] for row in rows if "candidate_dtype" in row}
        ),
        "dynall_order": "rhg -> adv -> rdgrft -> cor (icedyn.F90:130-135)",
        "rows": rows,
        "failed_rows": failed,
        "plant": {"field": plant_field, "magnitude": _PLANT_MAGNITUDE},
        "coverage": {
            "kt1_scored_fields": sorted(fields),
            "uninformative_oracle_zero_fields": sorted(UNINFORMATIVE_ZERO_FIELDS),
            "prather_moment_leaves": 5 * len(ICE_RHEO_TRACERS),
            "prather_moment_status": "UNMEASURED-ENTRY-FRAMES-DO-NOT-CARRY-MOMENTS",
            "trajectory_kt2_to_kt720": "UNMEASURED",
            "candidate_restart": "UNMEASURED",
        },
        "artifacts": {
            str(path): _sha256(path)
            for path in (frame1_path, frame2_path, mesh_path, init_path)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plant-field", action="store_true")
    args = parser.parse_args()
    report = run_gate(args.root, plant_field=args.plant_field)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
