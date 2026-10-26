#!/usr/bin/env python3
"""Offline score of round-337 S2 groups held by the round-5 records.

Only independently reconstructable groups are scored: runoff tracer content
from O1 ``sst`` plus the final runoff mass, and native-face stress from final
T stress plus masks.  Missing S0/SSS-restoring/post-ice entry operands are
reported unmeasured; oracle output substitution is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm import constants  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.freshwater import nemo_orca2_surface_runoff  # noqa: E402
from legoesm.ocean.forcing.nemo_surface import nemo_t_stress_to_uv  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_o1_numerical_gate as o1_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)


GROUPS = ("runoff_temperature_content", "runoff_salinity_content", "utauU", "vtauV")
UNMEASURED = {
    "S0_F_record_selection_and_temporal_blend": "entry records/times absent",
    "S0_M_seven_surface_means": "instantaneous/prior mean state absent",
    "S2_sss_restoring_and_fwb_merge": "post-ice entry, SSS target, rnfmsk absent",
    "S2_complete_qns_emp_sfx_taum_qsr_merge": "post-ice entry boundary absent",
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(candidate, oracle, mask) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(candidate.shape == oracle.shape == mask.shape, "score shape mismatch")
    lhs = np.ascontiguousarray(candidate[mask]).view(np.uint64)
    rhs = np.ascontiguousarray(oracle[mask]).view(np.uint64)
    unequal = lhs != rhs
    return {
        "words": int(lhs.size),
        "unequal": int(np.count_nonzero(unequal)),
        "over_bar": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(np.abs(candidate[mask] - oracle[mask]))) if lhs.size else 0.0,
        "status": "AT_BAR" if not unequal.any() else "DEBT",
    }


def validate(record_root: Path, *, plant_group: str | None) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    require(jax.default_backend() == "cpu", "round-337 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")

    o1_path = record_root / "oracle_sbcblk_o1_kt00000001.bin"
    surface_path = record_root / "oracle_ocean_surface_input_kt00000001.bin"
    mesh_path = record_root / "mesh_mask_0000.nc"
    for path in (o1_path, surface_path, mesh_path):
        require(path.is_file(), f"missing immutable input {path}")
    _, o1_output = o1_gate.read_o1(o1_path)
    fields = ladder.read_surface_fields(surface_path, kt=1, rank=0)
    with Dataset(mesh_path) as dataset:
        tmask = np.asarray(dataset["tmask"][0, 0], dtype=np.float64)
        umask = np.asarray(dataset["umask"][0, 0], dtype=np.float64)
        vmask = np.asarray(dataset["vmask"][0, 0], dtype=np.float64)

    sst = np.asarray(o1_output["sst"], dtype=np.float64)
    rnf = np.asarray(fields["rnf"], dtype=np.float64)
    oracle_rnf_tsc = np.asarray(fields["rnf_tsc"], dtype=np.float64)
    wet = tmask > 0.5
    runoff_fn = jax.jit(
        lambda runoff, temperature, mask: nemo_orca2_surface_runoff(
            runoff,
            temperature,
            mask,
            runoff_factor=jnp.asarray(1.0, dtype=jnp.float64),
            reference_density=jnp.asarray(constants.rho_ocean_nemo, dtype=jnp.float64),
        )
    )
    runoff = runoff_fn(jnp.asarray(rnf), jnp.asarray(sst), jnp.asarray(tmask))
    current_temperature = np.asarray(runoff.temperature_content)
    current_salinity = np.asarray(runoff.salinity_content)

    utau = np.asarray(fields["utau"], dtype=np.float64)
    vtau = np.asarray(fields["vtau"], dtype=np.float64)
    oracle_u = np.asarray(fields["utauU"], dtype=np.float64)[:, :-1]
    oracle_v = np.asarray(fields["vtauV"], dtype=np.float64)[:-1, :]
    utau_east = np.concatenate((utau[:, 1:], utau[:, -1:]), axis=1)
    vtau_north = np.concatenate((vtau[1:, :], vtau[-1:, :]), axis=0)
    tmask_east = np.concatenate((tmask[:, 1:], tmask[:, -1:]), axis=1)
    tmask_north = np.concatenate((tmask[1:, :], tmask[-1:, :]), axis=0)
    stress_fn = jax.jit(nemo_t_stress_to_uv)
    current_u, current_v = stress_fn(
        jnp.asarray(utau),
        jnp.asarray(vtau),
        jnp.asarray(utau_east),
        jnp.asarray(vtau_north),
        jnp.asarray(tmask),
        jnp.asarray(tmask_east),
        jnp.asarray(tmask_north),
        jnp.asarray(umask),
        jnp.asarray(vmask),
    )
    current_u = np.asarray(current_u)[:, :-1]
    current_v = np.asarray(current_v)[:-1, :]
    # The acquisition canonicalizes source-undefined dry faces to zero; score
    # only NEMO's native owned wet staggering, as preregistered.
    u_defined = umask[:, :-1] > 0.5
    v_defined = vmask[:-1, :] > 0.5

    current = {
        "runoff_temperature_content": current_temperature,
        "runoff_salinity_content": current_salinity,
        "utauU": current_u,
        "vtauV": current_v,
    }
    oracle = {
        "runoff_temperature_content": oracle_rnf_tsc[..., 0],
        "runoff_salinity_content": oracle_rnf_tsc[..., 1],
        "utauU": oracle_u,
        "vtauV": oracle_v,
    }
    masks = {
        "runoff_temperature_content": wet,
        "runoff_salinity_content": wet,
        "utauU": u_defined,
        "vtauV": v_defined,
    }
    if plant_group is not None:
        planted = np.array(current[plant_group], copy=True)
        target = tuple(np.argwhere(masks[plant_group])[0])
        planted[target] = np.nextafter(planted[target], np.inf)
        current[plant_group] = planted

    rows = {
        group: score(current[group], oracle[group], masks[group])
        for group in GROUPS
    }
    after_unequal = sum(int(row["unequal"]) for row in rows.values())
    if plant_group is not None:
        require(rows[plant_group]["unequal"] == 1,
                f"{plant_group} one-ULP plant did not fire once")
        raise GateError(f"planted {plant_group} word rejected through real scorer")
    require(
        after_unequal == 0,
        f"current independently scored S2 groups have {after_unequal} unequal words: "
        f"{json.dumps(rows, sort_keys=True)}",
    )
    return {
        "format": "orca2-round337-surface-offline-v1",
        "status": "PASS_MEASURED_GROUPS_AT_BAR",
        "execution": {"backend": jax.default_backend(), "dtype": "float64", "jit": True},
        "records": {
            "root": str(record_root.resolve()),
            "o1_sha256": sha256(o1_path),
            "surface_sha256": sha256(surface_path),
            "mesh_sha256": sha256(mesh_path),
        },
        "rows": rows,
        "unmeasured_needs_acquisition": UNMEASURED,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-group", choices=GROUPS)
    args = parser.parse_args()
    try:
        report = validate(args.record_root, plant_group=args.plant_group)
        code = 0
    except GateError as error:
        report = {"status": "FAIL", "error": str(error), "plant_group": args.plant_group}
        code = 1
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
