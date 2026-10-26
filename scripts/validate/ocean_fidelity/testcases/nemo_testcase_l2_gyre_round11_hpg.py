#!/usr/bin/env python3
"""GYRE stage-2 EOS/HPG literal-operand discriminators (JIT, fp64)."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from nemo_testcase_l2_gyre_phase3_gate import (
    BAR,
    CASE,
    DIMS,
    expected_masks,
    require,
    score,
    sha256,
)
from nemo_testcase_l2_gyre_stage3_completion_gate import (
    read_stage2_hpg_operands,
)
from nemo_testcase_phase3_eos_gate import (
    nemo_literal_density,
    parse_teos10_density_coefficients,
)
from nemo_testcase_state_ulp_probe import ulp_distance
from legoesm.ocean.fidelity.provenance import worktree_stamp

ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round11_oracle_stage2_v5"
)
OPERAND_NPZ = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round11_composition_v4/hpg_operands.npz"
)
EOS_ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round12_oracle_eos_v2"
)

EOS_COEFFICIENT_NAMES = (
    "EOS000", "EOS100", "EOS200", "EOS300", "EOS400", "EOS500", "EOS600",
    "EOS010", "EOS110", "EOS210", "EOS310", "EOS410", "EOS510",
    "EOS020", "EOS120", "EOS220", "EOS320", "EOS420",
    "EOS030", "EOS130", "EOS230", "EOS330", "EOS040", "EOS140", "EOS240",
    "EOS050", "EOS150", "EOS060", "EOS001", "EOS101", "EOS201", "EOS301",
    "EOS401", "EOS011", "EOS111", "EOS211", "EOS311", "EOS021", "EOS121",
    "EOS221", "EOS031", "EOS131", "EOS041", "EOS002", "EOS102", "EOS202",
    "EOS012", "EOS112", "EOS022", "EOS003", "EOS103", "EOS013",
)
EOS_ARRAY_NAMES = (
    "T", "S", "pdep", "zh", "zt", "zs", "ztm",
    "zn0", "zn1", "zn2", "zn3", "zn", "prd",
)


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_hpg_literal(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, kmm, krhs, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_HPGLT_1", f"{path}: bad magic")
    require(
        (version, kt, kmm, krhs, nx, ny, nz, bits)
        == (1, 1, 3, 2, *DIMS, 64),
        f"{path}: bad header {header}",
    )
    n3 = nx * ny * nz
    n2 = nx * ny
    require(values.size == 6 * n3 + 2 * n2, f"{path}: bad payload")
    names = ("zhpi_u", "zhpi_v", "zuap_u", "zuap_v", "sum_u", "sum_v")
    result = {
        name: _xyz(values[i * n3:(i + 1) * n3], nx, ny, nz)
        for i, name in enumerate(names)
    }
    offset = 6 * n3
    result["r1_e1u"] = _xy(values[offset:offset + n2], nx, ny)
    result["r1_e2v"] = _xy(values[offset + n2:], nx, ny)
    return result


def read_eos_operands(path: Path) -> dict:
    """Read the config-local eos_insitu source-order stream."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, knn, kmm, krhs, nx, ny, nz, bits, neos = header
    require(magic == "NEMO_L2_EOSOP_1", f"{path}: bad magic")
    require(
        (version, kt, stage, knn, kmm, krhs, nx, ny, nz, bits, neos)
        == (1, 1, 2, 3, 3, 2, *DIMS, 64, -1),
        f"{path}: bad header {header}",
    )
    n3 = nx * ny * nz
    nscalar = 6 + len(EOS_COEFFICIENT_NAMES)
    require(
        values.size == nscalar + len(EOS_ARRAY_NAMES) * n3,
        f"{path}: bad payload {values.size}",
    )
    scalars = dict(zip(
        ("rdeltaS", "r1_S0", "r1_T0", "r1_Z0", "rho0", "r1_rho0"),
        values[:6], strict=True))
    coefficients = dict(zip(
        EOS_COEFFICIENT_NAMES,
        values[6:nscalar], strict=True))
    arrays = {}
    for index, name in enumerate(EOS_ARRAY_NAMES):
        start = nscalar + index * n3
        arrays[name] = _xyz(values[start:start + n3], nx, ny, nz)
    return {
        "header": {
            "version": version, "kt": kt, "stage": stage, "Knn": knn,
            "Kmm": kmm, "Krhs": krhs, "neos": neos,
        },
        "scalars": scalars,
        "coefficients": coefficients,
        "arrays": arrays,
    }


def _operand_row(name: str, oracle, candidate, mask) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape, f"{name}: shape mismatch")
    require(bool(use.any()), f"{name}: empty mask")
    require(np.all(np.isfinite(candidate[use])), f"{name}: nonfinite candidate")
    different = oracle[use] != candidate[use]
    absolute = np.abs(candidate[use] - oracle[use])
    return {
        "name": name,
        "status": "BIT-EXACT" if not bool(different.any()) else "DEBT",
        "absolute_max": float(absolute.max(initial=0.0)),
        "ulp_max": int(ulp_distance(candidate[use], oracle[use]).max(initial=0)),
        "differing_cells": int(different.sum()),
        "n": int(use.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }


def run_eos(oracle_root: Path, operand_npz: Path, plant: bool = False) -> dict:
    """Walk every wet-cell stage-2 eos_insitu operand in source order."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
    from legoesm.ocean.eos import (
        _ROQUET_TEOS10,
        nemo_bn2_live_ladders,
        nemo_teos10_density_anomaly_ratio,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    path = oracle_root / "oracle_rkstage2_eos_operands_kt00000001.bin"
    require(time_level_for_dump(path.name) == "now", "EOS dump is not Kmm/now")
    oracle_record = read_eos_operands(path)
    oracle = oracle_record["arrays"]
    hpg_operand_path = oracle_root / "oracle_rkstage2_hpg_operands_kt00000001.bin"
    hpg_operands = read_stage2_hpg_operands(hpg_operand_path)
    require(
        np.array_equal(oracle["prd"], hpg_operands["rhd"]),
        "EOS recorder did not capture the prd consumed by stage-2 dyn_hpg",
    )
    with np.load(operand_npz) as loaded:
        T_stage = np.asarray(loaded["candidate_stage1_T"])
        S_stage = np.asarray(loaded["candidate_stage1_S"])
        ssh_stage = np.asarray(loaded["candidate_stage1_ssh"])

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    nlev = card.recipe.z_coord.n_levels
    land = card.recipe.initial_state.land_mask.data
    T = neumann_fill_cgrid(jnp.asarray(T_stage), land, grid=card.recipe.grid)
    S = neumann_fill_cgrid(jnp.asarray(S_stage), land, grid=card.recipe.grid)
    eta = jnp.asarray(ssh_stage)
    H = card.recipe.initial_state.H_bathy.data

    def evaluate(t_value, s_value, depth):
        return jax.jit(lambda t, s, d, m: nemo_teos10_density_anomaly_ratio(
            t, s, jnp.zeros_like(d), rho0=cfg.rho_0,
            geometric_depth_m=d, tmask=m,
            return_intermediates=True))(
                t_value, s_value, depth, jnp.asarray(active))

    quotient_depth = nemo_bn2_live_ladders(
        card.recipe.z_coord, eta, H, r3t_evaluation="quotient")[0]
    reciprocal_depth = nemo_bn2_live_ladders(
        card.recipe.z_coord, eta, H, r3t_evaluation="nemo_reciprocal")[0]
    candidate_sets = {
        "production_quotient": tuple(
            np.asarray(x) for x in evaluate(T, S, quotient_depth)),
        "reciprocal_arm": tuple(
            np.asarray(x) for x in evaluate(T, S, reciprocal_depth)),
    }
    source_coefficients = parse_teos10_density_coefficients()
    numpy_values = nemo_literal_density(
        np.asarray(T), np.asarray(S), np.asarray(reciprocal_depth),
        source_coefficients, return_intermediates=True)
    np_depth, np_zh, np_zt, np_zs, np_zn0, np_zn1, np_zn2, np_zn3, np_zn = numpy_values
    np_ztm = active.astype(np.float64)
    np_prd = (np_zn * oracle_record["scalars"]["r1_rho0"] - 1.0) * np_ztm
    candidate_sets["numpy_reciprocal"] = (
        np.asarray(T), np.asarray(S), np_depth, np_zh, np_zt, np_zs, np_ztm,
        np_zn0, np_zn1, np_zn2, np_zn3, np_zn, np_prd,
    )
    oracle_T = jnp.asarray(oracle["T"][..., :nlev])
    oracle_S = jnp.asarray(oracle["S"][..., :nlev])
    oracle_depth = jnp.asarray(oracle["pdep"][..., :nlev])
    candidate_sets["oracle_input_jit"] = tuple(
        np.asarray(x) for x in evaluate(oracle_T, oracle_S, oracle_depth))
    oracle_numpy_values = nemo_literal_density(
        np.asarray(oracle_T), np.asarray(oracle_S), np.asarray(oracle_depth),
        source_coefficients, return_intermediates=True)
    (on_depth, on_zh, on_zt, on_zs, on_zn0, on_zn1, on_zn2, on_zn3,
     on_zn) = oracle_numpy_values
    on_ztm = active.astype(np.float64)
    on_prd = (on_zn * oracle_record["scalars"]["r1_rho0"] - 1.0) * on_ztm
    candidate_sets["oracle_input_numpy"] = (
        np.asarray(oracle_T), np.asarray(oracle_S), on_depth, on_zh, on_zt,
        on_zs, on_ztm, on_zn0, on_zn1, on_zn2, on_zn3, on_zn, on_prd,
    )

    rows = []
    mask = active[..., :nlev]
    first_departure = {}
    for set_name, values in candidate_sets.items():
        for index, (name, candidate) in enumerate(zip(EOS_ARRAY_NAMES, values, strict=True)):
            trial = np.asarray(candidate)[..., :nlev]
            if plant and set_name == "production_quotient" and index == 0:
                trial = trial.copy()
                trial[tuple(np.argwhere(mask)[0])] += 1.0
            row = _operand_row(
                f"{CASE}.kt1.stage2.eos.{set_name}.{name}",
                oracle[name][..., :nlev], trial, mask)
            rows.append(row)
            if row["status"] != "BIT-EXACT" and set_name not in first_departure:
                first_departure[set_name] = name

    scalar_expected = {
        "rdeltaS": _ROQUET_TEOS10["rdeltaS"],
        "r1_S0": _ROQUET_TEOS10["r1_S0"],
        "r1_T0": _ROQUET_TEOS10["r1_T0"],
        "r1_Z0": _ROQUET_TEOS10["r1_Z0"],
        "rho0": cfg.rho_0,
        "r1_rho0": 1.0 / cfg.rho_0,
    }
    scalar_rows = []
    for name, expected in scalar_expected.items():
        observed = oracle_record["scalars"][name]
        scalar_rows.append(_operand_row(
            f"scalar.{name}", np.asarray([observed]), np.asarray([expected]),
            np.asarray([True])))
    for name in EOS_COEFFICIENT_NAMES:
        scalar_rows.append(_operand_row(
            f"coefficient.{name}",
            np.asarray([oracle_record["coefficients"][name]]),
            np.asarray([_ROQUET_TEOS10[name]]), np.asarray([True])))

    production_prd = next(
        row for row in rows if row["name"].endswith("production_quotient.prd"))
    oracle_input_prd = next(
        row for row in rows if row["name"].endswith("oracle_input_jit.prd"))
    if oracle_input_prd["status"] == "BIT-EXACT":
        owner_label = "CONFIRMED_XLA_SOURCE_ASSOCIATION_OWNER_FIXED"
    else:
        owner_label = "EOS_LITERAL_ASSOCIATION_REMAINS_OPEN"

    if plant:
        planted = next(row for row in rows if row["name"].endswith("production_quotient.T"))
        require(planted["status"] == "DEBT" and planted["absolute_max"] >= 1.0,
                "planted EOS operand violation did not fire")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round12-eos-v1",
        "case": CASE,
        "status": (
            "BIT-EXACT-OUTPUT"
            if production_prd["status"] == "BIT-EXACT" else "DEBT"),
        "owner_label": owner_label,
        "production_prd_bit_exact": production_prd["status"] == "BIT-EXACT",
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "oracle_dump": str(path),
        "oracle_dump_sha256": sha256(path),
        "hpg_operand_dump": str(hpg_operand_path),
        "hpg_operand_dump_sha256": sha256(hpg_operand_path),
        "eos_prd_equals_hpg_rhd_bits": True,
        "candidate_stage_artifact": str(operand_npz),
        "candidate_stage_artifact_sha256": sha256(operand_npz),
        "header": oracle_record["header"],
        "first_non_bit_exact_operand": first_departure,
        "rows": rows,
        "scalar_and_coefficient_rows": scalar_rows,
        "planted_control": plant,
    }


def run(oracle_root: Path, operand_npz: Path, plant: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_hpg_sco_literal_cgrid,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    literal_path = oracle_root / "oracle_rkstage2_hpg_literal_kt00000001.bin"
    operand_path = oracle_root / "oracle_rkstage2_hpg_operands_kt00000001.bin"
    literal = read_hpg_literal(literal_path)
    oracle_operands = read_stage2_hpg_operands(operand_path)
    with np.load(operand_npz) as loaded:
        candidate_operands = {
            name: np.asarray(loaded[f"candidate_{name}"])
            for name in ("rhd", "e3w", "gdept_z0")
        }

    card = build_nemo_testcase_card(CASE)
    masks = expected_masks(card)

    def evaluate(operands):
        return tuple(np.asarray(x) for x in nemo_hpg_sco_literal_cgrid(
            jnp.asarray(operands["rhd"]),
            jnp.asarray(operands["e3w"]),
            jnp.asarray(operands["gdept_z0"]),
            card.recipe.grid,
            card.recipe.model_config.constants.g,
            return_components=True,
        ))

    candidate = evaluate(candidate_operands)
    exact_operand = evaluate(oracle_operands)
    candidate_names = ("sum_u", "sum_v", "zhpi_u", "zhpi_v", "zuap_u", "zuap_v")
    candidate_map = dict(zip(candidate_names, candidate))
    exact_map = dict(zip(candidate_names, exact_operand))
    # The shared operator returns legoESM's redundant west/south faces.  The
    # WRITE-only NEMO record and gate masks use its owned east/north A2D cells.
    candidate_map = {
        name: value[:, 1:] if name.endswith("_u") else value[1:]
        for name, value in candidate_map.items()
    }
    exact_map = {
        name: value[:, 1:] if name.endswith("_u") else value[1:]
        for name, value in exact_map.items()
    }
    oracle_map = {name: literal[name] for name in candidate_names}

    rows = []
    nlev = card.recipe.z_coord.n_levels
    for source_name, values in (
        ("candidate_operands", candidate_map),
        ("oracle_operands", exact_map),
    ):
        for name in candidate_names:
            component = name[-1]
            trial = values[name][..., :nlev]
            if plant and source_name == "candidate_operands" and name == "sum_u":
                trial = trial.copy()
                trial[tuple(np.argwhere(masks[component])[0])] += 1.0
            row = score(
                f"{CASE}.kt1.stage2.hpg_literal.{source_name}.{name}",
                oracle_map[name][..., :nlev], trial, masks[component],
            )
            row["effective_stage_output_bar"] = BAR / 7200.0
            row["effective_status"] = (
                "AT-EFFECTIVE-BAR"
                if row["absolute_max"] <= BAR / 7200.0 else "DEBT")
            rows.append(row)

    metric_rows = [
        score(
            f"{CASE}.kt1.stage2.hpg_literal.metric.r1_e1u",
            literal["r1_e1u"],
            1.0 / np.asarray(card.recipe.grid.dx_u[:, 1:]),
            np.ones_like(literal["r1_e1u"], dtype=bool),
        ),
        score(
            f"{CASE}.kt1.stage2.hpg_literal.metric.r1_e2v",
            literal["r1_e2v"],
            1.0 / np.asarray(card.recipe.grid.dy_v[1:, :]),
            np.ones_like(literal["r1_e2v"], dtype=bool),
        ),
    ]
    exact_sum_rows = [row for row in rows if ".oracle_operands.sum_" in row["name"]]
    candidate_sum_rows = [row for row in rows if ".candidate_operands.sum_" in row["name"]]
    if all(row["effective_status"] == "AT-EFFECTIVE-BAR" for row in exact_sum_rows):
        owner = "CONFIRMED_UPSTREAM_OPERAND_OWNER"
    elif max(row["absolute_max"] for row in exact_sum_rows) < (
            0.1 * max(row["absolute_max"] for row in candidate_sum_rows)):
        owner = "PLAUSIBLE_UPSTREAM_OPERAND_DOMINANT"
    else:
        owner = "HPG_LITERAL_ARITHMETIC_REMAINS_OPEN"
    if plant:
        planted = next(row for row in rows if ".candidate_operands.sum_u" in row["name"])
        require(planted["status"] == "DEBT" and planted["absolute_max"] >= 1.0,
                "planted HPG literal violation did not fire")

    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round11-hpg-v1",
        "case": CASE,
        "status": "DEBT",
        "owner_label": owner,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "stage_rDt_s": 7200.0,
        "effective_tendency_bar": BAR / 7200.0,
        "oracle_literal": str(literal_path),
        "oracle_literal_sha256": sha256(literal_path),
        "oracle_operands": str(operand_path),
        "oracle_operands_sha256": sha256(operand_path),
        "candidate_operands": str(operand_npz),
        "candidate_operands_sha256": sha256(operand_npz),
        "rows": rows,
        "metric_rows": metric_rows,
        "planted_control": plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hpg", "eos"), default="hpg")
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--operand-npz", type=Path, default=OPERAND_NPZ)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    oracle_root = (
        EOS_ORACLE_ROOT
        if args.mode == "eos" and args.oracle_root == ORACLE_ROOT
        else args.oracle_root
    )
    report = (
        run_eos(oracle_root, args.operand_npz, args.plant)
        if args.mode == "eos"
        else run(oracle_root, args.operand_npz, args.plant)
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
