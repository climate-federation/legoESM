#!/usr/bin/env python3
"""GYRE stage-2 HPG literal-operand discriminator (production JIT, fp64)."""

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

ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round11_oracle_stage2_v5"
)
OPERAND_NPZ = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round11_composition_v4/hpg_operands.npz"
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


def run(oracle_root: Path, operand_npz: Path, plant: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_hpg_sco_literal_cgrid,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
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
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--operand-npz", type=Path, default=OPERAND_NPZ)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.oracle_root, args.operand_npz, args.plant)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
