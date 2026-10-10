#!/usr/bin/env python3
"""Replay OMT-4's substep-3 SSH interpolation at alpha 0.07 and 0.09."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)

FLOOR = np.float64(2.0e-10)
TABLE_MAX = np.float64(0.0031271104752883805)
PLANTS = ("none", "deck-alpha", "alpha09-bit")


class GateError(RuntimeError):
    """The deck or literal replay no longer supports the alpha attribution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def resolved_filter(namelist: Path) -> tuple[int, float]:
    """Read active nn_bt_flt/rn_bt_alpha assignments, excluding comments."""

    values: dict[str, str] = {}
    for raw in namelist.read_text(encoding="utf-8").splitlines():
        line = raw.split("!", 1)[0]
        match = re.match(
            r"\s*(nn_bt_flt|rn_bt_alpha)\s*=\s*([^,\s]+)", line
        )
        if match:
            values[match.group(1)] = match.group(2)
    require(
        set(values) == {"nn_bt_flt", "rn_bt_alpha"},
        f"resolved barotropic-filter registry moved: {values}",
    )
    return int(values["nn_bt_flt"]), float(values["rn_bt_alpha"])


def _replay(operands: tuple[np.ndarray, ...], alpha: float):
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_ab3am4_coeff_arrays,
    )

    _, weights = nemo_ab3am4_coeff_arrays(3, alpha=alpha, ramp=True)
    w = weights[2]

    @jax.jit
    def literal(a, b, c, d):
        terms = tuple(
            nemo_source_round(w[index] * nemo_source_round(value))
            for index, value in enumerate((a, b, c, d))
        )
        answer = nemo_source_round(terms[0] + terms[1])
        answer = nemo_source_round(answer + terms[2])
        return nemo_source_round(answer + terms[3])

    arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in operands)
    return np.asarray(jax.device_get(literal(*arrays))), np.asarray(weights)


def _ulp_gap(left: float, right: float) -> int:
    values = np.asarray([left, right], dtype=np.float64).view(np.uint64)
    return int(abs(int(values[0]) - int(values[1])))


def measure(record: Path, namelist: Path, expect_commit: str,
            *, plant: str = "none") -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(plant in PLANTS, f"unknown plant {plant!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-235 alpha replay worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-235 alpha replay commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-235 alpha replay requires production JIT on CPU")

    nn_filter, deck_alpha = resolved_filter(namelist)
    if plant == "deck-alpha":
        deck_alpha = 0.07
    require((nn_filter, deck_alpha) == (3, 0.09),
            f"resolved OMT-4 filter moved: {(nn_filter, deck_alpha)}")

    oracle = phase3.read_bt_substeps(
        record, expected_dims=(94, 152), expected_ncycle=65)
    operands = (
        oracle["eta_exit"][2],
        oracle["eta_entry"][2],
        oracle["eta_entry"][1],
        oracle["eta_entry"][0],
    )
    target = np.asarray(oracle["eta_pgf"][2], dtype=np.float64)
    replay07, weights07 = _replay(operands, 0.07)
    replay09, weights09 = _replay(operands, deck_alpha)
    if plant == "alpha09-bit":
        replay09 = replay09.copy()
        replay09.flat[0] = np.nextafter(replay09.flat[0], np.inf)
    row07 = r228._difference(replay07, target)
    row09 = r228._difference(replay09, target)
    require(row09["unequal"] == 0,
            "deck-alpha replay is not bit-exact against recorded eta_pgf")
    require(row07["max_abs"] is not None and row07["max_abs"] > FLOOR,
            "shared-alpha control stayed at the floor")
    require(_ulp_gap(row07["max_abs"], float(TABLE_MAX)) <= 1,
            f"shared-alpha replay did not reproduce table maximum: {row07['max_abs']}")
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round235-bt-alpha-replay-v1",
        "status": "PASS_R235_BT_ALPHA_OWNER",
        "worktree": stamp,
        "execution": "offline-production-jit-cpu-fp64-x64-libm",
        "resolved_namelist": {"nn_bt_flt": nn_filter, "rn_bt_alpha": deck_alpha},
        "substep": 3,
        "source_statement": "dynspg_ts.f90:1522-1557,604-612",
        "alpha07_weights": weights07[2].tolist(),
        "alpha09_weights": weights09[2].tolist(),
        "alpha07_vs_nemo": row07,
        "alpha09_vs_nemo": row09,
        "alpha07_table_max_ulp_gap": _ulp_gap(
            row07["max_abs"], float(TABLE_MAX)),
        "predictions": {
            "R235-A1": "CONFIRMED_FILTER3_ALPHA0.09",
            "R235-A2": "CONFIRMED_BIT_EXACT",
            "R235-A3": "CONFIRMED_REPRODUCES_TABLE_MAX",
            "R235-A4": "CONFIRMED_DECISION_REQUIRED_BEFORE_CONFIG_FIELD",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--namelist", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.record, args.namelist, args.expect_commit, plant=args.plant)
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R235_BT_ALPHA_OWNER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
