#!/usr/bin/env python3
"""Cellwise ORCA2 gate for NEMO's frozen barotropic EEN coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (  # noqa: E402
    _nemo_literal_een_coefficients,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
)

NX, NY = 94, 152
OWNED_NX, OWNED_NY = NX - 4, NY - 4
FIELDS = (
    "ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
    "ffv_nw", "ffv_ne", "ffv_sw", "ffv_se",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_coefficients(path: Path) -> dict[str, np.ndarray]:
    """Read the rank-0 A2D(0) stream with an exact header/EOF walk."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L2_ENECO_1", f"bad EEN magic {magic!r}")
    require(header == (1, 1, 1, 3, NX, NY, 64), f"bad EEN header {header}")
    count = OWNED_NX * OWNED_NY
    require(values.size == len(FIELDS) * count, "bad EEN payload size")
    require(np.isfinite(values).all(), "non-finite EEN payload")
    return {
        name: values[index * count:(index + 1) * count].reshape(
            (OWNED_NX, OWNED_NY), order="F"
        ).T
        for index, name in enumerate(FIELDS)
    }


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict:
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.shape == expected.shape and actual.size, "empty EEN score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite defined EEN cell")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
    }


def validate(deck_root: Path, oracle_root: Path, *, plant: bool) -> dict:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "EEN gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    path = oracle_root / "oracle_bt_ene_coeff_kt00000001.bin"
    oracle = read_coefficients(path)
    card = build_orca2_zps_card(deck_root)
    require(card.recipe.model_config.vorticity_scheme == "een_total",
            "ORCA2 card does not select EEN")
    require(card.recipe.model_config.barotropic.barotropic_coriolis == "een_metric",
            "ORCA2 card does not select metric-complete barotropic EEN")

    # This is the production shared coefficient builder selected by
    # barotropic_een_coefficient_evaluation='nemo_literal'.  JIT the whole
    # eight-field source program; do not reimplement it in the gate.
    candidate = jax.jit(
        lambda eta: _nemo_literal_een_coefficients(
            eta, card.recipe.z_coord, jnp.float64, scheme="een")
    )(card.recipe.initial_state.eta.data)
    candidate = {name: np.asarray(value)[:, :OWNED_NX]
                 for name, value in candidate.items()}
    operands = card.recipe.z_coord.nemo_een_barotropic
    masks = {
        "u": np.asarray(operands.umask, dtype=bool)[:, :OWNED_NX].any(axis=-1),
        "v": np.asarray(operands.vmask, dtype=bool)[:, :OWNED_NX].any(axis=-1),
    }

    rows = []
    for name in FIELDS:
        mask = masks[name[2]]
        # The control starts from an exact oracle/oracle arm, so its nonzero
        # exit cannot be borrowed from the measured production debt below.
        trial = (oracle[name] if plant else candidate[name]).copy()
        if plant and name == "ffu_nw":
            index = tuple(np.argwhere(mask)[0])
            trial[index] = np.nextafter(trial[index], np.inf)
        rows.append({"field": name, **score(trial, oracle[name], mask)})
    if plant:
        planted = rows[0]
        require(planted["unequal"] == 1, "binding EEN plant did not fire once")
        raise GateError(
            "planted EEN coefficient cell rejected through scorer "
            f"({planted['unequal']}/{planted['count']})"
        )

    first = next((row["field"] for row in rows if row["status"] != "AT_BAR"), None)
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O4-EXT-A/dyn_cor_2D_init-frozen-EEN-coefficients",
        "result": "AT_BAR" if first is None else "DEBT",
        "first_over_bar_field": first,
        "owner": "GYRE_OWNER_SHARED_EXTERNAL_MODE",
        "rows": rows,
        "record": {"path": str(path), "sha256": sha256(path)},
        "execution": {
            "backend": jax.default_backend(),
            "jax_disable_jit": False,
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 owned wet A2D(0) U/V cells",
            "rank0_global_mapping": "card latitude all 148; longitude 0:90",
        },
        "source": {
            "selector": "dynvor.F90:889-892 (ln_dynvor_een -> np_EEN=3)",
            "call_order": "dynspg_ts.F90:306-326 (init, then WRITE-only dump)",
            "coefficient_program": "dynspg_ts.F90:1495-1569",
            "first_consumer": "dynspg_ts.F90:1674-1697",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, plant=args.plant)
    except (GateError, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
