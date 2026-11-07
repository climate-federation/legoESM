#!/usr/bin/env python3
"""Score NEMO's separately prognostic depth-mean velocity after kt=1.

The oracle ``BTFRM`` record is written after ``stp_2D`` and contains
``uu_b/vv_b(:,:,Kaa)`` followed by ``un_adv/vn_adv``.  A NEMO-identity card
must reproduce the first pair bit for bit when both models enter from the
same kt=1 state.  Later pair rows are downstream trajectory consistency and
belong in the card-specific trajectory register instead of this equal-input
operator gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp


DEFAULT_ORACLE_ROOTS = {
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"),
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10"),
    "GYRE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
        "round19_oracle_v2_external"),
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bt_pair(path: Path) -> dict[str, np.ndarray | int]:
    """Read the Kaa pair from the shared lane-1 ``BTFRM`` schema."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        version, kt, kaa, nx, ny, bits = struct.unpack("=6i", handle.read(24))
        values = np.fromfile(handle, dtype=np.float64)
    require(magic == "NEMO_L1_BTFRM_1", f"{path}: bad magic {magic!r}")
    require((version, kt, kaa, bits) == (1, 1, 3, 64),
            f"{path}: bad header {(version, kt, kaa, nx, ny, bits)!r}")
    count = nx * ny
    require(values.size == 4 * count, f"{path}: bad payload length")

    def xy(block: np.ndarray) -> np.ndarray:
        return block.reshape((nx, ny), order="F")[2:-2, 2:-2].T

    return {
        "kt": kt,
        "Kaa": kaa,
        "uu_b": xy(values[:count]),
        "vv_b": xy(values[count:2 * count]),
    }


def score(
    name: str, oracle, candidate, mask, *, plant: bool,
    allow_empty_no_active_face: bool = False,
) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == mask.shape,
            f"{name}: shape mismatch")
    no_active_face = not bool(mask.any())
    if no_active_face:
        require(allow_empty_no_active_face, f"{name}: empty wet-face mask")
        mask = np.ones(oracle.shape, dtype=bool)
    if plant:
        candidate = candidate.copy()
        index = tuple(np.argwhere(mask)[0])
        scale = max(float(np.max(np.abs(oracle[mask]))), 1.0)
        candidate[index] += 3.0 * np.spacing(scale)
    unequal = oracle[mask].view(np.uint64) != candidate[mask].view(np.uint64)
    row = {
        "name": name,
        "status": "AT-BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "n": int(unequal.size),
        "absolute_max": float(
            np.max(np.abs(candidate[mask] - oracle[mask]), initial=0.0)),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
        "criterion": "bit identity on equal-input kt1 Kaa",
    }
    if no_active_face and row["status"] == "AT-BAR":
        row["status"] = "UNINFORMATIVE"
        row["reason"] = (
            "no active meridional velocity face; all stored values were "
            "nevertheless checked bitwise")
    return row


def run(case: str, oracle_root: Path, *, plant: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import git_sha

    revision = git_sha()
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy is not active")
    card = build_nemo_testcase_card(case)
    state = card.recipe.initial_state
    require(state.uu_b is not None and state.vv_b is not None,
            f"{case}: NEMO identity state has no uu_b/vv_b pair")
    record_path = oracle_root / "oracle_bt_frames_kt00000001.bin"
    require(record_path.is_file(), f"missing {record_path}")
    oracle = read_bt_pair(record_path)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    step_kwargs = {}
    if case == "GYRE-zco":
        gyre_gate_path = Path(__file__).with_name(
            "nemo_testcase_l2_gyre_phase3_gate.py")
        spec = importlib.util.spec_from_file_location(
            "gyre_phase3_surface_forcing", gyre_gate_path)
        require(spec is not None and spec.loader is not None,
                "cannot load the certified GYRE surface-forcing adapter")
        gyre_gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gyre_gate)
        freshwater, surface = gyre_gate._surface_forcings(card, state, 1)
        step_kwargs = {"freshwater": freshwater, "surface_forcing": surface}
    after = model.step(state, dt=card.dt_s, **step_kwargs)
    require(after.uu_b is not None and after.vv_b is not None,
            f"{case}: step dropped uu_b/vv_b")
    wet = np.asarray(state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active)[..., 0] & wet
    u_mask = active & np.roll(active, -1, axis=1)
    u_mask[:, -1] = False
    v_mask = active & np.roll(active, -1, axis=0)
    v_mask[-1] = False
    rows = [
        score(
            f"{case}.kt1.after.uu_b", oracle["uu_b"],
            np.asarray(after.uu_b.data)[:, 1:], u_mask, plant=plant),
        score(
            f"{case}.kt1.after.vv_b", oracle["vv_b"],
            np.asarray(after.vv_b.data)[1:, :], v_mask, plant=False,
            allow_empty_no_active_face=True),
    ]
    if plant:
        require(rows[0]["status"] == "DEBT" and rows[0]["unequal"] == 1,
                "three-ULP carried-state plant did not fire exactly once")
        raise GateError(
            f"planted carried-state violation rejected ({rows[0]['unequal']} / "
            f"{rows[0]['n']})")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-prognostic-barotropic-state-gate-v1",
        "status": "AT-BAR" if all(
            row["status"] in ("AT-BAR", "UNINFORMATIVE") for row in rows)
        else "DEBT",
        "case": case,
        "boundary": "kt1 external-mode Kaa uu_b/vv_b write",
        "rows": rows,
        "oracle_record": {
            "path": str(record_path), "sha256": sha256(record_path),
            "Kaa": oracle["Kaa"],
        },
        "execution": {
            "git_sha": revision, "backend": jax.default_backend(),
            "production_jit": True, "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(DEFAULT_ORACLE_ROOTS), required=True)
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            args.case, args.oracle_root or DEFAULT_ORACLE_ROOTS[args.case],
            plant=args.plant)
    except (GateError, OSError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    return 0 if result["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
