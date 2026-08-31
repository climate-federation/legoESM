#!/usr/bin/env python3
"""First-divergence trajectory gate for the certified NEMO testcases.

The NEMO records are the Nbb/before entry state.  legoESM's card initial state
therefore maps to kt=1 and one completed ``model.step`` maps to kt=2.  The gate
stops immediately after the first step containing an over-bar field.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

BAR = 1.0e-15
DEFAULT_ORACLE_ROOTS = {
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_3"),
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_3"),
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def read_entry(path: Path, case: str) -> dict:
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, step, nbb, nx, ny, nz, ntr, bits = struct.unpack(
            "=8i", fh.read(32))
        data = np.fromfile(fh, dtype=np.float64)
    expected = (206, 7, 101) if case == "OVERFLOW-zps" else (134, 7, 21)
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *expected, 2, 64),
        f"{path}: bad header")
    count = nx * ny * nz
    require(data.size == 4 * count + nx * ny, f"{path}: bad payload length")

    def xyz(values):
        return values.reshape((nx, ny, nz), order="F")[
            2:-2, 2:-2].transpose(1, 0, 2)

    return {
        "step": step,
        "Nbb": nbb,
        "T": xyz(data[:count]),
        "S": xyz(data[count:2 * count]),
        "u": xyz(data[2 * count:3 * count]),
        "v": xyz(data[3 * count:4 * count]),
        "ssh": data[4 * count:].reshape((nx, ny), order="F")[
            2:-2, 2:-2].T,
    }


def expected_masks(card):
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    v = active & np.roll(active, -1, axis=0)
    v[-1] = False
    return {"T": active, "S": active, "u": u, "v": v, "ssh": wet}


def lego_fields(state):
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        # NEMO's local interior stores one x record per T column.  The phase-2
        # geometry gate established these stagger mappings exactly.
        "u": np.asarray(state.u.data)[:, 1:, :],
        "v": np.asarray(state.v.data)[1:, :, :],
        "ssh": np.asarray(state.eta.data),
    }


def score(
    name: str, oracle, candidate, mask, *, plant=False,
    allow_empty_no_active_face=False,
) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate is {candidate.dtype}")
    no_active_face = not bool(use.any())
    if no_active_face:
        require(allow_empty_no_active_face, f"{name}: empty mask")
        # Inventory the structurally absent face array and retain a gross
        # nonzero control over all stored points. It is not an alignment row.
        use = np.ones(oracle.shape, dtype=bool)
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(use)[0])] += 1.0
    require(np.all(np.isfinite(candidate[use])), f"{name}: candidate nonfinite")
    exact = bool(np.array_equal(oracle[use], candidate[use]))
    scale = max(float(np.max(np.abs(oracle[use]))), 1.0)
    error = float(np.max(np.abs(candidate[use] - oracle[use]))) / scale
    status = "AT-BAR" if error <= BAR else "DEBT"
    row = {
        "name": name,
        "status": status,
        "exact": exact,
        "normalized_max_abs": error,
        "bar": BAR,
        "n": int(use.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if no_active_face and status == "AT-BAR":
        row["status"] = "UNMEASURED"
        row["reason"] = (
            "no active meridional velocity face in the three-row closed tank; "
            "all stored values were nevertheless checked against zero")
    return row


def run(case: str, oracle_root: Path, max_step: int, *, plant=False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_nemo_testcase_card(case)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    steps = []
    exact_prefix = True
    first_over_bar = None

    for kt in range(1, max_step + 1):
        path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        require(path.is_file(), f"missing {path}")
        oracle = read_entry(path, case)
        require(oracle["step"] == kt, f"{path}: step mismatch")
        candidate = lego_fields(state)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(oracle[field])
            if field != "ssh":
                reference = reference[..., :nlev]
            rows.append(score(
                f"{case}.kt{kt}.before.{field}", reference,
                candidate[field], masks[field], plant=plant and kt == 1
                and field == "T", allow_empty_no_active_face=field == "v"))
        exact_here = all(row["exact"] for row in rows)
        over = [row["name"].rsplit(".", 1)[-1] for row in rows
                if row["status"] == "DEBT"]
        steps.append({
            "kt": kt,
            "Nbb": oracle["Nbb"],
            "exact_prefix_entering": exact_prefix,
            "exact_at_step": exact_here,
            "rows": rows,
        })
        exact_prefix = exact_prefix and exact_here
        if over:
            first_over_bar = {"kt": kt, "fields": over}
            break
        if kt < max_step:
            state = model.step(state, dt=card.dt_s)

    cfg = card.recipe.model_config
    return {
        "format": "nemo-testcase-l1-phase3-trajectory-v1",
        "case": case,
        "status": "AT-BAR" if first_over_bar is None else "DEBT",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "bar": BAR,
        "oracle_root": str(oracle_root),
        "selectors": {
            "eos": cfg.eos,
            "eos_depth": cfg.eos_depth,
            "barotropic_time_filter": cfg.barotropic.barotropic_time_filter,
            "n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps,
            "vertical_momentum_scheme": cfg.vertical_momentum_scheme,
            "tracer_time_integrator": cfg.tracer_time_integrator,
        },
        "first_over_bar": first_over_bar,
        "steps": steps,
        "unmeasured": [
            "NEMO per-term tendencies at the first divergent step",
            "OVERFLOW BBL transport until LOCK is owned",
            "trajectory after the first over-bar step",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(DEFAULT_ORACLE_ROOTS), required=True)
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    require(args.max_step >= 1, "max-step must be positive")
    report = run(
        args.case, args.oracle_dir or DEFAULT_ORACLE_ROOTS[args.case],
        args.max_step, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)
