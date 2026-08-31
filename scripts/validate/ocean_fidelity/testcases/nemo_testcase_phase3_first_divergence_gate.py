#!/usr/bin/env python3
"""Own the LOCK kt=2 first divergence with stage-level oracle evidence.

This gate is intentionally diagnostic and red.  It establishes that the live
EOS/HPG stage-1 RHS is at the campaign bar, then inventories the first active
program mismatch: NEMO's stage-3 tracer operator consumes the stage-2 Kmm
transport, whereas legoESM's split step builds every tracer substage from the
single final corrected velocity.  The source/call-site register is written to
the JSON report; numerical rows are computed from the committed instrument's
binary stage records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


BAR = 1.0e-15
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_stage_kt1")
NEMO_ROOT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
EXPECTED_DIMS = (134, 7, 21)
EXPECTED_LEVELS = {
    1: {"Kaa": 3, "Kmm": 1},
    2: {"Kaa": 2, "Kmm": 3},
    3: {"Kaa": 3, "Kmm": 2},
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[
        2:-2, 2:-2].transpose(1, 0, 2)


def read_stage(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, level, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *EXPECTED_DIMS, 2, 64),
        f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload length")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "kt": kt,
        "stage": stage,
        "Kaa": level,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count:2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count:3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count:4 * count], nx, ny, nz),
        "ssh": values[4 * count:].reshape((nx, ny), order="F")[2:-2, 2:-2].T,
    }


def read_transport(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, level, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_TRANSP_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, bits) == (1, *EXPECTED_DIMS, 64),
        f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 3 * count, f"{path}: bad payload length")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "kt": kt,
        "stage": stage,
        "Kmm": level,
        "Fu": _xyz(values[:count], nx, ny, nz),
        "Fv": _xyz(values[count:2 * count], nx, ny, nz),
        "Fw": _xyz(values[2 * count:], nx, ny, nz),
    }


def read_rhs(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, level, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, bits) == (1, *EXPECTED_DIMS, 64),
        f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 2 * count, f"{path}: bad payload length")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "kt": kt,
        "level": level,
        "u": _xyz(values[:count], nx, ny, nz),
        "v": _xyz(values[count:], nx, ny, nz),
    }


def score(name: str, oracle, candidate, mask, *, plant=False) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate is {candidate.dtype}")
    require(bool(active.any()), f"{name}: empty mask")
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(active)[0])] += 1.0
    require(np.all(np.isfinite(candidate[active])), f"{name}: candidate non-finite")
    scale = max(float(np.max(np.abs(oracle[active]))), 1.0)
    error = float(np.max(np.abs(candidate[active] - oracle[active]))) / scale
    return {
        "name": name,
        "status": "AT-BAR" if error <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "normalized_max_abs": error,
        "bar": BAR,
        "n": int(active.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }


def validate_registry(stages: dict, transports: dict, *, plant=False) -> list[dict]:
    rows = []
    for stage in (1, 2, 3):
        kaa = stages[stage]["Kaa"]
        kmm = transports[stage]["Kmm"]
        if plant and stage == 1:
            kmm = 2
        expected = EXPECTED_LEVELS[stage]
        ok = (
            stages[stage]["kt"] == transports[stage]["kt"] == 1
            and stages[stage]["stage"] == transports[stage]["stage"] == stage
            and kaa == expected["Kaa"]
            and kmm == expected["Kmm"]
        )
        rows.append({
            "name": f"LOCK_EXCHANGE-zco.stage{stage}.time_levels",
            "status": "VERIFIED" if ok else "DEBT",
            "observed": {"Kaa": kaa, "Kmm": kmm},
            "expected": expected,
        })
    return rows


def expected_masks(card) -> dict:
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    return {"u": u, "T": active}


def run(root: Path, *, plant_rhs=False, plant_registry=False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_lock_exchange_zco_card

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_lock_exchange_zco_card()
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    stages = {}
    transports = {}
    artifacts = {}
    for stage in (1, 2, 3):
        stage_path = root / f"oracle_stage_kt00000001_s{stage}.bin"
        transport_path = root / f"oracle_transport_kt00000001_s{stage}.bin"
        require(stage_path.is_file(), f"missing {stage_path}")
        require(transport_path.is_file(), f"missing {transport_path}")
        stages[stage] = read_stage(stage_path)
        transports[stage] = read_transport(transport_path)
        artifacts[stage_path.name] = sha256(stage_path)
        artifacts[transport_path.name] = sha256(transport_path)
    rhs_path = root / "oracle_rhs_kt00000001.bin"
    require(rhs_path.is_file(), f"missing {rhs_path}")
    rhs = read_rhs(rhs_path)
    require((rhs["kt"], rhs["level"]) == (1, 3), "RHS is not kt=1/Nrhs=3")
    artifacts[rhs_path.name] = sha256(rhs_path)

    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    tendency = model.tendencies(
        card.recipe.initial_state, dt=card.dt_s, momentum_only=True)
    candidate_rhs_u = np.asarray(tendency.du_dt.data)[:, 1:, :]
    rows = [score(
        "LOCK_EXCHANGE-zco.kt1.stage1.full_u_rhs",
        rhs["u"][..., :nlev], candidate_rhs_u, masks["u"], plant=plant_rhs)]
    rows.extend(validate_registry(stages, transports, plant=plant_registry))

    # LOCK has e3u=1 m at every active level.  NEMO zFu is e2u*e3u times
    # the Kmm velocity plus its barotropic transport correction
    # (stprk3_stg.F90:257-303).  Compare that actual stage-3 tracer transport
    # with the final Kaa velocity that legoESM currently substitutes for every
    # tracer substage (ocean_model_latlon_cgrid.py:4554-4561,4618-4629).
    e2u = np.asarray(card.recipe.grid.dy_u)[:, 1:]
    stage3_transport_velocity = (
        transports[3]["Fu"][..., :nlev] / e2u[..., None])
    rows.append(score(
        "LOCK_EXCHANGE-zco.kt1.stage3.Kmm_transport_vs_final_Kaa_velocity",
        stages[3]["u"][..., :nlev], stage3_transport_velocity, masks["u"]))

    source_files = {
        "stprk3": NEMO_ROOT / "src/OCE/stprk3.F90",
        "stprk3_stg": NEMO_ROOT / "src/OCE/stprk3_stg.F90",
        "traadv_fct": NEMO_ROOT / "src/OCE/TRA/traadv_fct.F90",
    }
    for name, path in source_files.items():
        require(path.is_file(), f"missing source {path}")
        artifacts[f"source:{name}"] = sha256(path)
    failed = [row["name"] for row in rows if row["status"] == "DEBT"]
    return {
        "format": "nemo-testcase-l1-phase3-first-divergence-v1",
        "case": card.case,
        "status": "AT-BAR" if not failed else "DEBT",
        "first_over_bar_step": 2,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "candidate_dtypes": {
            "rhs_u": str(candidate_rhs_u.dtype),
            "stage3_transport_velocity": str(stage3_transport_velocity.dtype),
        },
        "bar": BAR,
        "oracle_root": str(root),
        "rows": rows,
        "failed_rows": failed,
        "ownership": {
            "stage1_eos_hpg": (
                "measured by the full at-rest u RHS; advection, viscosity, and "
                "f=0 vorticity are structural zeros"),
            "tracer_transport_time_level": (
                "NEMO stage 3 consumes Kmm=2 zFu/zFv/zFw; legoESM's split "
                "step constructs one transport from final state_new.u/v and "
                "reuses it in all tracer substages"),
            "fct_stage_kernel": (
                "key_RK3 dispatches fct_up1_2stp, while legoESM fct2 uses its "
                "one-step low-order predictor inside the generic WS wrapper"),
        },
        "source_register": {
            "NEMO_stage_calls": "src/OCE/stprk3.F90:184-207",
            "NEMO_Kmm_transport": "src/OCE/stprk3_stg.F90:257-303",
            "NEMO_stage_barotropic_correction": "src/OCE/stprk3_stg.F90:433-446",
            "NEMO_tracer_transport_call": "src/OCE/stprk3_stg.F90:456-519",
            "NEMO_RK3_FCT_dispatch": "src/OCE/TRA/traadv_fct.F90:153-161",
            "NEMO_FCT_two_step": "src/OCE/TRA/traadv_fct.F90:470-641",
            "lego_final_transport": (
                "packages/ocean/legoesm/ocean/dynamics/"
                "ocean_model_latlon_cgrid.py:4554-4561,4618-4629"),
            "lego_WS_wrapper": (
                "packages/ocean/legoesm/ocean/dynamics/"
                "ocean_model_latlon_cgrid.py:5242-5251"),
        },
        "artifacts_sha256": artifacts,
        "unmeasured": [
            "individual FCT limiter coefficients and antidiffusive fluxes",
            "post-stage1 momentum split between per-stage barotropic correction and later RHS terms",
            "kt>=3 trajectory (stopped at first over-bar step)",
            "OVERFLOW-zps trajectory (LOCK dependency remains red)",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-rhs", action="store_true")
    parser.add_argument("--plant-registry", action="store_true")
    args = parser.parse_args()
    report = run(
        args.oracle_dir, plant_rhs=args.plant_rhs,
        plant_registry=args.plant_registry)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, OSError, UnicodeError, struct.error) as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)
