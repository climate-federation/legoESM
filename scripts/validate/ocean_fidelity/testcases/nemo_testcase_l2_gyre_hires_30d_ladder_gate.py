#!/usr/bin/env python3
"""Certified CPU exact gate for r4_smag day-boundary stage entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_hires_ladder_gate import (  # noqa: E402
    ADMISSIONS_PATH,
    FIELDS,
    NEMO_DIMS,
    coefficient_branch_census,
    exact_score,
)
from nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E402
    expected_masks,
    lego_fields,
    read_entry,
)

DEFAULT_ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round6/oracle_gyre_r4_smag_daily_stage/run"
)
DEFAULT_SCALAR_ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round7/oracle_gyre_r4_smag_daily_stage_scalar/run"
)
STEPS_PER_DAY = 24
NDAYS = 30


def admit_daily_oracle(oracle_root: Path, math_flavour: str) -> dict:
    """Admit the immutable daily sequence and its compiler-math flavour."""
    registry = json.loads(ADMISSIONS_PATH.read_text(encoding="utf-8"))
    key = f"r4_smag_30d:{math_flavour}"
    try:
        admission = registry["admissions"][key]
    except KeyError as exc:
        raise ValueError(f"unregistered daily oracle {key!r}") from exc
    root = Path(admission["root"])
    if root.resolve() != oracle_root.resolve():
        raise ValueError(f"{key}: unregistered oracle root {oracle_root}")
    manifest = Path(admission["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"ACQUISITION_NEEDED: {manifest}")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(manifest) != admission["manifest_sha256"]:
        raise ValueError(f"{key}: manifest digest mismatch")
    checked = {}
    for relative, expected in admission["files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"{key}: digest mismatch for {relative}")
        checked[relative] = actual
    return {**admission, "key": key, "checked_files": checked}


def day_boundary_schedule(ndays: int = NDAYS) -> tuple[int, ...]:
    """Return entry indices for initial state plus each completed day."""
    if ndays < 1:
        raise ValueError("ndays must be positive")
    return tuple(1 + STEPS_PER_DAY * day for day in range(ndays + 1))


def register_daily_entry_time_levels(ndays: int = NDAYS) -> None:
    """Register only the day-boundary dumps emitted by the daily writer."""
    from legoesm.ocean.fidelity.time_levels import register_dump

    source = (
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
        "GYRE_OMIP_L2_P3_DAILY/BLD/ppsrc/nemo/stprk3.f90:90-100 writes "
        "ts/uu/vv/ssh(...,Nbb) at kt=1,25,...,721"
    )
    for kt in day_boundary_schedule(ndays):
        register_dump(
            f"oracle_step_entry_kt{kt:08d}.bin", "before", source)


def run(
    oracle_root: Path,
    output: Path,
    *,
    max_day: int = NDAYS,
    math_flavour: str = "vector",
) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_vertex_mask,
        nemo_smagorinsky_bilaplacian_coefficients_cgrid,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("30-day ladder certification requires CPU")
    if not 1 <= max_day <= NDAYS:
        raise ValueError(f"max_day must be in 1..{NDAYS}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    admission = admit_daily_oracle(oracle_root, math_flavour)
    register_daily_entry_time_levels(max_day)

    final_kt = day_boundary_schedule(max_day)[-1]
    namelist = GYRE_R4_SMAG_NAMELIST._replace(
        nn_itend=final_kt, nn_stock=final_kt, nn_write=STEPS_PER_DAY)
    card = build_nemo_testcase_card("GYRE-zco", namelist=namelist)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    masks = expected_masks(card)
    state = card.recipe.initial_state
    boundaries = set(day_boundary_schedule(max_day))
    rows_by_day = []
    controls = []
    day30_branch_census = None
    cell3 = (
        np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
        * np.asarray(state.land_mask.data)[..., None]
    )
    vertex_mask = jax.vmap(
        lambda level: compute_vertex_mask(level, grid=card.recipe.grid),
        in_axes=-1, out_axes=-1,
    )(cell3)

    for kt in range(1, final_kt + 1):
        if kt in boundaries:
            day = (kt - 1) // STEPS_PER_DAY
            path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
            if not path.is_file():
                raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
            oracle = read_entry(path, dims=NEMO_DIMS)
            if (oracle["kt"], oracle["Nbb"]) != (kt, 1):
                raise ValueError(f"{path}: expected odd kt with Nbb=1")
            candidate = lego_fields(state)
            rows = []
            for field in FIELDS:
                reference = (
                    oracle[field] if field == "ssh" else oracle[field][..., :30])
                rows.append(exact_score(
                    f"GYRE-zco-r4_smag.day{day}.{field}",
                    reference, candidate[field], masks[field]))
                if day == 0:
                    planted = exact_score(
                        f"control.one_ulp.{field}", reference,
                        candidate[field], masks[field], plant_one_ulp=True)
                    controls.append({
                        **planted,
                        "status": "FIRED" if planted["n_unequal"] else "MISSED",
                    })
            rows_by_day.append({"day": day, "kt": kt, "Nbb": 1, "rows": rows})
            if day == NDAYS:
                # Rule 12: classify the coefficient on NEMO's own before
                # velocity, not on the already-diverged legoESM trajectory.
                u_input = np.concatenate(
                    [np.zeros((oracle["u"].shape[0], 1, 30)),
                     oracle["u"][..., :30]], axis=1)
                v_input = np.concatenate(
                    [np.zeros((1, oracle["v"].shape[1], 30)),
                     oracle["v"][..., :30]], axis=0)
                _, got_f, diagnostics = (
                    nemo_smagorinsky_bilaplacian_coefficients_cgrid(
                        jnp.asarray(u_input), jnp.asarray(v_input),
                        card.recipe.grid, 3.5, card.dt_s,
                        mask=state.land_mask.data,
                        cell_mask=jnp.asarray(cell3),
                        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
                        vertex_mask=vertex_mask, return_diagnostics=True,
                    )
                )
                day30_branch_census = coefficient_branch_census(
                    jax.tree_util.tree_map(np.asarray, diagnostics),
                    np.asarray(cell3, dtype=bool),
                    np.asarray(got_f != 0.0, dtype=bool),
                )
        if kt == final_kt:
            break
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        state = model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        # Bound device retention during this intentionally per-step certified
        # path.  This is not the fused GPU long-run program.
        if kt % STEPS_PER_DAY == 0:
            state = jax.tree_util.tree_map(
                lambda leaf: np.asarray(leaf)
                if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
                state,
            )

    passed = (
        all(row["status"] == "AT-BAR"
            for boundary in rows_by_day for row in boundary["rows"])
        and all(control["status"] == "FIRED" for control in controls)
    )
    report = {
        "format": "nemo-testcase-l2-gyre-r4-smag-30d-exact-ladder-v1",
        "worktree": worktree_stamp(),
        "case": "GYRE-zco-r4_smag",
        "oracle_root": str(oracle_root),
        "oracle_admission": admission,
        "oracle_math_flavour": math_flavour,
        "policy": "fp64/x64/cpu/libm certified; per-step loop",
        "entry_semantics": (
            "kt=1 is initial state; kt=1+24*d is the state after d complete days"
        ),
        "acceptance": "zero unequal active float64 values at all 31 boundaries",
        "namelist": namelist._asdict(),
        "boundaries": rows_by_day,
        "day30_smagorinsky_branch_census_on_NEMO_state": day30_branch_census,
        "flavour_disposition": (
            "REPORT_ONLY: vector-math NEMO is not the certified scalar-libm oracle"
            if math_flavour == "vector" else
            "CERTIFYING: scalar-libm NEMO oracle"
        ),
        "controls": controls,
        "verdict": "PASS" if passed else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-day", type=int, default=NDAYS)
    parser.add_argument("--math-flavour", choices=("vector", "scalar"),
                        default="vector")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, args.output, max_day=args.max_day,
                     math_flavour=args.math_flavour)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
