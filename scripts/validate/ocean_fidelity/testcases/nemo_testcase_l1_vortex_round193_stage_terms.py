#!/usr/bin/env python3
"""Production-JIT VORTEX stage-2/3 momentum operator walk.

The NEMO record is cumulative.  HPG overwrites Krhs on the RK3 branch; VOR,
KEG and ZAD are successive differences.  legoESM publishes the already
computed component from the production-jitted step through its existing
WRITE-only operator exposure.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nemo_testcase_l1_vortex"))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _strip3, _u_full, _v_full, read_bt_frame, read_stage,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    BAR, GateError, expected_masks, lego_fields, read_entry, require, score,
)

CASE = "VORTEX_VEC-zco"
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round192/"
    "oracle_stage23_terms")
BOUNDARIES = ("hpg", "vor", "keg", "zad")


def read_stage_terms(root: Path, stage: int) -> dict[str, np.ndarray]:
    """Read every named group using the acquisition's self-describing parser."""
    checker_path = HERE / "nemo_testcase_l1_vortex" / "check_records.py"
    spec = importlib.util.spec_from_file_location("vortex_record_checker", checker_path)
    require(spec is not None and spec.loader is not None, "cannot load record checker")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    path = root / f"oracle_stage_terms_kt00000001_s{stage}.bin"
    parsed = checker.parse_record(path)
    require(parsed["stage"] == stage, f"{path}: parsed the wrong stage")
    raw = path.read_bytes()
    offset = 16 + 4 * 15
    out = {}
    for name, meta in parsed["groups"].items():
        offset += 32
        count = meta["doubles"]
        values = np.frombuffer(raw[offset:offset + 8 * count], dtype=np.float64).copy()
        offset += 8 * count
        shape = tuple(meta["shape"])
        if meta["rank"] == 3:
            out[name] = _strip3(values, *shape)
        else:
            nx, ny = shape
            out[name] = values.reshape((nx, ny), order="F")[2:-2, 2:-2].T
    require(offset == len(raw), f"{path}: parser did not consume the record")
    return out


def nemo_components(groups: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return NEMO's cumulative accumulator at each compiled boundary."""
    return {name: (groups[f"{name}_u"], groups[f"{name}_v"])
            for name in BOUNDARIES}


def _external(root: Path, entry2: dict):
    import jax.numpy as jnp
    frame = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin", expect_step=1)
    return (
        jnp.asarray(entry2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )


def run(root: Path, *, allow_dirty: bool = False, plant: str | None = None,
        source_order: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "the stage walk must run on CPU")

    valid_plants = tuple(f"s{s}.{op}.{face}" for s in (2, 3)
                         for op in BOUNDARIES for face in ("u", "v"))
    require(plant is None or plant in valid_plants,
            f"unknown plant {plant!r}; expected one of {valid_plants}")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASE,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASE,
                        expect_interior=interior)
    stages = {s: read_stage(root / f"oracle_stage_kt00000001_s{s}.bin",
                            expect_step=1, expect_stage=s) for s in (1, 2, 3)}
    external = _external(root, entry2)
    seed = _seed_from_record(card.recipe.initial_state, entry1, nlev)

    def stage_override(stage: int):
        source = stages[stage - 1]
        return (
            stage,
            jnp.asarray(_u_full(source["u"][..., :nlev])),
            jnp.asarray(_v_full(source["v"][..., :nlev])),
            jnp.asarray(source["T"][..., :nlev]),
            jnp.asarray(source["S"][..., :nlev]),
            jnp.asarray(source["ssh"]),
        )

    def model_step(hooks):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return model.step(seed, dt=card.dt_s)

    def owned(values, face):
        values = np.asarray(values.data if hasattr(values, "data") else values)
        return values[:, 1:, :nlev] if face == "u" else values[1:, :, :nlev]

    rows = []
    for stage in (2, 3):
        groups = read_stage_terms(root, stage)
        reference = nemo_components(groups)
        for operator in BOUNDARIES:
            hooks = _NEMOWSRK3TestHooks(
                stage_barotropic_output_override=external,
                stage_entry_override=stage_override(stage),
                expose_momentum_operator=f"after_{operator}",
                expose_momentum_operator_stage=stage,
                nemo_stage_rhs_accumulation_order_arm=source_order,
            )
            fields = lego_fields(model_step(hooks))
            for face, index in (("u", 0), ("v", 1)):
                candidate = np.asarray(fields[face])[..., :nlev]
                planted = plant == f"s{stage}.{operator}.{face}"
                if planted:
                    candidate = candidate.copy()
                    where = tuple(d // 2 for d in candidate.shape)
                    candidate[where] += 1.0
                ref = reference[operator][index][..., :nlev]
                row = score(f"{CASE}.stage{stage}.{operator}.{face}",
                            ref, candidate, masks[face])
                active = np.asarray(masks[face], dtype=bool)
                delta = candidate - ref
                row["cells_unequal"] = int(np.count_nonzero(
                    (candidate != ref)[active]))
                row["max_abs"] = float(np.max(np.abs(delta[active])))
                ref_peak = float(np.max(np.abs(ref[active])))
                row["relative_max_abs"] = row["max_abs"] / max(ref_peak, 1.0e-300)
                row["execution_regime"] = "production_step_jit"
                row["nemo_boundary"] = f"cumulative accumulator after {operator}"
                row["planted"] = planted
                # Operator-local exactness is bit equality. AT-BAR is retained
                # as a separate trajectory classification, never as exactness.
                row["bit_exact"] = row["cells_unequal"] == 0
                rows.append(row)

    # Existing stage-output calibration, still through the production step.
    calibration = []
    for stage in (2, 3):
        hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            stage_entry_override=stage_override(stage),
            expose_momentum_stage=stage if stage == 2 else 0,
        )
        fields = lego_fields(model_step(hooks))
        stage_rows = {}
        for face in ("u", "v"):
            row = score(f"{CASE}.stage{stage}.output.{face}",
                        np.asarray(stages[stage][face])[..., :nlev],
                        np.asarray(fields[face])[..., :nlev], masks[face])
            stage_rows[face] = row
        calibration.append({"stage": stage, "rows": stage_rows})

    first = None
    for stage in (2, 3):
        for operator in BOUNDARIES:
            selected = [r for r in rows if f".stage{stage}.{operator}." in r["name"]]
            if any(not r["bit_exact"] for r in selected):
                first = {"stage": stage, "operator": operator}
                break
        if first:
            break
    report = {
        "format": "nemo-testcase-l1-vortex-round193-stage-terms-v1",
        "case": CASE, "oracle_root": str(root), "legoesm_git_sha": sha,
        "precision_policy": "fp64/libm", "jax_backend": jax.default_backend(),
        "execution_regime": "production_step_jit", "bar": BAR,
        "source_order_arm": source_order,
        "plant": plant, "rows": rows, "calibration": calibration,
        "first_non_bit_operator": first,
        "status": "PLANT-FIRED" if plant else ("DEBT" if first else "BIT"),
    }
    if plant:
        planted_rows = [r for r in rows if r["planted"]]
        require(len(planted_rows) == 1 and planted_rows[0]["cells_unequal"] > 0
                and planted_rows[0]["max_abs"] > 1.0e-12,
                "planted output violation did not fire its own row")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--plant")
    parser.add_argument("--source-order", action="store_true",
                        help="enable the existing private NEMO accumulator-order arm")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, allow_dirty=args.allow_dirty,
                     plant=args.plant, source_order=args.source_order)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    for row in report["rows"]:
        print(f"{row['name']:<52} unequal={row['cells_unequal']:<7d} "
              f"max={row['max_abs']:.17e} exact={row['bit_exact']}")
    print("first_non_bit_operator:", report["first_non_bit_operator"])
    print("STATUS", report["status"])
    # Plants deliberately exit nonzero; log scrapers must never read PASS.
    return 1 if args.plant else 0


if __name__ == "__main__":
    sys.exit(main())
