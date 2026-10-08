#!/usr/bin/env python3
"""Round-72 source-ordered replay of GYRE's kt=2 tracer stages 1 and 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round54_tracer_decomposition as round54  # noqa: E402
import nemo_testcase_l2_gyre_round66_content_operands as round66  # noqa: E402
import nemo_testcase_l2_gyre_round67_ldf_order as round67  # noqa: E402
import nemo_testcase_l2_gyre_round71_fct_stage2_gate as round71  # noqa: E402
from legoesm.ocean.dynamics import (  # noqa: E402
    ocean_model_latlon_cgrid as model_module,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
TRACERS = ("T", "S")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def to_lego(array: np.ndarray, nlev: int | None = None) -> np.ndarray:
    """Map an owned NEMO i,j[,k] array to legoESM j,i[,k]."""
    value = np.ascontiguousarray(np.asarray(array).swapaxes(0, 1))
    return value if nlev is None else value[..., :nlev]


def replay_update(
    fields: dict[str, np.ndarray], tracer: str, dt: np.float64,
    wet: np.ndarray, *, plant_kaa_ulp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate compiled stprk3_stg.f90:899-901 without reassociation."""
    qbb = np.float64(1.0) + to_lego(fields["r3t_Kbb"])[..., None]
    qmm = np.float64(1.0) + to_lego(fields["r3t_Kmm"])[..., None]
    qaa = np.float64(1.0) + to_lego(fields["r3t_Kaa"])[..., None]
    kbb = to_lego(fields[f"Kbb_{tracer}"], wet.shape[-1])
    krhs = to_lego(fields[f"after_sbc_{tracer}"], wet.shape[-1])
    expected = to_lego(fields[f"Kaa_{tracer}"], wet.shape[-1])
    if plant_kaa_ulp and tracer == "T":
        expected = np.array(expected, copy=True)
        index = tuple(np.argwhere(wet)[0])
        expected[index] = np.nextafter(expected[index], np.float64(np.inf))
    left = qbb * kbb
    tendency = dt * qmm
    tendency = tendency * krhs
    tendency = tendency * wet.astype(np.float64)
    replay = (left + tendency) / qaa
    return replay, expected


def _round66_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        expect_commit=args.expect_commit,
        expect_record_commit=args.expect_krhs_commit,
        output=args.output,
        entry_root=ROOT / "year_owners/nemo_seed0",
        stage_root=ROOT / "round46/oracle_kt2_stage",
        krhs_record=(ROOT / "round64/oracle_krhs_split/"
                     "oracle_krhs_split_kt00000002.bin"),
        krhs_stamp=(ROOT / "round64/oracle_krhs_split/"
                    "oracle_krhs_split_kt00000002.bin.stamp"),
        producer_commit=ROOT / "round64/oracle_krhs_split/producer_commit.txt",
        admission=ROOT / "round64/oracle_krhs_split/round64_admission.json",
    )


def _capture_seeded_context(args):
    captures = []
    real = round66._capture_final_content_call

    def wrapper(card, seeded, freshwater, surface):
        result = real(card, seeded, freshwater, surface)
        captures.append((card, seeded, freshwater, surface, result[0]))
        return result

    round66._capture_final_content_call = wrapper
    try:
        base = round66.measure(_round66_args(args))
    finally:
        round66._capture_final_content_call = real
    require(len(captures) == 1, "expected one oracle-seeded kt2 context")
    return base, captures[0]


def _live_trace(card, seeded, freshwater, surface):
    source_rates: list[tuple[np.ndarray, ...]] = []
    real_pair = model_module._nemo_ws_rk3_tracer_pair_step

    def sink(*values):
        source_rates.append(tuple(np.asarray(value) for value in values))

    def capture_pair(*values, **kwargs):
        if kwargs.get("stop_after_stage") == 1:
            flattened = tuple(
                value for pair in kwargs["stage_source_rates"] for value in pair)
            jax.debug.callback(sink, *flattened, ordered=True)
        return real_pair(*values, **kwargs)

    model_module._nemo_ws_rk3_tracer_pair_step = capture_pair
    try:
        model = model_module.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                expose_live_stage_operands=True),
        )
        trace = jax.device_get(model.step(
            seeded, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    finally:
        model_module._nemo_ws_rk3_tracer_pair_step = real_pair
    require(source_rates, "stage source-rate callback did not fire")
    for duplicate in source_rates[1:]:
        require(all(np.array_equal(got, want)
                    for got, want in zip(duplicate, source_rates[0], strict=True)),
                "stage source-rate callbacks differ")
    return trace, source_rates[0]


def _live_stage_rows(trace, sources, card, records, masks) -> tuple[dict, dict]:
    nlev = card.recipe.z_coord.n_levels
    active = jnp.asarray(card.recipe.z_coord.is_active, dtype=jnp.float64)
    rows: dict[str, dict] = {}
    arrays: dict[str, dict] = {}

    @jax.jit
    def cen2_rhs(tracer, p_u, p_v, p_w, thickness):
        return model_module._nemo_cen2_tracer_rhs(
            tracer, p_u, p_v, p_w, thickness, active, card.recipe.grid)

    for stage in (1, 2):
        record = records[stage]["fields"]
        state = trace.stage_states[stage - 1]
        geometry = trace.stage_geometry[stage - 1]
        qco_index = ((0, 0, 1) if stage == 1 else (0, 1, 2))
        live_qco = tuple(np.asarray(trace.stage_qco[index][0])
                         for index in qco_index)
        live_transport = (
            np.asarray(geometry[7])[:, 1:, :nlev],
            np.asarray(geometry[8])[1:, :, :nlev],
            np.asarray(geometry[2]) * np.asarray(card.recipe.grid.area_T)[..., None],
        )
        tracer_arrays = {}
        stage_rows = {
            "zero_T": round54.field_stats(
                np.zeros_like(to_lego(record["zero_T"], nlev)),
                to_lego(record["zero_T"], nlev), masks["T"]),
            "zero_S": round54.field_stats(
                np.zeros_like(to_lego(record["zero_S"], nlev)),
                to_lego(record["zero_S"], nlev), masks["S"]),
            "zFu": round54.field_stats(
                live_transport[0], to_lego(record["zFu"], nlev), masks["u"]),
            "zFv": round54.field_stats(
                live_transport[1], to_lego(record["zFv"], nlev), masks["v"]),
            "zFw": round54.field_stats(
                live_transport[2], to_lego(record["zFw"]),
                np.ones_like(live_transport[2], dtype=bool)),
        }
        for name, state_index in (("T", 2), ("S", 3)):
            live_kmm = np.asarray(state[state_index])
            live_kbb = np.asarray(trace.stage_states[0][state_index])
            rhs = jax.device_get(cen2_rhs(
                state[state_index], geometry[7], geometry[8],
                geometry[2] * jnp.asarray(card.recipe.grid.area_T)[..., None],
                geometry[3]))
            after_adv = np.asarray(rhs)
            after_sbc = after_adv + np.asarray(sources[(stage - 1) * 2
                                                       + TRACERS.index(name)])
            live_kaa = np.asarray(trace.stage_states[stage][state_index])
            tracer_arrays[name] = {
                "Kbb": live_kbb,
                "Kmm": live_kmm,
                "after_advection": after_adv,
                "after_sbc": after_sbc,
                "Kaa": live_kaa,
            }
            for boundary, live_value, record_name in (
                ("Kbb", live_kbb, f"Kbb_{name}"),
                ("Kmm", live_kmm, f"Kmm_{name}"),
                ("after_advection", after_adv, f"after_advection_{name}"),
                ("after_sbc", after_sbc, f"after_sbc_{name}"),
                ("Kaa", live_kaa, f"Kaa_{name}"),
            ):
                stage_rows[f"{boundary}_{name}"] = round54.field_stats(
                    live_value, to_lego(record[record_name], nlev), masks[name])
        for label, live_value, record_name in zip(
            ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa"), live_qco,
            ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa"), strict=True,
        ):
            stage_rows[label] = round54.field_stats(
                live_value, to_lego(record[record_name]), masks["T"][..., 0])
        rows[str(stage)] = stage_rows
        arrays[str(stage)] = {
            "transport": live_transport,
            "tracers": tracer_arrays,
        }
    return rows, arrays


def _first_non_bit(rows: dict) -> str | None:
    order = []
    for stage in ("1", "2"):
        order.extend(
            f"stage{stage}.{name}" for name in (
                "zero_T", "zero_S", "Kbb_T", "Kbb_S", "Kmm_T", "Kmm_S",
                "zFu", "zFv", "zFw", "after_advection_T",
                "after_advection_S", "after_sbc_T", "after_sbc_S",
                "r3t_Kbb", "r3t_Kmm", "r3t_Kaa", "Kaa_T", "Kaa_S"))
    for key in order:
        stage, name = key.split(".")
        if rows[stage.removeprefix("stage")][name]["cells_unequal"]:
            return key
    return None


def measure(args) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "round-72 producer worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "round-72 commit stamp mismatch")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")

    admission = json.loads((args.record_root / "round71_admission.json").read_text())
    require(admission["verdict"] == "PASS", "round-71 twin admission failed")
    require((admission["byte_identical_records"],
             len(admission["classified_changed_records"]),
             admission["admitted_difference_count"]) == (45, 20, 132),
            "round-71 admission census changed")
    producer = (args.record_root / "producer_commit.txt").read_text().strip()
    require(producer.lower() == args.expect_record_commit.lower(),
            "round-71 producer commit mismatch")
    records = {}
    for stage in (1, 2):
        path = args.record_root / (
            f"oracle_rktracer_operands_kt00000002_s{stage}.bin")
        round71.check_stamp(path.with_name(path.name + ".stamp"), producer)
        records[stage] = round71.read_record(path, stage)
    for name in TRACERS:
        require(np.array_equal(records[1]["fields"][f"Kbb_{name}"],
                               records[2]["fields"][f"Kbb_{name}"]),
                f"record Kbb {name} bridge changed")
        require(np.array_equal(records[1]["fields"][f"Kaa_{name}"],
                               records[2]["fields"][f"Kmm_{name}"]),
                f"record Kaa1/Kmm2 {name} bridge changed")

    base, context = _capture_seeded_context(args)
    card, seeded, freshwater, surface, round66_trace = context
    require(np.float64(card.dt_s) == np.float64(14400.0),
            f"resolved GYRE dt changed: {card.dt_s!r}")
    masks = gate.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels

    replay_rows = {}
    for stage, dt in ((1, np.float64(card.dt_s / 3.0)),
                      (2, np.float64(card.dt_s / 2.0))):
        replay_rows[str(stage)] = {}
        for name in TRACERS:
            replay, expected = replay_update(
                records[stage]["fields"], name, dt, masks[name],
                plant_kaa_ulp=args.plant_kaa_ulp)
            replay_rows[str(stage)][name] = round54.field_stats(
                replay, expected, masks[name])
    self_replay_exact = all(
        row["cells_unequal"] == 0
        for stage in replay_rows.values() for row in stage.values())

    trace, sources = _live_trace(card, seeded, freshwater, surface)
    require(round67.pytree_exact_census(
        trace.state_after, round66_trace.state_after)["cells_unequal"] == 0,
        "independent live trace changed ordinary state")
    live_rows, live_arrays = _live_stage_rows(
        trace, sources, card, records, masks)
    first_non_bit = _first_non_bit(live_rows)

    round71_kmm = {
        "T": (17994, 8.369461070856232e-7),
        "S": (16769, 6.794565621248694e-8),
    }
    kmm_reproduction = {}
    for name in TRACERS:
        row = live_rows["2"][f"Kaa_{name}"]
        expected_count, expected_max = round71_kmm[name]
        kmm_reproduction[name] = bool(
            row["cells_unequal"] == expected_count
            and row["max_abs"] == expected_max)

    stage1_exact = all(
        row["cells_unequal"] == 0 for row in live_rows["1"].values())
    predicted_boundary = first_non_bit in ("stage2.zFu", "stage2.zFv")
    causal = {"executed": False}
    if predicted_boundary or args.plant_transport_null:
        target = (
            jnp.asarray(to_lego(records[2]["fields"]["zFu"], nlev)),
            jnp.asarray(to_lego(records[2]["fields"]["zFv"], nlev)),
            jnp.asarray(to_lego(records[2]["fields"]["zFw"])),
        )
        if args.plant_transport_null:
            target = tuple(jnp.asarray(value)
                           for value in live_arrays["2"]["transport"])
        hooks = model_module._NEMOWSRK3TestHooks(
            expose_tracer_stage=2,
            stage2_tracer_transport_override=target)
        target_state = jax.device_get(model_module.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks).step(
                seeded, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
        target_rows = {}
        criteria = {}
        for name in TRACERS:
            candidate = np.asarray(gate.lego_fields(target_state)[name])
            oracle = to_lego(records[2]["fields"][f"Kaa_{name}"], nlev)
            live = live_arrays["2"]["tracers"][name]["Kaa"]
            target_rows[name] = {
                "candidate_vs_oracle": round54.field_stats(
                    candidate, oracle, masks[name]),
                "candidate_vs_live": round54.field_stats(
                    candidate, live, masks[name]),
            }
            live_row = live_rows["2"][f"Kaa_{name}"]
            candidate_row = target_rows[name]["candidate_vs_oracle"]
            criteria[name] = {
                "active_movement": (
                    target_rows[name]["candidate_vs_live"]["cells_unequal"] > 0),
                "max_improves_fourfold": (
                    candidate_row["max_abs"] * 4.0 <= live_row["max_abs"]),
                "unequal_cells_halved": (
                    candidate_row["cells_unequal"] * 2
                    <= live_row["cells_unequal"]),
            }
        causal = {"executed": True, "rows": target_rows, "criteria": criteria}

    causal_confirmed = bool(
        causal["executed"]
        and all(all(values.values()) for values in causal["criteria"].values()))
    controls = {
        "self_replay_exact": self_replay_exact,
        "independent_state_exact": True,
        "round71_kmm_reproduced": kmm_reproduction,
        "stage1_exact": stage1_exact,
        "predicted_first_boundary": predicted_boundary,
        "causal_prediction": causal_confirmed,
    }
    all_controls = (
        self_replay_exact and all(kmm_reproduction.values()) and stage1_exact
        and predicted_boundary and causal_confirmed)
    return {
        "format": "nemo-testcase-l2-gyre-round72-tracer-stage-v1",
        "status": "CONFIRMED" if all_controls else "REFUTED",
        "worktree": stamp,
        "record_producer": producer,
        "record_admission": {"exact": 45, "total": 65, "changed": 20,
                             "admitted": 132},
        "dtype": {
            "record": str(records[1]["fields"]["Kaa_T"].dtype),
            "live_tracer": str(np.asarray(trace.stage_states[0][2]).dtype),
            "live_geometry": str(np.asarray(trace.stage_geometry[0][3]).dtype),
        },
        "resolved_dt_s": float(card.dt_s),
        "replay_rows": replay_rows,
        "live_rows": live_rows,
        "first_non_bit": first_non_bit,
        "causal_stage2_transport": causal,
        "controls": controls,
        "base_round66_status": base["status"],
        "plants": {"kaa_ulp": args.plant_kaa_ulp,
                   "transport_null": args.plant_transport_null},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--record-root", type=Path, default=(
        ROOT / "round71/oracle_fct_stage2"))
    parser.add_argument("--plant-kaa-ulp", action="store_true")
    parser.add_argument("--plant-transport-null", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    row = report["live_rows"]["2"]["Kaa_T"]
    print(
        f"ROUND72 TRACER STAGE {report['status']}: "
        f"first_non_bit={report['first_non_bit']} "
        f"Kaa_T={row['max_abs']:.12e}")
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
