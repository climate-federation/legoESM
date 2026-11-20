#!/usr/bin/env python3
"""Compiled C4 ORCA2+SI3 year runner with monthly restart checkpoints.

The coupled recipe remains the sole owner of model arithmetic.  This module
loads the admitted deck through the C3 readers, compiles the existing coupled
pair once, observes completed daily states through the Decision-160/D188 metric
functions, and writes one restart-complete checkpoint per no-leap month.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_common as common,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round340_standalone_gate as r340,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round340_score_composed_card as r340_score,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round342_s4_year_metrics as year_metrics,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round342_s4_year_driver as year_driver,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    scorer as c_ladder_scorer,
)


CHECKPOINT_FORMAT = "nemo-coupled-c4-month-checkpoint-v1"
C4_METRIC_FORMAT = "nemo-coupled-c4-month-metrics-v1"
PAIRS_PER_DAY = year_metrics.STEPS_PER_DAY // 2
TOTAL_PAIRS = 1460
ICE_KEYS = (
    "ICE_AREA_NH",
    "ICE_AREA_SH",
    "ICE_VOLUME_NH",
    "ICE_VOLUME_SH",
)


class C4InputReader:
    """Materialize one year-clock forcing record at a time from the deck."""

    def __init__(self, reference_root: Path, n_steps: int, forcing_template: Any, card: Any):
        from legoesm.ocean.fidelity.nemo_testcase_recipe import (
            build_orca2_damping_for_step,
        )

        self.reference_root = reference_root
        self.n_steps = n_steps
        self.forcing_template = forcing_template
        self.card = card
        self._build_sponge = build_orca2_damping_for_step
        self.mapped_core = year_driver.load_core_window_for_steps(
            reference_root, r340_score.DEFAULT_CORE_MANIFEST, n_steps
        )
        self.runoff_12 = year_driver._load_monthly_12(
            reference_root / "runoff_core_monthly.nc", "sorunoff"
        )
        self.sss_12 = year_driver._load_monthly_12(
            reference_root / "sss_data.nc", "sss"
        )
        state = card.ocean.recipe.initial_state
        self.wet = np.asarray(state.land_mask.data) > 0.5
        self.active = np.asarray(card.ocean.recipe.z_coord.is_active) > 0.5

    def record(self, kt: int) -> tuple[Any, Any]:
        """Use the round-342 year clock for CORE, monthly fields, chl, and sponge."""

        from legoesm.ocean import bulk_flux_omip as bulk

        require(1 <= kt <= self.n_steps, f"C4 input step {kt} outside 1..{self.n_steps}")
        dt_s = int(self.card.ocean.dt_s)
        isecsbc = kt * dt_s
        retained_kt = kt if kt % 2 else kt - 1
        retained_clock = retained_kt * dt_s
        idx_6h = retained_clock // 21600
        idx_daily = retained_clock // 86400
        idx_monthly = int(np.searchsorted(
            year_driver.MONTH_END_SEC, retained_clock, side="right"
        ))
        if self.mapped_core["precip_raw"].shape[0] == 1:
            idx_monthly = 0
        atmosphere = bulk.NemoStandaloneAtmosphereFields(*(
            jnp.asarray(self.mapped_core[name][index], dtype=jnp.float64)
            for name, index in (
                ("wndi", idx_6h),
                ("wndj", idx_6h),
                ("tair", idx_6h),
                ("humi", idx_6h),
                ("slp", idx_6h),
                ("qsr_down", idx_daily),
                ("qlw_down", idx_daily),
                ("precip_raw", idx_monthly),
                ("snow_raw", idx_monthly),
            )
        ))
        s0f = bulk.NemoStandaloneS0FInputs(
            atmosphere,
            atmosphere,
            isecsbc,
            isecsbc - 1,
            isecsbc + 1,
            year_driver._monthly_pair_for_clock(self.runoff_12, isecsbc),
            year_driver._monthly_pair_for_clock(self.sss_12, isecsbc),
        )
        forcing = bulk.NemoORCA2SI3Forcing(
            s0f,
            self.forcing_template.o17,
            jnp.asarray(
                ladder.chlorophyll_at_step(
                    self.reference_root, kt, self.wet, dt_s=self.card.ocean.dt_s
                ),
                dtype=jnp.float64,
            ),
            self.forcing_template.geothermal_heat_flux_mwm2,
        )
        sponge = self._build_sponge(
            self.reference_root,
            self.active,
            kt=kt,
            dt_s=self.card.ocean.dt_s,
            mask_evaluation="nemo_tmask_multiply",
        )
        return forcing, sponge


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _jsonable_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    require(array.dtype.kind != "O", "C4 carry contains an object-dtype leaf")
    return array


def _carry_payload(carry: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    arrays = [_jsonable_array(leaf) for leaf in leaves]
    schema = [
        {"name": f"carry_{index:05d}", "shape": list(array.shape), "dtype": array.dtype.str}
        for index, array in enumerate(arrays)
    ]
    return (
        {row["name"]: array for row, array in zip(schema, arrays, strict=True)},
        {"leaf_schema": schema, "treedef": str(treedef)},
    )


def checkpoint_input_identity(reference_root: Path) -> dict[str, str | None]:
    """Bind resumed state to its deck receipt and coupled program sources."""

    from legoesm.ocean import bulk_flux_omip as bulk

    receipt = reference_root / "C_DECK_RECEIPT.json"
    return {
        "deck_receipt_sha256": common.sha256(receipt) if receipt.is_file() else None,
        "runner_sha256": common.sha256(Path(__file__)),
        "coupled_source_sha256": common.sha256(Path(bulk.__file__)),
    }


def checkpoint_name(month: int, completed_kt: int) -> str:
    require(1 <= month <= 12, f"invalid C4 month {month}")
    require(completed_kt > 0, f"invalid C4 completed step {completed_kt}")
    return f"c4_month{month:02d}_step{completed_kt:08d}.npz"


def write_checkpoint(
    path: Path,
    carry: Any,
    *,
    month: int,
    completed_kt: int,
    reference_root: Path,
    month_report: dict[str, Any],
) -> dict[str, Any]:
    """Atomically store the complete pytree carry and its scored month."""

    require(not path.exists(), f"refusing to overwrite C4 checkpoint {path}")
    require(int(np.asarray(carry.kt)) == completed_kt + 1,
            "C4 checkpoint carry kt is not the next step")
    payload, tree = _carry_payload(carry)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "month": month,
        "completed_kt": completed_kt,
        "next_kt": completed_kt + 1,
        "reference_root": str(reference_root.resolve()),
        "input_identity": checkpoint_input_identity(reference_root.resolve()),
        **tree,
    }
    payload["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    payload["__month_report_json__"] = np.asarray(
        json.dumps(month_report, sort_keys=True, allow_nan=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": path.name,
        "month": month,
        "completed_kt": completed_kt,
        "sha256": common.sha256(path),
        "leaf_count": len(tree["leaf_schema"]),
    }


def inspect_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read checkpoint metadata/month metrics without materializing carry leaves."""

    require(path.is_file(), f"missing C4 checkpoint {path}")
    with np.load(path, allow_pickle=False) as archive:
        require("__metadata_json__" in archive and "__month_report_json__" in archive,
                f"{path.name}: not a C4 checkpoint")
        metadata = json.loads(str(archive["__metadata_json__"].item()))
        report = json.loads(str(archive["__month_report_json__"].item()))
    require(metadata.get("format") == CHECKPOINT_FORMAT,
            f"{path.name}: unsupported checkpoint format {metadata.get('format')!r}")
    return metadata, report


def validate_month_checkpoint(
    path: Path, metadata: dict[str, Any], report: dict[str, Any]
) -> int:
    """Bind a checkpoint filename, carry clock, and metric window to one month."""

    month = int(metadata.get("month", 0))
    require(1 <= month <= 12, f"{path.name}: invalid C4 month {month}")
    window = year_metrics.calendar_month_day_ranges()[month - 1]
    completed_kt = int(window["end_day"]) * year_metrics.STEPS_PER_DAY
    require(path.name == checkpoint_name(month, completed_kt),
            f"{path.name}: C4 checkpoint filename is not its calendar boundary")
    require(
        int(metadata.get("completed_kt", -1)) == completed_kt
        and int(metadata.get("next_kt", -1)) == completed_kt + 1,
        f"{path.name}: carry clock is not the end of {window['label']}",
    )
    require(report.get("month") == month and report.get("label") == window["label"],
            f"{path.name}: embedded metric month disagrees with its calendar boundary")
    for key in ("start_day", "end_day", "available_days", "saved_kts", "complete_month"):
        require(report.get(key) == window[key],
                f"{path.name}: embedded metric {key} is not the admitted month window")
    return month


def validate_checkpoint_prefix(
    output_dir: Path, reference_root: Path, through_month: int
) -> None:
    """Require a contiguous same-deck checkpoint prefix before a resumed write."""

    paths = sorted(output_dir.glob("c4_month??_step*.npz"))
    require(len(paths) == through_month,
            f"resume directory has {len(paths)} checkpoints, expected {through_month}")
    observed = []
    for path in paths:
        metadata, report = inspect_checkpoint(path)
        month = validate_month_checkpoint(path, metadata, report)
        require(Path(metadata["reference_root"]).resolve() == reference_root.resolve(),
                f"{path.name}: resumed checkpoint deck differs from this run")
        observed.append(month)
    require(observed == list(range(1, through_month + 1)),
            f"resume checkpoint months are {observed}, expected 1..{through_month}")


def restore_checkpoint(
    path: Path,
    template: Any,
    reference_root: Path,
    *,
    validate_calendar: bool = True,
) -> tuple[Any, dict, dict]:
    """Restore a carry using the fresh deck-built pytree as its exact schema."""

    metadata, report = inspect_checkpoint(path)
    if validate_calendar:
        validate_month_checkpoint(path, metadata, report)
    require(Path(metadata["reference_root"]).resolve() == reference_root.resolve(),
            f"{path.name}: checkpoint reference deck differs from this run")
    require(metadata.get("input_identity") == checkpoint_input_identity(reference_root),
            f"{path.name}: deck receipt or coupled program changed since checkpoint")
    template_leaves, treedef = jax.tree_util.tree_flatten(template)
    schema = metadata.get("leaf_schema", [])
    require(metadata.get("treedef") == str(treedef),
            f"{path.name}: carry pytree definition differs from the deck template")
    require(len(schema) == len(template_leaves),
            f"{path.name}: carry leaf count differs from the deck template")
    restored = []
    with np.load(path, allow_pickle=False) as archive:
        for row, template_leaf in zip(schema, template_leaves, strict=True):
            name = str(row["name"])
            require(name in archive, f"{path.name}: missing carry leaf {name}")
            value = np.asarray(archive[name])
            require(list(value.shape) == row["shape"] and value.dtype.str == row["dtype"],
                    f"{path.name}: carry leaf {name} disagrees with its schema")
            expected = np.asarray(template_leaf)
            require(value.shape == expected.shape and value.dtype == expected.dtype,
                    f"{path.name}: carry leaf {name} differs from the deck template")
            restored.append(jnp.asarray(value))
    carry = jax.tree_util.tree_unflatten(treedef, restored)
    require(int(np.asarray(carry.kt)) == int(metadata["next_kt"]),
            f"{path.name}: restored carry kt disagrees with checkpoint metadata")
    return carry, metadata, report


def read_nemo_ice_day(reference_root: Path, day: int) -> dict[str, np.ndarray]:
    """Read category-summed SI3 ``a_i`` and ``v_i`` from both NEMO ranks."""

    import netCDF4

    kt = day * year_metrics.STEPS_PER_DAY
    rank_fields: list[dict[str, np.ndarray]] = []
    for rank in (0, 1):
        path = reference_root / f"ORCA2_{kt:08d}_restart_ice_{rank:04d}.nc"
        require(path.is_file(), f"missing NEMO SI3 restart {path}")
        with netCDF4.Dataset(path) as dataset:
            fields = {}
            for output, source in (("area", "a_i"), ("volume", "v_i")):
                require(source in dataset.variables, f"{path.name}: missing SI3 {source}")
                raw = np.ma.filled(dataset.variables[source][0], 0.0)
                fields[output] = np.asarray(raw, dtype=np.float64).sum(axis=0)
            rank_fields.append(fields)
    out = {
        name: np.concatenate((rank_fields[0][name], rank_fields[1][name]), axis=1)
        for name in ("area", "volume")
    }
    for name, value in out.items():
        require(value.shape == (148, 180),
                f"day {day}: NEMO SI3 {name} shape {value.shape} != (148, 180)")
        require(bool(np.isfinite(value).all()), f"day {day}: non-finite NEMO SI3 {name}")
    return out


def candidate_ice_fields(carry: Any, nn_hls: int) -> dict[str, np.ndarray]:
    """Observe live category aggregates on the owned NEMO T-point layout."""

    aggregated = carry.ice.ice.aggregated
    result = {}
    for output, source in (("area", aggregated.at_i), ("volume", aggregated.vt_i)):
        value = np.asarray(getattr(source, "data", source), dtype=np.float64)
        if nn_hls:
            value = value[nn_hls:-nn_hls, nn_hls:-nn_hls]
        result[output] = value.T
        require(result[output].shape == (148, 180),
                f"candidate SI3 {output} shape {result[output].shape} != (148, 180)")
    return result


def ice_day_metrics(
    candidate: dict[str, np.ndarray],
    nemo: dict[str, np.ndarray],
    geo: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Hemispheric SI3 area/volume totals using the D188 unique T-cell area."""

    area_weight = np.asarray(geo["area_t"], dtype=np.float64)
    latitude = np.asarray(geo["gphit"], dtype=np.float64)
    rows = {}
    for field, label, scale, units in (
        ("area", "ICE_AREA", 1.0e12, "1e6 km2"),
        ("volume", "ICE_VOLUME", 1.0e9, "km3"),
    ):
        # SI3 assigns gphit > 0 to its northern branch and ELSEWHERE to the
        # southern branch (icestp.F90); keep the diagnostic a true partition.
        for hemisphere, mask in (("NH", latitude > 0.0), ("SH", latitude <= 0.0)):
            nemo_total = float(np.sum(nemo[field] * area_weight * mask) / scale)
            candidate_total = float(np.sum(candidate[field] * area_weight * mask) / scale)
            require(nemo_total > 0.0, f"{label}_{hemisphere}: NEMO total is not positive")
            rows[f"{label}_{hemisphere}"] = {
                "nemo": nemo_total,
                "candidate": candidate_total,
                "units": units,
            }
    return rows


def monthly_ice_metrics(daily_rows: list[dict[str, dict[str, float]]]) -> dict[str, dict]:
    require(bool(daily_rows), "cannot aggregate an empty C4 SI3 month")
    output = {}
    for key in ICE_KEYS:
        nemo_mean = float(np.mean([row[key]["nemo"] for row in daily_rows]))
        candidate_mean = float(np.mean([row[key]["candidate"] for row in daily_rows]))
        absolute = abs(candidate_mean - nemo_mean)
        output[key] = {
            "metric": "relative_abs_monthly_mean",
            "nemo_monthly_mean": nemo_mean,
            "candidate_monthly_mean": candidate_mean,
            "absolute_difference": absolute,
            "relative_abs_difference": absolute / abs(nemo_mean),
            "units": daily_rows[0][key]["units"],
        }
    return output


def build_month_report(
    month_index: int,
    daily_rows: list[dict[str, Any]],
    daily_maps: list[dict[str, np.ndarray]],
    ice_rows: list[dict[str, dict[str, float]]],
    geo: dict[str, np.ndarray],
) -> dict[str, Any]:
    window = year_metrics.calendar_month_day_ranges()[month_index - 1]
    require(window["complete_month"], f"{window['label']}: calendar month is not complete")
    require([row["day"] for row in daily_rows] == window["available_days"],
            f"{window['label']}: daily accumulator does not cover its calendar window")
    summary = year_metrics.pilot_metrics.monthly(daily_rows, daily_maps, geo)
    metrics = year_metrics._attach_decision185_reference(summary, daily_rows)
    metrics.update(monthly_ice_metrics(ice_rows))
    return {
        **window,
        "format": C4_METRIC_FORMAT,
        "status": "EVALUATED_COMPLETE_MONTH",
        "bar_mode": "DECISION_188_PASS_FAIL",
        "verdict": None,
        "metrics": metrics,
    }


def compile_pair(
    carry: Any,
    odd_forcing: Any,
    even_forcing: Any,
    odd_sponge: Any,
    even_sponge: Any,
    card: Any,
    model: Any,
):
    """Compile the existing C3 odd/even coupled-pair program exactly once."""

    from legoesm.ocean import bulk_flux_omip as bulk

    compile_count = 0

    def pair_program(entry, odd_forcing, even_forcing, odd_sponge, even_sponge):
        nonlocal compile_count
        compile_count += 1
        next_carry, _traces = bulk.nemo_orca2_si3_coupled_pair(
            entry,
            odd_forcing,
            card,
            even_forcing=even_forcing,
            model=model,
            sponge=odd_sponge,
            even_sponge=even_sponge,
        )
        return next_carry

    compiled = jax.jit(pair_program)
    started = time.perf_counter()
    executable = compiled.lower(
        carry, odd_forcing, even_forcing, odd_sponge, even_sponge
    ).compile()
    compile_seconds = time.perf_counter() - started
    require(compile_count == 1, f"C4 pair compiled {compile_count} times; expected exactly 1")
    return executable, compile_seconds, lambda: compile_count


def _advance_pairs(executable, carry, forcing, sponge, start: int, stop: int):
    timings = []
    for pair_index in range(start, stop):
        odd_index = 2 * pair_index
        started = time.perf_counter()
        carry = executable(
            carry,
            forcing[odd_index],
            forcing[odd_index + 1],
            sponge[odd_index],
            sponge[odd_index + 1],
        )
        jax.block_until_ready(carry)
        timings.append(time.perf_counter() - started)
    return carry, timings


def _first_tree_bit_difference(left: Any, right: Any) -> dict[str, Any] | None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    require(left_tree == right_tree, "restart identity changed the carry pytree")
    for index, (got, want) in enumerate(zip(left_leaves, right_leaves, strict=True)):
        got_array = np.asarray(got)
        want_array = np.asarray(want)
        if (
            got_array.shape != want_array.shape
            or got_array.dtype != want_array.dtype
            or got_array.tobytes() != want_array.tobytes()
        ):
            return {
                "leaf": index,
                "got_shape": list(got_array.shape),
                "want_shape": list(want_array.shape),
                "got_dtype": got_array.dtype.str,
                "want_dtype": want_array.dtype.str,
            }
    return None


def run_restart_identity(reference_root: Path, output_dir: Path) -> dict[str, Any]:
    """Prove four real compiled pairs equal a two-pair checkpoint continuation."""

    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    reference_root = reference_root.resolve()
    total_pairs, split_pairs = 4, 2
    carry, forcing, sponge, card, model = c_ladder_scorer._c2_cold_start(
        reference_root, n_steps=2 * total_pairs
    )
    executable, compile_seconds, compile_count = compile_pair(
        carry, forcing[0], forcing[1], sponge[0], sponge[1], card, model
    )
    straight, straight_seconds = _advance_pairs(
        executable, carry, forcing, sponge, 0, total_pairs
    )
    split, first_seconds = _advance_pairs(
        executable, carry, forcing, sponge, 0, split_pairs
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "c4_restart_identity_after_pair0002.npz"
    written = write_checkpoint(
        checkpoint,
        split,
        month=1,
        completed_kt=2 * split_pairs,
        reference_root=reference_root,
        month_report={
            "format": "nemo-coupled-c4-restart-identity-v1",
            "pairs_completed": split_pairs,
            "verdict": None,
        },
    )
    restored, _metadata, _report = restore_checkpoint(
        checkpoint, carry, reference_root, validate_calendar=False
    )
    continued, second_seconds = _advance_pairs(
        executable, restored, forcing, sponge, split_pairs, total_pairs
    )
    difference = _first_tree_bit_difference(continued, straight)
    require(difference is None, f"C4 restart identity first differing leaf: {difference}")
    require(compile_count() == 1,
            f"C4 restart identity retraced {compile_count()} times; expected exactly 1")
    return {
        "format": "nemo-coupled-c4-restart-identity-v1",
        "status": "PASS_C4_RESTART_IDENTITY_BITWISE",
        "backend": jax.default_backend(),
        "fp64": bool(jax.config.jax_enable_x64),
        "total_pairs": total_pairs,
        "split_after_pairs": split_pairs,
        "compile_count": compile_count(),
        "compile_seconds": compile_seconds,
        "straight_pair_seconds": straight_seconds,
        "split_pair_seconds": first_seconds + second_seconds,
        "first_difference": difference,
        "checkpoint": written,
    }


def run_c4(
    reference_root: Path,
    output_dir: Path,
    *,
    resume: Path | None = None,
    pair_limit: int | None = None,
    profile_only: bool = False,
) -> dict[str, Any]:
    """Run the compiled trajectory; full mode writes exactly 12 month checkpoints."""

    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() in ("cpu", "gpu"),
            f"C4 requires a compiled CPU/GPU backend, got {jax.default_backend()}")
    reference_root = reference_root.resolve()
    require(reference_root.is_dir(), f"missing C4 reference deck {reference_root}")
    if profile_only:
        require(resume is None, "C4 profile mode does not resume checkpoints")
        require(type(pair_limit) is int and 1 <= pair_limit <= 4,
                "C4 profile mode requires --pair-limit in [1, 4]")
        n_steps = 2 * pair_limit
    else:
        require(pair_limit is None, "a verdict-bearing C4 year cannot use --pair-limit")
        n_steps = 2 * TOTAL_PAIRS

    carry, seed_forcing, _seed_sponge, card, model = c_ladder_scorer._c2_cold_start(
        reference_root, n_steps=2
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resumed_month = 0
    if resume is not None:
        resume = resume.resolve()
        require(resume.parent == output_dir,
                "C4 resume checkpoint must be in the output checkpoint directory")
        carry, metadata, _report = restore_checkpoint(resume, carry, reference_root)
        resumed_month = int(metadata["month"])
        validate_checkpoint_prefix(output_dir, reference_root, resumed_month)
        require(resume.name == checkpoint_name(
                    resumed_month, int(metadata["completed_kt"])
                ),
                "C4 resume path is not the terminal checkpoint in its prefix")
    elif not profile_only:
        require(not list(output_dir.glob("c4_month??_step*.npz")),
                f"refusing a cold C4 run in nonempty checkpoint directory {output_dir}")
    start_pair = (int(np.asarray(carry.kt)) - 1) // 2
    stop_pair = pair_limit if profile_only else TOTAL_PAIRS
    require(0 <= start_pair <= stop_pair,
            f"C4 start pair {start_pair} is outside requested stop pair {stop_pair}")
    if start_pair == stop_pair:
        require(not profile_only and resumed_month == 12,
                "only a complete month-12 checkpoint may end a resumed C4 run")
        return {
            "format": "nemo-coupled-c4-run-v1",
            "status": "PASS_C4_RUN_12_CHECKPOINTS",
            "reference_root": str(reference_root),
            "backend": jax.default_backend(),
            "fp64": bool(jax.config.jax_enable_x64),
            "compiled": False,
            "compile_count": 0,
            "compile_seconds": 0.0,
            "pairs_executed": 0,
            "pair_wall_seconds": [],
            "mean_pair_wall_seconds": None,
            "resumed_month": 12,
            "already_complete": True,
            "checkpoints": [],
        }

    inputs = C4InputReader(reference_root, n_steps, seed_forcing[0], card)
    first_kt = 2 * start_pair + 1
    first_forcing, first_sponge = inputs.record(first_kt)
    second_forcing, second_sponge = inputs.record(first_kt + 1)
    executable, compile_seconds, compile_count = compile_pair(
        carry,
        first_forcing,
        second_forcing,
        first_sponge,
        second_sponge,
        card,
        model,
    )
    metric_context = None
    if not profile_only:
        geo = year_metrics.pilot_metrics.static_geometry(reference_root)
        mld = year_metrics.K0MLDEvaluator(reference_root)
        cover = r340.read_decision171_cover(r340_score.DEFAULT_ICE_COVER)
        metric_context = (geo, mld, cover)

    month_rows: list[dict[str, Any]] = []
    month_maps: list[dict[str, np.ndarray]] = []
    month_ice: list[dict[str, dict[str, float]]] = []
    checkpoints = []
    pair_seconds = []
    nn_hls = int(card.ice.state.nn_hls)
    for pair_index in range(start_pair, stop_pair):
        odd_index = 2 * pair_index
        if pair_index == start_pair:
            odd_forcing, even_forcing = first_forcing, second_forcing
            odd_sponge, even_sponge = first_sponge, second_sponge
        else:
            odd_forcing, odd_sponge = inputs.record(odd_index + 1)
            even_forcing, even_sponge = inputs.record(odd_index + 2)
        started = time.perf_counter()
        carry = executable(
            carry,
            odd_forcing,
            even_forcing,
            odd_sponge,
            even_sponge,
        )
        jax.block_until_ready(carry)
        pair_seconds.append(time.perf_counter() - started)
        completed_kt = 2 * (pair_index + 1)
        if profile_only or completed_kt % year_metrics.STEPS_PER_DAY:
            continue

        geo, mld, cover = metric_context
        day = completed_kt // year_metrics.STEPS_PER_DAY
        candidate = ladder._candidate_fields(carry.ocean)
        nemo = year_metrics.pilot_metrics.read_nemo_day(reference_root, day)
        row, maps = year_metrics.pilot_metrics.day_metrics(candidate, nemo, geo, mld, cover)
        year_metrics._augment_day_single_arm_stats(row, candidate, nemo, geo, cover, maps)
        month_rows.append({"day": day, "kt": completed_kt, **row})
        month_maps.append(maps)
        month_ice.append(ice_day_metrics(
            candidate_ice_fields(carry, nn_hls),
            read_nemo_ice_day(reference_root, day),
            geo,
        ))

        month_number = next(
            (index for index, window in enumerate(year_metrics.calendar_month_day_ranges(), 1)
             if completed_kt == window["end_day"] * year_metrics.STEPS_PER_DAY),
            None,
        )
        if month_number is None:
            continue
        require(month_number == resumed_month + len(checkpoints) + 1,
                f"C4 month cadence skipped or repeated month {month_number}")
        month_report = build_month_report(
            month_number, month_rows, month_maps, month_ice, geo
        )
        path = output_dir / checkpoint_name(month_number, completed_kt)
        checkpoints.append(write_checkpoint(
            path,
            carry,
            month=month_number,
            completed_kt=completed_kt,
            reference_root=reference_root,
            month_report=month_report,
        ))
        month_rows, month_maps, month_ice = [], [], []

    require(compile_count() == 1,
            f"C4 pair retraced {compile_count()} times; expected exactly 1")
    if not profile_only:
        require(int(np.asarray(carry.kt)) == 2921, f"C4 final next kt is {carry.kt}, expected 2921")
        require(resumed_month + len(checkpoints) == 12,
                f"C4 wrote through month {resumed_month + len(checkpoints)}, expected 12")
    return {
        "format": "nemo-coupled-c4-run-v1",
        "status": (
            f"PROFILE_C4_{len(pair_seconds)}_PAIRS_NO_VERDICT"
            if profile_only else "PASS_C4_RUN_12_CHECKPOINTS"
        ),
        "reference_root": str(reference_root),
        "backend": jax.default_backend(),
        "fp64": bool(jax.config.jax_enable_x64),
        "compiled": True,
        "compile_count": compile_count(),
        "compile_seconds": compile_seconds,
        "pairs_executed": len(pair_seconds),
        "pair_wall_seconds": pair_seconds,
        "mean_pair_wall_seconds": float(np.mean(pair_seconds)),
        "resumed_month": resumed_month,
        "checkpoints": checkpoints,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--restart-identity", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.restart_identity:
        require(args.resume is None and args.pair_limit is None and not args.profile_only,
                "--restart-identity cannot be combined with resume/profile options")
        report = run_restart_identity(args.reference, args.output_dir)
    else:
        report = run_c4(
            args.reference,
            args.output_dir,
            resume=args.resume,
            pair_limit=args.pair_limit,
            profile_only=args.profile_only,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
