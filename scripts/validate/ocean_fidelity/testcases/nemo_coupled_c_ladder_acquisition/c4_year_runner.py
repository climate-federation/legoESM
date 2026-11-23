#!/usr/bin/env python3
"""Compiled C4 ORCA2+SI3 year runner with monthly restart checkpoints.

The coupled recipe remains the sole owner of model arithmetic.  This module
loads the admitted deck through the C3 readers, compiles the existing coupled
pair once, observes completed daily states through the Decision-160/D188 metric
functions, and writes one restart-complete checkpoint per no-leap month.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import get_policy

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
    nemo_testcase_l4_orca2_round342_s4_year_driver as year_driver,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round342_s4_year_metrics as year_metrics,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    deck as c_ladder_deck,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    scorer as c_ladder_scorer,
)

CHECKPOINT_FORMAT = "nemo-coupled-c4-month-checkpoint-v1"
C4_METRIC_FORMAT = "nemo-coupled-c4-month-metrics-v3"
LEGACY_C4_METRIC_FORMAT = "nemo-coupled-c4-month-metrics-v2"
SUPPORTED_C4_METRIC_FORMATS = (LEGACY_C4_METRIC_FORMAT, C4_METRIC_FORMAT)
IN_FLIGHT_44512_INPUT_IDENTITY = {
    "deck_receipt_sha256": "8b916b74567cd0495a0dce986f1bd77ef281e7984aa1bbdd26a1537dc9a21254",
    "runner_sha256": "7d675a75cd32d71f7d1e8d3b9bb236af09a23652229429d274d146ad75e76495",
    "coupled_source_sha256": "a3f93e902b8decb157937989d581497d8b5207339434ff6849e7ee5dc1eb55a6",
    "ice_package_source_sha256": "424362b575d38649a29b1e6fa72837387cf9260b980d813bf36346c58406b0c1",
}
IN_FLIGHT_44512_LEAF_COUNT = 179
PAIRS_PER_DAY = year_metrics.STEPS_PER_DAY // 2
TOTAL_PAIRS = 1460
ICE_UNIT_SCALE = 1.0e12
ICE_KEYS = (
    "ICE_AREA_NH",
    "ICE_AREA_SH",
    "ICE_VOLUME_NH",
    "ICE_VOLUME_SH",
)
ICE_MISPLACEMENT_KEYS = (
    "ICE_MISPLACED_AREA_NH",
    "ICE_MISPLACED_AREA_SH",
    "ICE_MISPLACED_VOLUME_NH",
    "ICE_MISPLACED_VOLUME_SH",
)
SNOW_MISPLACEMENT_KEYS = (
    "SNOW_MISPLACED_VOLUME_NH",
    "SNOW_MISPLACED_VOLUME_SH",
)
DAILY_MAP_FIELDS = ("sst", "ice_area", "ice_volume", "snow_volume")
MATH_CHOICES = c_ladder_scorer.COUPLED_MATH_CHOICES


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
            reference_root, coupled_core_manifest(reference_root), n_steps
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


def _source_tree_sha256(root: Path) -> str:
    """Hash every Python source path and byte in one package tree."""

    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    require(bool(files), f"no Python sources found under {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def coupled_core_manifest(reference_root: Path) -> Path:
    """Return the manifest belonging to the source deck staged as REF_C4.

    The manifest validates model-input CORE fields.  It is not a scoring mask.
    """

    receipt_path = reference_root / "C_DECK_RECEIPT.json"
    require(receipt_path.is_file(), f"missing coupled deck receipt {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("status") == "PASS_EXACT_R339_R5_DECK_ICE5",
            f"{receipt_path}: coupled deck receipt did not pass")
    require(receipt.get("rung") == "C4", f"{receipt_path}: receipt is not C4")
    try:
        verified = c_ladder_deck.verify(
            reference_root, "C4", require_resolved_output=True
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    require(verified.get("resolved_output_checked") is True,
            f"{reference_root}: executed SI3 controls were not verified")
    source_deck = Path(c_ladder_deck.OVERLAYS["source_deck"])
    manifest = source_deck / "input_files.sha256"
    require(manifest.is_file(), f"missing coupled source-deck manifest {manifest}")
    return manifest


def resolved_c4_controls(reference_root: Path) -> dict[str, Any]:
    """Read the executed coupled ocean/ice controls before a C4 run."""

    from legoesm.core.nemo_namelist import parse_namelist

    resolved_path = reference_root / "output.namelist.dyn"
    require(resolved_path.is_file(), f"missing coupled resolved namelist {resolved_path}")
    parsed = parse_namelist(resolved_path, "ocean")

    def scalar(group: str, name: str) -> Any:
        values = parsed.get(("ocean", group, name))
        require(isinstance(values, list) and len(values) == 1,
                f"{resolved_path}: unresolved scalar {group}.{name}")
        return values[0]

    output_path = reference_root / "ocean.output"
    require(output_path.is_file(), f"missing coupled run log {output_path}")
    calendars = re.findall(
        r'The IOIPSL calendar is\s+"([^"]+)"',
        output_path.read_text(encoding="utf-8", errors="replace"),
    )
    require(len(set(calendars)) == 1, f"{output_path}: unresolved executed calendar")
    ice = c_ladder_scorer._coupled_resolved_namelist(reference_root)
    controls = {
        "nn_it000": int(scalar("namrun", "nn_it000")),
        "nn_itend": int(scalar("namrun", "nn_itend")),
        "nn_stock": int(scalar("namrun", "nn_stock")),
        "ln_rstart": bool(scalar("namrun", "ln_rstart")),
        "rn_Dt": float(scalar("namdom", "rn_dt")),
        "nn_fsbc": int(scalar("namsbc", "nn_fsbc")),
        "nn_ice": int(scalar("namsbc", "nn_ice")),
        "nn_leapy": int(scalar("namrun", "nn_leapy")),
        "calendar": calendars[0],
        "ice_dimensions": tuple(int(ice[name]) for name in ("jpl", "nlay_i", "nlay_s")),
        "si3_switches": {
            name: ice[name]
            for name in (
                "rn_simin", "rn_sinew", "nn_icesal", "ln_pnd", "ln_pnd_LEV",
                "ln_pnd_TOPO", "ln_pnd_lids", "ln_pnd_alb", "nn_snwfra",
                "ln_drgice_imp",
            )
        },
    }
    require(controls == {
        "nn_it000": 1,
        "nn_itend": 2920,
        "nn_stock": 8,
        "ln_rstart": False,
        "rn_Dt": 10800.0,
        "nn_fsbc": 2,
        "nn_ice": 2,
        "nn_leapy": 0,
        "calendar": "noleap",
        "ice_dimensions": (5, 10, 5),
        "si3_switches": {
            "rn_simin": 0.1,
            "rn_sinew": 0.75,
            "nn_icesal": 4.0,
            "ln_pnd": True,
            "ln_pnd_LEV": True,
            "ln_pnd_TOPO": False,
            "ln_pnd_lids": True,
            "ln_pnd_alb": True,
            "nn_snwfra": 2.0,
            "ln_drgice_imp": True,
        },
    }, f"resolved coupled C4 controls moved: {controls}")
    return controls


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
        {
            "leaf_schema": schema,
            "leaf_count": treedef.num_leaves,
            "treedef": str(treedef),
        },
    )


def checkpoint_input_identity(reference_root: Path) -> dict[str, str]:
    """Bind resumed state to its deck receipt and coupled program sources."""

    from legoesm.ice import si3_o17
    from legoesm.ocean import bulk_flux_omip as bulk

    receipt = reference_root / "C_DECK_RECEIPT.json"
    require(receipt.is_file(), f"missing required coupled deck receipt {receipt}")
    return {
        "deck_receipt_sha256": common.sha256(receipt),
        "runner_sha256": common.sha256(Path(__file__)),
        "coupled_source_sha256": common.sha256(Path(bulk.__file__)),
        "ice_package_source_sha256": _source_tree_sha256(Path(si3_o17.__file__).resolve().parent),
    }


def _resume_identity_is_compatible(stored: Any, current: dict[str, str]) -> bool:
    return stored == current


def checkpoint_identity_is_scorable(stored: Any, current: dict[str, str]) -> bool:
    """Admit current evidence or the operator-named in-flight 44512 acquisition."""

    return stored == current or stored == IN_FLIGHT_44512_INPUT_IDENTITY


def checkpoint_math(metadata: dict[str, Any]) -> str:
    """Return an explicit math stamp, admitting only the named legacy run."""

    math_mode = metadata.get("math")
    if math_mode is not None:
        require(math_mode in MATH_CHOICES,
                f"C4 checkpoint has invalid math selection {math_mode!r}")
        return str(math_mode)
    require(
        metadata.get("input_identity") == IN_FLIGHT_44512_INPUT_IDENTITY,
        "C4 checkpoint has no admitted math selection",
    )
    return "libm-cpu"


def checkpoint_name(month: int, completed_kt: int) -> str:
    require(1 <= month <= 12, f"invalid C4 month {month}")
    require(completed_kt > 0, f"invalid C4 completed step {completed_kt}")
    return f"c4_month{month:02d}_step{completed_kt:08d}.npz"


def daily_maps_name(month: int) -> str:
    require(1 <= month <= 12, f"invalid C4 daily-map month {month}")
    return f"c4_daily_maps_month{month:02d}.npz"


def _write_npz_exclusive(path: Path, payload: dict[str, np.ndarray], label: str) -> None:
    """Atomically write one NPZ while refusing an existing destination."""

    require(not path.exists(), f"refusing to overwrite C4 {label} {path}")
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


def write_daily_maps(
    path: Path,
    month: int,
    records: list[dict[str, Any]],
    geo: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Write Decision-237 float32 daily surface maps for one complete month."""

    window = year_metrics.calendar_month_day_ranges()[month - 1]
    days = [int(record["day"]) for record in records]
    require(days == window["available_days"],
            f"{window['label']}: daily-map accumulator does not cover its calendar window")
    area_t = np.asarray(geo["area_t"])
    unique_wet = (np.asarray(geo["tmask"])[..., 0] > 0) & (area_t > 0.0)
    static = {
        "gphit": np.asarray(geo["gphit"]),
        "glamt": np.asarray(geo["glamt"]),
        "area_t": area_t,
        "wet": unique_wet,
    }
    for name, value in static.items():
        require(value.shape == (148, 180),
                f"Decision-237 {name} shape {value.shape} != (148, 180)")
        require(bool(np.isfinite(value).all()),
                f"Decision-237 {name} contains non-finite values")
    payload: dict[str, np.ndarray] = {
        "day": np.asarray(days, dtype=np.float32),
        **{name: np.asarray(value, dtype=np.float32) for name, value in static.items()},
    }
    for field in DAILY_MAP_FIELDS:
        for side in ("nemo", "candidate"):
            key = f"{side}_{field}"
            values = np.asarray([record[key] for record in records], dtype=np.float32)
            require(values.shape == (len(days), 148, 180),
                    f"Decision-237 {key} shape {values.shape} is invalid")
            require(bool(np.isfinite(values).all()),
                    f"Decision-237 {key} contains non-finite values")
            payload[key] = values
    _write_npz_exclusive(path, payload, "daily maps")
    return {
        "path": path.name,
        "month": month,
        "days": days,
        "dtype": "float32",
        "sha256": common.sha256(path),
    }


def write_checkpoint(
    path: Path,
    carry: Any,
    *,
    month: int,
    completed_kt: int,
    reference_root: Path,
    month_report: dict[str, Any],
    math_mode: str,
    input_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically store the complete pytree carry and its scored month."""

    require(not path.exists(), f"refusing to overwrite C4 checkpoint {path}")
    require(int(np.asarray(carry.kt)) == completed_kt + 1,
            "C4 checkpoint carry kt is not the next step")
    require(math_mode in MATH_CHOICES, f"invalid C4 math selection {math_mode!r}")
    payload, tree = _carry_payload(carry)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "month": month,
        "completed_kt": completed_kt,
        "next_kt": completed_kt + 1,
        "reference_root": str(reference_root.resolve()),
        "math": math_mode,
        "input_identity": (
            checkpoint_input_identity(reference_root.resolve())
            if input_identity is None else input_identity
        ),
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
        "math": math_mode,
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
        schema = metadata.get("leaf_schema", [])
        require(isinstance(schema, list) and bool(schema),
                f"{path.name}: checkpoint has no carry leaf schema")
        treedef_text = metadata.get("treedef")
        require(isinstance(treedef_text, str) and bool(treedef_text),
                f"{path.name}: checkpoint has no carry treedef")
        treedef_leaf_count = treedef_text.count("*")
        require(treedef_leaf_count > 0,
                f"{path.name}: checkpoint carry treedef has no leaves")
        leaf_count = metadata.get("leaf_count")
        if metadata.get("input_identity") == IN_FLIGHT_44512_INPUT_IDENTITY:
            require(treedef_leaf_count == IN_FLIGHT_44512_LEAF_COUNT,
                    f"{path.name}: in-flight checkpoint must contain exactly "
                    f"{IN_FLIGHT_44512_LEAF_COUNT} leaves")
            require(leaf_count in (None, IN_FLIGHT_44512_LEAF_COUNT),
                    f"{path.name}: in-flight checkpoint must contain exactly "
                    f"{IN_FLIGHT_44512_LEAF_COUNT} leaves")
            # The already-running checkpoint format predates the explicit
            # treedef count.  Its exact admitted identity and schema retain the
            # count needed for scoring only; restore still requires the live
            # deck-built treedef below.
            leaf_count = IN_FLIGHT_44512_LEAF_COUNT
        require(type(leaf_count) is int and leaf_count > 0,
                f"{path.name}: checkpoint has no valid carry treedef leaf count")
        require(leaf_count == treedef_leaf_count,
                f"{path.name}: carry leaf count differs from its treedef")
        require(len(schema) == leaf_count,
                f"{path.name}: carry schema differs from its treedef leaf count")
        names = [str(row.get("name", "")) for row in schema]
        expected_names = [f"carry_{index:05d}" for index in range(leaf_count)]
        require(names == expected_names,
                f"{path.name}: carry leaf schema order differs from its treedef")
        expected = set(names) | {"__metadata_json__", "__month_report_json__"}
        require(set(archive.files) == expected,
                f"{path.name}: checkpoint archive leaves differ from its carry schema")
        for row in schema:
            value = np.asarray(archive[row["name"]])
            require(list(value.shape) == row.get("shape")
                    and value.dtype.str == row.get("dtype"),
                    f"{path.name}: carry leaf {row['name']} disagrees with its schema")
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
) -> tuple[dict[str, str], str, str]:
    """Require a contiguous same-deck checkpoint prefix before a resumed write."""

    paths = sorted(output_dir.glob("c4_month??_step*.npz"))
    require(len(paths) == through_month,
            f"resume directory has {len(paths)} checkpoints, expected {through_month}")
    observed = []
    identities = []
    metric_formats = []
    carry_schemas = []
    carry_treedefs = []
    math_modes = []
    for path in paths:
        metadata, report = inspect_checkpoint(path)
        month = validate_month_checkpoint(path, metadata, report)
        require(Path(metadata["reference_root"]).resolve() == reference_root.resolve(),
                f"{path.name}: resumed checkpoint deck differs from this run")
        observed.append(month)
        identities.append(metadata.get("input_identity"))
        metric_formats.append(report.get("format"))
        carry_schemas.append(metadata.get("leaf_schema"))
        carry_treedefs.append(metadata.get("treedef"))
        math_modes.append(checkpoint_math(metadata))
    require(observed == list(range(1, through_month + 1)),
            f"resume checkpoint months are {observed}, expected 1..{through_month}")
    require(all(identity == identities[0] for identity in identities),
            "resume checkpoint input_identity changed between months")
    require(all(metric_format == metric_formats[0] for metric_format in metric_formats),
            "resume checkpoint metric format changed between months")
    require(all(schema == carry_schemas[0] for schema in carry_schemas),
            "resume checkpoint carry schema changed between months")
    require(all(treedef == carry_treedefs[0] for treedef in carry_treedefs),
            "resume checkpoint carry treedef changed between months")
    require(all(math_mode == math_modes[0] for math_mode in math_modes),
            "resume checkpoint math selection changed between months")
    require(metric_formats[0] in SUPPORTED_C4_METRIC_FORMATS,
            f"unsupported resumed metric format {metric_formats[0]!r}")
    return identities[0], metric_formats[0], math_modes[0]


def validate_daily_map_prefix(output_dir: Path, through_month: int) -> None:
    """Require one immutable Decision-237 map file per resumed month."""

    paths = sorted(output_dir.glob("c4_daily_maps_month??.npz"))
    expected = [daily_maps_name(month) for month in range(1, through_month + 1)]
    require([path.name for path in paths] == expected,
            f"resume daily-map files are {[path.name for path in paths]}, expected {expected}")


def restore_checkpoint(
    path: Path,
    template: Any,
    reference_root: Path,
    *,
    math_mode: str,
    validate_calendar: bool = True,
) -> tuple[Any, dict, dict]:
    """Restore a carry using the fresh deck-built pytree as its exact schema."""

    metadata, report = inspect_checkpoint(path)
    if validate_calendar:
        validate_month_checkpoint(path, metadata, report)
    require(Path(metadata["reference_root"]).resolve() == reference_root.resolve(),
            f"{path.name}: checkpoint reference deck differs from this run")
    require(checkpoint_math(metadata) == math_mode,
            f"{path.name}: checkpoint math differs from this run")
    require(_resume_identity_is_compatible(
                metadata.get("input_identity"), checkpoint_input_identity(reference_root)
            ),
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
    carry = carry._replace(kt=int(np.asarray(carry.kt)))
    return carry, metadata, report


def read_nemo_ice_day(reference_root: Path, day: int) -> dict[str, np.ndarray]:
    """Read category-summed SI3 area, ice volume, and snow volume by rank."""

    import netCDF4

    kt = day * year_metrics.STEPS_PER_DAY
    rank_fields: list[dict[str, np.ndarray]] = []
    for rank in (0, 1):
        path = reference_root / f"ORCA2_{kt:08d}_restart_ice_{rank:04d}.nc"
        require(path.is_file(), f"missing NEMO SI3 restart {path}")
        with netCDF4.Dataset(path) as dataset:
            require("kt_ice" in dataset.variables, f"{path.name}: missing SI3 kt_ice")
            kt_ice = np.ma.asarray(dataset.variables["kt_ice"][:])
            require(int(np.ma.count_masked(kt_ice)) == 0 and kt_ice.size == 1,
                    f"{path.name}: invalid or fill-valued SI3 kt_ice")
            kt_value = float(np.asarray(kt_ice).item())
            require(math.isfinite(kt_value) and kt_value == float(kt),
                    f"{path.name}: SI3 kt_ice {kt_value} != restart kt {kt}")
            fields = {}
            for output, source in (
                ("area", "a_i"),
                ("volume", "v_i"),
                ("snow_volume", "v_s"),
            ):
                require(source in dataset.variables, f"{path.name}: missing SI3 {source}")
                raw = np.ma.asarray(dataset.variables[source][0])
                require(int(np.ma.count_masked(raw)) == 0,
                        f"{path.name}: SI3 {source} contains fill/masked values")
                fields[output] = np.asarray(raw, dtype=np.float64).sum(axis=0)
            rank_fields.append(fields)
    out = {
        name: np.concatenate((rank_fields[0][name], rank_fields[1][name]), axis=1)
        for name in ("area", "volume", "snow_volume")
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
    for output, source in (
        ("area", aggregated.at_i),
        ("volume", aggregated.vt_i),
        ("snow_volume", aggregated.vt_s),
    ):
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
        ("area", "ICE_AREA", ICE_UNIT_SCALE, "1e6 km2"),
        ("volume", "ICE_VOLUME", ICE_UNIT_SCALE, "1e3 km3"),
    ):
        # NEMO icewri.F90:222-241 assigns ff_t > 0 to NH and ELSEWHERE to SH,
        # and reports both area and volume after multiplying by 1.e-12.
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
        nemo_daily = np.asarray([row[key]["nemo"] for row in daily_rows], dtype=np.float64)
        candidate_daily = np.asarray(
            [row[key]["candidate"] for row in daily_rows], dtype=np.float64
        )
        require(bool(np.isfinite(nemo_daily).all() and np.isfinite(candidate_daily).all()),
                f"{key}: non-finite daily ice total")
        require(bool((nemo_daily > 0.0).all()), f"{key}: NEMO daily total is not positive")
        require(bool((candidate_daily >= 0.0).all()),
                f"{key}: candidate daily total is negative")
        nemo_mean = float(np.mean(nemo_daily))
        candidate_mean = float(np.mean(candidate_daily))
        absolute = abs(candidate_mean - nemo_mean)
        output[key] = {
            "metric": "relative_abs_monthly_mean_and_mean_daily_relative_abs",
            "nemo_monthly_mean": nemo_mean,
            "candidate_monthly_mean": candidate_mean,
            "absolute_difference": absolute,
            "relative_abs_difference": absolute / abs(nemo_mean),
            "mean_daily_relative_abs_difference": float(
                np.mean(np.abs(candidate_daily - nemo_daily) / np.abs(nemo_daily))
            ),
            "units": daily_rows[0][key]["units"],
        }
    return output


def monthly_ice_map_metrics(
    candidate_days: list[dict[str, np.ndarray]],
    nemo_days: list[dict[str, np.ndarray]],
    geo: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Area-weight monthly-mean ice/snow map displacement for Decision 238."""

    require(bool(candidate_days) and len(candidate_days) == len(nemo_days),
            "cannot aggregate empty or unequal C4 SI3 map windows")
    area_weight = np.asarray(geo["area_t"], dtype=np.float64)
    latitude = np.asarray(geo["gphit"], dtype=np.float64)
    require(area_weight.shape == latitude.shape == (148, 180),
            "C4 SI3 map geometry is not the owned (148, 180) T-point layout")
    output: dict[str, dict[str, Any]] = {}
    for field, label, units, verdict_bearing in (
        ("area", "ICE_MISPLACED_AREA", "1e6 km2", True),
        ("volume", "ICE_MISPLACED_VOLUME", "1e3 km3", True),
        ("snow_volume", "SNOW_MISPLACED_VOLUME", "1e3 km3", False),
    ):
        candidate = np.asarray([day[field] for day in candidate_days], dtype=np.float64)
        nemo = np.asarray([day[field] for day in nemo_days], dtype=np.float64)
        expected_shape = (len(candidate_days), 148, 180)
        require(candidate.shape == nemo.shape == expected_shape,
                f"{label}: daily map shape is not {expected_shape}")
        require(bool(np.isfinite(candidate).all() and np.isfinite(nemo).all()),
                f"{label}: daily map contains non-finite values")
        require(bool((candidate >= 0.0).all() and (nemo >= 0.0).all()),
                f"{label}: daily map contains negative values")
        candidate_mean = np.mean(candidate, axis=0)
        nemo_mean = np.mean(nemo, axis=0)
        for hemisphere, mask in (("NH", latitude > 0.0), ("SH", latitude <= 0.0)):
            nemo_total = float(np.sum(area_weight * nemo_mean * mask) / ICE_UNIT_SCALE)
            candidate_total = float(
                np.sum(area_weight * candidate_mean * mask) / ICE_UNIT_SCALE
            )
            misplaced = float(
                np.sum(area_weight * np.abs(candidate_mean - nemo_mean) * mask)
                / ICE_UNIT_SCALE
            )
            require(nemo_total > 0.0, f"{label}_{hemisphere}: NEMO total is not positive")
            output[f"{label}_{hemisphere}"] = {
                "metric": "area_weighted_abs_difference_of_monthly_mean_maps",
                "candidate_value": misplaced,
                "nemo_monthly_mean_hemisphere_total": nemo_total,
                "candidate_monthly_mean_hemisphere_total": candidate_total,
                "units": units,
                "verdict_bearing": verdict_bearing,
            }
    return output


def daily_map_record(
    day: int,
    candidate_ocean: dict[str, np.ndarray],
    nemo_ocean: dict[str, np.ndarray],
    candidate_ice: dict[str, np.ndarray],
    nemo_ice: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Collect one Decision-237 map row on the shared owned T-point layout."""

    record: dict[str, Any] = {
        "day": day,
        "candidate_sst": np.asarray(candidate_ocean["T"])[..., 0],
        "nemo_sst": np.asarray(nemo_ocean["T"])[..., 0],
    }
    for output, source in (
        ("ice_area", "area"),
        ("ice_volume", "volume"),
        ("snow_volume", "snow_volume"),
    ):
        record[f"candidate_{output}"] = np.asarray(candidate_ice[source])
        record[f"nemo_{output}"] = np.asarray(nemo_ice[source])
    for key, value in record.items():
        if key == "day":
            continue
        require(value.shape == (148, 180),
                f"day {day}: Decision-237 {key} shape {value.shape} != (148, 180)")
        require(bool(np.isfinite(value).all()),
                f"day {day}: Decision-237 {key} contains non-finite values")
    return record


def build_month_report(
    month_index: int,
    daily_rows: list[dict[str, Any]],
    daily_maps: list[dict[str, np.ndarray]],
    ice_rows: list[dict[str, dict[str, float]]],
    candidate_ice_maps: list[dict[str, np.ndarray]],
    nemo_ice_maps: list[dict[str, np.ndarray]],
    geo: dict[str, np.ndarray],
    *,
    metric_format: str = C4_METRIC_FORMAT,
    ice_sst_sss_mask_source: str = "Decision-188 ice_cover_clim.nc",
) -> dict[str, Any]:
    window = year_metrics.calendar_month_day_ranges()[month_index - 1]
    require(window["complete_month"], f"{window['label']}: calendar month is not complete")
    require([row["day"] for row in daily_rows] == window["available_days"],
            f"{window['label']}: daily accumulator does not cover its calendar window")
    summary = year_metrics.pilot_metrics.monthly(daily_rows, daily_maps, geo)
    metrics = year_metrics._attach_decision185_reference(summary, daily_rows)
    metrics.update(monthly_ice_metrics(ice_rows))
    metrics.update(monthly_ice_map_metrics(candidate_ice_maps, nemo_ice_maps, geo))
    return {
        **window,
        "format": metric_format,
        "status": "EVALUATED_COMPLETE_MONTH",
        "bar_mode": "DECISION_188_PASS_FAIL",
        "verdict": None,
        "ice_sst_sss_mask_source": ice_sst_sss_mask_source,
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

    c_ladder_scorer._validate_sponge_steps(
        (odd_sponge, even_sponge), int(np.asarray(carry.kt))
    )
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


def _normalize_compile_carry(carry: Any) -> Any:
    """Keep the resumed ``kt`` leaf weakly typed like the cold Python integer."""

    return carry._replace(kt=int(np.asarray(carry.kt)))


def _advance_pairs(executable, carry, forcing, sponge, start: int, stop: int):
    timings = []
    for pair_index in range(start, stop):
        odd_index = 2 * pair_index
        c_ladder_scorer._validate_sponge_steps(
            (sponge[odd_index], sponge[odd_index + 1]), int(np.asarray(carry.kt))
        )
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


def _run_restart_identity(
    reference_root: Path, output_dir: Path, *, math_mode: str
) -> dict[str, Any]:
    """Prove four real compiled pairs equal a two-pair checkpoint continuation."""

    jax.config.update("jax_enable_x64", True)
    reference_root = reference_root.resolve()
    total_pairs, split_pairs = 4, 2
    carry, forcing, sponge, card, model = c_ladder_scorer._c2_cold_start(
        reference_root, n_steps=2 * total_pairs, math_mode=math_mode
    )
    carry = _normalize_compile_carry(carry)
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
        math_mode=math_mode,
        month_report={
            "format": "nemo-coupled-c4-restart-identity-v1",
            "pairs_completed": split_pairs,
            "verdict": None,
        },
    )
    restored, _metadata, _report = restore_checkpoint(
        checkpoint,
        carry,
        reference_root,
        math_mode=math_mode,
        validate_calendar=False,
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
        "math": math_mode,
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


def run_restart_identity(
    reference_root: Path, output_dir: Path, *, math_mode: str
) -> dict[str, Any]:
    """Run the restart proof under the explicitly selected coupled math."""

    with c_ladder_scorer.coupled_math_scope(math_mode):
        return _run_restart_identity(
            reference_root, output_dir, math_mode=math_mode
        )


def _run_c4(
    reference_root: Path,
    output_dir: Path,
    *,
    math_mode: str,
    resume: Path | None = None,
    pair_limit: int | None = None,
    profile_only: bool = False,
) -> dict[str, Any]:
    """Run the compiled trajectory; full mode writes exactly 12 month checkpoints."""

    jax.config.update("jax_enable_x64", True)
    reference_root = reference_root.resolve()
    require(reference_root.is_dir(), f"missing C4 reference deck {reference_root}")
    controls = resolved_c4_controls(reference_root)
    if profile_only:
        require(resume is None, "C4 profile mode does not resume checkpoints")
        require(type(pair_limit) is int and 1 <= pair_limit <= 4,
                "C4 profile mode requires --pair-limit in [1, 4]")
        n_steps = 2 * pair_limit
    else:
        require(pair_limit is None, "a verdict-bearing C4 year cannot use --pair-limit")
        n_steps = controls["nn_itend"]
        require(n_steps == 2 * TOTAL_PAIRS and n_steps % 2 == 0,
                f"resolved C4 step count {n_steps} is not {2 * TOTAL_PAIRS}")

    carry, seed_forcing, _seed_sponge, card, model = c_ladder_scorer._c2_cold_start(
        reference_root, n_steps=2, math_mode=math_mode
    )
    carry = _normalize_compile_carry(carry)
    require(float(card.ocean.dt_s) == controls["rn_Dt"],
            f"coupled card dt {card.ocean.dt_s} != resolved rn_Dt {controls['rn_Dt']}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resumed_month = 0
    run_input_identity = checkpoint_input_identity(reference_root)
    metric_format = C4_METRIC_FORMAT
    if resume is not None:
        resume = resume.resolve()
        require(resume.parent == output_dir,
                "C4 resume checkpoint must be in the output checkpoint directory")
        carry, metadata, resumed_report = restore_checkpoint(
            resume, carry, reference_root, math_mode=math_mode
        )
        resumed_month = int(metadata["month"])
        prefix_identity, prefix_metric_format, prefix_math = validate_checkpoint_prefix(
            output_dir, reference_root, resumed_month
        )
        validate_daily_map_prefix(output_dir, resumed_month)
        require(prefix_identity == metadata.get("input_identity"),
                "terminal resume checkpoint identity differs from its prefix")
        require(prefix_metric_format == resumed_report.get("format"),
                "terminal resume checkpoint metric format differs from its prefix")
        require(prefix_math == math_mode,
                "terminal resume checkpoint math differs from this run")
        run_input_identity = prefix_identity
        metric_format = prefix_metric_format
        require(resume.name == checkpoint_name(
                    resumed_month, int(metadata["completed_kt"])
                ),
                "C4 resume path is not the terminal checkpoint in its prefix")
    elif not profile_only:
        require(not list(output_dir.glob("c4_month??_step*.npz")),
                f"refusing a cold C4 run in nonempty checkpoint directory {output_dir}")
        require(not list(output_dir.glob("c4_daily_maps_month??.npz")),
                f"refusing a cold C4 run with existing daily maps in {output_dir}")
    start_pair = (int(np.asarray(carry.kt)) - 1) // 2
    stop_pair = pair_limit if profile_only else TOTAL_PAIRS
    expected_months = len(year_metrics.calendar_month_day_ranges())
    require(0 <= start_pair <= stop_pair,
            f"C4 start pair {start_pair} is outside requested stop pair {stop_pair}")
    if start_pair == stop_pair:
        require(not profile_only and resumed_month == expected_months,
                "only the final monthly checkpoint may end a resumed C4 run")
        return {
            "format": "nemo-coupled-c4-run-v1",
            "status": f"PASS_C4_RUN_{expected_months}_CHECKPOINTS",
            "math": math_mode,
            "reference_root": str(reference_root),
            "backend": jax.default_backend(),
            "fp64": bool(jax.config.jax_enable_x64),
            "compiled": False,
            "compile_count": 0,
            "compile_seconds": 0.0,
            "pairs_executed": 0,
            "pair_wall_seconds": [],
            "mean_pair_wall_seconds": None,
            "resumed_month": expected_months,
            "already_complete": True,
            "resolved_controls": controls,
            "metric_format": metric_format,
            "checkpoints": [],
            "daily_maps": [],
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
        mld = year_metrics.K0MLDEvaluator(
            reference_root, precision_policy=get_policy()
        )
        cover = r340.read_decision171_cover(year_driver.DEFAULT_ICE_COVER_PATH)
        mask_source = "Decision-188 ice_cover_clim.nc"
        metric_context = (geo, mld, cover, mask_source)

    month_rows: list[dict[str, Any]] = []
    month_maps: list[dict[str, np.ndarray]] = []
    month_ice: list[dict[str, dict[str, float]]] = []
    month_candidate_ice_maps: list[dict[str, np.ndarray]] = []
    month_nemo_ice_maps: list[dict[str, np.ndarray]] = []
    month_daily_maps: list[dict[str, Any]] = []
    checkpoints = []
    daily_map_outputs = []
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
        c_ladder_scorer._validate_sponge_steps(
            (odd_sponge, even_sponge), int(np.asarray(carry.kt))
        )
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

        geo, mld, cover, mask_source = metric_context
        day = completed_kt // year_metrics.STEPS_PER_DAY
        candidate = ladder._candidate_fields(carry.ocean)
        nemo = year_metrics.pilot_metrics.read_nemo_day(reference_root, day)
        row, maps = year_metrics.pilot_metrics.day_metrics(candidate, nemo, geo, mld, cover)
        year_metrics._augment_day_single_arm_stats(row, candidate, nemo, geo, cover, maps)
        month_rows.append({"day": day, "kt": completed_kt, **row})
        month_maps.append(maps)
        candidate_ice = candidate_ice_fields(carry, nn_hls)
        nemo_ice = read_nemo_ice_day(reference_root, day)
        month_ice.append(ice_day_metrics(candidate_ice, nemo_ice, geo))
        month_candidate_ice_maps.append(candidate_ice)
        month_nemo_ice_maps.append(nemo_ice)
        month_daily_maps.append(daily_map_record(
            day, candidate, nemo, candidate_ice, nemo_ice
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
            month_number,
            month_rows,
            month_maps,
            month_ice,
            month_candidate_ice_maps,
            month_nemo_ice_maps,
            geo,
            metric_format=metric_format,
            ice_sst_sss_mask_source=mask_source,
        )
        daily_map_outputs.append(write_daily_maps(
            output_dir / daily_maps_name(month_number),
            month_number,
            month_daily_maps,
            geo,
        ))
        path = output_dir / checkpoint_name(month_number, completed_kt)
        checkpoints.append(write_checkpoint(
            path,
            carry,
            month=month_number,
            completed_kt=completed_kt,
            reference_root=reference_root,
            math_mode=math_mode,
            month_report=month_report,
            input_identity=run_input_identity,
        ))
        month_rows, month_maps, month_ice = [], [], []
        month_candidate_ice_maps, month_nemo_ice_maps, month_daily_maps = [], [], []

    require(compile_count() == 1,
            f"C4 pair retraced {compile_count()} times; expected exactly 1")
    if not profile_only:
        final_next_kt = 2 * stop_pair + 1
        require(int(np.asarray(carry.kt)) == final_next_kt,
                f"C4 final next kt is {carry.kt}, expected {final_next_kt}")
        require(resumed_month + len(checkpoints) == expected_months,
                f"C4 wrote through month {resumed_month + len(checkpoints)}, "
                f"expected {expected_months}")
    return {
        "format": "nemo-coupled-c4-run-v1",
        "status": (
            f"PROFILE_C4_{len(pair_seconds)}_PAIRS_NO_VERDICT"
            if profile_only else f"PASS_C4_RUN_{expected_months}_CHECKPOINTS"
        ),
        "math": math_mode,
        "reference_root": str(reference_root),
        "backend": jax.default_backend(),
        "fp64": bool(jax.config.jax_enable_x64),
        "resolved_controls": controls,
        "metric_format": metric_format,
        "compiled": True,
        "compile_count": compile_count(),
        "compile_seconds": compile_seconds,
        "pairs_executed": len(pair_seconds),
        "pair_wall_seconds": pair_seconds,
        "mean_pair_wall_seconds": float(np.mean(pair_seconds)),
        "resumed_month": resumed_month,
        "checkpoints": checkpoints,
        "daily_maps": daily_map_outputs,
    }


def run_c4(
    reference_root: Path,
    output_dir: Path,
    *,
    math_mode: str,
    resume: Path | None = None,
    pair_limit: int | None = None,
    profile_only: bool = False,
) -> dict[str, Any]:
    """Run C4 under exact CPU or round-344 fast-literal math."""

    with c_ladder_scorer.coupled_math_scope(math_mode):
        return _run_c4(
            reference_root,
            output_dir,
            math_mode=math_mode,
            resume=resume,
            pair_limit=pair_limit,
            profile_only=profile_only,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--math", choices=MATH_CHOICES, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--restart-identity", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.math == "libm-cpu" and jax.default_backend() != "cpu":
        parser.error("--math libm-cpu requires the CPU backend")
    if args.restart_identity:
        require(args.resume is None and args.pair_limit is None and not args.profile_only,
                "--restart-identity cannot be combined with resume/profile options")
        report = run_restart_identity(
            args.reference, args.output_dir, math_mode=args.math
        )
    else:
        report = run_c4(
            args.reference,
            args.output_dir,
            math_mode=args.math,
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
