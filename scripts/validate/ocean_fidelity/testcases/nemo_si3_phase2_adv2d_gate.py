#!/usr/bin/env python3
"""Fail-closed fp64 gate for legoESM SI3 lane-3 rung 3.2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import jax
import netCDF4
import numpy as np

POINTWISE_BAR = 1.0e-15
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final")
REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE1 = Path(__file__).with_name("nemo_si3_oracle_gate.py")
PHASE2_1D = Path(__file__).with_name("nemo_si3_phase2_gate.py")
MANIFEST = Path(__file__).with_name("manifests") / "ice_adv2d_l3.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle_gate = _load("nemo_si3_oracle_gate_adv2d", PHASE1)
base_gate = _load("nemo_si3_phase2_gate_adv2d", PHASE2_1D)
GateError = base_gate.GateError
require = base_gate.require
_score = base_gate._score
_entry_field = base_gate._entry_field
_restart_field = base_gate._restart_field
_restart_2d_field = base_gate._restart_2d_field
_moment_restart_name = base_gate._moment_restart_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frame_header(header: dict[str, int], *, kt: int, card) -> None:
    """Reject a mislabeled/retyped oracle frame before using its arrays."""

    expected = {
        "kt": kt,
        "jpi": card.wet_global_xy.shape[0] + 2 * card.halo_width,
        "jpj": card.wet_global_xy.shape[1] + 2 * card.halo_width,
        "jpl": card.jpl,
        "nlay_i": card.nlay_i,
        "nlay_s": card.nlay_s,
        "storage_bits": 64,
    }
    require(header == expected, f"oracle frame header at kt={kt}: {header} != {expected}")


def _read_oracle_frame(root: Path, *, kt: int, card) -> dict[str, np.ndarray]:
    path = root / f"oracle_ice_step_entry_kt{kt:08d}.bin"
    header, frame = oracle_gate.read_frame(path)
    _validate_frame_header(header, kt=kt, card=card)
    return cast(dict[str, np.ndarray], frame)


def frame_provenance(root: Path, card) -> dict:
    """Hash and validate the complete ordered oracle step-entry sequence."""

    paths = sorted(root.glob("oracle_ice_step_entry_kt*.bin"))
    require(
        len(paths) == card.n_steps,
        f"expected {card.n_steps} oracle frames, found {len(paths)}",
    )
    hashes: list[str] = []
    header0: dict[str, int] | None = None
    for kt, path in enumerate(paths, start=1):
        expected_name = f"oracle_ice_step_entry_kt{kt:08d}.bin"
        require(
            path.name == expected_name,
            f"oracle frame sequence: {path.name} != {expected_name}",
        )
        header, _ = oracle_gate.read_frame(path)
        _validate_frame_header(header, kt=kt, card=card)
        if header0 is None:
            header0 = header
        else:
            require(
                {key: value for key, value in header.items() if key != "kt"}
                == {key: value for key, value in header0.items() if key != "kt"},
                f"oracle frame shape changed at kt={kt}",
            )
        hashes.append(_sha256(path))
    assert header0 is not None
    return {
        "frame_count": len(paths),
        "header": header0,
        "first_sha256": hashes[0],
        "last_sha256": hashes[-1],
        "ordered_frame_hash_aggregate": hashlib.sha256(
            "\n".join(hashes).encode()
        ).hexdigest(),
    }


def mesh_coverage(root: Path, *, plant: bool = False) -> dict[str, int]:
    manifest = json.loads(MANIFEST.read_text())
    return cast(
        dict[str, int],
        oracle_gate.check_manifest(root, "3.2", manifest, plant_unaccounted=plant),
    )


def candidate_frame_contract(*, plant: bool = False) -> dict[str, dict[str, str]]:
    verified = {
        "v_i",
        "v_s",
        "a_i",
        "t_su",
        "oa_i",
        "u_ice",
        "v_ice",
        "e_s",
        "e_i",
        "sv_i",
        "szv_i",
        "a_ip",
        "v_ip",
        "v_il",
    }
    waived = {
        "stress1_i": "WAIVED-INACTIVE: ADV2D prescribes velocity (icedyn.F90:159-166)",
        "stress2_i": "WAIVED-INACTIVE: ADV2D prescribes velocity (icedyn.F90:159-166)",
        "stress12_i": "WAIVED-INACTIVE: ADV2D prescribes velocity (icedyn.F90:159-166)",
    }
    unmeasured = {
        "snwice_mass": "UNMEASURED: derived snow-plus-ice diagnostic",
        "snwice_mass_b": "UNMEASURED: before-level derived diagnostic",
    }
    contract = {
        name: {"status": "VERIFIED", "reason": "scored at every common boundary"}
        for name in verified
    }
    contract.update(
        {name: {"status": "WAIVED", "reason": reason} for name, reason in waived.items()}
    )
    contract.update(
        {name: {"status": "UNMEASURED", "reason": reason} for name, reason in unmeasured.items()}
    )
    if plant:
        contract["PLANTED_UNACCOUNTED_FRAME_ARRAY"] = {
            "status": "VERIFIED",
            "reason": "planted",
        }
    expected = {row[0] for row in oracle_gate.FRAME_REGISTRY}
    require(
        set(contract) == expected,
        f"ICE_ADV2D candidate frame coverage mismatch: "
        f"missing={sorted(expected - set(contract))}, extra={sorted(set(contract) - expected)}",
    )
    return dict(sorted(contract.items()))


def oracle_surface_temperature_c(root: Path) -> np.ndarray:
    with netCDF4.Dataset(root / "output.init_ice.nc") as ds:
        value: np.ndarray = np.asarray(ds["sst"][0]).T
    require(value.shape == (99, 99), f"ICE_ADV2D sst shape {value.shape}")
    return value


def geometry_gate(root: Path, card, rows: list[dict], dtypes: dict, *, plant: bool = False) -> None:
    grid = card.grid
    with netCDF4.Dataset(root / "mesh_mask.nc") as ds:
        lon_u = np.asarray(grid.lon_T) + 0.5 * card.dx_m / grid.radius
        lat_v = np.asarray(grid.lat_T) + 0.5 * card.dy_m / grid.radius
        fields = {
            "glamt": np.asarray(grid.lon_T) * grid.radius / 1000.0,
            "glamu": lon_u * grid.radius / 1000.0,
            "glamv": np.asarray(grid.lon_T) * grid.radius / 1000.0,
            "glamf": lon_u * grid.radius / 1000.0,
            "gphit": np.asarray(grid.lat_T) * grid.radius / 1000.0,
            "gphiu": np.asarray(grid.lat_T) * grid.radius / 1000.0,
            "gphiv": lat_v * grid.radius / 1000.0,
            "gphif": lat_v * grid.radius / 1000.0,
            "e1t": np.asarray(grid.dx_T),
            "e1u": np.asarray(grid.dx_u)[:, 1:],
            "e1v": np.asarray(grid.dx_v)[1:],
            "e1f": np.asarray(grid.dx_u)[:, 1:],
            "e2t": np.asarray(grid.dy_T),
            "e2u": np.asarray(grid.dy_u)[:, 1:],
            "e2v": np.asarray(grid.dy_v)[1:],
            "e2f": np.asarray(grid.dy_v)[1:],
            "ff_t": np.asarray(grid.f_T),
            "ff_f": np.asarray(grid.f_v)[1:],
        }
        for name, value in fields.items():
            _score(
                rows,
                dtypes,
                f"geometry.{name}",
                np.asarray(ds[name][0]),
                value,
                plant=plant and name == "e1t",
            )
        all_wet: np.ndarray = np.ones((99, 99), dtype=bool)
        for name in ("tmask", "umask", "vmask", "fmask"):
            _score(
                rows,
                dtypes,
                f"geometry.{name}",
                np.asarray(ds[name][0, 0]).astype(bool),
                all_wet,
                exact=True,
            )


def state_field(card, state, name: str) -> np.ndarray:
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS

    halo = card.halo_width
    area = card.dx_m * card.dy_m
    return cast(
        np.ndarray,
        np.asarray(state.contents)[halo:-halo, halo:-halo, ICE_ADV2D_TRACERS.index(name)] / area,
    )


def _entry_tracer(frame: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name.startswith(("e_s_l", "e_i_l")):
        return cast(np.ndarray, _entry_field(frame, name))
    if name.startswith("szv_i_l"):
        layer = int(name.rsplit("_l", 1)[1]) - 1
        return cast(np.ndarray, np.asarray(oracle_gate._interior(frame["szv_i"])[..., layer, 0]))
    return cast(np.ndarray, _entry_field(frame, name))


def _restart_tracer(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    return cast(np.ndarray, _restart_field(ds, name))


def _bulk_salinity(card, state) -> np.ndarray:
    halo = card.halo_width
    return cast(
        np.ndarray,
        np.asarray(state.bulk_salt_diagnostic)[halo:-halo, halo:-halo],
    )


def _moment_name(moment: str, tracer: str) -> str:
    base = {
        "v_i": "ice",
        "v_s": "sn",
        "a_i": "a",
        "oa_i": "age",
        "e_s_l01": "c0_l01",
        "e_s_l02": "c0_l02",
        "e_s_l03": "c0_l03",
        "e_i_l01": "e_l01",
        "e_i_l02": "e_l02",
        "e_i_l03": "e_l03",
        "szv_i_l01": "si_l01",
        "szv_i_l02": "si_l02",
        "szv_i_l03": "si_l03",
        "a_ip": "ap",
        "v_ip": "vp",
        "v_il": "vl",
    }[tracer]
    return moment + base


def _validate_row_registry(root: Path, card, rows: list[dict]) -> None:
    """Require the exact preregistered geometry/boundary/moment roster."""

    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS
    from legoesm.ice.transport import SI3_PRATHER_MOMENT_NAMES

    geometry = {
        f"geometry.{name}"
        for name in (
            "glamt",
            "glamu",
            "glamv",
            "glamf",
            "gphit",
            "gphiu",
            "gphiv",
            "gphif",
            "e1t",
            "e1u",
            "e1v",
            "e1f",
            "e2t",
            "e2u",
            "e2v",
            "e2f",
            "ff_t",
            "ff_f",
            "tmask",
            "umask",
            "vmask",
            "fmask",
        )
    }
    boundary_fields = set(ICE_ADV2D_TRACERS) | {"sv_i", "t_surface", "u_ice", "v_ice"}
    boundary = {f"entry.kt00000001.{name}" for name in boundary_fields}
    boundary |= {
        f"trajectory.post_step_{step:08d}.{name}"
        for step in range(1, card.n_steps + 1)
        for name in boundary_fields
    }
    expected_moments = {
        _moment_name(moment, tracer)
        for moment in SI3_PRATHER_MOMENT_NAMES
        for tracer in ICE_ADV2D_TRACERS
    }
    restart_path = oracle_gate.run_files(root)["restart"]
    with netCDF4.Dataset(restart_path) as ds:
        discovered_moments = {
            name for name in ds.variables if name.startswith(("sx", "sy"))
        }
    require(
        discovered_moments == expected_moments,
        "oracle active Prather-moment roster mismatch: "
        f"missing={sorted(expected_moments - discovered_moments)}, "
        f"extra={sorted(discovered_moments - expected_moments)}",
    )
    moments = {f"restart_moment.{name}" for name in discovered_moments}
    expected = geometry | boundary | moments
    names = [str(row["name"]) for row in rows]
    require(len(names) == len(set(names)), "duplicate registered comparison row")
    require(
        set(names) == expected,
        "comparison roster mismatch: "
        f"missing={sorted(expected - set(names))}, extra={sorted(set(names) - expected)}",
    )


def comparison_gate(root: Path, card, rows: list[dict], dtypes: dict, *, plant: bool = False):
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
        ICE_ADV2D_TRACERS,
        step_ice_adv2d_card,
    )
    from legoesm.ice.transport import SI3_PRATHER_MOMENT_NAMES

    state = card.initial_state
    first = None
    frame = _read_oracle_frame(root, kt=1, card=card)
    for name in ICE_ADV2D_TRACERS:
        row = _score(
            rows,
            dtypes,
            f"entry.kt00000001.{name}",
            _entry_tracer(frame, name),
            state_field(card, state, name),
            plant=plant and name == "v_i",
        )
        first = row if first is None and row["status"] == "DEBT" else first
    row = _score(
        rows,
        dtypes,
        "entry.kt00000001.sv_i",
        _entry_field(frame, "sv_i"),
        _bulk_salinity(card, state),
    )
    first = row if first is None and row["status"] == "DEBT" else first
    halo = card.halo_width
    for name, oracle, lego in (
        (
            "t_surface",
            _entry_field(frame, "t_su"),
            np.asarray(state.t_surface)[halo:-halo, halo:-halo],
        ),
        (
            "u_ice",
            oracle_gate._interior(frame["u_ice"]),
            np.asarray(state.u_ice)[halo:-halo, halo:-halo],
        ),
        (
            "v_ice",
            oracle_gate._interior(frame["v_ice"]),
            np.asarray(state.v_ice)[halo:-halo, halo:-halo],
        ),
    ):
        row = _score(rows, dtypes, f"entry.kt00000001.{name}", oracle, lego)
        first = row if first is None and row["status"] == "DEBT" else first

    trajectory = ICE_ADV2D_TRACERS
    for completed in range(1, card.n_steps):
        state = step_ice_adv2d_card(card, state, completed_steps=completed - 1)
        kt = completed + 1
        frame = _read_oracle_frame(root, kt=kt, card=card)
        for name in trajectory:
            row = _score(
                rows,
                dtypes,
                f"trajectory.post_step_{completed:08d}.{name}",
                _entry_tracer(frame, name),
                state_field(card, state, name),
            )
            first = row if first is None and row["status"] == "DEBT" else first
        row = _score(
            rows,
            dtypes,
            f"trajectory.post_step_{completed:08d}.sv_i",
            _entry_field(frame, "sv_i"),
            _bulk_salinity(card, state),
        )
        first = row if first is None and row["status"] == "DEBT" else first
        comparisons = (
            (
                "t_surface",
                _entry_field(frame, "t_su"),
                np.asarray(state.t_surface)[halo:-halo, halo:-halo],
            ),
            (
                "u_ice",
                oracle_gate._interior(frame["u_ice"]),
                np.asarray(state.u_ice)[halo:-halo, halo:-halo],
            ),
            (
                "v_ice",
                oracle_gate._interior(frame["v_ice"]),
                np.asarray(state.v_ice)[halo:-halo, halo:-halo],
            ),
        )
        for name, oracle, lego in comparisons:
            row = _score(rows, dtypes, f"trajectory.post_step_{completed:08d}.{name}", oracle, lego)
            first = row if first is None and row["status"] == "DEBT" else first

    state = step_ice_adv2d_card(card, state, completed_steps=card.n_steps - 1)
    restart = oracle_gate.run_files(root)["restart"]
    with netCDF4.Dataset(restart) as ds:
        for name in trajectory:
            row = _score(
                rows,
                dtypes,
                f"trajectory.post_step_{card.n_steps:08d}.{name}",
                _restart_tracer(ds, name),
                state_field(card, state, name),
            )
            first = row if first is None and row["status"] == "DEBT" else first
        row = _score(
            rows,
            dtypes,
            f"trajectory.post_step_{card.n_steps:08d}.sv_i",
            _restart_field(ds, "sv_i"),
            _bulk_salinity(card, state),
        )
        first = row if first is None and row["status"] == "DEBT" else first
        for name, oracle, lego in (
            (
                "t_surface",
                _restart_field(ds, "t_su"),
                np.asarray(state.t_surface)[halo:-halo, halo:-halo],
            ),
            (
                "u_ice",
                _restart_2d_field(ds, "u_ice"),
                np.asarray(state.u_ice)[halo:-halo, halo:-halo],
            ),
            (
                "v_ice",
                _restart_2d_field(ds, "v_ice"),
                np.asarray(state.v_ice)[halo:-halo, halo:-halo],
            ),
        ):
            row = _score(
                rows, dtypes, f"trajectory.post_step_{card.n_steps:08d}.{name}", oracle, lego
            )
            first = row if first is None and row["status"] == "DEBT" else first
        for moment_index, moment in enumerate(SI3_PRATHER_MOMENT_NAMES):
            for tracer_index, tracer in enumerate(ICE_ADV2D_TRACERS):
                restart_name = _moment_name(moment, tracer)
                row = _score(
                    rows,
                    dtypes,
                    f"restart_moment.{restart_name}",
                    _restart_field(ds, restart_name),
                    np.asarray(state.moments[moment_index])[halo:-halo, halo:-halo, tracer_index],
                )
                first = row if first is None and row["status"] == "DEBT" else first
    return state, first


def restart_control(card, *, plant: bool = False, drop: bool = False, retype: bool = False) -> dict:
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
        load_ice_adv2d_restart,
        save_ice_adv2d_restart,
        step_ice_adv2d_card,
    )

    state = card.initial_state
    for completed in range(7):
        state = step_ice_adv2d_card(card, state, completed_steps=completed)
    with tempfile.TemporaryDirectory(prefix="si3-adv2d-restart-") as directory:
        path = Path(directory) / "state.npz"
        save_ice_adv2d_restart(path, card, state, completed_steps=7)
        if plant or drop or retype:
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: archive[name].copy() for name in archive.files}
            if plant:
                payload["moment_4"][20, 20, 0] += 1.0
            if drop:
                payload.pop("moment_4")
            if retype:
                payload["moment_4"] = payload["moment_4"].astype(np.float32)
            np.savez(path, **payload)
        restored, clock = load_ice_adv2d_restart(path, card)
        require(clock == 7, "ICE_ADV2D restart clock changed")
        require(
            all(
                np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(
                    jax.tree_util.tree_leaves(state),
                    jax.tree_util.tree_leaves(restored),
                    strict=True,
                )
            ),
            "ICE_ADV2D restart state or moment carry changed",
        )
        direct = step_ice_adv2d_card(card, state, completed_steps=7)
        resumed = step_ice_adv2d_card(card, restored, completed_steps=clock)
        require(
            all(
                np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(direct, resumed, strict=True)
            ),
            "ICE_ADV2D restart continuation changed",
        )
    return {"status": "VERIFIED", "state_arrays": 10, "moment_leaves": 80}


def run_gate(
    root: Path,
    *,
    plant_mesh: bool = False,
    plant_frame: bool = False,
    plant_geometry: bool = False,
    plant_state: bool = False,
    plant_moment: bool = False,
    drop_moment: bool = False,
    retype_moment: bool = False,
) -> tuple[dict, int]:
    from legoesm.core.precision import PrecisionPolicy, get_policy
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
        build_ice_adv2d_card,
        validate_ice_adv2d_card,
    )

    require(jax.default_backend() == "cpu", "CPU backend required")
    inventory = mesh_coverage(root, plant=plant_mesh)
    contract = candidate_frame_contract(plant=plant_frame)
    card = build_ice_adv2d_card(oracle_surface_temperature_c(root))
    validate_ice_adv2d_card(card)
    require(get_policy() == PrecisionPolicy.fp64(), "fp64 policy required")
    frames = frame_provenance(root, card)
    rows: list[dict] = []
    dtypes: dict = {}
    geometry_gate(root, card, rows, dtypes, plant=plant_geometry)
    _, first = comparison_gate(root, card, rows, dtypes, plant=plant_state)
    restart = restart_control(card, plant=plant_moment, drop=drop_moment, retype=retype_moment)
    _validate_row_registry(root, card, rows)
    debt = [row for row in rows if row["status"] == "DEBT"]
    unmeasured = [
        "candidate_frame.snwice_mass",
        "candidate_frame.snwice_mass_b",
        "thermodynamics",
        "rheology",
        "ridging_rafting",
        "landfast_L16_deferred_lane4",
        "coupled_ice_ocean",
        "production_restart_integration",
    ]
    numeric_status = "DEBT" if debt else "AT-BAR"
    report = {
        # Coverage debt takes precedence: a numerically AT-BAR future run must
        # not turn green while required candidate fields remain UNMEASURED.
        "status": "UNMEASURED" if unmeasured else numeric_status,
        "numeric_status": numeric_status,
        "scope": "ICE_ADV2D_OMIP_L3_LEGOESM_PHASE2",
        "backend": jax.default_backend(),
        "precision_policy": "fp64",
        "pointwise_bar": POINTWISE_BAR,
        "inventory_counts": inventory,
        "candidate_frame_contract": contract,
        "rows": rows,
        "dtypes": dtypes,
        "trajectory": {"compared_step_entries": card.n_steps, "first_divergence": first},
        "oracle_frame_sequence": frames,
        "restart_carry_control": restart,
        "provenance": {
            "oracle_root": str(root),
            "oracle_input_sha256": {
                "mesh_mask.nc": _sha256(root / "mesh_mask.nc"),
                "output.init_ice.nc": _sha256(root / "output.init_ice.nc"),
                oracle_gate.run_files(root)["restart"].name: _sha256(
                    oracle_gate.run_files(root)["restart"]
                ),
                "ordered_ice_step_entry_frames": frames[
                    "ordered_frame_hash_aggregate"
                ],
            },
            "git_parent_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "manifest_sha256": _sha256(MANIFEST),
            "implementation_sha256": {
                "transport.py": _sha256(REPO_ROOT / "packages/ice/legoesm/ice/transport.py"),
                "nemo_adv2d_testcase_recipe.py": _sha256(
                    REPO_ROOT / "packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py"
                ),
                "nemo_si3_phase2_adv2d_gate.py": _sha256(Path(__file__).resolve()),
            },
        },
        "unmeasured": unmeasured,
    }
    require(len(rows) == 9822, f"corrected registered row count {len(rows)} != 9822")
    return report, 1 if debt or unmeasured else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-mesh", action="store_true")
    parser.add_argument("--plant-frame", action="store_true")
    parser.add_argument("--plant-geometry", action="store_true")
    parser.add_argument("--plant-state", action="store_true")
    parser.add_argument("--plant-moment", action="store_true")
    parser.add_argument("--drop-moment", action="store_true")
    parser.add_argument("--retype-moment", action="store_true")
    args = parser.parse_args()
    report, code = run_gate(
        args.run_dir.resolve(),
        plant_mesh=args.plant_mesh,
        plant_frame=args.plant_frame,
        plant_geometry=args.plant_geometry,
        plant_state=args.plant_state,
        plant_moment=args.plant_moment,
        drop_moment=args.drop_moment,
        retype_moment=args.retype_moment,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, oracle_gate.GateError) as exc:
        print(f"DEBT: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
