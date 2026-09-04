#!/usr/bin/env python3
"""Coverage-first legoESM gate for SI3 lane-3 rung 3.1.

The only implemented physics is the source-pinned ICE_ADV1D Prather arm.
Every result is classified by the preregistered 1e-15 pointwise bar; the
script reports the first divergence and exits nonzero when any scored row is
outside it.
"""

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
VALID = {"VERIFIED", "WAIVED", "UNMEASURED"}
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d/final")
PHASE1 = Path(__file__).with_name("nemo_si3_oracle_gate.py")
MANIFEST = Path(__file__).with_name("manifests") / "ice_adv1d_l3.json"
REPO_ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location("nemo_si3_oracle_gate", PHASE1)
assert _SPEC and _SPEC.loader
oracle_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle_gate)


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(
    rows: list[dict],
    dtypes: dict[str, dict[str, str]],
    name: str,
    oracle,
    lego,
    *,
    exact: bool = False,
    plant: bool = False,
) -> dict:
    oracle = np.asarray(oracle)
    lego = np.asarray(lego)
    require(oracle.shape == lego.shape, f"{name}: shape {oracle.shape} != {lego.shape}")
    dtypes[name] = {"oracle": str(oracle.dtype), "legoesm": str(lego.dtype)}
    if np.issubdtype(lego.dtype, np.floating):
        require(lego.dtype == np.float64, f"{name}: legoESM dtype {lego.dtype} != float64")
    require(bool(np.all(np.isfinite(oracle))), f"{name}: oracle nonfinite")
    require(bool(np.all(np.isfinite(lego))), f"{name}: legoESM nonfinite")
    if plant:
        lego = lego.copy()
        lego.flat[0] += 1.0
    if oracle.dtype == np.bool_ or lego.dtype == np.bool_:
        absolute = np.not_equal(lego, oracle).astype(np.float64)
        oracle_magnitude = oracle.astype(np.float64)
    else:
        absolute = np.abs(lego - oracle)
        oracle_magnitude = np.abs(oracle)
    max_abs = float(np.max(absolute, initial=0.0))
    nonzero = int(np.count_nonzero(absolute))
    oracle_max_abs = float(np.max(oracle_magnitude, initial=0.0))
    relative_max_abs = (
        max_abs / oracle_max_abs
        if oracle_max_abs > 0.0
        else (0.0 if max_abs == 0.0 else None)
    )
    if exact:
        error = 0.0 if np.array_equal(oracle, lego) else float("inf")
        bar = 0.0
    else:
        scale = max(oracle_max_abs, 1.0)
        error = max_abs / scale
        bar = POINTWISE_BAR
    row = {
        "name": name,
        "status": "AT-BAR" if error <= bar else "DEBT",
        "arithmetic_class": "EXACT" if exact else "POINTWISE",
        "normalized_max_abs": error,
        "max_abs": max_abs,
        "oracle_max_abs": oracle_max_abs,
        "relative_max_abs": relative_max_abs,
        "bar": bar,
        "n": int(oracle.size),
        "bitwise_nonzero_over_n": f"{nonzero} / {oracle.size}",
    }
    rows.append(row)
    return row


def _mesh_coverage(root: Path, plant_unaccounted: bool) -> dict[str, int]:
    manifest = json.loads(MANIFEST.read_text())
    counts = oracle_gate.check_manifest(
        root, "3.1", manifest, plant_unaccounted=plant_unaccounted
    )
    for namespace, ledger in manifest["entries"].items():
        for name, item in ledger.items():
            require(item["status"] in VALID, f"{namespace}.{name}: bad disposition")
            require(bool(item["reason"].strip()), f"{namespace}.{name}: empty reason")
    return cast(dict[str, int], counts)


def candidate_frame_contract(
    *, plant_unaccounted: bool = False
) -> dict[str, dict[str, str]]:
    """Account for every phase-1 step-entry field, with no silent subset."""

    verified = {
        "v_i", "v_s", "a_i", "t_su", "oa_i", "u_ice", "v_ice", "e_s", "e_i"
    }
    waived = {
        "a_ip": "WAIVED-INACTIVE: ORCA1 ponds are off (cfg:151-152)",
        "v_ip": "WAIVED-INACTIVE: ORCA1 ponds are off (cfg:151-152)",
        "v_il": "WAIVED-INACTIVE: ORCA1 ponds are off (cfg:151-152)",
        "stress1_i": "WAIVED-INACTIVE: ADV1D prescribes velocity (icedyn.F90:144-157)",
        "stress2_i": "WAIVED-INACTIVE: ADV1D prescribes velocity (icedyn.F90:144-157)",
        "stress12_i": "WAIVED-INACTIVE: ADV1D prescribes velocity (icedyn.F90:144-157)",
    }
    unmeasured = {
        "sv_i": "UNMEASURED: ORCA1 option 2 versus phase-1 option-4 oracle",
        "szv_i": "UNMEASURED: phase-1 option-4 layer salinity is outside card",
        "snwice_mass": (
            "UNMEASURED: nonzero snow-plus-ice mass carry (iceistate.F90:400) "
            "is not an independent card transport tracer"
        ),
        "snwice_mass_b": (
            "UNMEASURED: before-level snow-plus-ice mass carry is outside the card"
        ),
    }
    contract = {
        name: {"status": "VERIFIED", "reason": "scored at every available boundary"}
        for name in verified
    }
    contract.update(
        {name: {"status": "WAIVED", "reason": reason} for name, reason in waived.items()}
    )
    contract.update(
        {name: {"status": "UNMEASURED", "reason": reason} for name, reason in unmeasured.items()}
    )
    if plant_unaccounted:
        contract["PLANTED_UNACCOUNTED_FRAME_ARRAY"] = {
            "status": "VERIFIED",
            "reason": "planted violation",
        }
    expected = {item[0] for item in oracle_gate.FRAME_REGISTRY}
    require(
        set(contract) == expected,
        f"candidate frame coverage mismatch: missing={sorted(expected - set(contract))}, "
        f"extra={sorted(set(contract) - expected)}",
    )
    return dict(sorted(contract.items()))


def _oracle_surface_temperature_c(root: Path) -> np.ndarray:
    """Load the phase-1 oracle's resolved `sst_m` diagnostic."""

    with netCDF4.Dataset(root / "output.init_ice.nc") as ds:
        value: np.ndarray = np.asarray(ds["sst"][0]).T
    require(value.shape == (59, 59), f"oracle sst shape {value.shape} != (59, 59)")
    require(bool(np.all(np.isfinite(value))), "oracle sst contains nonfinite values")
    return value


def geometry_gate(
    root: Path,
    card,
    rows: list[dict],
    dtypes: dict[str, dict[str, str]],
    *,
    plant_geometry: bool,
) -> None:
    grid = card.grid
    with netCDF4.Dataset(root / "mesh_mask.nc") as ds:
        lon_u = np.asarray(grid.lon_T) + 0.5 * card.dx_m / grid.radius
        lat_v = np.asarray(grid.lat_T) + 0.5 * card.dy_m / grid.radius
        horizontal = {
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
        for name, lego in horizontal.items():
            _score(
                rows,
                dtypes,
                f"geometry.{name}",
                np.asarray(ds[name][0]),
                lego,
                plant=plant_geometry and name == "e1t",
            )

        wet_yx = np.asarray(card.wet_global_xy).T
        umask = wet_yx & np.roll(wet_yx, -1, axis=1)
        umask[:, -1] = False
        vmask = wet_yx & np.roll(wet_yx, -1, axis=0)
        vmask[-1] = False
        fmask = umask & np.roll(umask, -1, axis=0)
        fmask[-1] = False
        for name, lego in {
            "tmask": wet_yx,
            "umask": umask,
            "vmask": vmask,
            "fmask": fmask,
        }.items():
            _score(
                rows,
                dtypes,
                f"geometry.{name}",
                np.asarray(ds[name][0, 0]).astype(bool),
                lego,
                exact=True,
            )


def _entry_field(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name.startswith("e_s_l"):
        result = oracle_gate._interior(arrays["e_s"])[
            ..., int(name[-2:]) - 1, 0
        ]
        return cast(np.ndarray, np.asarray(result))
    if name.startswith("e_i_l"):
        result = oracle_gate._interior(arrays["e_i"])[
            ..., int(name[-2:]) - 1, 0
        ]
        return cast(np.ndarray, np.asarray(result))
    return cast(np.ndarray, np.asarray(oracle_gate._interior(arrays[name])[..., 0]))


def _restart_field(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    result: np.ndarray = np.asarray(ds[name][0, 0]).T
    return result


def _restart_2d_field(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    result: np.ndarray = np.asarray(ds[name][0]).T
    return result


def state_field(card, state, name: str) -> np.ndarray:
    from legoesm.ice.fidelity.nemo_testcase_recipe import ICE_ADV1D_TRACERS

    halo = card.halo_width
    area = card.dx_m * card.dy_m
    index = ICE_ADV1D_TRACERS.index(name)
    result: np.ndarray = (
        np.asarray(state.contents)[halo:-halo, halo:-halo, index] / area
    )
    return result


def _moment_restart_name(moment: str, tracer: str) -> str | None:
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
    }.get(tracer)
    return None if base is None else moment + base


def comparison_gate(
    root: Path,
    card,
    rows: list[dict],
    dtypes: dict[str, dict[str, str]],
    *,
    plant_state: bool,
) -> tuple[object, dict]:
    from legoesm.ice.fidelity.nemo_testcase_recipe import (
        ICE_ADV1D_TRACERS,
        step_ice_adv1d_card,
    )
    from legoesm.ice.transport import SI3_PRATHER_MOMENT_NAMES

    state = card.initial_state
    first_divergence = None
    _, entry = oracle_gate.read_frame(root / "oracle_ice_step_entry_kt00000001.bin")
    for name in ICE_ADV1D_TRACERS:
        row = _score(
            rows,
            dtypes,
            f"entry.kt00000001.{name}",
            _entry_field(entry, name),
            state_field(card, state, name),
            plant=plant_state and name == "v_i",
        )
        if row["status"] == "DEBT" and first_divergence is None:
            first_divergence = row
    halo = card.halo_width
    row = _score(
        rows,
        dtypes,
        "entry.kt00000001.t_surface",
        _entry_field(entry, "t_su"),
        np.asarray(state.t_surface)[halo:-halo, halo:-halo],
    )
    if row["status"] == "DEBT" and first_divergence is None:
        first_divergence = row
    row = _score(
        rows,
        dtypes,
        "entry.kt00000001.u_ice",
        oracle_gate._interior(entry["u_ice"]),
        np.asarray(state.u_ice)[halo:-halo, halo:-halo],
    )
    if row["status"] == "DEBT" and first_divergence is None:
        first_divergence = row
    row = _score(
        rows,
        dtypes,
        "entry.kt00000001.v_ice",
        oracle_gate._interior(entry["v_ice"]),
        np.asarray(state.v_ice)[halo:-halo, halo:-halo],
    )
    if row["status"] == "DEBT" and first_divergence is None:
        first_divergence = row

    compared_trajectory = tuple(name for name in ICE_ADV1D_TRACERS if name != "sv_i")
    for completed_step in range(1, card.n_steps):
        state = step_ice_adv1d_card(card, state)
        kt = completed_step + 1
        _, entry = oracle_gate.read_frame(
            root / f"oracle_ice_step_entry_kt{kt:08d}.bin"
        )
        for name in compared_trajectory:
            row = _score(
                rows,
                dtypes,
                f"trajectory.post_step_{completed_step:08d}.{name}",
                _entry_field(entry, name),
                state_field(card, state, name),
            )
            if row["status"] == "DEBT" and first_divergence is None:
                first_divergence = row
        row = _score(
            rows,
            dtypes,
            f"trajectory.post_step_{completed_step:08d}.t_surface",
            _entry_field(entry, "t_su"),
            np.asarray(state.t_surface)[halo:-halo, halo:-halo],
        )
        if row["status"] == "DEBT" and first_divergence is None:
            first_divergence = row
        for name in ("u_ice", "v_ice"):
            row = _score(
                rows,
                dtypes,
                f"trajectory.post_step_{completed_step:08d}.{name}",
                oracle_gate._interior(entry[name]),
                np.asarray(getattr(state, name))[halo:-halo, halo:-halo],
            )
            if row["status"] == "DEBT" and first_divergence is None:
                first_divergence = row

    state = step_ice_adv1d_card(card, state)
    restart = oracle_gate.run_files(root)["restart"]
    with netCDF4.Dataset(restart) as ds:
        for name in compared_trajectory:
            row = _score(
                rows,
                dtypes,
                f"trajectory.post_step_{card.n_steps:08d}.{name}",
                _restart_field(ds, name),
                state_field(card, state, name),
            )
            if row["status"] == "DEBT" and first_divergence is None:
                first_divergence = row
        row = _score(
            rows,
            dtypes,
            f"trajectory.post_step_{card.n_steps:08d}.t_surface",
            _restart_field(ds, "t_su"),
            np.asarray(state.t_surface)[halo:-halo, halo:-halo],
        )
        if row["status"] == "DEBT" and first_divergence is None:
            first_divergence = row
        for name in ("u_ice", "v_ice"):
            row = _score(
                rows,
                dtypes,
                f"trajectory.post_step_{card.n_steps:08d}.{name}",
                _restart_2d_field(ds, name),
                np.asarray(getattr(state, name))[halo:-halo, halo:-halo],
            )
            if row["status"] == "DEBT" and first_divergence is None:
                first_divergence = row
        for moment_index, moment in enumerate(SI3_PRATHER_MOMENT_NAMES):
            for tracer_index, tracer in enumerate(ICE_ADV1D_TRACERS):
                restart_name = _moment_restart_name(moment, tracer)
                if restart_name is None:
                    continue
                row = _score(
                    rows,
                    dtypes,
                    f"restart_moment.{restart_name}",
                    _restart_field(ds, restart_name),
                    np.asarray(state.moments[moment_index])[
                        halo:-halo, halo:-halo, tracer_index
                    ],
                )
                if row["status"] == "DEBT" and first_divergence is None:
                    first_divergence = row
    return state, {
        "status": "AT-BAR" if first_divergence is None else "DEBT",
        "first_divergence": first_divergence,
        "compared_step_entries": card.n_steps,
        "compared_core_fields": list(compared_trajectory)
        + ["t_surface", "u_ice", "v_ice"],
    }


def restart_carry_control(
    card,
    *,
    plant_moment: bool = False,
    drop_moment: bool = False,
    retype_moment: bool = False,
) -> dict:
    """Bitwise complete-state split-continuation and moment controls."""
    from legoesm.ice.fidelity.nemo_testcase_recipe import (
        load_ice_adv1d_restart,
        save_ice_adv1d_restart,
        step_ice_adv1d_card,
    )

    split = card.initial_state
    for _ in range(7):
        split = step_ice_adv1d_card(card, split)
    with tempfile.TemporaryDirectory(prefix="si3-prather-restart-") as directory:
        restart = Path(directory) / "state.npz"
        save_ice_adv1d_restart(restart, card, split, completed_steps=7)
        if plant_moment or drop_moment or retype_moment:
            with np.load(restart, allow_pickle=False) as archive:
                payload = {name: archive[name].copy() for name in archive.files}
            if plant_moment:
                payload["moment_0"][20, 20, 0] += 1.0
            if drop_moment:
                payload.pop("moment_4")
            if retype_moment:
                payload["moment_0"] = payload["moment_0"].astype(np.float32)
            np.savez(restart, **payload)  # type: ignore[arg-type]
        try:
            restored, completed_steps = load_ice_adv1d_restart(restart, card)
        except ValueError as exc:
            raise GateError(str(exc)) from exc
    require(completed_steps == 7, f"restart clock {completed_steps} != 7")
    uninterrupted = step_ice_adv1d_card(card, split)
    resumed = step_ice_adv1d_card(card, restored)
    equal = all(
        np.array_equal(
            np.asarray(getattr(uninterrupted, name)),
            np.asarray(getattr(resumed, name)),
        )
        for name in ("contents", "u_ice", "v_ice", "t_surface")
    ) and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(uninterrupted.moments, resumed.moments, strict=True)
    )
    require(equal, "prather moment carry changed split continuation")
    return {
        "status": "VERIFIED",
        "state_arrays": 9,
        "moment_arrays": 5,
        "split_after_steps": 7,
    }


def inactive_source_arm_gate(card) -> dict[str, dict]:
    """Measure the two executed-but-inactive cleanup arms for this rung."""
    import jax.numpy as jnp
    from legoesm.ice.fidelity.nemo_testcase_recipe import (
        ICE_ADV1D_TRACERS,
        apply_ice_adv1d_zapsmall,
    )
    from legoesm.ice.transport import (
        _si3_prather_x_substep,
        advect_si3_prather_1d,
    )

    state = card.initial_state
    area = jnp.full(state.u_ice.shape, card.dx_m * card.dy_m, jnp.float64)
    wet = jnp.pad(card.wet_global_xy, card.halo_width)
    min_pre_zapneg = float("inf")
    max_zapneg_change = 0.0
    max_snow = 0.0

    def zapneg_potential_change(contents) -> float:
        vi = contents[..., ICE_ADV1D_TRACERS.index("v_i")]
        vs = contents[..., ICE_ADV1D_TRACERS.index("v_s")]
        ai = contents[..., ICE_ADV1D_TRACERS.index("a_i")]
        age = contents[..., ICE_ADV1D_TRACERS.index("oa_i")]
        salt = contents[..., ICE_ADV1D_TRACERS.index("sv_i")]
        ai_after_volume = jnp.where(vi <= 0.0, 0.0, ai)
        candidates = [
            jnp.where(vi <= 0.0, ai, 0.0),
            jnp.where(
                (salt < 0.0) | (ai_after_volume <= 0.0) | (vi <= 0.0),
                salt,
                0.0,
            ),
            jnp.where((vi < 0.0) | (ai_after_volume <= 0.0), vi, 0.0),
            jnp.where((vs < 0.0) | (ai_after_volume <= 0.0), vs, 0.0),
            jnp.where(age < 0.0, age, 0.0),
        ]
        for name in ("e_i_l01", "e_i_l02", "e_i_l03"):
            value = contents[..., ICE_ADV1D_TRACERS.index(name)]
            candidates.append(
                jnp.where(
                    (value < 0.0) | (ai_after_volume <= 0.0) | (vi <= 0.0),
                    value,
                    0.0,
                )
            )
        for name in ("e_s_l01", "e_s_l02", "e_s_l03"):
            value = contents[..., ICE_ADV1D_TRACERS.index(name)]
            candidates.append(
                jnp.where(
                    (value < 0.0) | (ai_after_volume <= 0.0) | (vs <= 0.0),
                    value,
                    0.0,
                )
            )
        return float(jnp.max(jnp.abs(jnp.stack(candidates))))

    for _ in range(card.n_steps):
        after_first, _, _ = _si3_prather_x_substep(
            state.contents,
            state.moments,
            card.prescribed_u_ice * card.dy_m,
            area,
            wet,
            card.dt_s / 2,
            initial_area=area,
            first_sweep=True,
            halo_width=card.halo_width,
            subcycle_index=1,
            subcycles=2,
        )
        contents, moments, _ = advect_si3_prather_1d(
            state.contents,
            state.moments,
            card.prescribed_u_ice,
            state.v_ice,
            area,
            wet,
            card.dt_s,
            dx=card.dx_m,
            dy=card.dy_m,
            halo_width=card.halo_width,
            ice_volume_index=ICE_ADV1D_TRACERS.index("v_i"),
            concentration_index=ICE_ADV1D_TRACERS.index("a_i"),
            subcycles=2,
        )
        min_pre_zapneg = min(
            min_pre_zapneg,
            float(jnp.min(after_first)),
            float(jnp.min(contents)),
        )
        max_zapneg_change = max(
            max_zapneg_change,
            zapneg_potential_change(after_first),
            zapneg_potential_change(contents),
        )
        snow_indices = [
            ICE_ADV1D_TRACERS.index(name)
            for name in ("v_s", "e_s_l01", "e_s_l02", "e_s_l03")
        ]
        max_snow = max(max_snow, float(jnp.max(jnp.abs(contents[..., snow_indices]))))
        state = apply_ice_adv1d_zapsmall(card, state, contents, moments)
    require(min_pre_zapneg >= 0.0, f"pre-zapneg content minimum {min_pre_zapneg}")
    require(max_zapneg_change == 0.0, f"zapneg potential change {max_zapneg_change}")
    require(max_snow == 0.0, f"snow content maximum {max_snow}")
    return {
        "Hsnow_pra": {
            "status": "WAIVED-INACTIVE",
            "reason": "zero snow makes icedyn_adv_pra.F90:1117-1129 non-binding",
            "measured_max_abs_snow_content": max_snow,
        },
        "ice_var_zapneg": {
            "status": "WAIVED-INACTIVE",
            "reason": "icevar.F90:759-837 would change no candidate field",
            "measured_min_content": min_pre_zapneg,
            "measured_max_potential_state_change": max_zapneg_change,
        },
    }


def run_gate(
    root: Path,
    *,
    plant_unaccounted: bool = False,
    plant_unaccounted_frame: bool = False,
    plant_geometry: bool = False,
    plant_state: bool = False,
    plant_moment: bool = False,
    drop_moment: bool = False,
    retype_moment: bool = False,
) -> tuple[dict, int]:
    from legoesm.core.precision import PrecisionPolicy, get_policy
    from legoesm.ice.fidelity.nemo_testcase_recipe import (
        build_ice_adv1d_card,
        validate_ice_adv1d_card,
    )

    require(jax.default_backend() == "cpu", f"CPU required, got {jax.default_backend()}")
    counts = _mesh_coverage(root, plant_unaccounted)
    frame_contract = candidate_frame_contract(
        plant_unaccounted=plant_unaccounted_frame
    )
    card = build_ice_adv1d_card(_oracle_surface_temperature_c(root))
    validate_ice_adv1d_card(card)
    source_arm_contract = inactive_source_arm_gate(card)
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    rows: list[dict] = []
    dtypes: dict[str, dict[str, str]] = {}
    geometry_gate(root, card, rows, dtypes, plant_geometry=plant_geometry)
    _, trajectory = comparison_gate(
        root, card, rows, dtypes, plant_state=plant_state
    )
    restart_control = restart_carry_control(
        card,
        plant_moment=plant_moment,
        drop_moment=drop_moment,
        retype_moment=retype_moment,
    )
    debt = [row for row in rows if row["status"] == "DEBT"]
    oracle_files = [
        root / "mesh_mask.nc",
        root / "output.init_ice.nc",
        oracle_gate.run_files(root)["restart"],
    ]
    oracle_files.extend(
        root / f"oracle_ice_step_entry_kt{kt:08d}.bin"
        for kt in range(1, card.n_steps + 1)
    )
    report = {
        "status": "AT-BAR" if not debt else "DEBT",
        "scope": "ICE_ADV1D_OMIP_L3_LEGOESM_PHASE2",
        "backend": jax.default_backend(),
        "precision_policy": "fp64",
        "pointwise_bar": POINTWISE_BAR,
        "provenance": {
            "oracle_root": str(root),
            "oracle_input_sha256": {
                path.name: _sha256(path) for path in oracle_files
            },
            "manifest": str(MANIFEST),
            "manifest_sha256": _sha256(MANIFEST),
            "phase1_gate": str(PHASE1),
            "phase1_gate_sha256": _sha256(PHASE1),
            "git_parent_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "implementation_sha256": {
                "transport.py": _sha256(
                    REPO_ROOT / "packages/ice/legoesm/ice/transport.py"
                ),
                "nemo_testcase_recipe.py": _sha256(
                    REPO_ROOT
                    / "packages/ice/legoesm/ice/fidelity/nemo_testcase_recipe.py"
                ),
                "nemo_si3_phase2_gate.py": _sha256(Path(__file__).resolve()),
            },
        },
        "inventory_counts": counts,
        "candidate_frame_contract": frame_contract,
        "source_arm_contract": source_arm_contract,
        "rows": rows,
        "dtypes": dtypes,
        "trajectory": trajectory,
        "restart_carry_control": restart_control,
        "unmeasured": [
            "rung_3.2",
            "rung_3.3",
            "within_step_xy_split_states",
            "sv_i_trajectory_against_nn_icesal4_oracle",
            "option4_layer_salinity_moments",
            "pond_fields_and_moments",
            "thermodynamics",
            "rheology",
            "ridging_rafting",
            "landfast_L16_deferred_lane4",
            "coupled_ice_ocean_matching",
            "production_run_restart_integration_of_opt_in_card_state",
            "zapsmall_zapneg_separation",
            "snwice_mass_and_before_level_candidate_alignment",
        ],
    }
    return report, 0 if not debt else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-unaccounted", action="store_true")
    parser.add_argument("--plant-unaccounted-frame", action="store_true")
    parser.add_argument("--plant-geometry", action="store_true")
    parser.add_argument("--plant-state", action="store_true")
    parser.add_argument("--plant-moment", action="store_true")
    parser.add_argument("--drop-moment", action="store_true")
    parser.add_argument("--retype-moment", action="store_true")
    args = parser.parse_args()
    report, code = run_gate(
        args.run_dir.resolve(),
        plant_unaccounted=args.plant_unaccounted,
        plant_unaccounted_frame=args.plant_unaccounted_frame,
        plant_geometry=args.plant_geometry,
        plant_state=args.plant_state,
        plant_moment=args.plant_moment,
        drop_moment=args.drop_moment,
        retype_moment=args.retype_moment,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, oracle_gate.GateError) as exc:
        print(f"DEBT: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
