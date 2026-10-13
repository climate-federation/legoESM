#!/usr/bin/env python3
"""Measure raw-versus-conservative dg50 initialization shock at 6/24/48 h."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3_600.0
DEFAULT_START_DAYS = tuple(range(731, 1092, 40))
NO_JUMP_FACTOR = 2.0
FAST_ENERGY_TRANSIENT_FACTOR = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wet_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    wet = np.asarray(mask, dtype=bool)
    jj, ii = np.where(wet)
    if wet.ndim != 2 or jj.size == 0:
        raise ValueError("projection mask must be a nonempty 2-D rectangle")
    bounds = int(jj.min()), int(jj.max()), int(ii.min()), int(ii.max())
    expected = np.zeros_like(wet)
    expected[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1] = True
    if not np.array_equal(wet, expected):
        raise ValueError("projection mask is not one rectangle")
    return bounds


def divergent_velocity_projection(
        u: np.ndarray, v: np.ndarray, mask: np.ndarray, spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closed-wall discrete gradient carrying input divergence."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    wet = np.asarray(mask, dtype=bool)
    if u.shape != wet.shape or v.shape != wet.shape:
        raise ValueError("native U/V and T mask shapes differ")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("projection spacing must be positive finite")
    j0, j1, i0, i1 = _wet_rectangle(wet)
    divergence = (
        u - np.roll(u, 1, axis=1)
        + v - np.roll(v, 1, axis=0)
    ) / spacing_m
    rhs = divergence[j0:j1 + 1, i0:i1 + 1]
    rhs = rhs - np.mean(rhs, dtype=np.float64)
    ny, nx = rhs.shape
    ky = np.arange(ny, dtype=np.float64)
    kx = np.arange(nx, dtype=np.float64)
    eigenvalue = (
        2.0 * (np.cos(np.pi * ky / ny) - 1.0)[:, None]
        + 2.0 * (np.cos(np.pi * kx / nx) - 1.0)[None, :]
    ) / spacing_m ** 2
    rhs_hat = dctn(rhs, type=2, norm="ortho")
    phi_hat = np.zeros_like(rhs_hat)
    phi_hat[1:, :] = rhs_hat[1:, :] / eigenvalue[1:, :]
    phi_hat[0, 1:] = rhs_hat[0, 1:] / eigenvalue[0, 1:]
    phi = idctn(phi_hat, type=2, norm="ortho")

    u_div = np.zeros_like(u)
    v_div = np.zeros_like(v)
    u_div[j0:j1 + 1, i0:i1] = np.diff(phi, axis=1) / spacing_m
    v_div[j0:j1, i0:i1 + 1] = np.diff(phi, axis=0) / spacing_m
    return u_div, v_div


def fast_divergent_energy(
        u: np.ndarray, v: np.ndarray, mask: np.ndarray, spacing_m: float,
) -> float:
    """Basin-mean KE of the closed-wall divergent velocity component."""
    u_div, v_div = divergent_velocity_projection(u, v, mask, spacing_m)
    wet = np.asarray(mask, dtype=bool)
    u_center = 0.5 * (u_div + np.roll(u_div, 1, axis=1))
    v_center = 0.5 * (v_div + np.roll(v_div, 1, axis=0))
    return float(np.mean(
        0.5 * (u_center[wet] ** 2 + v_center[wet] ** 2),
        dtype=np.float64))


def initialization_ratios(
        raw_zeta: np.ndarray, conservative_zeta: np.ndarray,
        target_zeta: np.ndarray, wet: np.ndarray,
        leads: tuple[int, ...] = (6, 24, 48),
) -> dict[str, dict[str, float]]:
    """Shock/closure ratios: ||raw-cons|| / ||cons-target|| at lead hours."""
    raw = np.asarray(raw_zeta, dtype=np.float64)
    conservative = np.asarray(conservative_zeta, dtype=np.float64)
    target = np.asarray(target_zeta, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    if raw.shape != conservative.shape or raw.shape != target.shape:
        raise ValueError("vorticity trajectory shapes differ")
    if raw.ndim != 3 or wet.shape != raw.shape[1:]:
        raise ValueError("vorticity trajectories must be (hour, y, x)")
    result = {}
    for lead in leads:
        if lead < 1 or lead > raw.shape[0]:
            raise ValueError(f"lead hour {lead} is unavailable")
        shock = float(np.sqrt(np.mean(
            (raw[lead - 1][wet] - conservative[lead - 1][wet]) ** 2,
            dtype=np.float64)))
        closure = float(np.sqrt(np.mean(
            (conservative[lead - 1][wet] - target[lead - 1][wet]) ** 2,
            dtype=np.float64)))
        result[str(lead)] = {
            "initialization_shock_vorticity_rms_s-1": shock,
            "genuine_closure_vorticity_rms_s-1": closure,
            "shock_to_closure_ratio": (
                float(shock / closure) if closure > 0.0 else float("inf")),
        }
    return result


def _state_admission(record: dict, label: str) -> dict:
    """Apply the pre-registered per-start gate to one initialized state."""
    error = np.asarray(
        record[f"{label}_relative_vorticity_error_rms_s-1"],
        dtype=np.float64)
    fast = np.asarray(
        record[f"{label}_fast_divergent_ke_m2_s-2"], dtype=np.float64)
    ssh = np.asarray(record[f"{label}_ssh_tendency_rms_m_s"], dtype=np.float64)
    if error.shape != (48,) or fast.shape != (48,) or ssh.shape != (48,) \
            or not all(np.all(np.isfinite(x)) for x in (error, fast, ssh)):
        raise ValueError(f"{label} admission requires 48 finite hourly values")
    jump_limit = NO_JUMP_FACTOR * error[23] / HOURS_PER_DAY
    late_fast = float(np.mean(fast[24:48], dtype=np.float64))
    fast_limit = FAST_ENERGY_TRANSIENT_FACTOR * late_fast
    no_jump = bool(error[0] <= jump_limit)
    fast_bounded = bool(np.max(fast[:6]) <= fast_limit)
    return {
        "no_jump": no_jump,
        "zeta_error_hour1_s-1": float(error[0]),
        "zeta_error_hour24_s-1": float(error[23]),
        "zeta_error_hour1_limit_s-1": float(jump_limit),
        "fast_energy_bounded": fast_bounded,
        "fast_energy_max_hours1_6_m2_s-2": float(np.max(fast[:6])),
        "fast_energy_mean_hours25_48_m2_s-2": late_fast,
        "fast_energy_limit_m2_s-2": fast_limit,
        "ssh_tendency_rms_m_s": {
            str(hour): float(ssh[hour - 1]) for hour in (1, 6, 24, 48)
        },
        "admitted": bool(no_jump and fast_bounded),
    }


def apply_registered_admission(report: dict, hourly_root: Path) -> dict:
    """Augment a completed ten-start report with fail-closed admissions."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        restricted_relative_vorticity,
    )

    records = report.get("records", [])
    if len(records) != 10:
        raise ValueError("balanced admission requires exactly ten start records")
    geometry = cgrid_geometry(SWG_DG50_NAMELIST)
    wet = np.asarray(geometry["mask"]["T"], dtype=bool)
    spacing = float(geometry["spacing_m"])
    for record in records:
        target, provenance = _load_hourly_target(
            hourly_root / f"start_day{int(record['start_day']):04d}")
        target_zeta = np.stack([
            restricted_relative_vorticity(
                {kind: target[kind][hour] for kind in ("T", "U", "V")},
                geometry)
            for hour in range(48)
        ])
        truth_fast = np.asarray([
            fast_divergent_energy(
                target["U"][hour], target["V"][hour], wet, spacing)
            for hour in range(48)
        ])
        truth_rms = np.sqrt(np.mean(
            target_zeta[:, wet] ** 2, axis=1, dtype=np.float64))
        record["hourly_target"] = provenance
        record["truth_relative_vorticity_rms_s-1"] = {
            str(hour): float(truth_rms[hour - 1]) for hour in (1, 6, 24, 48)
        }
        record["truth_fast_divergent_ke_m2_s-2"] = {
            "max_hours1_6": float(np.max(truth_fast[:6])),
            "mean_hours25_48": float(np.mean(truth_fast[24:48])),
        }
        record["admission"] = {
            label: _state_admission(record, label)
            for label in ("conservative", "raw")
        }
    conservative = [
        record["admission"]["conservative"]["admitted"] for record in records
    ]
    raw = [record["admission"]["raw"]["admitted"] for record in records]
    report["admission_rule"] = {
        "pre_registered_manifest": (
            "manifests/nemo_testcases_l1_swg_online_round4_preregister.json"),
        "no_jump": "error_h1 <= 2 * error_h24 / 24",
        "fast_energy": "max_h1_6 <= 2 * mean_h25_48",
        "per_start": True,
        "all_ten_required": True,
    }
    report["admission_summary"] = {
        "conservative_pass_count": int(sum(conservative)),
        "raw_pass_count": int(sum(raw)),
        "conservative_admitted_for_training": bool(all(conservative)),
        "raw_admitted_for_training": bool(all(raw)),
    }
    return report


def _load_coarse_state(state, forcing, path: Path):
    import jax.numpy as jnp

    with np.load(path, allow_pickle=False) as saved:
        fields = [np.asarray(saved[name], dtype=np.float64)
                  for name in ("ssh_m", "u_m_s", "v_m_s")]
    expected = tuple(state.eta.data.shape)
    if any(field.shape != expected for field in fields):
        raise ValueError(f"{path}: coarse state shape differs from {expected}")
    if not all(np.all(np.isfinite(field)) for field in fields):
        raise ValueError(f"{path}: coarse state is non-finite")
    u = np.zeros(state.u.data.shape, dtype=np.float64)
    v = np.zeros(state.v.data.shape, dtype=np.float64)
    u[:, 1:, 0], v[1:, :, 0] = fields[1], fields[2]
    return state._replace(
        eta=state.eta.replace(data=jnp.asarray(fields[0])),
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        tau_x_prev=jnp.asarray(forcing.tau_i_native, dtype=jnp.float64),
        tau_y_prev=jnp.asarray(forcing.tau_j_native, dtype=jnp.float64),
    )


def _build_hourly_stepper(model, dt_s: float, forcing, steps_per_hour: int):
    import jax

    def step(state, _):
        return model.step(state, dt_s, surface_forcing=forcing), None

    def hour(state, _):
        state, _ = jax.lax.scan(step, state, None, length=steps_per_hour)
        fields = (state.eta.data, state.u.data[:, 1:, 0],
                  state.v.data[1:, :, 0])
        return state, fields

    @jax.jit
    def run_48h(state):
        return jax.lax.scan(hour, state, None, length=48)

    return run_48h


def _load_hourly_target(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "legoesm-swg-double-gyre-hourly-target-v1" \
            or manifest.get("status") != "complete" \
            or manifest.get("completed_days", 0) < 2:
        raise ValueError(f"incomplete hourly target segment: {root}")
    if set(manifest.get("available_sampling", ())) != {
            "hourly_conservative_coarse_target"}:
        raise ValueError("hourly segment stores an unexpected sampling stream")
    stacked = {kind: [] for kind in ("T", "U", "V")}
    keys = {"T": "ssh_hourly_m", "U": "u_hourly_m_s",
            "V": "v_hourly_m_s"}
    for day in (1, 2):
        path = root / f"day{day:04d}.npz"
        if sha256(path) != manifest.get("daily_sha256", {}).get(path.name):
            raise ValueError(f"hourly target digest differs: {path}")
        with np.load(path, allow_pickle=False) as saved:
            if any(name in saved.files for name in ("ssh_m", "u_m_s", "v_m_s")):
                raise ValueError("hourly target leaks native fine fields")
            for kind, key in keys.items():
                stacked[kind].append(np.asarray(saved[key], dtype=np.float64))
    return ({kind: np.concatenate(values, axis=0)
             for kind, values in stacked.items()}, {
        "root": str(root.resolve()),
        "manifest_sha256": sha256(manifest_path),
    })


def run_experiment(raw_root: Path, conservative_root: Path,
                   hourly_root: Path, start_days: tuple[int, ...]) -> dict:
    """Run paired 48-hour dg50 forecasts and score initialization shock."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        build_swg_card,
        swg_surface_wind,
    )
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        restricted_relative_vorticity,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    if len(start_days) != 10 or len(set(start_days)) != 10:
        raise ValueError("balanced-start experiment requires ten unique starts")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if not bool(jax.config.jax_enable_x64) or jax.default_backend() != "cpu":
        raise RuntimeError("balanced-start experiment requires CPU fp64")
    card = build_swg_card(SWG_DG50_NAMELIST)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(SWG_DG50_NAMELIST)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64))
    steps_per_hour = int(SECONDS_PER_HOUR / card.dt_s)
    if steps_per_hour * card.dt_s != SECONDS_PER_HOUR:
        raise ValueError("dg50 timestep does not divide one hour")
    run_48h = _build_hourly_stepper(
        model, card.dt_s, forcing, steps_per_hour)
    geometry = cgrid_geometry(SWG_DG50_NAMELIST)
    wet = np.asarray(geometry["mask"]["T"], dtype=bool)
    records = []
    input_hashes = {}
    for start_day in start_days:
        paths = {
            "raw": raw_root / f"day{start_day:04d}.npz",
            "conservative": conservative_root / f"day{start_day:04d}.npz",
        }
        target, target_provenance = _load_hourly_target(
            hourly_root / f"start_day{start_day:04d}")
        trajectories = {}
        for label, path in paths.items():
            input_hashes[str(path.resolve())] = sha256(path)
            initial = _load_coarse_state(
                card.recipe.initial_state, forcing, path)
            _, samples = run_48h(initial)
            jax.block_until_ready(samples[0])
            trajectories[label] = {
                kind: np.asarray(value, dtype=np.float64)
                for kind, value in zip(("T", "U", "V"), samples, strict=True)
            }
        zeta = {}
        divergent_energy = {}
        ssh_tendency = {}
        for label, trajectory in trajectories.items():
            zeta[label] = np.stack([
                restricted_relative_vorticity(
                    {kind: trajectory[kind][hour] for kind in ("T", "U", "V")},
                    geometry)
                for hour in range(48)
            ])
            divergent_energy[label] = [
                fast_divergent_energy(
                    trajectory["U"][hour], trajectory["V"][hour], wet,
                    float(geometry["spacing_m"]))
                for hour in range(48)
            ]
            with np.load(paths[label], allow_pickle=False) as saved:
                initial_ssh = np.asarray(
                    saved["ssh_m"], dtype=np.float64).copy()
            prior = np.concatenate((initial_ssh[None], trajectory["T"][:-1]))
            ssh_tendency[label] = [
                float(np.sqrt(np.mean(
                    ((trajectory["T"][hour] - prior[hour])
                     / SECONDS_PER_HOUR)[wet] ** 2, dtype=np.float64)))
                for hour in range(48)
            ]
        target_zeta = np.stack([
            restricted_relative_vorticity(
                {kind: target[kind][hour] for kind in ("T", "U", "V")},
                geometry)
            for hour in range(48)
        ])
        zeta_error = {
            label: [
                float(np.sqrt(np.mean(
                    (values[hour][wet] - target_zeta[hour][wet]) ** 2,
                    dtype=np.float64)))
                for hour in range(48)
            ]
            for label, values in zeta.items()
        }
        records.append({
            "start_day": start_day,
            "hourly_target": target_provenance,
            "ratios": initialization_ratios(
                zeta["raw"], zeta["conservative"], target_zeta, wet),
            "raw_fast_divergent_ke_m2_s-2": divergent_energy["raw"],
            "conservative_fast_divergent_ke_m2_s-2": (
                divergent_energy["conservative"]),
            "raw_relative_vorticity_error_rms_s-1": zeta_error["raw"],
            "conservative_relative_vorticity_error_rms_s-1": (
                zeta_error["conservative"]),
            "raw_ssh_tendency_rms_m_s": ssh_tendency["raw"],
            "conservative_ssh_tendency_rms_m_s": ssh_tendency["conservative"],
        })
    summary = {}
    for lead in (6, 24, 48):
        values = np.asarray([
            item["ratios"][str(lead)]["shock_to_closure_ratio"]
            for item in records], dtype=np.float64)
        summary[str(lead)] = {
            "median_shock_to_closure_ratio": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "schema": "swg-online-balanced-start-v1",
        "definition": (
            "shock=RMS(zeta_raw-zeta_conservative); "
            "closure=RMS(zeta_conservative-zeta_filtered_fine)"),
        "start_days": list(start_days),
        "input_sha256": input_hashes,
        "records": records,
        "summary": summary,
        "adjusted_start": "not constructed",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--conservative-root", type=Path)
    parser.add_argument("--hourly-root", type=Path, required=True)
    parser.add_argument("--existing-json", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--start-days", type=int, nargs="+",
                        default=DEFAULT_START_DAYS)
    args = parser.parse_args(argv)
    try:
        if args.existing_json is None:
            if args.raw_root is None or args.conservative_root is None:
                raise ValueError("raw and conservative roots are required")
            report = run_experiment(
                args.raw_root, args.conservative_root, args.hourly_root,
                tuple(args.start_days))
        else:
            if args.raw_root is not None or args.conservative_root is not None:
                raise ValueError("existing-json cannot be combined with run roots")
            report = json.loads(args.existing_json.read_text())
        report = apply_registered_admission(report, args.hourly_root)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    summary = report["admission_summary"]
    print(
        f"{'PASS' if summary['conservative_admitted_for_training'] else 'FAIL'} "
        f"conservative={summary['conservative_pass_count']}/10 "
        f"raw={summary['raw_pass_count']}/10")
    return 0 if summary["conservative_admitted_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
