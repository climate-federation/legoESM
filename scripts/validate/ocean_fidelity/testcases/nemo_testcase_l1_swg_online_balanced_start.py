#!/usr/bin/env python3
"""Admit dg50 starts against same-operator truth, fast energy relative to truth.

The tool reports; it does not select the training objective (owner decision).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn

HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3_600.0
DEFAULT_START_DAYS = tuple(range(731, 1092, 40))
NO_JUMP_FACTOR = 2.0
# A start passes only if its no-closure forecast carries at most this many
# times the divergent (fast) KE of the same-operator coarse-grained truth:
# in the worst single hour of hours 1-6 and in the hours-25-48 mean.
FAST_ENERGY_TRUTH_FACTOR = 2.0
# Sea-level rebalancing preserves volume by construction; more than this
# fraction of the basin volume means the instrument, not the start, failed.
REBALANCE_VOLUME_FRACTION = 1.0e-12
WALL_RING_OPTIONS = ("weighted", "excluded")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fast_divergent_energy(
        u: np.ndarray, v: np.ndarray, mask: np.ndarray, spacing_m: float,
) -> float:
    """Basin-mean KE of the closed-wall divergent velocity component."""
    from legoesm.training.swg_online import closed_wall_divergent_part

    _, u_div, v_div = closed_wall_divergent_part(u, v, mask, spacing_m)
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


def state_admission(*, error, fast, truth_fast, ssh_tendency,
                    rebalance: dict | None = None,
                    volume_bound_m3: float | None = None) -> dict:
    """The registered per-start rule, judged against same-operator truth.

    A start is admitted only if (i) no jump: the hour-1 vorticity error is at
    most twice the hour-24 error over 24; (ii) its no-closure forecast's fast
    (closed-wall divergent) KE over the truth's is at most 2 in the worst
    hour of hours 1-6 AND for the hours-25-48 means; (iii) fail closed: the
    truth's fast KE is positive at every hour, and a rebalanced start
    converged with its volume unchanged to ``volume_bound_m3``.
    """
    series = {}
    for name, value in (("error", error), ("fast", fast),
                        ("truth_fast", truth_fast),
                        ("ssh_tendency", ssh_tendency)):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (48,) or not np.all(np.isfinite(array)) \
                or np.any(array < 0.0):
            raise ValueError(
                f"{name} admission requires 48 finite non-negative hourly values")
        series[name] = array
    error, fast = series["error"], series["fast"]
    truth_fast, ssh = series["truth_fast"], series["ssh_tendency"]
    failed = []
    truth_positive = bool(np.all(truth_fast > 0.0))
    if not truth_positive:
        failed.append("truth fast energy is not positive at every hour")
    early = (float(np.max(fast[:6] / truth_fast[:6]))
             if truth_positive else float("inf"))
    late = (float(np.mean(fast[24:48]) / np.mean(truth_fast[24:48]))
            if truth_positive else float("inf"))
    if not early <= FAST_ENERGY_TRUTH_FACTOR:
        failed.append("hours 1-6 fast energy exceeds twice the truth's")
    if not late <= FAST_ENERGY_TRUTH_FACTOR:
        failed.append("hours 25-48 mean fast energy exceeds twice the truth's")
    jump_limit = NO_JUMP_FACTOR * error[23] / HOURS_PER_DAY
    no_jump = bool(error[0] <= jump_limit)
    if not no_jump:
        failed.append("vorticity error jumps in hour 1")
    if rebalance is not None:
        if "failed" in rebalance:
            failed.append(f"rebalancing failed: {rebalance['failed']}")
        elif volume_bound_m3 is None \
                or not abs(rebalance["volume_change_m3"]) <= volume_bound_m3:
            failed.append("rebalancing changed the basin volume")
    return {
        "no_jump": no_jump,
        "zeta_error_hour1_s-1": float(error[0]),
        "zeta_error_hour24_s-1": float(error[23]),
        "zeta_error_hour1_limit_s-1": float(jump_limit),
        "fast_energy_worst_hour1_6_over_truth": early,
        "fast_energy_mean25_48_over_truth": late,
        "fast_energy_limit_over_truth": FAST_ENERGY_TRUTH_FACTOR,
        "fast_energy_max_hours1_6_m2_s-2": float(np.max(fast[:6])),
        "fast_energy_mean_hours25_48_m2_s-2": float(np.mean(fast[24:48])),
        "truth_fast_energy_max_hours1_6_m2_s-2": float(np.max(truth_fast[:6])),
        "truth_fast_energy_mean_hours25_48_m2_s-2": float(
            np.mean(truth_fast[24:48])),
        "ssh_tendency_rms_m_s": {
            str(hour): float(ssh[hour - 1]) for hour in (1, 6, 24, 48)
        },
        "failed": failed,
        "admitted": not failed,
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
        target, _, provenance = _load_hourly_target(
            hourly_root / f"start_day{int(record['start_day']):04d}",
            operator="symmetric")
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
            label: state_admission(
                error=record[f"{label}_relative_vorticity_error_rms_s-1"],
                fast=record[f"{label}_fast_divergent_ke_m2_s-2"],
                truth_fast=truth_fast,
                ssh_tendency=record[f"{label}_ssh_tendency_rms_m_s"])
            for label in ("conservative", "raw")
        }
    conservative = [
        record["admission"]["conservative"]["admitted"] for record in records
    ]
    raw = [record["admission"]["raw"]["admitted"] for record in records]
    report["admission_rule"] = _admission_rule()
    # Legacy diagnostic of STORED starts; training starts are rebalanced and
    # are admitted only by --dual-operator.
    report["admission_summary"] = {
        "conservative_pass_count": int(sum(conservative)),
        "raw_pass_count": int(sum(raw)),
        "conservative_stored_start_admitted": bool(all(conservative)),
        "raw_stored_start_admitted": bool(all(raw)),
    }
    return report


def _admission_rule() -> dict:
    return {
        "no_jump": "error_h1 <= 2 * error_h24 / 24",
        "fast_energy": (
            "max_h1_6(fast/truth_fast) <= 2 and "
            "mean_h25_48(fast)/mean_h25_48(truth_fast) <= 2, "
            "truth = same-operator coarse-grained truth"),
        "fail_closed": (
            "non-finite input, non-positive truth fast energy, failed "
            "rebalancing, or a rebalancing volume change above "
            f"{REBALANCE_VOLUME_FRACTION:g} of the basin volume rejects"),
        "per_start": True,
    }


def _load_coarse_state(state, forcing, path: Path):
    with np.load(path, allow_pickle=False) as saved:
        fields = [np.asarray(saved[name], dtype=np.float64)
                  for name in ("ssh_m", "u_m_s", "v_m_s")]
    return _coarse_state_from_fields(state, forcing, dict(zip(
        ("T", "U", "V"), fields, strict=True)))


def _coarse_state_from_fields(state, forcing, fields):
    """Build a dg50 state from one explicit native T/U/V mapping."""
    import jax.numpy as jnp
    from legoesm.training.swg_online import coarse_state_from_native

    values = [np.asarray(fields[kind], dtype=np.float64)
              for kind in ("T", "U", "V")]
    expected = tuple(state.eta.data.shape)
    if any(field.shape != expected for field in values):
        raise ValueError(f"coarse state shape differs from {expected}")
    if not all(np.all(np.isfinite(field)) for field in values):
        raise ValueError("coarse state is non-finite")
    return coarse_state_from_native(state, *values)._replace(
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


def _load_hourly_target(root: Path, *, operator: str,
                        ) -> tuple[dict[str, np.ndarray], dict | None, dict]:
    """First 48 hourly targets (and a v2 stream's t0) via the lane loader.

    ``load_content_pinned_hourly_segment`` checks every day file's digest,
    its 24 records, the absolute-hour lineage, finiteness and, for schema v2,
    the truth provenance.  The manifest is pinned to its present digest,
    which the report records.
    """
    from legoesm.training.swg_online import load_content_pinned_hourly_segment

    manifest_sha256 = sha256(root / "manifest.json")
    record = load_content_pinned_hourly_segment(
        root, expected_manifest_sha256=manifest_sha256, operator=operator)
    first = 1 if record.includes_initial_state else 0
    fields = dict(zip(("T", "U", "V"),
                      (record.ssh_m, record.u_m_s, record.v_m_s), strict=True))
    if fields["T"].shape[0] - first < 48:
        raise ValueError(f"hourly target segment holds under 48 hours: {root}")
    target = {kind: value[first:first + 48] for kind, value in fields.items()}
    initial = ({kind: value[0] for kind, value in fields.items()}
               if first else None)
    return target, initial, {
        "root": str(root.resolve()),
        "manifest_sha256": manifest_sha256,
        "operator": operator,
    }


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
        target, _, target_provenance = _load_hourly_target(
            hourly_root / f"start_day{start_day:04d}", operator="symmetric")
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


def _top_octave_variance_share(zeta: np.ndarray, wet: np.ndarray) -> float:
    jj, ii = np.where(wet)
    block = zeta[:, jj.min():jj.max() + 1, ii.min():ii.max() + 1]
    power = np.mean(np.abs(dstn(
        block, type=1, axes=(-2, -1), norm="ortho")) ** 2, axis=0)
    ky, kx = np.indices(power.shape)
    radius = np.sqrt((ky + 1.0) ** 2 + (kx + 1.0) ** 2)
    top = radius >= 0.5 * np.max(radius)
    return float(np.sum(power[top]) / np.sum(power))


def build_dual_context() -> dict:
    """Build the CPU fp64 dg50 model, stepper and rebalancer once."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        build_swg_card,
        swg_surface_wind,
    )
    from legoesm.ocean.fidelity.swg_coarse_graining import cgrid_geometry
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.training.swg_online import (
        START_REBALANCE_MAX_ITERATIONS,
        START_REBALANCE_TOLERANCE,
        build_closure_geometry,
        build_start_rebalancer,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if not bool(jax.config.jax_enable_x64) or jax.default_backend() != "cpu":
        raise RuntimeError("dual-operator admission requires CPU fp64")
    card = build_swg_card(SWG_DG50_NAMELIST)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(SWG_DG50_NAMELIST)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
        tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
    geometry = cgrid_geometry(SWG_DG50_NAMELIST)
    wet = np.asarray(geometry["mask"]["T"], dtype=bool)
    volume = float(np.sum(
        geometry["reference_thickness"]["T"][wet] * geometry["area"]["T"][wet]))
    return {
        "card": card,
        "forcing": forcing,
        "run_48h": _build_hourly_stepper(
            model, card.dt_s, forcing, int(SECONDS_PER_HOUR / card.dt_s)),
        "rebalance": build_start_rebalancer(
            card, tolerance=START_REBALANCE_TOLERANCE,
            max_iterations=START_REBALANCE_MAX_ITERATIONS),
        "rebalance_rule": {
            "tolerance": START_REBALANCE_TOLERANCE,
            "max_iterations": START_REBALANCE_MAX_ITERATIONS,
        },
        "closure_geometry": build_closure_geometry(card),
        "wet": wet,
        "spacing_m": float(geometry["spacing_m"]),
        "volume_bound_m3": REBALANCE_VOLUME_FRACTION * volume,
    }


def evaluate_operator_start(context: dict, truth: dict, initial: dict, *,
                            objective: str) -> dict:
    """Admit one stored and one rebalanced start against same-operator truth.

    ``truth`` and ``initial`` are native T/U/V mappings (48 hourly states and
    the stored t0).  Training starts from the rebalanced state, so its
    admission is the one that counts; the stored start is a diagnostic.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.training.swg_online import (
        deterministic_trajectory_loss,
        diagnose_native_trajectory,
    )

    card = context["card"]
    geometry = context["closure_geometry"]
    interior = np.asarray(geometry.vertex_interior, dtype=bool)
    wet = context["wet"]
    spacing = context["spacing_m"]
    target_native = tuple(np.asarray(truth[kind][:48], dtype=np.float64)
                          for kind in ("T", "U", "V"))
    if any(value.shape[0] != 48 for value in target_native):
        raise ValueError("dual-operator admission requires 48 truth hours")
    diagnosed_target = np.asarray(diagnose_native_trajectory(
        tuple(jnp.asarray(value) for value in target_native), geometry,
        objective=objective).relative_vorticity_s1)
    truth_fast = np.asarray([fast_divergent_energy(
        target_native[1][hour], target_native[2][hour], wet, spacing)
        for hour in range(48)])
    truth_rms = np.sqrt(np.mean(
        diagnosed_target[:, interior] ** 2, axis=1, dtype=np.float64))
    stored = {kind: np.asarray(initial[kind], dtype=np.float64)
              for kind in ("T", "U", "V")}

    def forecast_admission(start, rebalance):
        state = _coarse_state_from_fields(
            card.recipe.initial_state, context["forcing"], start)
        _, samples = context["run_48h"](state)
        jax.block_until_ready(samples[0])
        predicted = tuple(np.asarray(value, dtype=np.float64)
                          for value in samples)
        if not all(np.all(np.isfinite(value)) for value in predicted):
            raise ValueError("no-closure forecast is non-finite")
        zeta = np.asarray(diagnose_native_trajectory(
            tuple(jnp.asarray(value) for value in predicted), geometry,
            objective=objective).relative_vorticity_s1)
        error = np.sqrt(np.mean(
            (zeta - diagnosed_target)[:, interior] ** 2, axis=1,
            dtype=np.float64))
        fast = np.asarray([fast_divergent_energy(
            predicted[1][hour], predicted[2][hour], wet, spacing)
            for hour in range(48)])
        prior = np.concatenate((start["T"][None], predicted[0][:-1]))
        ssh_tendency = np.sqrt(np.mean(
            ((predicted[0] - prior) / SECONDS_PER_HOUR)[:, wet] ** 2,
            axis=1, dtype=np.float64))
        return state_admission(
            error=error, fast=fast, truth_fast=truth_fast,
            ssh_tendency=ssh_tendency, rebalance=rebalance,
            volume_bound_m3=context["volume_bound_m3"])

    try:
        balanced_ssh, rebalance = context["rebalance"](
            stored["T"], stored["U"], stored["V"])
        change = (balanced_ssh - stored["T"])[wet]
        rebalance |= {
            "ssh_change_rms_m": float(np.sqrt(np.mean(change**2))),
            "ssh_change_max_abs_m": float(np.max(np.abs(change))),
        }
        admission = forecast_admission(
            stored | {"T": balanced_ssh}, rebalance)
    except ValueError as error:
        rebalance = {"failed": str(error)}
        admission = {"admitted": False, "failed": [str(error)]}
    checkerboard = 0.1 * np.where(np.indices(wet.shape).sum(axis=0) % 2,
                                  -1.0, 1.0)
    planted = (target_native[0], target_native[1] + checkerboard,
               target_native[2])
    return {
        "objective": objective,
        "truth_fast_divergent_ke_m2_s-2": {
            "max_hours1_6": float(np.max(truth_fast[:6])),
            "mean_hours25_48": float(np.mean(truth_fast[24:48])),
        },
        "truth_relative_vorticity_rms_s-1": {
            str(hour): float(truth_rms[hour - 1]) for hour in (1, 6, 24, 48)
        },
        "rebalance": rebalance,
        "admission": admission,
        "stored_start_admission": forecast_admission(stored, None),
        "target_top_octave_vorticity_variance_share": (
            _top_octave_variance_share(diagnosed_target, wet)),
        "checkerboard_0.1m_s_eq7_loss": {
            ring: float(deterministic_trajectory_loss(
                tuple(jnp.asarray(value)[None] for value in planted),
                tuple(jnp.asarray(value)[None] for value in target_native),
                geometry, objective=objective,
                beta_m1_s1=card.resolved_namelist.rn_beta_m_s,
                basin_length_m=card.resolved_namelist.rn_domsiz_m,
                wall_ring=ring))
            for ring in WALL_RING_OPTIONS
        },
    }


def summarize_dual(records: list[dict]) -> dict:
    """Count admissions per operator; never select the objective."""
    counts = {
        key: {label: sum(bool(item["operators"][label][key]["admitted"])
                         for item in records) for label in ("box", "filtered")}
        for key in ("admission", "stored_start_admission")
    }
    return {
        "rebalanced_pass_count": counts["admission"],
        "stored_start_pass_count": counts["stored_start_admission"],
        "starts": len(records),
        "all_rebalanced_starts_admitted": all(
            item["operators"][label]["admission"]["admitted"]
            for item in records for label in ("box", "filtered")),
        "objective_selection": "owner decision; this tool selects nothing",
    }


def run_dual_operator_experiment(hourly_root: Path,
                                 start_days: tuple[int, ...]) -> dict:
    """Admit rebalanced box/filtered starts against same-operator truth."""
    if len(start_days) != 10 or len(set(start_days)) != 10:
        raise ValueError("dual-operator admission requires ten unique starts")
    context = build_dual_context()
    records = []
    for start_day in start_days:
        record = {"start_day": start_day, "operators": {}}
        for label, objective in (("box", "symmetric"),
                                 ("filtered", "template")):
            target, initial, provenance = _load_hourly_target(
                hourly_root / f"start_day{start_day:04d}",
                operator=objective)
            if initial is None:
                raise ValueError("dual-operator segment lacks its stored t0 state")
            record["operators"][label] = evaluate_operator_start(
                context, target, initial, objective=objective) | {
                    "target": provenance}
        records.append(record)
    return {
        "schema": "swg-online-balanced-start-dual-v3",
        "start_days": list(start_days),
        "records": records,
        "admission_rule": _admission_rule() | {
            "start": "stored t0 with sea level rebalanced",
            "rebalance": context["rebalance_rule"],
        },
        "summary": summarize_dual(records),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--conservative-root", type=Path)
    parser.add_argument("--hourly-root", type=Path, required=True)
    parser.add_argument("--existing-json", type=Path)
    parser.add_argument("--dual-operator", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--start-days", type=int, nargs="+",
                        default=DEFAULT_START_DAYS)
    args = parser.parse_args(argv)
    try:
        if args.dual_operator:
            if args.existing_json is not None or args.raw_root is not None \
                    or args.conservative_root is not None:
                raise ValueError(
                    "dual-operator mode excludes legacy roots and existing JSON")
            report = run_dual_operator_experiment(
                args.hourly_root, tuple(args.start_days))
        elif args.existing_json is None:
            if args.raw_root is None or args.conservative_root is None:
                raise ValueError("raw and conservative roots are required")
            report = run_experiment(
                args.raw_root, args.conservative_root, args.hourly_root,
                tuple(args.start_days))
        else:
            if args.raw_root is not None or args.conservative_root is not None:
                raise ValueError("existing-json cannot be combined with run roots")
            report = json.loads(args.existing_json.read_text())
        if not args.dual_operator:
            report = apply_registered_admission(report, args.hourly_root)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    if args.dual_operator:
        summary = report["summary"]
        passed = summary["all_rebalanced_starts_admitted"]
        rebalanced = summary["rebalanced_pass_count"]
        stored = summary["stored_start_pass_count"]
        print(
            f"{'PASS' if passed else 'FAIL'} rebalanced "
            f"box={rebalanced['box']}/{summary['starts']} "
            f"filtered={rebalanced['filtered']}/{summary['starts']} "
            f"(stored box={stored['box']} filtered={stored['filtered']}); "
            f"objective={summary['objective_selection']}")
        return 0 if passed else 1
    summary = report["admission_summary"]
    print(
        f"{'PASS' if summary['conservative_stored_start_admitted'] else 'FAIL'} "
        f"legacy stored starts: conservative="
        f"{summary['conservative_pass_count']}/10 "
        f"raw={summary['raw_pass_count']}/10 (training starts are rebalanced; "
        "admit them with --dual-operator)")
    return 0 if summary["conservative_stored_start_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
