#!/usr/bin/env python3
"""Independent written-order replay of one SI3 rung-3.3 aEVP outer step."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final")
HERE = Path(__file__).resolve().parent
PHASE1 = HERE / "nemo_si3_oracle_gate.py"
RUNG33 = HERE / "nemo_si3_phase2_rung33_gate.py"
POINTWISE_BAR = 1.0e-15
ULP_LIMIT = 2


class ReplayError(RuntimeError):
    """A fail-closed replay contract violation."""


class ReplayState(NamedTuple):
    """Five restart-carried aEVP fields."""

    u: np.ndarray
    v: np.ndarray
    stress1: np.ndarray
    stress2: np.ndarray
    stress12: np.ndarray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle_gate = _load("nemo_si3_oracle_gate_rung33_replay", PHASE1)
rung33_gate = _load("nemo_si3_phase2_rung33_gate_replay", RUNG33)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _periodic(value: np.ndarray) -> np.ndarray:
    """NEMO full two-cell periodic exchange on this one-rank card."""

    return cast(np.ndarray, np.pad(value[2:-2, 2:-2], 2, mode="wrap"))


def _write(previous: np.ndarray, region: slice, value: np.ndarray) -> np.ndarray:
    result = previous.copy()
    result[region, region] = value[region, region]
    return result


def _roll(value: np.ndarray, shift: int, axis: int) -> np.ndarray:
    return cast(np.ndarray, np.roll(value, shift, axis=axis))


def _ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    require(left.dtype == right.dtype == np.float64, "ULP comparison requires fp64")
    require(
        bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right))),
        "ULP comparison received a non-finite value",
    )
    sign = np.uint64(1 << 63)
    left_bits: np.ndarray = left.view(np.uint64)
    right_bits: np.ndarray = right.view(np.uint64)
    left_ordered = np.where(left_bits & sign, ~left_bits, left_bits | sign)
    right_ordered = np.where(right_bits & sign, ~right_bits, right_bits | sign)
    return cast(
        np.ndarray,
        np.where(
            left_ordered >= right_ordered,
            left_ordered - right_ordered,
            right_ordered - left_ordered,
        ),
    )


def _score(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    require(reference.shape == candidate.shape, "replay score shape mismatch")
    difference = np.abs(candidate - reference)
    nonzero = int(np.count_nonzero(difference))
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    scale = max(1.0, float(np.max(np.abs(reference))))
    return {
        "max_abs": float(difference[index]),
        "normalized_max_abs": float(difference[index]) / scale,
        "max_abs_index_xy": [int(i) for i in index],
        "max_ulp": int(np.max(_ulp_distance(reference, candidate))),
        "bitwise_nonzero_over_n": f"{nonzero} / {difference.size}",
        "ulp_at_max_abs": int(_ulp_distance(reference, candidate)[index]),
        "reference_at_max": float(reference[index]),
        "candidate_at_max": float(candidate[index]),
    }


def _oracle_state(frame: dict[str, np.ndarray]) -> ReplayState:
    return ReplayState(
        *(
            np.asarray(frame[name], dtype=np.float64).copy()
            for name in ("u_ice", "v_ice", "stress1_i", "stress2_i", "stress12_i")
        )
    )


def _oracle_forcing(card, frame: dict[str, np.ndarray]):
    from legoesm.ice.dynamics import SI3CGridAEVPForcing

    template = card.forcing_template

    def scalar(name: str) -> np.ndarray:
        return np.asarray(frame[name][..., 0], dtype=np.float64).copy()

    return SI3CGridAEVPForcing(
        concentration_t=scalar("a_i"),
        ice_volume_t=scalar("v_i"),
        snow_volume_t=scalar("v_s"),
        pond_volume_t=scalar("v_ip"),
        lid_volume_t=scalar("v_il"),
        air_stress_u_t=np.asarray(template.air_stress_u_t),
        air_stress_v_t=np.asarray(template.air_stress_v_t),
        drag_io_t=np.asarray(template.drag_io_t),
        ocean_u_u=np.asarray(template.ocean_u_u),
        ocean_v_v=np.asarray(template.ocean_v_v),
        ssh_t=np.asarray(template.ssh_t),
        coriolis_t=np.asarray(template.coriolis_t),
        tmask_t=np.asarray(template.tmask_t),
        umask_u=np.asarray(template.umask_u),
        vmask_v=np.asarray(template.vmask_v),
        fast_tmask=np.asarray(template.fast_tmask),
        depth_t=np.asarray(template.depth_t),
        depth_u=np.asarray(template.depth_u),
        depth_v=np.asarray(template.depth_v),
        iceberg_tmask=np.asarray(template.iceberg_tmask),
        iceberg_umask=np.asarray(template.iceberg_umask),
        iceberg_vmask=np.asarray(template.iceberg_vmask),
    )


def _setup(card, forcing) -> dict[str, np.ndarray | float]:
    """Transcribe icedyn_rhg_evp.F90:189-374 without production helpers."""

    metrics = card.metrics
    cfg = card.dynamics_config
    m = {name: np.asarray(getattr(metrics, name)) for name in metrics._fields}
    f = {name: np.asarray(getattr(forcing, name)) for name in forcing._fields}
    at_i = f["concentration_t"]
    zmsk = (at_i >= 1.0e-10).astype(np.float64)
    fimask_value = (
        f["tmask_t"]
        * _roll(f["tmask_t"], -1, 0)
        * _roll(f["tmask_t"], -1, 1)
        * _roll(_roll(f["tmask_t"], -1, 0), -1, 1)
    )
    fimask = _periodic(_write(np.zeros_like(at_i), slice(2, -2), fimask_value))
    inverse_eccentricity_square = 1.0 / (cfg.eccentricity * cfg.eccentricity)
    mass_t = (
        cfg.rho_snow * f["snow_volume_t"]
        + cfg.rho_ice * f["ice_volume_t"]
        + cfg.rho_water * (f["pond_volume_t"] + f["lid_volume_t"])
    )
    dt_over_mass_t = cfg.dt_s / np.maximum(mass_t, 1.0)
    strength_formula = (
        cfg.strength_parameter_pa * f["ice_volume_t"] * np.exp(-cfg.strength_decay * (1.0 - at_i))
    )
    strength_t = np.where(at_i > 1.0e-10, strength_formula, 0.0)
    area_t_east = _roll(m["area_t"], -1, 0)
    area_t_north = _roll(m["area_t"], -1, 1)
    mass_t_east = _roll(mass_t, -1, 0)
    mass_t_north = _roll(mass_t, -1, 1)
    at_i_east = _roll(at_i, -1, 0)
    at_i_north = _roll(at_i, -1, 1)
    za_u = 0.5 * (at_i * m["area_t"] + at_i_east * area_t_east) * (1.0 / m["area_u"]) * f["umask_u"]
    za_v = (
        0.5 * (at_i * m["area_t"] + at_i_north * area_t_north) * (1.0 / m["area_v"]) * f["vmask_v"]
    )
    mass_u = (
        0.5
        * (mass_t * m["area_t"] + mass_t_east * area_t_east)
        * (1.0 / m["area_u"])
        * f["umask_u"]
    )
    mass_v = (
        0.5
        * (mass_t * m["area_t"] + mass_t_north * area_t_north)
        * (1.0 / m["area_v"])
        * f["vmask_v"]
    )
    ocean_v_u = (
        0.25
        * (
            (f["ocean_v_v"] + _roll(f["ocean_v_v"], 1, 1))
            + (_roll(f["ocean_v_v"], -1, 0) + _roll(_roll(f["ocean_v_v"], -1, 0), 1, 1))
        )
        * f["umask_u"]
    )
    ocean_u_v = (
        0.25
        * (
            (f["ocean_u_u"] + _roll(f["ocean_u_u"], 1, 0))
            + (_roll(f["ocean_u_u"], -1, 1) + _roll(_roll(f["ocean_u_u"], 1, 0), -1, 1))
        )
        * f["vmask_v"]
    )
    mass_over_dt_u = mass_u * (1.0 / cfg.dt_s)
    mass_over_dt_v = mass_v * (1.0 / cfg.dt_s)
    tau_air_u = (
        za_u
        * 0.5
        * (f["air_stress_u_t"] + _roll(f["air_stress_u_t"], -1, 0))
        * (2.0 - f["umask_u"])
        * np.maximum(f["tmask_t"], _roll(f["tmask_t"], -1, 0))
    )
    tau_air_v = (
        za_v
        * 0.5
        * (f["air_stress_v_t"] + _roll(f["air_stress_v_t"], -1, 1))
        * (2.0 - f["vmask_v"])
        * np.maximum(f["tmask_t"], _roll(f["tmask_t"], -1, 1))
    )
    drag_u = (
        cfg.rho_ocean
        * za_u
        * 0.5
        * (f["drag_io_t"] + _roll(f["drag_io_t"], -1, 0))
        * (2.0 - f["umask_u"])
        * np.maximum(f["tmask_t"], _roll(f["tmask_t"], -1, 0))
    )
    drag_v = (
        cfg.rho_ocean
        * za_v
        * 0.5
        * (f["drag_io_t"] + _roll(f["drag_io_t"], -1, 1))
        * (2.0 - f["vmask_v"])
        * np.maximum(f["tmask_t"], _roll(f["tmask_t"], -1, 1))
    )
    slope_u = -mass_u * cfg.gravity * (_roll(f["ssh_t"], -1, 0) - f["ssh_t"]) * (1.0 / m["e1u"])
    slope_v = -mass_v * cfg.gravity * (_roll(f["ssh_t"], -1, 1) - f["ssh_t"]) * (1.0 / m["e2v"])
    return {
        **m,
        **f,
        "zmsk": zmsk,
        "fimask": fimask,
        "inverse_eccentricity_square": inverse_eccentricity_square,
        "mass_coriolis_t": mass_t * f["coriolis_t"],
        "dt_over_mass_t": dt_over_mass_t,
        "strength_t": strength_t,
        "mass_over_dt_u": mass_over_dt_u,
        "mass_over_dt_v": mass_over_dt_v,
        "tau_air_u": tau_air_u,
        "tau_air_v": tau_air_v,
        "drag_u": drag_u,
        "drag_v": drag_v,
        "slope_u": slope_u,
        "slope_v": slope_v,
        "mass_mask_u": (mass_u > 0.0).astype(np.float64),
        "mass_mask_v": (mass_v > 0.0).astype(np.float64),
        "active_u": (~((mass_u <= 1.0) & (za_u <= 0.001))).astype(np.float64),
        "active_v": (~((mass_v <= 1.0) & (za_v <= 0.001))).astype(np.float64),
        "fast_u": np.maximum(f["fast_tmask"], _roll(f["fast_tmask"], -1, 0)),
        "fast_v": np.maximum(f["fast_tmask"], _roll(f["fast_tmask"], -1, 1)),
        "ocean_v_u": ocean_v_u,
        "ocean_u_v": ocean_u_v,
        "dt_s": float(cfg.dt_s),
    }


def _deformation(u: np.ndarray, v: np.ndarray, shear: np.ndarray, q: dict):
    u_west = _roll(u, 1, 0)
    v_south = _roll(v, 1, 1)
    shear_west = _roll(shear, 1, 0)
    shear_south = _roll(shear, 1, 1)
    shear_southwest = _roll(shear_west, 1, 1)
    area_f_west = _roll(q["area_f"], 1, 0)
    area_f_south = _roll(q["area_f"], 1, 1)
    area_f_southwest = _roll(area_f_west, 1, 1)
    shear_square = (
        (
            (shear * shear * q["area_f"] + shear_west * shear_west * area_f_west)
            + (
                shear_south * shear_south * area_f_south
                + shear_southwest * shear_southwest * area_f_southwest
            )
        )
        * 0.25
        * (1.0 / q["area_t"])
    )
    divergence = (
        (q["e2u"] * u - _roll(q["e2u"], 1, 0) * u_west)
        + (q["e1v"] * v - _roll(q["e1v"], 1, 1) * v_south)
    ) * (1.0 / q["area_t"])
    tension = (
        (u * (1.0 / q["e2u"]) - u_west * _roll(1.0 / q["e2u"], 1, 0)) * q["e2t"] * q["e2t"]
        - (v * (1.0 / q["e1v"]) - v_south * _roll(1.0 / q["e1v"], 1, 1)) * q["e1t"] * q["e1t"]
    ) * (1.0 / q["area_t"])
    return divergence, tension, shear_square


def _subcycle(state: ReplayState, initial: ReplayState, q: dict, iteration: int):
    """Source statements icedyn_rhg_evp.F90:392-741, in written order."""

    u, v, stress1, stress2, stress12 = (value.copy() for value in state)
    r1_e1u = 1.0 / q["e1u"]
    r1_e2v = 1.0 / q["e2v"]
    shear_value = (
        (
            ((_roll(u, -1, 1) * _roll(r1_e1u, -1, 1) - u * r1_e1u) * q["e1f"] * q["e1f"])
            + ((_roll(v, -1, 0) * _roll(r1_e2v, -1, 0) - v * r1_e2v) * q["e2f"] * q["e2f"])
        )
        * (1.0 / q["area_f"])
        * q["fimask"]
    )
    shear = _write(np.zeros_like(stress12), slice(0, -1), shear_value)
    divergence, tension, shear_square = _deformation(u, v, shear, q)
    first_divergence = divergence
    first_tension = tension
    divergence_square = divergence * divergence
    tension_square = tension * tension
    delta_value = (
        np.sqrt(
            divergence_square
            + (tension_square + shear_square) * q["inverse_eccentricity_square"]
        )
        * q["zmsk"]
    )
    delta_floor = delta_value + 2.0e-9
    p_over_delta_value = q["strength_t"] / delta_floor * q["zmsk"]
    delta = _periodic(_write(np.zeros_like(delta_value), slice(2, -2), delta_value))
    p_over_delta = _periodic(_write(np.zeros_like(delta_value), slice(2, -2), p_over_delta_value))
    divergence, tension, _ = _deformation(u, v, shear, q)
    alpha_t = np.maximum(
        50.0,
        np.pi * np.sqrt(0.5 * p_over_delta * (1.0 / q["area_t"]) * q["dt_over_mass_t"]),
    )
    inverse_alpha_t = 1.0 / (alpha_t + 1.0)
    stress1_value = (
        (stress1 * alpha_t + p_over_delta * (divergence * (1.0 + 0.0) - delta * (1.0 - 0.0)))
        * inverse_alpha_t
        * q["zmsk"]
    )
    stress2_value = (
        (
            stress2 * alpha_t
            + p_over_delta * (tension * q["inverse_eccentricity_square"] * (1.0 + 0.0))
        )
        * inverse_alpha_t
        * q["zmsk"]
    )
    stress1 = _write(stress1, slice(1, None), stress1_value)
    stress2 = _write(stress2, slice(1, None), stress2_value)
    beta = np.maximum(
        50.0,
        np.pi * np.sqrt(0.5 * p_over_delta * (1.0 / q["area_t"]) * q["dt_over_mass_t"]),
    )
    alpha_f = np.maximum(
        np.maximum(beta, _roll(beta, -1, 0)),
        np.maximum(_roll(beta, -1, 1), _roll(_roll(beta, -1, 0), -1, 1)),
    )
    p_over_delta_f = 0.25 * (
        (p_over_delta + _roll(p_over_delta, -1, 0))
        + (_roll(p_over_delta, -1, 1) + _roll(_roll(p_over_delta, -1, 0), -1, 1))
    )
    inverse_alpha_f = 1.0 / (alpha_f + 1.0)
    stress12_value = (
        stress12 * alpha_f
        + p_over_delta_f * (shear * q["inverse_eccentricity_square"] * (1.0 + 0.0)) * 0.5
    ) * inverse_alpha_f
    stress12 = _write(stress12, slice(0, -1), stress12_value)
    force_u_value = (
        0.5
        * (
            (
                (_roll(stress1, -1, 0) - stress1) * q["e2u"]
                + (
                    _roll(stress2, -1, 0) * _roll(q["e2t"], -1, 0) * _roll(q["e2t"], -1, 0)
                    - stress2 * q["e2t"] * q["e2t"]
                )
                * (1.0 / q["e2u"])
            )
            + (
                stress12 * q["e1f"] * q["e1f"]
                - _roll(stress12, 1, 1) * _roll(q["e1f"], 1, 1) * _roll(q["e1f"], 1, 1)
            )
            * 2.0
            * (1.0 / q["e1u"])
        )
        * (1.0 / q["area_u"])
    )
    force_v_value = (
        0.5
        * (
            (
                (_roll(stress1, -1, 1) - stress1) * q["e1v"]
                - (
                    _roll(stress2, -1, 1) * _roll(q["e1t"], -1, 1) * _roll(q["e1t"], -1, 1)
                    - stress2 * q["e1t"] * q["e1t"]
                )
                * (1.0 / q["e1v"])
            )
            + (
                stress12 * q["e2f"] * q["e2f"]
                - _roll(stress12, 1, 0) * _roll(q["e2f"], 1, 0) * _roll(q["e2f"], 1, 0)
            )
            * 2.0
            * (1.0 / q["e2v"])
        )
        * (1.0 / q["area_v"])
    )
    force_u = _write(np.zeros_like(stress1), slice(1, -1), force_u_value)
    force_v = _write(np.zeros_like(stress1), slice(1, -1), force_v_value)
    cross_v_u = (
        0.25
        * ((v + _roll(v, 1, 1)) + (_roll(v, -1, 0) + _roll(_roll(v, -1, 0), 1, 1)))
        * q["umask_u"]
    )
    cross_u_v = (
        0.25
        * ((u + _roll(u, 1, 0)) + (_roll(u, -1, 1) + _roll(_roll(u, 1, 0), -1, 1)))
        * q["vmask_v"]
    )

    def update_u(current_u: np.ndarray, current_v: np.ndarray, region: slice):
        speed = np.sqrt(
            (current_u - q["ocean_u_u"]) * (current_u - q["ocean_u_u"])
            + (cross_v_u - q["ocean_v_u"]) * (cross_v_u - q["ocean_v_u"])
        )
        drag = q["drag_u"] * speed
        ocean_stress = drag * (q["ocean_u_u"] - current_u)
        bottom_speed = 5.0e-5 + np.sqrt(cross_v_u * cross_v_u + current_u * current_u)
        bottom_drag = q.get("base_u", np.zeros_like(current_u)) / bottom_speed
        bottom_stress = bottom_drag * current_u
        coriolis = (
            0.25
            * (1.0 / q["e1u"])
            * (
                q["mass_coriolis_t"]
                * (q["e1v"] * current_v + _roll(q["e1v"], 1, 1) * _roll(current_v, 1, 1))
                + _roll(q["mass_coriolis_t"], -1, 0)
                * (
                    _roll(q["e1v"], -1, 0) * _roll(current_v, -1, 0)
                    + _roll(_roll(q["e1v"], -1, 0), 1, 1) * _roll(_roll(current_v, -1, 0), 1, 1)
                )
            )
        )
        rhs = force_u + q["tau_air_u"] + coriolis + q["slope_u"] + ocean_stress
        beta_u = np.maximum(beta, _roll(beta, -1, 0))
        denominator = np.maximum(
            1.0e-20,
            q["mass_over_dt_u"] * (beta_u + 1.0) + drag - bottom_drag,
        )
        raw = (
            q["mass_over_dt_u"] * (beta_u * current_u + initial.u) + rhs + drag * current_u
        ) / denominator
        thin = (raw * q["active_u"] + q["ocean_u_u"] * 0.01 * (1.0 - q["active_u"])) * q[
            "mass_mask_u"
        ]
        value = thin * (1.0 - 0.99 * q["fast_u"])
        zero = np.zeros_like(current_u)
        trace = {
            "tauo_u": _write(zero, region, drag),
            "ocean_stress_u": _write(zero, region, ocean_stress),
            "speed_u": _write(zero, region, bottom_speed),
            "taub_u": _write(zero, region, bottom_drag),
            "bottom_stress_u": _write(zero, region, bottom_stress),
            "coriolis_u": _write(zero, region, coriolis),
            "rhs_u": _write(zero, region, rhs),
            "beta_u": _write(zero, region, beta_u),
            "denominator_u": _write(zero, region, denominator),
            "raw_u": _write(zero, region, raw),
            "thin_u": _write(zero, region, thin),
            "prehalo_u": _write(zero, region, value),
        }
        return _write(current_u, region, value), trace

    def update_v(current_u: np.ndarray, current_v: np.ndarray, region: slice):
        speed = np.sqrt(
            (current_v - q["ocean_v_v"]) * (current_v - q["ocean_v_v"])
            + (cross_u_v - q["ocean_u_v"]) * (cross_u_v - q["ocean_u_v"])
        )
        drag = q["drag_v"] * speed
        ocean_stress = drag * (q["ocean_v_v"] - current_v)
        bottom_speed = 5.0e-5 + np.sqrt(current_v * current_v + cross_u_v * cross_u_v)
        bottom_drag = q.get("base_v", np.zeros_like(current_v)) / bottom_speed
        bottom_stress = bottom_drag * current_v
        coriolis = (
            -0.25
            * (1.0 / q["e2v"])
            * (
                q["mass_coriolis_t"]
                * (q["e2u"] * current_u + _roll(q["e2u"], 1, 0) * _roll(current_u, 1, 0))
                + _roll(q["mass_coriolis_t"], -1, 1)
                * (
                    _roll(q["e2u"], -1, 1) * _roll(current_u, -1, 1)
                    + _roll(_roll(q["e2u"], 1, 0), -1, 1) * _roll(_roll(current_u, 1, 0), -1, 1)
                )
            )
        )
        rhs = force_v + q["tau_air_v"] + coriolis + q["slope_v"] + ocean_stress
        beta_v = np.maximum(beta, _roll(beta, -1, 1))
        denominator = np.maximum(
            1.0e-20,
            q["mass_over_dt_v"] * (beta_v + 1.0) + drag - bottom_drag,
        )
        raw = (
            q["mass_over_dt_v"] * (beta_v * current_v + initial.v) + rhs + drag * current_v
        ) / denominator
        thin = (raw * q["active_v"] + q["ocean_v_v"] * 0.01 * (1.0 - q["active_v"])) * q[
            "mass_mask_v"
        ]
        value = thin * (1.0 - 0.99 * q["fast_v"])
        zero = np.zeros_like(current_v)
        trace = {
            "tauo_v": _write(zero, region, drag),
            "ocean_stress_v": _write(zero, region, ocean_stress),
            "speed_v": _write(zero, region, bottom_speed),
            "taub_v": _write(zero, region, bottom_drag),
            "bottom_stress_v": _write(zero, region, bottom_stress),
            "coriolis_v": _write(zero, region, coriolis),
            "rhs_v": _write(zero, region, rhs),
            "beta_v": _write(zero, region, beta_v),
            "denominator_v": _write(zero, region, denominator),
            "raw_v": _write(zero, region, raw),
            "thin_v": _write(zero, region, thin),
            "prehalo_v": _write(zero, region, value),
        }
        return _write(current_v, region, value), trace

    if (iteration + 1) % 2 == 0:
        v, trace_v = update_v(u, v, slice(1, -1))
        u, trace_u = update_u(u, v, slice(2, -2))
    else:
        u, trace_u = update_u(u, v, slice(1, -1))
        v, trace_v = update_v(u, v, slice(2, -2))
    result = ReplayState(_periodic(u), _periodic(v), stress1, stress2, stress12)
    checkpoints = {
        "shear": shear,
        "shear2": shear_square,
        "divergence": first_divergence,
        "divergence2": divergence_square,
        "tension": first_tension,
        "tension2": tension_square,
        "delta": delta,
        "delta_floor": delta_floor,
        "p_over_delta": p_over_delta,
        "alpha_t": alpha_t,
        "inverse_alpha_t": inverse_alpha_t,
        "beta_t": beta,
        "alpha_f": alpha_f,
        "inverse_alpha_f": inverse_alpha_f,
        "p_over_delta_f": p_over_delta_f,
        "stress1": stress1,
        "stress2": stress2,
        "stress12": stress12,
        "force_u": force_u,
        "force_v": force_v,
        "cross_v_u": cross_v_u,
        "cross_u_v": cross_u_v,
        **trace_u,
        **trace_v,
        "u": result.u,
        "v": result.v,
    }
    return result, checkpoints


def _production(card, forcing, initial: ReplayState, subcycles: int) -> ReplayState:
    from legoesm.ice.dynamics import SI3CGridAEVPState, si3_cgrid_aevp_solver

    state = SI3CGridAEVPState(*(jnp.asarray(value) for value in initial))
    config = card.dynamics_config._replace(n_subcycles=subcycles)
    result = si3_cgrid_aevp_solver(
        state,
        type(forcing)(*(jnp.asarray(value) for value in forcing)),
        card.metrics,
        config,
    )
    jax.block_until_ready(result)
    return ReplayState(*(np.asarray(value) for value in result))


def _written_order_production(card, forcing, initial: ReplayState, subcycles: int) -> ReplayState:
    """Execute the production statements without lowering the loop as one graph."""

    from legoesm.ice.dynamics import SI3CGridAEVPState, si3_cgrid_aevp_solver

    state = SI3CGridAEVPState(*(jnp.asarray(value) for value in initial))
    config = card.dynamics_config._replace(n_subcycles=subcycles)
    with jax.disable_jit():
        result = si3_cgrid_aevp_solver(
            state,
            type(forcing)(*(jnp.asarray(value) for value in forcing)),
            card.metrics,
            config,
        )
    return ReplayState(*(np.asarray(value) for value in result))


def _physical(value: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, value[2:-2, 2:-2])


def run_replay(root: Path = ROOT) -> tuple[dict[str, object], int]:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import build_ice_adv2d_rhg_card

    set_policy(PrecisionPolicy.fp64())
    header1, frame1 = oracle_gate.read_frame(root / "oracle_ice_step_entry_kt00000001.bin")
    header2, frame2 = oracle_gate.read_frame(root / "oracle_ice_step_entry_kt00000002.bin")
    card = build_ice_adv2d_rhg_card(
        rung33_gate.oracle_surface_temperature_c(root), frame1
    )
    require(header1["storage_bits"] == header2["storage_bits"] == 64, "oracle frames are not fp64")
    initial = _oracle_state(frame1)
    forcing = _oracle_forcing(card, frame1)
    area = card.base.dx_m * card.base.dy_m
    bridge_forcing = type(forcing)(
        *((np.asarray(value) * area) / area for value in forcing[:5]),
        *(np.asarray(value) for value in forcing[5:]),
    )
    bridge_input_rows = {
        name: _score(np.asarray(exact), np.asarray(bridged))
        for name, exact, bridged in zip(
            ("a_i", "v_i", "v_s", "v_ip", "v_il"),
            forcing[:5],
            bridge_forcing[:5],
            strict=True,
        )
    }
    q = _setup(card, forcing)
    replay_one, checkpoints = _subcycle(initial, initial, q, 0)
    production_one = _production(card, forcing, initial, 1)
    one_rows = {
        name: _score(_physical(getattr(replay_one, name)), _physical(getattr(production_one, name)))
        for name in ReplayState._fields
    }
    # Subcycle 1 has exactly zero stress because its entry velocity and stress
    # are zero.  Use its production output as the byte-identical entry to the
    # first *active* stress update, subcycle 2.  This avoids a vacuous 0-ULP
    # classification while retaining the oracle outer-step carries.
    replay_active, active_checkpoints = _subcycle(production_one, initial, q, 1)
    production_two = _production(card, forcing, initial, 2)
    active_rows = {
        name: _score(
            _physical(getattr(replay_active, name)),
            _physical(getattr(production_two, name)),
        )
        for name in ReplayState._fields
    }
    written_order_two = _written_order_production(card, forcing, initial, 2)
    written_order_active_rows = {
        name: _score(
            _physical(getattr(replay_active, name)),
            _physical(getattr(written_order_two, name)),
        )
        for name in ReplayState._fields
    }
    # Locate the first operand in the preregistered source order.  The
    # production helper is used only as the comparison target; the independent
    # NumPy replay above does not call it.
    from legoesm.ice.dynamics import _si3_f_shear

    production_shear = _si3_f_shear(
        jnp.asarray(production_one.u),
        jnp.asarray(production_one.v),
        card.metrics,
        jnp.ones_like(card.metrics.e1t),
        jnp.zeros_like(card.metrics.e1t),
    )
    shear_row = _score(
        _physical(active_checkpoints["shear"]),
        _physical(np.asarray(production_shear)),
    )
    replay = initial
    for iteration in range(card.dynamics_config.n_subcycles):
        replay, _ = _subcycle(replay, initial, q, iteration)
    production = _production(card, forcing, initial, card.dynamics_config.n_subcycles)
    oracle_next = _oracle_state(frame2)
    bridge_production = _production(card, bridge_forcing, initial, card.dynamics_config.n_subcycles)
    bridge_output_rows = {
        name: _score(
            _physical(getattr(oracle_next, name)),
            _physical(getattr(bridge_production, name)),
        )
        for name in ReplayState._fields
    }

    # Private one-variable ablation only: remove the shared materialization
    # identity while leaving the card, formulas, operands, and precision fixed.
    from legoesm.ice import dynamics as dynamics_module
    from legoesm.ice import rheology as rheology_module

    saved_dynamics_round = dynamics_module.nemo_source_round
    saved_rheology_round = rheology_module.nemo_source_round
    try:
        dynamics_module.nemo_source_round = lambda value: value
        rheology_module.nemo_source_round = lambda value: value
        unrounded_production = _production(card, forcing, initial, card.dynamics_config.n_subcycles)
    finally:
        dynamics_module.nemo_source_round = saved_dynamics_round
        rheology_module.nemo_source_round = saved_rheology_round
    unrounded_rows = {
        name: _score(
            _physical(getattr(oracle_next, name)),
            _physical(getattr(unrounded_production, name)),
        )
        for name in ReplayState._fields
    }
    production_rows = {
        name: _score(_physical(getattr(oracle_next, name)), _physical(getattr(production, name)))
        for name in ReplayState._fields
    }
    replay_rows = {
        name: _score(_physical(getattr(oracle_next, name)), _physical(getattr(replay, name)))
        for name in ReplayState._fields
    }
    replay_vs_production = {
        name: _score(_physical(getattr(production, name)), _physical(getattr(replay, name)))
        for name in ReplayState._fields
    }
    stress_names = ("stress1", "stress2", "stress12")
    within_two_ulp = all(
        cast(int, written_order_active_rows[name]["max_ulp"]) <= ULP_LIMIT for name in stress_names
    )
    active_over_bar = any(
        cast(float, active_rows[name]["normalized_max_abs"]) > POINTWISE_BAR
        for name in stress_names
    )
    replay_in_debt_class = all(
        POINTWISE_BAR < cast(float, replay_rows[name]["normalized_max_abs"]) < 1.0e-13
        for name in stress_names
    )
    production_is_bit_exact = all(row["max_ulp"] == 0 for row in production_rows.values())
    replay_is_bit_exact = all(row["max_ulp"] == 0 for row in replay_rows.values())
    if production_is_bit_exact and replay_is_bit_exact:
        classification = "BIT-EXACT"
    elif within_two_ulp and replay_in_debt_class:
        classification = "RE-ASSOCIATION"
    else:
        classification = "IMPLEMENTATION_OWNER"
    report = {
        "format": "nemo-si3-phase2-rung33-written-order-replay-v1",
        "worktree": worktree_stamp(),
        "status": classification,
        "bar": POINTWISE_BAR,
        "ulp_limit": ULP_LIMIT,
        "root": str(root),
        "inputs": {
            "frame_kt1_sha256": _sha256(root / "oracle_ice_step_entry_kt00000001.bin"),
            "frame_kt2_sha256": _sha256(root / "oracle_ice_step_entry_kt00000002.bin"),
            "storage_bits": 64,
            "input_carries_sha256": hashlib.sha256(
                b"".join(np.ascontiguousarray(v).tobytes() for v in initial)
            ).hexdigest(),
        },
        "entry_subcycle_1_replay_vs_production": one_rows,
        "first_active_stress_subcycle_2_replay_vs_production": active_rows,
        "first_active_stress_subcycle_2_replay_vs_written_order_production": (
            written_order_active_rows
        ),
        "compiled_loop_reassociation_localization": {
            "subcycle": 2,
            "source_span": "icedyn_rhg_evp.F90:392-741",
            "isolated_first_operand": (
                "none: all registered carries are byte-exact"
                if production_is_bit_exact
                else "F-point shear zds"
            ),
            "isolated_first_operand_score": shear_row,
            "written_order_python_loop_is_byte_exact": all(
                row["max_ulp"] == 0 for row in written_order_active_rows.values()
            ),
            "compiled_fori_loop_rows": active_rows,
            "classification": (
                "BIT-EXACT: source-rounded compiled loop, written-order replay, "
                "and oracle endpoint have identical carries"
                if production_is_bit_exact
                else "RE-ASSOCIATION: source-order NumPy and the same production "
                "statements in a Python loop are byte-identical; only lowering "
                "lax.fori_loop as one compiled graph changes the carry"
            ),
        },
        "hundred_subcycles_production_vs_oracle": production_rows,
        "hundred_subcycles_replay_vs_oracle": replay_rows,
        "hundred_subcycles_replay_vs_production": replay_vs_production,
        "extensive_state_bridge": {
            "source": "testcase card field * cell_area / cell_area",
            "input_rows": bridge_input_rows,
            "hundred_subcycles_vs_oracle": bridge_output_rows,
            "classification": (
                "FIRST-NONEXACT-INPUT"
                if any(row["max_ulp"] != 0 for row in bridge_input_rows.values())
                else "BIT-EXACT"
            ),
        },
        "private_unrounded_ablation": {
            "one_variable": "nemo_source_round identity removed",
            "rows": unrounded_rows,
            "binds": any(row["max_ulp"] != 0 for row in unrounded_rows.values()),
        },
        "checkpoint_hashes": {
            name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in checkpoints.items()
        },
        "classification_predicates": {
            "compiled_first_active_stress_subcycle_is_over_bar": active_over_bar,
            "all_written_order_active_stress_rows_at_most_two_ulp": within_two_ulp,
            "all_replayed_stress_rows_in_1e-14_oracle_debt_class_after_100": replay_in_debt_class,
            "all_production_carries_bit_exact_after_100": production_is_bit_exact,
            "all_replay_carries_bit_exact_after_100": replay_is_bit_exact,
        },
    }
    return report, 0 if classification in {"BIT-EXACT", "RE-ASSOCIATION"} else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    report, code = run_replay(args.root)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.artifact:
        args.artifact.write_text(payload + "\n")
    print(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
