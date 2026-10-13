#!/usr/bin/env python3
"""Measure fixed-seed SWG twin divergence and the first SSH e-folding time."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

BALANCE_RATIO_BAR = 0.07
MINIMUM_FIT_POINTS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_efold_crossing(days: np.ndarray, rms: np.ndarray,
                         initial_rms: float) -> float | None:
    """Log-interpolate the first crossing of e times the initial RMS."""
    days = np.asarray(days, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)
    if days.ndim != 1 or rms.shape != days.shape or days.size < 2:
        raise ValueError("e-folding curve must have at least two paired points")
    if not np.all(np.isfinite(days)) or not np.all(np.isfinite(rms)) \
            or initial_rms <= 0.0 or np.any(rms <= 0.0):
        raise ValueError("e-folding inputs must be positive finite")
    threshold = np.e * initial_rms
    above = np.flatnonzero(rms >= threshold)
    if above.size == 0:
        return None
    hi = int(above[0])
    if hi == 0:
        return float(days[0])
    lo = hi - 1
    fraction = ((np.log(threshold) - np.log(rms[lo]))
                / (np.log(rms[hi]) - np.log(rms[lo])))
    return float(days[lo] + fraction * (days[hi] - days[lo]))


def rotational_velocity_from_vorticity(
        zeta: np.ndarray, wet: np.ndarray, spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a closed-basin Dirichlet streamfunction diagnostic."""
    zeta = np.asarray(zeta, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    if zeta.shape != wet.shape or not np.all(np.isfinite(zeta)):
        raise ValueError("vorticity and wet mask must be finite and conformable")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("rotational inversion spacing must be positive finite")
    jj, ii = np.where(wet)
    j0, j1 = int(jj.min()), int(jj.max())
    i0, i1 = int(ii.min()), int(ii.max())
    rectangle = np.zeros_like(wet)
    rectangle[j0:j1 + 1, i0:i1 + 1] = True
    if not np.array_equal(rectangle, wet):
        raise ValueError("rotational inversion requires one wet rectangle")
    rhs = zeta[j0:j1 + 1, i0:i1 + 1]
    ny, nx = rhs.shape
    ky = np.arange(1, ny + 1, dtype=np.float64)
    kx = np.arange(1, nx + 1, dtype=np.float64)
    eigenvalue = (
        2.0 * (np.cos(np.pi * ky / (ny + 1)) - 1.0)[:, None]
        + 2.0 * (np.cos(np.pi * kx / (nx + 1)) - 1.0)[None, :]
    ) / spacing_m ** 2
    psi = idstn(
        dstn(rhs, type=1, norm="ortho") / eigenvalue,
        type=1, norm="ortho")
    padded = np.pad(psi, 1)
    u_block = -(padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * spacing_m)
    v_block = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * spacing_m)
    u = np.zeros_like(zeta)
    v = np.zeros_like(zeta)
    u[j0:j1 + 1, i0:i1 + 1] = u_block
    v[j0:j1 + 1, i0:i1 + 1] = v_block
    return u, v


def balanced_fit(days: np.ndarray, zeta_rms: np.ndarray,
                 divergence_rms: np.ndarray) -> dict:
    """Fit after fast-mode contamination falls below the registered 7%."""
    days = np.asarray(days, dtype=np.float64)
    zeta = np.asarray(zeta_rms, dtype=np.float64)
    divergence = np.asarray(divergence_rms, dtype=np.float64)
    if days.ndim != 1 or zeta.shape != days.shape \
            or divergence.shape != days.shape:
        raise ValueError("balanced fit curves must be paired 1-D arrays")
    if np.any(zeta <= 0.0) or np.any(divergence < 0.0) \
            or not all(np.all(np.isfinite(x)) for x in (days, zeta, divergence)):
        raise ValueError("balanced fit curves must be finite and positive")
    ratio = divergence / zeta
    candidates = [index for index in range(days.size)
                  if days.size - index >= MINIMUM_FIT_POINTS
                  and np.all(ratio[index:] <= BALANCE_RATIO_BAR)]
    if not candidates:
        raise ValueError("no balanced fit window satisfies the registered rule")
    start = candidates[0]
    x = days[start:]
    y = np.log(zeta[start:])
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2, dtype=np.float64))
    total = float(np.sum((y - np.mean(y)) ** 2, dtype=np.float64))
    if slope <= 0.0 or total <= 0.0:
        raise ValueError("balanced fit does not have positive exponential growth")
    return {
        "rule": (
            "first lead whose divergence/vorticity RMS is <=0.07 for all "
            "remaining leads, retaining at least 20 points"),
        "balance_ratio_bar": BALANCE_RATIO_BAR,
        "minimum_points": MINIMUM_FIT_POINTS,
        "start_day": float(x[0]),
        "stop_day": float(x[-1]),
        "points": int(x.size),
        "slope_day-1": float(slope),
        "efolding_days": float(1.0 / slope),
        "r_squared": float(1.0 - residual / total),
    }


def measure_twins(control_root: Path, perturbed_root: Path) -> dict:
    """Compare two complete fine-deck daily runs from one restart."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
    )

    manifests = []
    for root in (control_root, perturbed_root):
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete" \
                or manifest.get("sampling") != "daily_snapshot" \
                or manifest.get("completed_days") != manifest.get("days"):
            raise ValueError(f"incomplete daily-snapshot twin: {root}")
        manifests.append((manifest, sha256(path)))
    control, perturbed = (item[0] for item in manifests)
    if control["days"] != perturbed["days"] \
            or control["namelist"] != perturbed["namelist"]:
        raise ValueError("twin clocks or resolved decks differ")
    expected = json.loads(json.dumps(SWG_MADDISON_NAMELIST._asdict()))
    for field in expected.keys() - {"nn_itend", "nn_stock"}:
        if control["namelist"].get(field) != expected[field]:
            raise ValueError(f"twins differ from fine deck at {field}")
    c_restart, p_restart = control.get("restart"), perturbed.get("restart")
    if c_restart is None or p_restart is None:
        raise ValueError("both twins must declare their restart")
    for field in ("path", "sha256", "source_day", "source_step"):
        if c_restart.get(field) != p_restart.get(field):
            raise ValueError(f"twin restart provenance differs at {field}")
    if c_restart.get("perturb_amplitude_m") not in (None, 0.0):
        raise ValueError("control twin is perturbed")
    perturb_kind = p_restart.get("perturb_kind", "white-ssh")
    amplitude = p_restart.get("perturb_amplitude_m")
    seed = p_restart.get("perturb_seed")
    if perturb_kind == "white-ssh":
        if not isinstance(amplitude, (int, float)) \
                or amplitude <= 0.0 or seed != 0:
            raise ValueError(
                "white-SSH twin must use positive amplitude and fixed seed 0")
    elif perturb_kind == "balanced-vorticity":
        balanced = p_restart.get("balanced_perturbation", {})
        amplitude = balanced.get("amplitude_s-1")
        if not isinstance(amplitude, (int, float)) or amplitude <= 0.0:
            raise ValueError("balanced twin must declare a positive amplitude")
    else:
        raise ValueError(f"unsupported perturbation kind {perturb_kind!r}")
    card = build_swg_card(SWG_MADDISON_NAMELIST)
    tmask = np.asarray(card.recipe.land_mask, dtype=bool)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    umask = np.asarray(u_full[:, 1:], dtype=bool)
    vmask = np.asarray(v_full[1:, :], dtype=bool)
    if perturb_kind == "white-ssh":
        noise = np.random.default_rng(seed).standard_normal(tmask.shape)
        initial_ssh_rms = float(amplitude * np.sqrt(
            np.mean(noise[tmask] ** 2, dtype=np.float64)))
    else:
        initial_ssh_rms = 0.0
    days = int(control["days"])
    lead = np.arange(days + 1, dtype=np.float64)
    ssh_rms = np.empty(days + 1, dtype=np.float64)
    u_rms = np.empty(days + 1, dtype=np.float64)
    v_rms = np.empty(days + 1, dtype=np.float64)
    ssh_rms[0], u_rms[0], v_rms[0] = initial_ssh_rms, 0.0, 0.0
    zeta_rms = np.zeros(days + 1, dtype=np.float64)
    divergence_rms = np.zeros(days + 1, dtype=np.float64)
    rotational_velocity_rms = np.zeros(days + 1, dtype=np.float64)
    spacing_m = float(SWG_MADDISON_NAMELIST.rn_dx_m
                      / SWG_MADDISON_NAMELIST.nn_SWG)
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        restricted_relative_vorticity,
    )
    geometry = cgrid_geometry(SWG_MADDISON_NAMELIST)
    for day in range(1, days + 1):
        name = f"day{day:04d}.npz"
        paths = (control_root / name, perturbed_root / name)
        for manifest, path in zip((control, perturbed), paths, strict=True):
            if sha256(path) != manifest["daily_sha256"].get(name):
                raise ValueError(f"twin daily digest mismatch: {path}")
        with np.load(paths[0], allow_pickle=False) as left, \
                np.load(paths[1], allow_pickle=False) as right:
            delta_ssh = np.asarray(right["ssh_m"] - left["ssh_m"])
            delta_u = np.asarray(right["u_m_s"] - left["u_m_s"])
            delta_v = np.asarray(right["v_m_s"] - left["v_m_s"])
        if not all(np.all(np.isfinite(value))
                   for value in (delta_ssh, delta_u, delta_v)):
            raise ValueError(f"non-finite twin difference at lead day {day}")
        ssh_rms[day] = np.sqrt(np.mean(delta_ssh[tmask] ** 2, dtype=np.float64))
        u_rms[day] = np.sqrt(np.mean(delta_u[umask] ** 2, dtype=np.float64))
        v_rms[day] = np.sqrt(np.mean(delta_v[vmask] ** 2, dtype=np.float64))
        delta_zeta = restricted_relative_vorticity(
            {"T": delta_ssh, "U": delta_u, "V": delta_v}, geometry)
        delta_divergence = (
            delta_u - np.roll(delta_u, 1, axis=1)
            + delta_v - np.roll(delta_v, 1, axis=0)
        ) / spacing_m
        zeta_rms[day] = np.sqrt(np.mean(
            delta_zeta[tmask] ** 2, dtype=np.float64))
        divergence_rms[day] = np.sqrt(np.mean(
            delta_divergence[tmask] ** 2, dtype=np.float64))
        rotational_u, rotational_v = rotational_velocity_from_vorticity(
            delta_zeta, tmask, spacing_m)
        rotational_velocity_rms[day] = np.sqrt(np.mean(
            rotational_u[tmask] ** 2 + rotational_v[tmask] ** 2,
            dtype=np.float64))
    crossing = first_efold_crossing(lead, ssh_rms, initial_ssh_rms)
    if crossing is None:
        fit_efold = None
    else:
        fit_stop = max(2, int(np.ceil(crossing)) + 1)
        slope = np.polyfit(lead[:fit_stop], np.log(ssh_rms[:fit_stop]), 1)[0]
        fit_efold = float(1.0 / slope) if slope > 0.0 else None
    fit = balanced_fit(lead[1:], zeta_rms[1:], divergence_rms[1:])
    return {
        "schema": "swg-online-predictability-v2",
        "control_root": str(control_root.resolve()),
        "perturbed_root": str(perturbed_root.resolve()),
        "control_manifest_sha256": manifests[0][1],
        "perturbed_manifest_sha256": manifests[1][1],
        "source_restart_sha256": c_restart["sha256"],
        "perturb_kind": perturb_kind,
        "perturb_amplitude": float(amplitude),
        "perturb_amplitude_units": (
            "m" if perturb_kind == "white-ssh" else "s-1"),
        "perturb_seed": seed,
        "lead_days": lead.tolist(),
        "ssh_rms_m": ssh_rms.tolist(),
        "u_rms_m_s": u_rms.tolist(),
        "v_rms_m_s": v_rms.tolist(),
        "relative_vorticity_rms_s-1": zeta_rms.tolist(),
        "divergence_rms_s-1": divergence_rms.tolist(),
        "rotational_velocity_rms_m_s": rotational_velocity_rms.tolist(),
        "balanced_exponential_fit": fit,
        "first_ssh_efold_crossing_days": crossing,
        "early_log_linear_ssh_efold_days": fit_efold,
        "saturation_reached": None,
        "saturation_assessment": "not inferred from a partial deterministic twin",
    }


def forward_tangent_probe(restart: Path, *, steps: int) -> dict:
    """JVP of the fine production solver along the balanced perturbation."""
    import importlib.util

    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
        swg_surface_wind,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    if type(steps) is not int or steps <= 0:
        raise ValueError("tangent steps must be a positive integer")
    driver_path = Path(__file__).resolve().parents[3] / "run/run_swg_double_gyre.py"
    spec = importlib.util.spec_from_file_location("swg_double_gyre_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    admission = driver.admit_listed_snapshot(restart)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if not bool(jax.config.jax_enable_x64) or jax.default_backend() != "cpu":
        raise RuntimeError("tangent probe requires CPU fp64")
    card = build_swg_card(SWG_MADDISON_NAMELIST)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    umask = np.asarray(u_full[:, 1:], dtype=bool)
    vmask = np.asarray(v_full[1:, :], dtype=bool)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(SWG_MADDISON_NAMELIST)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
        tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
    state, restart_provenance = driver.restart_state_from_npz(
        card.recipe.initial_state, forcing, restart, dt_s=card.dt_s)
    du, dv, direction_provenance = driver.balanced_vorticity_direction(
        state, SWG_MADDISON_NAMELIST)
    du, dv = jnp.asarray(du), jnp.asarray(dv)

    def integrate(amplitude):
        perturbed = state._replace(
            u=state.u.replace(data=state.u.data + amplitude * du[..., None]),
            v=state.v.replace(data=state.v.data + amplitude * dv[..., None]))

        def body(carry, _):
            return model.step(
                carry, card.dt_s, surface_forcing=forcing), None

        final, _ = jax.lax.scan(body, perturbed, None, length=steps)
        return final.eta.data, final.u.data[:, 1:, 0], final.v.data[1:, :, 0]

    primal, tangent = jax.jvp(
        jax.jit(integrate), (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    jax.block_until_ready(tangent[0])
    wet = np.asarray(card.recipe.land_mask, dtype=bool)
    tangent_np = tuple(np.asarray(value) for value in tangent)
    if not all(np.all(np.isfinite(value)) for value in tangent_np):
        raise RuntimeError("forward tangent is non-finite")
    return {
        "schema": "swg-online-balanced-forward-tangent-v1",
        "restart": restart_provenance | {"manifest_admission": admission},
        "steps": steps,
        "lead_seconds": float(steps * card.dt_s),
        "direction": direction_provenance,
        "primal_all_finite": bool(all(np.all(np.isfinite(np.asarray(value)))
                                          for value in primal)),
        "tangent_ssh_rms_m_per_s-1": float(np.sqrt(np.mean(
            tangent_np[0][wet] ** 2, dtype=np.float64))),
        "tangent_u_rms_m_s_per_s-1": float(np.sqrt(np.mean(
            tangent_np[1][umask] ** 2, dtype=np.float64))),
        "tangent_v_rms_m_s_per_s-1": float(np.sqrt(np.mean(
            tangent_np[2][vmask] ** 2, dtype=np.float64))),
        "tangent_nonzero": bool(any(np.any(value != 0.0) for value in tangent_np)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--perturbed-root", type=Path)
    parser.add_argument("--tangent-restart", type=Path)
    parser.add_argument("--tangent-steps", type=int, default=1)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        tangent_mode = args.tangent_restart is not None
        twin_mode = args.control_root is not None or args.perturbed_root is not None
        if tangent_mode == twin_mode:
            raise ValueError("select exactly one complete twin or tangent probe")
        if tangent_mode:
            report = forward_tangent_probe(
                args.tangent_restart, steps=args.tangent_steps)
        else:
            if args.control_root is None or args.perturbed_root is None:
                raise ValueError("both twin roots are required")
            report = measure_twins(args.control_root, args.perturbed_root)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    if tangent_mode:
        print(f"PASS tangent_steps={report['steps']} nonzero={report['tangent_nonzero']}")
    else:
        print(
            f"PASS leads={len(report['lead_days']) - 1} "
            f"balanced_efold_days="
            f"{report['balanced_exponential_fit']['efolding_days']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
