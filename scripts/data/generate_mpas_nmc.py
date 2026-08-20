#!/usr/bin/env python
"""Generate full-state, same-verification-time MPAS NMC samples.

Model integrations must be run inside a batch allocation. Each sample stores
a 48-hour forecast, a 24-hour forecast valid at the same time, and their
difference for static background-error estimation.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import jax
import numpy as np
import yaml
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    MPASPrimitiveEquationModel,
)
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.state import HydrostaticState
from legoesm.driver.component_factory import compute_diffusion
from legoesm.driver.config import DycoreConfig
from legoesm.grids.factory import create_grid
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels
from legoesm.grids.voronoi import reconstruct_cell_velocity
from legoesm.training.era5_to_state import era5_to_mpas_carry
from legoesm.training.rda_era5 import load_rda_era5_slice

set_policy(PrecisionPolicy.fp64())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/4DVar_single/nmc.yaml"),
    )
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for section in ("era5", "model", "nmc"):
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    return config


def _state_from_carry(carry) -> HydrostaticState:
    return HydrostaticState(
        u=Field(carry.u, name="u_edge", dims=("edge", "level"), units="m/s"),
        v=None,
        T=Field(carry.T, name="T", dims=("cell", "level"), units="K"),
        p_s=Field(carry.p_s, name="p_s", dims=("cell",), units="Pa"),
        phis=Field(carry.phis, name="phis", dims=("cell",), units="m2/s2"),
        tracers={
            "q_v": Field(
                carry.q_v,
                name="q_v",
                dims=("cell", "level"),
                units="kg/kg",
            )
        },
    )


def _make_model(config: dict):
    mesh = create_grid(
        "mpas",
        int(config["resolution"]),
        lloyd_iterations=int(config["lloyd_iterations"]),
    )
    levels = int(config["levels"])
    if config["vertical_coordinate"] == "hybrid":
        sigma = make_hybrid_levels(levels, p_top_Pa=200.0, stretching=2.0)
    else:
        sigma = create_sigma_coordinate(levels)
    dycore = DycoreConfig(
        model_type="hydrostatic",
        discretization="mpas",
        dt=float(config["time_step_seconds"]),
        hyperdiff_scale=float(config["hyperdiffusion_scale"]),
        a_h_scale=float(config["horizontal_diffusion_scale"]),
        time_integrator="auto",
        mpas_nu_vert4_T=float(config["vertical_temperature_diffusion"]),
    )
    diffusion = compute_diffusion(mesh, dycore)
    model_config = MPASPrimitiveEquationConfig(
        nu_del2=diffusion.A_h,
        nu_del4=diffusion.hyperdiff,
        nu_del4_ps=diffusion.hyperdiff,
        K_h=diffusion.K_h_A,
        fix_mass=dycore.fix_mass and dycore.conservation_fixer,
        anchor_mass_to_initial=dycore.fix_mass and dycore.conservation_fixer,
        time_integrator=MPASPrimitiveEquationConfig().time_integrator,
        nu_vert4_T=dycore.mpas_nu_vert4_T,
    )
    return mesh, sigma, MPASPrimitiveEquationModel(mesh, sigma, model_config)


def _forecast(model, state: HydrostaticState, hours: int, dt: float):
    step = jax.jit(lambda value: model.step(value, dt))
    total_steps = int(round(hours * 3600.0 / dt))
    report_steps = max(1, int(round(6 * 3600.0 / dt)))
    for index in range(total_steps):
        state = step(state)
        if (index + 1) % report_steps == 0 or index + 1 == total_steps:
            jax.block_until_ready(state.T.data)
            elapsed = (index + 1) * dt / 3600.0
            print(f"forecast progress {elapsed:.1f}/{hours} h", flush=True)
    return state


def _cell_arrays(state: HydrostaticState, mesh) -> dict[str, np.ndarray]:
    u_cell, v_cell = reconstruct_cell_velocity(state.u.data, mesh)
    return {
        "u": np.asarray(jax.device_get(u_cell)),
        "v": np.asarray(jax.device_get(v_cell)),
        "T": np.asarray(jax.device_get(state.T.data)),
        "p_s": np.asarray(jax.device_get(state.p_s.data)),
        "phis": np.asarray(jax.device_get(state.phis.data)),
        "tracer_q_v": np.asarray(jax.device_get(state.tracers["q_v"].data)),
    }


def _save_sample(
    path: Path,
    verification: datetime,
    init_long: datetime,
    init_short: datetime,
    forecast_long: HydrostaticState,
    forecast_short: HydrostaticState,
    mesh,
) -> None:
    long_state = _cell_arrays(forecast_long, mesh)
    short_state = _cell_arrays(forecast_short, mesh)
    output: dict[str, np.ndarray] = {
        "verify_time": np.asarray(verification.isoformat()),
        "init_48h": np.asarray(init_long.isoformat()),
        "init_24h": np.asarray(init_short.isoformat()),
        "lat_cell": np.asarray(mesh.latCell),
        "lon_cell": np.asarray(mesh.lonCell),
    }
    for name in long_state:
        output[f"f48_{name}"] = long_state[name]
        output[f"f24_{name}"] = short_state[name]
        output[f"error_{name}"] = long_state[name] - short_state[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    print(f"saved {path}", flush=True)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    model_config = config["model"]
    nmc_config = config["nmc"]
    era5_root = os.environ.get("LEGOESM_ERA5_RDA_ROOT", config["era5"]["root"])
    first_time = datetime.fromisoformat(str(nmc_config["first_verification_time"]))
    total_count = int(nmc_config["sample_count"])
    count = total_count - args.case_offset if args.count is None else args.count
    if args.case_offset < 0 or count <= 0 or args.case_offset + count > total_count:
        raise ValueError("Requested case range is outside the configured sample range")
    output_directory = Path(nmc_config["output_directory"])
    mesh, sigma, model = _make_model(model_config)
    dt = float(model_config["time_step_seconds"])
    smoothing = int(model_config["smoothing_passes"])
    long_hours = int(nmc_config["long_forecast_hours"])
    short_hours = int(nmc_config["short_forecast_hours"])
    interval = int(nmc_config["interval_days"])

    def initial_state(time: datetime) -> HydrostaticState:
        era5 = load_rda_era5_slice(era5_root, time)
        carry = era5_to_mpas_carry(era5, mesh, sigma, smoothing_passes=smoothing)
        return _state_from_carry(carry)

    print(f"backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    for local_index in range(count):
        case_index = args.case_offset + local_index
        verification = first_time + timedelta(days=interval * case_index)
        init_long = verification - timedelta(hours=long_hours)
        init_short = verification - timedelta(hours=short_hours)
        path = (
            output_directory
            / "samples"
            / (f"mpas_nmc_case_{case_index + 1:03d}_{verification:%Y%m%d%H%M%S}.npz")
        )
        if path.exists() and not args.force:
            print(f"reuse {path}", flush=True)
            continue
        print(
            f"case={case_index + 1:03d} valid={verification.isoformat()} "
            f"init48={init_long.isoformat()} init24={init_short.isoformat()}",
            flush=True,
        )
        forecast_long = _forecast(model, initial_state(init_long), long_hours, dt)
        forecast_short = _forecast(model, initial_state(init_short), short_hours, dt)
        _save_sample(
            path,
            verification,
            init_long,
            init_short,
            forecast_long,
            forecast_short,
            mesh,
        )


if __name__ == "__main__":
    main()
