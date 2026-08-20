#!/usr/bin/env python
"""Propagate the MPAS 4DVar initial increment with the tangent-linear model."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.run.mpas_4dvar_single import common


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/4DVar_single/experiment.yaml"),
    )
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = common.load_config(args.config)
    output_directory = Path(config["output"]["directory"])
    sample = args.sample or Path(config["inputs"]["nmc_sample"])
    analysis_path = args.analysis or output_directory / "analysis_4dvar_state.npz"
    output_path = args.output or output_directory / "t500_tlm.npz"
    model_config = config["model"]
    background = common.state_from_sample(
        sample,
        str(config["background"]["sample_state"]),
    )
    analysis = common.state_from_analysis(analysis_path)
    mesh, sigma, model = common.make_grid_sigma_model(model_config)
    dt = float(model_config["time_step_seconds"])
    background_state = common.cell_state_to_edge_state(background, mesh)
    analysis_state = common.cell_state_to_edge_state(analysis, mesh)
    increment_state = jax.tree_util.tree_map(
        lambda analysis_value, background_value: analysis_value - background_value,
        analysis_state,
        background_state,
    )
    step = jax.checkpoint(lambda state: model.step(state, dt))
    hours = np.asarray(config["diagnostics"]["forecast_hours"], dtype=np.float64)
    checkpoints = tuple(int(round(hour * 3600.0 / dt)) for hour in hours)
    fields = []
    previous = 0
    for target in checkpoints:
        if target > previous:

            def body(carry, _):
                return jax.jvp(step, (carry[0],), (carry[1],)), None

            (background_state, increment_state), _ = jax.lax.scan(
                body,
                (background_state, increment_state),
                None,
                length=target - previous,
            )
            previous = target
        fields.append(
            jax.jvp(
                lambda state: common.t500(state, sigma),
                (background_state,),
                (increment_state,),
            )[1]
        )
    array = np.asarray(jax.device_get(jnp.stack(fields)))
    observation = config["observation"]
    observation_cell = common.nearest_cell(
        mesh,
        float(observation["latitude_degrees_north"]),
        float(observation["longitude_degrees_east"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        hours=hours,
        t500_4dvar_tangent_increment=array,
        observation_latitude_degrees=np.rad2deg(np.asarray(mesh.latCell)[observation_cell]),
        observation_longitude_degrees=np.rad2deg(np.asarray(mesh.lonCell)[observation_cell])
        % 360.0,
    )
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
